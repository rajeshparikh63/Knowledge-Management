"""
EntityVectorCache — per-organization FAISS index for entity-name embeddings.

Purpose
-------
Entity resolution during ingest needs to compare new entity names against every
existing canonical entity in the org's graph. Re-embedding 5k+ existing names
on every ingest is wasteful (and used to blow OpenAI's 2048-input cap). This
module caches each entity's embedding the first time we see it, so future
ingests only embed the truly-new names.

Storage
-------
One directory per org under settings.ENTITY_CACHE_DIR:

    {ENTITY_CACHE_DIR}/{org_id}/
        index.faiss   — IndexFlatIP over L2-normalized embeddings (cosine sim)
        names.json    — ordered list of canonical names; row i ↔ index vector i

Why FAISS IndexFlatIP
---------------------
Inner product on L2-normalized vectors == cosine similarity. We expose the
existing cosine-distance API (`distance = 1 - similarity`) so the rest of the
codebase doesn't change.

Concurrency
-----------
One asyncio.Lock per org guards reads/writes. The cache is purely an
in-process accelerator — multiple workers can race, but the worst case is a
duplicate embedding write (harmless, just consumes a bit of disk).

Atomicity
---------
Saves write to `.tmp` and rename, so a crash mid-save leaves the previous
known-good state intact.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silence "OMP: Error #15 — libomp already initialized" on macOS, which fires
# when faiss-cpu and numpy/torch each bring their own OpenMP runtime. The
# workaround is documented by Intel/LLVM; the practical impact is nil for
# read-mostly FAISS workloads. Must be set BEFORE faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

from app.logger import logger
from app.settings import settings


def _cache_dir_for(org_id: str) -> Path:
    return Path(settings.ENTITY_CACHE_DIR) / org_id


def _index_path(org_id: str) -> Path:
    return _cache_dir_for(org_id) / "index.faiss"


def _names_path(org_id: str) -> Path:
    return _cache_dir_for(org_id) / "names.json"


class _OrgCache:
    """In-memory state for one org's FAISS index + name list."""

    __slots__ = ("org_id", "index", "names", "name_to_pos", "lock", "_dirty")

    def __init__(self, org_id: str, dim: int):
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed")
        self.org_id = org_id
        # IndexFlatIP on L2-normalized vectors → cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.names: List[str] = []
        self.name_to_pos: Dict[str, int] = {}
        self.lock = asyncio.Lock()
        self._dirty = False

    def add(self, names: List[str], vectors: np.ndarray) -> int:
        """Append (name, vector) pairs that aren't already cached.

        Returns: number of new vectors actually added.
        """
        if not names:
            return 0
        # Filter out any names already cached
        new_names: List[str] = []
        new_vecs: List[np.ndarray] = []
        for nm, v in zip(names, vectors):
            if nm in self.name_to_pos:
                continue
            new_names.append(nm)
            new_vecs.append(v)
        if not new_names:
            return 0

        arr = np.asarray(new_vecs, dtype=np.float32)
        faiss.normalize_L2(arr)
        self.index.add(arr)
        for nm in new_names:
            self.name_to_pos[nm] = len(self.names)
            self.names.append(nm)
        self._dirty = True
        return len(new_names)

    def get_vectors(self, names: List[str]) -> Dict[str, np.ndarray]:
        """Return cached vectors for any of `names` that exist in the index."""
        if self.index.ntotal == 0:
            return {}
        out: Dict[str, np.ndarray] = {}
        # IndexFlatIP exposes reconstruct() so we can read the stored (normalized) vec back
        for nm in names:
            pos = self.name_to_pos.get(nm)
            if pos is None:
                continue
            try:
                out[nm] = self.index.reconstruct(int(pos))
            except RuntimeError:
                # Index out of bounds — shouldn't happen but be defensive
                continue
        return out

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Search the index for nearest neighbors of a single query.

        Returns: list of (name, cosine_distance) tuples, distance = 1 - similarity.
        """
        if self.index.ntotal == 0:
            return []
        q = np.asarray([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, min(top_k, self.index.ntotal))
        out: List[Tuple[str, float]] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.names[int(idx)], float(1.0 - sim)))
        return out

    def batch_search(
        self,
        query_vecs: np.ndarray,
        top_k: int = 1,
    ) -> List[List[Tuple[str, float]]]:
        """Search N queries in one FAISS call.

        Returns: list aligned with query_vecs; each element is a list of
        (name, cosine_distance) tuples for that query's top-k neighbors.
        """
        if self.index.ntotal == 0 or len(query_vecs) == 0:
            return [[] for _ in query_vecs]
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        sims, idxs = self.index.search(q, k)
        results: List[List[Tuple[str, float]]] = []
        for row_sims, row_idxs in zip(sims, idxs):
            row: List[Tuple[str, float]] = []
            for sim, idx in zip(row_sims, row_idxs):
                if idx == -1:
                    continue
                row.append((self.names[int(idx)], float(1.0 - sim)))
            results.append(row)
        return results

    def known_names(self) -> set:
        return set(self.name_to_pos.keys())


class EntityVectorCache:
    """Process-wide registry of per-org caches."""

    def __init__(self, dim: int = None):
        self.dim = dim or settings.EMBEDDING_DIM
        self._orgs: Dict[str, _OrgCache] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_or_load(self, org_id: str) -> _OrgCache:
        if org_id in self._orgs:
            return self._orgs[org_id]

        async with self._registry_lock:
            # Double-check after acquiring lock
            if org_id in self._orgs:
                return self._orgs[org_id]

            cache = _OrgCache(org_id, self.dim)
            await asyncio.to_thread(self._load_from_disk, cache)
            self._orgs[org_id] = cache
            return cache

    def _load_from_disk(self, cache: _OrgCache) -> None:
        idx_path = _index_path(cache.org_id)
        names_path = _names_path(cache.org_id)
        if not idx_path.exists() or not names_path.exists():
            return
        try:
            cache.index = faiss.read_index(str(idx_path))
            with open(names_path, "r") as f:
                data = json.load(f)
            cache.names = list(data.get("names", []))
            cache.name_to_pos = {nm: i for i, nm in enumerate(cache.names)}
            if cache.index.d != self.dim:
                logger.warning(
                    f"Entity cache dim mismatch for org={cache.org_id}: "
                    f"index has {cache.index.d}, expected {self.dim}. Resetting."
                )
                cache.index = faiss.IndexFlatIP(self.dim)
                cache.names = []
                cache.name_to_pos = {}
            if cache.index.ntotal != len(cache.names):
                logger.warning(
                    f"Entity cache count mismatch for org={cache.org_id}: "
                    f"index has {cache.index.ntotal} vectors, "
                    f"names file has {len(cache.names)}. Resetting."
                )
                cache.index = faiss.IndexFlatIP(self.dim)
                cache.names = []
                cache.name_to_pos = {}
            else:
                logger.info(
                    f"Loaded entity cache for org={cache.org_id}: "
                    f"{cache.index.ntotal} vectors"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load entity cache for org={cache.org_id}: {e}; "
                f"starting fresh"
            )
            cache.index = faiss.IndexFlatIP(self.dim)
            cache.names = []
            cache.name_to_pos = {}

    def _save_to_disk(self, cache: _OrgCache) -> None:
        if not cache._dirty:
            return
        cache_dir = _cache_dir_for(cache.org_id)
        os.makedirs(cache_dir, exist_ok=True)

        idx_path = _index_path(cache.org_id)
        names_path = _names_path(cache.org_id)
        tmp_idx = idx_path.with_suffix(".faiss.tmp")
        tmp_names = names_path.with_suffix(".json.tmp")

        faiss.write_index(cache.index, str(tmp_idx))
        with open(tmp_names, "w") as f:
            json.dump({"names": cache.names}, f)
        # Atomic rename
        os.replace(tmp_idx, idx_path)
        os.replace(tmp_names, names_path)
        cache._dirty = False
        logger.debug(
            f"Saved entity cache for org={cache.org_id}: "
            f"{cache.index.ntotal} vectors"
        )

    async def get_cached_vectors(
        self, org_id: str, names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Return {name: vector} for any of `names` already in the org's cache."""
        cache = await self._get_or_load(org_id)
        async with cache.lock:
            return cache.get_vectors(names)

    async def known_names(self, org_id: str) -> set:
        cache = await self._get_or_load(org_id)
        async with cache.lock:
            return cache.known_names()

    async def add_vectors(
        self,
        org_id: str,
        names: List[str],
        vectors: List[List[float]],
    ) -> int:
        """Cache new (name, vector) pairs and persist to disk.

        Vectors must be aligned with names. Returns count of NEW vectors added
        (already-cached names are skipped).
        """
        if not names:
            return 0
        cache = await self._get_or_load(org_id)
        arr = np.asarray(vectors, dtype=np.float32)
        async with cache.lock:
            added = cache.add(names, arr)
            if added > 0:
                await asyncio.to_thread(self._save_to_disk, cache)
            return added

    async def nearest(
        self,
        org_id: str,
        query_vec: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Search the org's cache for nearest entity names to query_vec."""
        cache = await self._get_or_load(org_id)
        async with cache.lock:
            return cache.search(query_vec, top_k=top_k)

    async def batch_nearest(
        self,
        org_id: str,
        query_vecs: np.ndarray,
        top_k: int = 1,
    ) -> List[List[Tuple[str, float]]]:
        """Search N query vectors against the org's cache in one FAISS call.

        Far cheaper than calling nearest() in a Python loop when N is large —
        FAISS does the entire SIMD-accelerated scan once.
        """
        cache = await self._get_or_load(org_id)
        async with cache.lock:
            return cache.batch_search(query_vecs, top_k=top_k)

    async def stats(self, org_id: str) -> Dict[str, int]:
        cache = await self._get_or_load(org_id)
        async with cache.lock:
            return {
                "vectors": int(cache.index.ntotal),
                "names": len(cache.names),
            }


# Singleton
_singleton: Optional[EntityVectorCache] = None


def get_entity_vector_cache() -> EntityVectorCache:
    global _singleton
    if _singleton is None:
        _singleton = EntityVectorCache()
    return _singleton
