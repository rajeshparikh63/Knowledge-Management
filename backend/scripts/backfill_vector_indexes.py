#!/usr/bin/env python3
"""
Backfill the Chunk.embedding VECTOR index on existing FalkorDB graphs.

Why this exists
---------------
`_ensure_indexes` used to call `db.idx.vector.createNodeIndex`, a procedure that
is NOT registered in the deployed FalkorDB. The call failed silently (wrapped in
`except: pass`), so no graph ever got a vector index and every `search()` ran a
full scan. `_ensure_indexes` is now fixed (DDL `CREATE VECTOR INDEX`), which
creates the index on each org's NEXT ingest — but graphs that are not
re-ingested keep scanning. This script creates the index on all existing graphs
immediately.

Safe by design:
  * Creating a vector index is additive and non-destructive.
  * Idempotent — graphs that already have the index are skipped.
  * Only touches graphs that actually contain Chunk nodes with embeddings
    (detected, not guessed from the name), so non-KG graphs
    (collecct_network*, sharepoint_structure*) and empty graphs are left alone.
  * DRY RUN by default. Pass --apply to actually create indexes.

Usage:
    uv run python scripts/backfill_vector_indexes.py            # dry run (report only)
    uv run python scripts/backfill_vector_indexes.py --apply    # create the indexes
    uv run python scripts/backfill_vector_indexes.py --apply --verify   # + probe each
    uv run python scripts/backfill_vector_indexes.py --only org_abc,org_def
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import falkordb

sys.path.insert(0, ".")
from app.settings import settings  # noqa: E402
from app.logger import logger  # noqa: E402


def _connect() -> falkordb.FalkorDB:
    return falkordb.FalkorDB(
        host=settings.GRAPH_DATABASE_URL,
        port=settings.GRAPH_DATABASE_PORT,
        username=settings.GRAPH_DATABASE_USERNAME or None,
        password=settings.GRAPH_DATABASE_PASSWORD or None,
        ssl=settings.GRAPH_DATABASE_SSL,
    )


def _has_vector_index(g) -> bool:
    """True if graph g already has a VECTOR index on Chunk.embedding."""
    try:
        rows = g.query("CALL db.indexes()").result_set
    except Exception:
        return False
    for row in rows:
        # row shape: [label, [fields...], {field: [types...]}, ...]
        label = row[0]
        types_map = row[2] if len(row) > 2 else {}
        if label == "Chunk" and isinstance(types_map, dict):
            for field, kinds in types_map.items():
                if field == "embedding" and "VECTOR" in list(kinds or []):
                    return True
    return False


def _sample_embedding_dim(g) -> Optional[int]:
    """Return the length of a stored Chunk.embedding, or None if no embeddings."""
    try:
        rows = g.query(
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
            "RETURN c.embedding LIMIT 1"
        ).result_set
    except Exception as e:
        logger.warning(f"dim probe failed: {e}")
        return None
    if not rows:
        return None
    vec = rows[0][0]
    try:
        return len(vec)
    except TypeError:
        return None


def _chunk_count(g) -> int:
    try:
        rows = g.query("MATCH (c:Chunk) RETURN count(c)").result_set
        return int(rows[0][0]) if rows else 0
    except Exception:
        return 0


def _create_vector_index(g, dim: int) -> None:
    g.query(
        f"CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{dimension: {int(dim)}, similarityFunction: 'cosine'}}"
    )


def _index_status(g) -> Optional[str]:
    """OPERATIONAL / UNDER CONSTRUCTION / etc. for the Chunk.embedding index."""
    try:
        for row in g.query("CALL db.indexes()").result_set:
            types_map = row[2] if len(row) > 2 else {}
            if (row[0] == "Chunk" and isinstance(types_map, dict)
                    and "VECTOR" in list(types_map.get("embedding") or [])):
                return row[7] if len(row) > 7 else None
    except Exception:
        pass
    return None


def _verify(g, dim: int) -> Tuple[bool, str]:
    """Probe the index. Large graphs index asynchronously, so a probe right
    after CREATE can fail while the index is still building — report that
    honestly rather than as a failure."""
    status = _index_status(g)
    if status and status != "OPERATIONAL":
        # Not ready yet — the CREATE succeeded, the build is in progress.
        return True, f"index status={status} (still building; will be usable shortly)"
    try:
        row = g.query(
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.embedding LIMIT 1"
        ).result_set
        if not row:
            return False, "no embedding to probe with"
        v = row[0][0]
        probe = g.query(
            "CALL db.idx.vector.queryNodes('Chunk','embedding',1,vecf32($v)) "
            "YIELD node, score RETURN node.id, score",
            {"v": v},
        ).result_set
        if probe:
            return True, f"probe ok (score={probe[0][1]:.4f})"
        # Index exists but empty result — likely mid-build.
        return True, f"index status={status or '?'} (building; probe returned no rows yet)"
    except Exception as e:
        if status and status != "OPERATIONAL":
            return True, f"index status={status} (building; probe not ready: {str(e)[:50]})"
        return False, f"probe failed: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Actually create indexes (default is a dry run)")
    ap.add_argument("--verify", action="store_true",
                    help="After creating, run a queryNodes probe on each graph")
    ap.add_argument("--only", help="Comma-separated graph names to restrict to")
    args = ap.parse_args()

    db = _connect()
    all_graphs: List[str] = db.list_graphs()
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\n{'=' * 70}\n  Vector index backfill — {mode}\n"
          f"  FalkorDB: {settings.GRAPH_DATABASE_URL}:{settings.GRAPH_DATABASE_PORT}\n"
          f"  {len(all_graphs)} graphs total\n{'=' * 70}\n")

    created, already, skipped_empty, failed, verified_ok = [], [], [], [], []

    for name in sorted(all_graphs):
        if only and name not in only:
            continue
        g = db.select_graph(name)

        n_chunks = _chunk_count(g)
        if n_chunks == 0:
            skipped_empty.append(name)
            continue  # not a KG graph, or empty — nothing to index

        if _has_vector_index(g):
            already.append(name)
            print(f"  ✓ {name:<52} already indexed ({n_chunks} chunks)")
            continue

        dim = _sample_embedding_dim(g)
        if not dim:
            skipped_empty.append(name)
            print(f"  · {name:<52} has {n_chunks} chunks but no embeddings — skip")
            continue

        if not args.apply:
            print(f"  → {name:<52} WOULD create index (dim={dim}, {n_chunks} chunks)")
            created.append(name)  # would-create, for the dry-run tally
            continue

        try:
            _create_vector_index(g, dim)
            msg = f"created index (dim={dim}, {n_chunks} chunks)"
            if args.verify:
                ok, vmsg = _verify(g, dim)
                msg += f" | {vmsg}"
                if ok:
                    verified_ok.append(name)
            print(f"  ✅ {name:<52} {msg}")
            created.append(name)
        except Exception as e:
            emsg = str(e).lower()
            if "already" in emsg or "exist" in emsg:
                already.append(name)
                print(f"  ✓ {name:<52} already indexed (race)")
            else:
                failed.append((name, str(e)[:100]))
                print(f"  ❌ {name:<52} FAILED: {str(e)[:80]}")

    verb = "would create" if not args.apply else "created"
    print(f"\n{'=' * 70}\n  Summary\n{'=' * 70}")
    print(f"  {verb:<16} {len(created)}")
    print(f"  already indexed  {len(already)}")
    print(f"  skipped (no KG)  {len(skipped_empty)}")
    if args.verify:
        print(f"  verified probe   {len(verified_ok)}/{len(created)}")
    print(f"  failed           {len(failed)}")
    for n, e in failed:
        print(f"      {n}: {e}")

    if not args.apply and created:
        print(f"\n  Dry run only. Re-run with --apply to create these "
              f"{len(created)} indexes.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
