"""
FalkorDB graph client — owns ingestion and retrieval.

Ingestion pipeline (called per document):
  1. Chunk text via Chonkie (SemanticChunker)
  2. Embed + LLM-extract entities/triples per chunk (parallel)
  3. Resolve entity names via normalize + cosine similarity (Option B)
  4. Bulk-write to FalkorDB: Chunk → MENTIONS → Entity → RELATES → Entity
  5. Stamp document_id on every chunk at write time (no provenance workaround needed)

Retrieval pipeline (4 phases, pre-filtered by document_id):
  1. Embed query
  2. MATCH chunks WHERE document_id IN $docs + vec.cosineDistance → top K
  3. Entity expansion via MENTIONS (same doc filter)
  4. RELATES triples from chunk set (confidence >= threshold)

Dynamic ontology (per org, in-memory, merged per doc):
  - First doc: LLM proposes entity types + relation types from sample text
  - Each subsequent doc: LLM proposes for that doc, merged into org's running schema
  - Shapes the extraction prompt so entities stay consistent across docs in the org
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Optional async progress reporter passed by callers (e.g. ingestion_service)
# Signature: stage_key, human-readable description, current, total
#   current/total may be None for stages without a count.
ProgressCallback = Callable[
    [str, str, Optional[int], Optional[int]], Awaitable[None]
]

import numpy as np
from openai import AsyncOpenAI
import pydantic

from app.logger import logger
from app.settings import settings
from clients.chunker_client import get_chunker_client
from clients.entity_vector_cache import get_entity_vector_cache

try:
    import falkordb
except ImportError:  # pragma: no cover
    falkordb = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_name(organization_id: str) -> str:
    return f"org_{organization_id}"


def _norm(s: str) -> str:
    return s.strip().lower()


def _chunk_id(document_id: str, idx: int) -> str:
    return f"{document_id}#{idx}"


def _triple_id(subj: str, pred: str, obj: str, chunk_id: str) -> str:
    return hashlib.sha1(f"{subj}|{pred}|{obj}|{chunk_id}".encode()).hexdigest()[:16]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


_JSON_DECODER = json.JSONDecoder()


def _parse_first_json(text: str) -> Any:
    """Parse the first valid JSON object/array in `text`, ignoring trailing junk.

    The LLM sometimes returns a valid JSON object followed by extra text
    (a comment, a second object, markdown commentary). `json.loads` rejects
    the whole response → we lose that chunk's entities. raw_decode reads
    only the first JSON value and returns the byte offset where it ended,
    so any trailing garbage is harmless.

    Also strips ``` fences first and skips any leading whitespace/text
    up to the first `{` or `[`.
    """
    cleaned = _strip_fences(text)
    # Find the first opening brace/bracket — handles "Here's the JSON:\n{...}"
    start = -1
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            start = i
            break
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", cleaned, 0)
    obj, _ = _JSON_DECODER.raw_decode(cleaned[start:])
    return obj


# ---------------------------------------------------------------------------
# Extraction schemas
# ---------------------------------------------------------------------------

class ExtractedEntity(pydantic.BaseModel):
    name: str
    type: str


class ExtractedTriple(pydantic.BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = pydantic.Field(ge=0.0, le=1.0)


class Extraction(pydantic.BaseModel):
    entities: List[ExtractedEntity] = pydantic.Field(default_factory=list)
    triples: List[ExtractedTriple] = pydantic.Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ontology prompts
# ---------------------------------------------------------------------------

_ONTOLOGY_PROMPT = """\
You are designing a knowledge-graph ontology for a corpus.

Read the SAMPLE TEXT and propose a compact ontology that captures the kinds of
things and relationships present. Aim for 6–12 entity types and 6–14 relation
types.

Rules:
- Entity labels: PascalCase, singular, no spaces, length >= 3.
- Relation labels: SCREAMING_SNAKE_CASE, verb phrase, length >= 3.
- Every relation MUST list at least one (source_label, target_label) pair from
  your entity labels.
- No duplicates.

Return ONLY valid JSON — no prose, no fences:
{{"entities": [{{"label": "...", "description": "..."}}, ...],
  "relations": [{{"label": "...", "description": "...",
                  "patterns": [["SrcLabel", "TgtLabel"], ...]}}, ...]}}

SAMPLE TEXT:
\"\"\"{sample}\"\"\"
"""


def _build_extract_prompt(entity_types: List[str], relation_types: List[str]) -> str:
    ent_list = ", ".join(entity_types) if entity_types else "Person, Organization, Product, Concept, Event, Location, Date"
    rel_list = ", ".join(relation_types) if relation_types else "FOUNDED, WORKS_FOR, LOCATED_IN, RELATED_TO, IS_A"
    return f"""\
You build a knowledge graph from a text passage.

PRECISION OVER RECALL. A small number of high-quality triples beats a long list of vague ones.

# Entity types for this corpus
{ent_list}

# Step 1 — Entities
Extract SPECIFIC named entities from the passage. Each entity must be a proper noun
or precise technical identifier. Reject generic nouns like "tool", "system", "user".
Assign one type from the list above (use the closest match).

# Step 2 — Triples
A triple records a CONCRETE FACT stated in this passage between two entities from Step 1.
- predicate: SCREAMING_SNAKE_CASE active verb from: {rel_list}
- confidence: 0.7–1.0 only (omit anything below 0.7)
- subject and object MUST exactly match entity names from Step 1

Return ONLY valid JSON:
{{"entities": [{{"name": "...", "type": "..."}}],
  "triples": [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]}}
"""


# ---------------------------------------------------------------------------
# OpenAI async client (embeddings + LLM)
# ---------------------------------------------------------------------------

def _openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _openrouter_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


# OpenAI embeddings cap input arrays at 2048 items per request — batch defensively
# so big entity-resolution lists don't fall back to "use normalized names as-is".
_EMBED_BATCH = 1024


async def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = _openai_client()

    # Fast path — single call for small lists
    if len(texts) <= _EMBED_BATCH:
        resp = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # Batched path — fan out, preserve original order
    batches = [
        texts[i : i + _EMBED_BATCH]
        for i in range(0, len(texts), _EMBED_BATCH)
    ]
    logger.info(
        f"Embedding {len(texts)} inputs in {len(batches)} batches "
        f"of up to {_EMBED_BATCH}"
    )
    responses = await asyncio.gather(
        *[
            client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=batch,
            )
            for batch in batches
        ]
    )
    out: List[List[float]] = []
    for resp in responses:
        out.extend(item.embedding for item in resp.data)
    return out


async def _llm_json(prompt: str, model: str) -> str:
    client = _openrouter_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Entity resolution (Option B: normalize + cosine similarity)
# ---------------------------------------------------------------------------

def _cosine_distance(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


async def _resolve_entities(
    names: List[str],
    threshold: float,
    existing_canonicals: Optional[List[str]] = None,
    organization_id: Optional[str] = None,
) -> Dict[str, str]:
    """Return mapping raw_name → canonical_name.

    Pipeline:
      1. Normalize (lowercase + strip) — exact dupes merged immediately.
      2. Use the per-org FAISS cache to fetch any already-embedded vectors for
         the new names; only call OpenAI for names never seen before.
      3. Persist freshly-embedded vectors to the cache for future ingests.
      4. Greedy merge:
           a. ONE batched FAISS call: for every new name, find its nearest
              existing canonical (FAISS SIMD scan over the on-disk index).
           b. For each new name, also check the small in-memory list of
              new-canonicals-added-during-this-call (rare, tiny — pure numpy).
           c. Whichever match is closest within `threshold` wins; else the
              name becomes a new canonical.

    Why FAISS over the old Python loop: comparing N new names against M
    existing canonicals used to be O(N·M) in Python (~265k cosine ops for
    50 × 5300). FAISS does the same scan in optimized SIMD C in milliseconds.

    Tradeoff vs the previous implementation:
      The "shorter name wins" tiebreaker now only applies WITHIN this call's
      new canonicals — we no longer rename existing FAISS-cached canonicals
      mid-merge (would require removing+reindexing). In practice the original
      canonical name is whatever the LLM first chose for that entity, and
      subsequent merges flow into it.
    """
    if not names:
        return {}

    norm_map: Dict[str, str] = {n: _norm(n) for n in names}
    unique_norms = list(dict.fromkeys(norm_map.values()))  # dedup, preserve order

    # Track which existing canonicals were seen in the new doc — those don't
    # need a separate FAISS round-trip since they self-match (distance 0).
    seed_set: set = set()
    if existing_canonicals:
        for ec in existing_canonicals:
            n = _norm(ec)
            if n:
                seed_set.add(n)

    if not unique_norms:
        return {}

    # ---- 1. Cache lookup + (selective) OpenAI embed ---------------------
    cache = get_entity_vector_cache() if organization_id else None
    vec_map: Dict[str, List[float]] = {}

    if cache is not None:
        try:
            cached = await cache.get_cached_vectors(organization_id, unique_norms)
            for nm, vec in cached.items():
                vec_map[nm] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except Exception as e:
            logger.warning(f"Entity cache read failed: {e}; will re-embed")

    missing = [n for n in unique_norms if n not in vec_map]
    logger.info(
        f"Entity resolution: {len(unique_norms)} new names "
        f"({len(vec_map)} cached, {len(missing)} to embed) "
        f"vs {len(seed_set)} existing canonicals"
    )

    if missing:
        try:
            new_embeddings = await _embed_texts(missing)
        except Exception as e:
            logger.warning(
                f"Entity resolution embedding failed: {e}; "
                f"using normalized names as-is"
            )
            return {n: norm_map[n] for n in names}

        for nm, vec in zip(missing, new_embeddings):
            vec_map[nm] = vec

        # Persist new vectors so the NEXT ingest skips OpenAI for these names
        if cache is not None:
            try:
                added = await cache.add_vectors(
                    organization_id, missing, new_embeddings
                )
                logger.info(f"Entity cache: added {added} new vectors")
            except Exception as e:
                logger.warning(f"Entity cache write failed: {e}")

    # ---- 1b. Warm-up: make sure FAISS knows about ALL existing graph
    # canonicals. Without this, the very first ingest after a clean deploy
    # has an empty FAISS index, so the batch search below can't actually
    # find the 5k+ entities that already live in FalkorDB → new entities
    # silently fail to merge with their already-existing counterparts.
    # Subsequent ingests find these vectors cached and skip OpenAI.
    if cache is not None and seed_set:
        try:
            known = await cache.known_names(organization_id)
            uncached_existing = [n for n in seed_set if n not in known]
            if uncached_existing:
                logger.info(
                    f"Cache warm-up: embedding {len(uncached_existing)} "
                    f"existing canonicals not yet in FAISS"
                )
                warmup_embeddings = await _embed_texts(uncached_existing)
                added_warm = await cache.add_vectors(
                    organization_id, uncached_existing, warmup_embeddings
                )
                logger.info(
                    f"Cache warm-up: added {added_warm} existing canonicals "
                    f"to FAISS index"
                )
        except Exception as e:
            logger.warning(
                f"Cache warm-up failed: {e}; merge quality will be degraded "
                f"this ingest but will self-heal on the next one"
            )

    # ---- 2. Batched FAISS search: each new norm → nearest existing canon
    # The cache now contains both old canonicals AND the names we just added,
    # so a self-match (distance ~0) is expected and harmless — we filter it
    # so a new name doesn't "merge with itself" before we've decided it's new.
    nearest_existing: Dict[str, Optional[Tuple[str, float]]] = {}
    if cache is not None and unique_norms:
        try:
            query_arr = np.asarray(
                [vec_map[n] for n in unique_norms], dtype=np.float32
            )
            # top_k=2 so when the top hit IS the query itself (self-match)
            # we can fall back to the second-nearest distinct neighbor.
            results = await cache.batch_nearest(organization_id, query_arr, top_k=2)
            for norm, hits in zip(unique_norms, results):
                pick: Optional[Tuple[str, float]] = None
                for hit_name, dist in hits:
                    if hit_name == norm:
                        continue  # skip self-match
                    pick = (hit_name, dist)
                    break
                nearest_existing[norm] = pick
        except Exception as e:
            logger.warning(f"FAISS batch search failed: {e}; falling back to no merges")

    # ---- 3. Greedy merge ------------------------------------------------
    # Walk new norms in order. For each:
    #   - candidate A: best match against existing canonicals (from FAISS)
    #   - candidate B: best match against new canonicals added so far this call
    # Pick whichever is closer; merge if within threshold; else become a new canonical.
    new_canon_names: List[str] = []
    new_canon_vecs: List[np.ndarray] = []
    norm_to_canon: Dict[str, str] = {}

    for norm in unique_norms:
        if norm not in vec_map:
            norm_to_canon[norm] = norm
            continue
        vec_np = np.asarray(vec_map[norm], dtype=np.float32)

        best_canon: Optional[str] = None
        best_dist: float = threshold + 1.0

        # Candidate A: nearest existing canonical (FAISS)
        a = nearest_existing.get(norm)
        if a is not None:
            cand_name, cand_dist = a
            # Only consider it if this name is actually an existing canonical
            # in FalkorDB. (Cache may contain new-from-this-doc names that
            # haven't been written to the graph yet.)
            if cand_name in seed_set and cand_dist < best_dist:
                best_dist = cand_dist
                best_canon = cand_name

        # Candidate B: nearest in-flight new canonical (small linear scan)
        if new_canon_vecs:
            arr = np.asarray(new_canon_vecs, dtype=np.float32)
            # Cosine distance via normalized dot product
            v_norm = vec_np / (np.linalg.norm(vec_np) + 1e-12)
            a_norms = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            sims = a_norms @ v_norm
            dists = 1.0 - sims
            i = int(np.argmin(dists))
            d = float(dists[i])
            if d < best_dist:
                best_dist = d
                best_canon = new_canon_names[i]

        if best_canon is not None and best_dist <= threshold:
            # In-call "shorter wins" tiebreaker only matters when both names
            # are from THIS call. For existing FAISS canonicals, keep the
            # name as-is (renaming a cached canonical would require index
            # surgery — out of scope here).
            in_flight_idx: Optional[int] = None
            try:
                in_flight_idx = new_canon_names.index(best_canon)
            except ValueError:
                pass

            if in_flight_idx is not None and len(norm) < len(best_canon):
                # The new norm is shorter than the in-call canonical it
                # matched → promote the new norm as the winner.
                winner = norm
                new_canon_names[in_flight_idx] = winner
                new_canon_vecs[in_flight_idx] = vec_np
                # Update any earlier-resolved norms that pointed at the loser
                for k in list(norm_to_canon.keys()):
                    if norm_to_canon[k] == best_canon:
                        norm_to_canon[k] = winner
                norm_to_canon[norm] = winner
            else:
                norm_to_canon[norm] = best_canon
        else:
            # Truly new entity for this call
            new_canon_names.append(norm)
            new_canon_vecs.append(vec_np)
            norm_to_canon[norm] = norm

    return {n: norm_to_canon.get(norm_map[n], norm_map[n]) for n in names}


# ---------------------------------------------------------------------------
# FalkorDB connection + index management
# ---------------------------------------------------------------------------

_BATCH = 500

_connections: Dict[str, Any] = {}
_conn_lock = asyncio.Lock()


def _get_falkor_connection(graph_name: str):
    """Return a synchronous FalkorDB graph handle (cached)."""
    if graph_name not in _connections:
        db = falkordb.FalkorDB(
            host=settings.GRAPH_DATABASE_URL,
            port=settings.GRAPH_DATABASE_PORT,
            username=settings.GRAPH_DATABASE_USERNAME or None,
            password=settings.GRAPH_DATABASE_PASSWORD or None,
            ssl=settings.GRAPH_DATABASE_SSL,
        )
        _connections[graph_name] = db.select_graph(graph_name)
    return _connections[graph_name]


def _ensure_indexes(g) -> None:
    """Idempotently create vector + range indexes."""
    # Vector index on Chunk.embedding. NOTE: the db.idx.vector.createNodeIndex
    # procedure is NOT registered in current FalkorDB — the previous call failed
    # on every ingest (silently, via `except: pass`), so no vector index ever
    # existed and `search()` always fell back to a full scan. The DDL form below
    # is the supported syntax; dimension must be inlined (OPTIONS rejects params).
    try:
        g.query(
            f"CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{dimension: {int(settings.EMBEDDING_DIM)}, "
            f"similarityFunction: 'cosine'}}"
        )
        logger.info("Created Chunk.embedding vector index")
    except Exception as e:
        # Expected after the first ingest: index already exists. Anything else
        # is a real problem and must NOT be swallowed silently again.
        msg = str(e).lower()
        if "already" not in msg and "exist" not in msg:
            logger.warning(f"Vector index creation failed: {e}")

    # Range indexes for fast property lookups
    for label, prop in [
        ("Chunk", "document_id"),
        ("Entity", "name"),
    ]:
        try:
            g.query(f"CREATE INDEX ON :{label}({prop})")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dynamic ontology (in-memory, merged per doc)
# ---------------------------------------------------------------------------

# Total characters sent to the ontology detector, spread over this many windows
# taken evenly across the document (see _spread_sample).
_ONTOLOGY_SAMPLE_CHARS = 12000
_ONTOLOGY_SAMPLE_WINDOWS = 6

class _OntologySchema:
    """Per-org ontology.

    `*_descriptions` hold the one-line definition the detector produces for each
    label. _ONTOLOGY_PROMPT has always asked for these; they used to be parsed
    and dropped. They are kept now because label strings alone are a weak signal
    for "is this type new?" — "Company" vs "Enterprise" are only moderately
    similar as strings, while their definitions are near-identical. Descriptions
    are also the schema format encoder extractors (GLiNER2) accept directly.

    `relation_patterns` holds the (source_label, target_label) pairs the detector
    proposes per relation, which is what makes inverse pairs detectable —
    HAS_EXECUTIVE and LEADS_ORGANIZATION connect the same two types in opposite
    directions.
    """

    def __init__(self):
        self.entity_labels: List[str] = []
        self.relation_labels: List[str] = []
        self.entity_descriptions: Dict[str, str] = {}
        self.relation_descriptions: Dict[str, str] = {}
        self.relation_patterns: Dict[str, List[List[str]]] = {}


_schemas: Dict[str, _OntologySchema] = {}
_schema_locks: Dict[str, asyncio.Lock] = {}


def _schema_lock(graph_name: str) -> asyncio.Lock:
    if graph_name not in _schema_locks:
        _schema_locks[graph_name] = asyncio.Lock()
    return _schema_locks[graph_name]


# Serializes the entity-resolution → graph-write critical section per org.
# Resolution reads the current entity set and resolves each doc's new names
# against it; if two documents did this concurrently they'd each miss the
# other's not-yet-written entities and create DUPLICATE canonicals (and race on
# the shared FAISS cache). Holding this lock from "load existing entities"
# through "write to graph" keeps cross-document dedup correct. The expensive
# per-chunk LLM extraction runs BEFORE this section, so it stays fully parallel
# — only this short tail (~load + resolve + write) serializes, which lets the
# worker run at high concurrency safely.
_write_locks: Dict[str, asyncio.Lock] = {}


def _write_lock(graph_name: str) -> asyncio.Lock:
    if graph_name not in _write_locks:
        _write_locks[graph_name] = asyncio.Lock()
    return _write_locks[graph_name]


# The org's ontology is persisted as a singleton node in its own graph so it
# survives worker recycle (worker_max_tasks_per_child), redeploys, and is shared
# across replicas — the in-memory _schemas dict alone loses all three. Dict
# fields (descriptions, patterns) are stored as JSON strings since FalkorDB
# node properties can't hold maps.
_SCHEMA_NODE_LABEL = "_OntologySchema"


def _load_schema_from_graph(graph_name: str) -> Optional[_OntologySchema]:
    """Read the persisted ontology for this graph, or None if absent."""
    try:
        g = _get_falkor_connection(graph_name)
        rows = g.query(
            f"MATCH (s:{_SCHEMA_NODE_LABEL} {{id: 'singleton'}}) "
            "RETURN s.entity_labels, s.relation_labels, s.entity_descriptions, "
            "s.relation_descriptions, s.relation_patterns"
        ).result_set
    except Exception as e:
        logger.warning(f"Could not load persisted schema for {graph_name}: {e}")
        return None

    if not rows:
        return None

    row = rows[0]
    schema = _OntologySchema()
    schema.entity_labels = list(row[0] or [])
    schema.relation_labels = list(row[1] or [])
    try:
        schema.entity_descriptions = json.loads(row[2]) if row[2] else {}
        schema.relation_descriptions = json.loads(row[3]) if row[3] else {}
        schema.relation_patterns = json.loads(row[4]) if row[4] else {}
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Corrupt persisted schema JSON for {graph_name}: {e}")
    return schema


def _save_schema_to_graph(graph_name: str, schema: _OntologySchema) -> None:
    """Persist the merged ontology as a singleton node."""
    try:
        g = _get_falkor_connection(graph_name)
        g.query(
            f"MERGE (s:{_SCHEMA_NODE_LABEL} {{id: 'singleton'}}) "
            "SET s.entity_labels = $el, s.relation_labels = $rl, "
            "s.entity_descriptions = $ed, s.relation_descriptions = $rd, "
            "s.relation_patterns = $rp",
            {
                "el": schema.entity_labels,
                "rl": schema.relation_labels,
                "ed": json.dumps(schema.entity_descriptions),
                "rd": json.dumps(schema.relation_descriptions),
                "rp": json.dumps(schema.relation_patterns),
            },
        )
    except Exception as e:
        logger.warning(f"Could not persist schema for {graph_name}: {e}")


async def _ensure_schema(organization_id: str, sample_text: str) -> _OntologySchema:
    """Detect ontology for this doc and merge into org's running schema."""
    graph_name = _graph_name(organization_id)
    async with _schema_lock(graph_name):
        # On a cold in-memory cache (fresh process, post-deploy, or a replica
        # that has never seen this org) load the persisted schema from the graph
        # rather than starting from empty — otherwise every restart re-discovers
        # the ontology from scratch and the type set drifts.
        existing = _schemas.get(graph_name)
        if existing is None:
            existing = await asyncio.to_thread(_load_schema_from_graph, graph_name)
            if existing is None:
                existing = _OntologySchema()
            _schemas[graph_name] = existing

        new_schema = await _detect_schema(sample_text, graph_name)
        merged = _merge_schemas(existing, new_schema, graph_name)
        _schemas[graph_name] = merged
        # Persist so it survives restart/recycle/deploy and reaches other replicas.
        await asyncio.to_thread(_save_schema_to_graph, graph_name, merged)
        return merged


def _spread_sample(text: str, budget: int = _ONTOLOGY_SAMPLE_CHARS,
                   windows: int = _ONTOLOGY_SAMPLE_WINDOWS) -> str:
    """Sample evenly across the document rather than taking the first N chars.

    Taking the head describes a long document by its front matter. On a 1.29M
    character annual report, 8000 leading characters is 0.6% of the text — all
    of it cover pages — which produced an ontology with Award, Website and
    HAS_ANNIVERSARY and nothing about credit risk or loan portfolios. Every
    entity extracted from the other 99.4% then has to fit that ontology.

    Short documents fall through unchanged.
    """
    text = text.strip()
    if len(text) <= budget:
        return text

    per_window = budget // windows
    # Space window starts evenly across the document; the last one is pulled
    # back from the end so the tail is represented too.
    span = len(text) - per_window
    starts = [round(i * span / (windows - 1)) for i in range(windows)]

    parts: List[str] = []
    for start in starts:
        piece = text[start: start + per_window]
        # Trim partial words at the edges so the LLM sees clean text.
        if start > 0:
            piece = piece.partition(" ")[2]
        parts.append(piece.strip())

    return "\n\n[...]\n\n".join(p for p in parts if p)


async def _detect_schema(sample_text: str, graph_name: str) -> _OntologySchema:
    # Sample across the whole document, not just the opening pages.
    sample = _spread_sample(sample_text)
    prompt = _ONTOLOGY_PROMPT.format(sample=sample)
    try:
        raw = await _llm_json(prompt, settings.ONTOLOGY_MODEL)
        data = _parse_first_json(raw)
    except Exception as e:
        logger.warning(f"Ontology detection failed for {graph_name}: {e}")
        return _OntologySchema()

    schema = _OntologySchema()
    seen_e: set = set()
    for item in data.get("entities", []):
        label = (item.get("label") or "").strip()
        if label and len(label) >= 3 and label not in seen_e:
            seen_e.add(label)
            schema.entity_labels.append(label)
            desc = (item.get("description") or "").strip()
            if desc:
                schema.entity_descriptions[label] = desc

    seen_r: set = set()
    for item in data.get("relations", []):
        label = (item.get("label") or "").strip()
        if label and len(label) >= 3 and label not in seen_r:
            seen_r.add(label)
            schema.relation_labels.append(label)
            desc = (item.get("description") or "").strip()
            if desc:
                schema.relation_descriptions[label] = desc
            patterns = [
                [str(p[0]).strip(), str(p[1]).strip()]
                for p in (item.get("patterns") or [])
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if patterns:
                schema.relation_patterns[label] = patterns

    logger.info(
        f"Ontology detected for {graph_name}: "
        f"{len(schema.entity_labels)} entity types, {len(schema.relation_labels)} relation types"
    )
    return schema


def _merge_schemas(
    existing: _OntologySchema,
    new: _OntologySchema,
    graph_name: str,
) -> _OntologySchema:
    merged = _OntologySchema()
    # Existing descriptions win — the first definition of a label stays
    # authoritative, so a later document cannot silently redefine a type.
    merged.entity_descriptions = dict(new.entity_descriptions)
    merged.entity_descriptions.update(existing.entity_descriptions)
    merged.relation_descriptions = dict(new.relation_descriptions)
    merged.relation_descriptions.update(existing.relation_descriptions)
    merged.relation_patterns = dict(new.relation_patterns)
    merged.relation_patterns.update(existing.relation_patterns)

    seen_e = set(existing.entity_labels)
    merged.entity_labels = list(existing.entity_labels)
    added_e = []
    for label in new.entity_labels:
        if label not in seen_e:
            merged.entity_labels.append(label)
            seen_e.add(label)
            added_e.append(label)

    seen_r = set(existing.relation_labels)
    merged.relation_labels = list(existing.relation_labels)
    added_r = []
    for label in new.relation_labels:
        if label not in seen_r:
            merged.relation_labels.append(label)
            seen_r.add(label)
            added_r.append(label)

    if added_e or added_r:
        logger.info(
            f"Schema extended for {graph_name}: "
            f"+{len(added_e)} entity types, +{len(added_r)} relation types — "
            f"now {len(merged.entity_labels)} entities, {len(merged.relation_labels)} relations"
        )
    return merged


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

async def _extract_chunk(
    text: str,
    schema: _OntologySchema,
    sem: asyncio.Semaphore,
) -> Extraction:
    prompt = _build_extract_prompt(schema.entity_labels, schema.relation_labels)
    full_prompt = prompt + f"\n\nPASSAGE:\n{text}"
    async with sem:
        try:
            raw = await _llm_json(full_prompt, settings.EXTRACTION_MODEL)
            data = _parse_first_json(raw)
            return Extraction.model_validate(data)
        except Exception as e:
            logger.warning(f"Chunk extraction failed: {e}")
            return Extraction()


# ---------------------------------------------------------------------------
# Graph writes
# ---------------------------------------------------------------------------

def _write_to_graph(
    g,
    chunks: List[Dict[str, Any]],
    entity_map: Dict[str, str],          # raw_name → canonical
    triple_rows: List[Dict[str, Any]],
    mention_rows: List[Tuple[str, str]],  # (chunk_id, canonical_entity_name)
    entity_types: Dict[str, str],         # canonical_name → type
) -> Dict[str, int]:
    # ---- Chunks ----
    logger.info(f"Writing {len(chunks)} chunks")
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i: i + _BATCH]
        g.query(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {id: row.id})
            SET c.document_id = row.document_id,
                c.text = row.text,
                c.embedding = vecf32(row.embedding)
            """,
            {"rows": batch},
        )

    # ---- Entities ----
    ent_rows = [
        {"name": name, "type": entity_types.get(name, "Unknown")}
        for name in set(entity_map.values())
    ]
    logger.info(f"Writing {len(ent_rows)} entities")
    for i in range(0, len(ent_rows), _BATCH):
        g.query(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.name})
            SET e.type = row.type
            """,
            {"rows": ent_rows[i: i + _BATCH]},
        )

    # ---- RELATES triples ----
    logger.info(f"Writing {len(triple_rows)} triples")
    for i in range(0, len(triple_rows), _BATCH):
        g.query(
            """
            UNWIND $rows AS row
            MATCH (s:Entity {name: row.subj}), (t:Entity {name: row.obj})
            MERGE (s)-[r:RELATES {triple_id: row.triple_id}]->(t)
            SET r.predicate = row.predicate,
                r.source_chunk = row.source_chunk,
                r.confidence = row.confidence
            """,
            {"rows": triple_rows[i: i + _BATCH]},
        )

    # ---- MENTIONS edges ----
    m_rows = [{"cid": c, "name": n} for c, n in mention_rows]
    logger.info(f"Writing {len(m_rows)} MENTIONS edges")
    for i in range(0, len(m_rows), _BATCH):
        g.query(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {id: row.cid}), (e:Entity {name: row.name})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            {"rows": m_rows[i: i + _BATCH]},
        )

    return {
        "chunks": len(chunks),
        "entities": len(ent_rows),
        "triples": len(triple_rows),
        "mentions": len(m_rows),
    }


# ---------------------------------------------------------------------------
# GraphRAGClient
# ---------------------------------------------------------------------------

class GraphRAGClient:
    """
    Drop-in replacement for the SDK-based GraphRAGClient.
    Public interface (ingest_text, ingest_chunks, search, delete_document, delete_org)
    is identical so ingestion_service.py requires no changes.
    """

    # ---- ingest ---------------------------------------------------------------

    async def ingest_text(
        self,
        text: str,
        organization_id: str,
        document_id: str,
        filename: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            logger.warning(f"Empty text for document {document_id}, skipping ingest")
            return {"ingested": False, "reason": "empty_text"}

        # Helper that swallows errors so a flaky callback never breaks ingest
        async def _emit(stage: str, desc: str,
                        current: Optional[int] = None,
                        total: Optional[int] = None) -> None:
            if progress_callback is None:
                return
            try:
                await progress_callback(stage, desc, current, total)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"progress_callback failed at {stage}: {e}")

        graph_name = _graph_name(organization_id)
        logger.info(f"Ingest start: doc={document_id} graph={graph_name} len={len(text)}")

        # Step 1: detect + merge ontology for this doc
        await _emit("schema_detect", "Detecting entity ontology")
        schema = await _ensure_schema(organization_id, text)

        # Step 2: chunk
        await _emit("chunking", "Splitting document into semantic chunks")
        chunker = get_chunker_client()
        try:
            raw_chunks = chunker.chunk_text(text, chunker_type="large_chunk")
            chunk_texts = [c.text for c in raw_chunks if c.text.strip()]
        except Exception as e:
            logger.error(f"Chunking failed for {document_id}: {e}")
            return {"ingested": False, "reason": f"chunking_failed: {e}"}

        if not chunk_texts:
            return {"ingested": False, "reason": "no_chunks"}

        logger.info(f"Chunked doc={document_id} into {len(chunk_texts)} chunks")

        # Step 3: embed + extract in parallel, reporting progress as each finishes
        total = len(chunk_texts)
        await _emit(
            "extracting",
            f"Embedding & extracting entities (0/{total} chunks)",
            0,
            total,
        )

        sem = asyncio.Semaphore(settings.LLM_CONCURRENCY)
        completed = {"n": 0}
        # Throttle UI updates: emit at most ~20 times across the whole batch
        # for big docs, but always emit for small docs.
        emit_every = max(1, total // 20)

        # Embed ALL chunks in ONE batched call (it fans out internally at
        # _EMBED_BATCH=1024 per request) instead of one request per chunk. The
        # per-chunk version fired `total` concurrent OpenAI requests with no cap
        # → 429s on large documents. Extraction still runs per-chunk under the
        # LLM_CONCURRENCY semaphore, concurrently with embedding.
        async def _embed_all() -> List[List[float]]:
            return await _embed_texts(chunk_texts)

        async def _extract_one(chunk_text: str) -> Extraction:
            extraction = await _extract_chunk(chunk_text, schema, sem)
            completed["n"] += 1
            n = completed["n"]
            if n == total or n % emit_every == 0:
                await _emit(
                    "extracting",
                    f"Embedding & extracting entities ({n}/{total} chunks)",
                    n,
                    total,
                )
            return extraction

        embeddings, extractions = await asyncio.gather(
            _embed_all(),
            asyncio.gather(*[_extract_one(t) for t in chunk_texts]),
        )

        results = [
            (_chunk_id(document_id, i), chunk_texts[i], embeddings[i], extractions[i])
            for i in range(total)
        ]

        # Step 4: collect all raw entity names for resolution
        all_entity_names: List[str] = []
        for _, _, _, extraction in results:
            for e in extraction.entities:
                if e.name.strip():
                    all_entity_names.append(e.name.strip())
            for tr in extraction.triples:
                for n in (tr.subject.strip(), tr.object.strip()):
                    if n:
                        all_entity_names.append(n)

        # ── Critical section: serialize resolution → write per org graph ────
        # Everything above (chunk embedding + LLM entity/triple extraction) ran
        # in parallel across documents. From here we read the current entity
        # set, resolve THIS doc's names against it, and write — which must not
        # interleave with another doc doing the same, or both miss each other's
        # not-yet-written entities and create duplicate canonicals. This tail is
        # short (~load + resolve + write), so high worker concurrency stays safe.
        async with _write_lock(graph_name):
            # Seed cross-document resolution with the org's existing canonicals.
            # The per-org FAISS cache already holds them, so when it is warm we
            # use it and SKIP the `MATCH (e:Entity) RETURN e.name` scan — that
            # scan is unbounded and grows linearly with the graph, right here
            # inside the serialized critical section. We only fall back to the
            # full graph load on a COLD cache (fresh process / post-deploy),
            # where it is needed once to warm FAISS or new entities can't merge
            # with existing ones and duplicate canonicals appear.
            existing_entity_names: List[str] = []
            try:
                cache = get_entity_vector_cache() if organization_id else None
                known = await cache.known_names(organization_id) if cache else set()
                if known:
                    existing_entity_names = list(known)
                    logger.info(
                        f"Seeded resolution from FAISS cache ({len(known)} names); "
                        f"skipped full graph scan"
                    )
                else:
                    g_pre = await asyncio.to_thread(_get_falkor_connection, graph_name)
                    rows = await asyncio.to_thread(
                        lambda: g_pre.query("MATCH (e:Entity) RETURN e.name").result_set
                    )
                    existing_entity_names = [row[0] for row in rows if row[0]]
                    logger.info(
                        f"Cold cache — loaded {len(existing_entity_names)} entities "
                        f"from graph to warm resolution"
                    )
            except Exception as e:
                logger.warning(f"Could not load existing entities for resolution: {e}")

            await _emit(
                "resolving",
                f"Resolving {len(set(all_entity_names))} entity names",
            )
            entity_map = await _resolve_entities(
                list(dict.fromkeys(all_entity_names)),
                threshold=settings.ENTITY_RESOLUTION_THRESHOLD,
                existing_canonicals=existing_entity_names,
                organization_id=organization_id,
            )

            # Step 5: build write payloads
            chunk_rows = []
            mention_rows: List[Tuple[str, str]] = []
            triple_rows: List[Dict[str, Any]] = []
            entity_types: Dict[str, str] = {}
            seen_triples: set = set()

            for cid, chunk_text, embedding, extraction in results:
                chunk_rows.append({
                    "id": cid,
                    "document_id": document_id,
                    "text": chunk_text,
                    "embedding": [float(x) for x in embedding],
                })

                for e in extraction.entities:
                    raw = e.name.strip()
                    if not raw:
                        continue
                    canon = entity_map.get(raw, _norm(raw))
                    entity_types.setdefault(canon, e.type)
                    mention_rows.append((cid, canon))

                for tr in extraction.triples:
                    if tr.confidence < settings.MIN_TRIPLE_CONFIDENCE:
                        continue
                    subj_raw = tr.subject.strip()
                    obj_raw = tr.object.strip()
                    if not subj_raw or not obj_raw:
                        continue
                    subj = entity_map.get(subj_raw, _norm(subj_raw))
                    obj = entity_map.get(obj_raw, _norm(obj_raw))
                    if subj == obj:
                        continue
                    if subj not in entity_types or obj not in entity_types:
                        continue
                    tid = _triple_id(subj, tr.predicate, obj, cid)
                    if tid in seen_triples:
                        continue
                    seen_triples.add(tid)
                    triple_rows.append({
                        "triple_id": tid,
                        "subj": subj,
                        "obj": obj,
                        "predicate": tr.predicate,
                        "source_chunk": cid,
                        "confidence": tr.confidence,
                    })

            # Deduplicate mention_rows
            mention_rows = list(set(mention_rows))

            # Step 6: write to FalkorDB (sync client, run in thread)
            await _emit(
                "writing",
                f"Writing {len(chunk_rows)} chunks · {len(entity_map)} entities · "
                f"{len(triple_rows)} relations to graph",
            )
            g = await asyncio.to_thread(_get_falkor_connection, graph_name)
            await asyncio.to_thread(_ensure_indexes, g)
            counts = await asyncio.to_thread(
                _write_to_graph, g, chunk_rows, entity_map,
                triple_rows, mention_rows, entity_types,
            )

        logger.info(
            f"Ingest complete: doc={document_id} "
            f"chunks={counts['chunks']} entities={counts['entities']} "
            f"triples={counts['triples']} mentions={counts['mentions']}"
        )
        return {
            "ingested": True,
            "graph": graph_name,
            "document_id": document_id,
            **counts,
        }

    async def ingest_chunks(
        self,
        chunks: List[str],
        organization_id: str,
        document_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        non_empty = [c for c in chunks if c and c.strip()]
        if not non_empty:
            return {"ingested": False, "reason": "no_chunks"}
        joined = "\n\n---\n\n".join(non_empty)
        return await self.ingest_text(
            text=joined,
            organization_id=organization_id,
            document_id=document_id,
            progress_callback=progress_callback,
        )

    # ---- search ---------------------------------------------------------------

    async def search(
        self,
        query: str,
        organization_id: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        graph_expand: bool = True,
        expand_k: int = 20,
        top_entities: int = 12,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        graph_name = _graph_name(organization_id)

        # Phase 1: embed query
        try:
            qv = (await _embed_texts([query.strip()]))[0]
        except Exception as e:
            logger.error(f"Search embedding failed: {e}")
            return []

        g = await asyncio.to_thread(_get_falkor_connection, graph_name)

        # Phase 2: pre-filtered vector search
        try:
            if document_ids:
                vector_rows = await asyncio.to_thread(
                    lambda: g.query(
                        """
                        MATCH (c:Chunk)
                        WHERE c.document_id IN $docs
                        WITH c, vec.cosineDistance(c.embedding, vecf32($qv)) AS dist
                        ORDER BY dist ASC LIMIT $k
                        RETURN c.id, c.document_id, c.text, dist
                        """,
                        {"qv": qv, "k": top_k, "docs": list(document_ids)},
                    ).result_set
                )
            else:
                # Use the vector index for ANN retrieval instead of scanning
                # every Chunk in the org and computing distance in-query. The
                # index returns the k nearest candidates; we recompute exact
                # cosine distance on just those k so `dist` semantics (lower =
                # closer) and ordering match the scoped path exactly.
                def _indexed_search():
                    # queryNodes returns `score` = cosine distance (0 = identical,
                    # verified against real data), already nearest-first; ORDER BY
                    # is belt-and-suspenders. Matches the scoped path's `dist`.
                    return g.query(
                        """
                        CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qv))
                        YIELD node, score
                        RETURN node.id, node.document_id, node.text, score
                        ORDER BY score ASC
                        """,
                        {"qv": qv, "k": top_k},
                    ).result_set

                def _scan_search():
                    return g.query(
                        """
                        MATCH (c:Chunk)
                        WITH c, vec.cosineDistance(c.embedding, vecf32($qv)) AS dist
                        ORDER BY dist ASC LIMIT $k
                        RETURN c.id, c.document_id, c.text, dist
                        """,
                        {"qv": qv, "k": top_k},
                    ).result_set

                try:
                    vector_rows = await asyncio.to_thread(_indexed_search)
                except Exception as idx_err:
                    # Index missing (e.g. never ingested) → fall back to scan so
                    # search still works rather than hard-failing.
                    logger.warning(
                        f"Vector index query failed ({idx_err}); falling back to scan"
                    )
                    vector_rows = await asyncio.to_thread(_scan_search)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        chunks: List[Dict[str, Any]] = [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "text": row[2],
                "score": float(row[3]),
                "shared_entities": None,
                "via": "vector",
            }
            for row in vector_rows
        ]
        vector_chunk_ids = [c["chunk_id"] for c in chunks]

        # Phase 3: entity expansion via MENTIONS
        anchors: List[Dict[str, Any]] = []
        anchor_names: List[str] = []
        if graph_expand and vector_chunk_ids:
            try:
                anchor_rows = await asyncio.to_thread(
                    lambda: g.query(
                        """
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                        WHERE c.id IN $cids
                        WITH e, count(DISTINCT c) AS chunk_count
                        ORDER BY chunk_count DESC LIMIT $top_n
                        RETURN e.name, e.type, chunk_count
                        """,
                        {"cids": vector_chunk_ids, "top_n": top_entities},
                    ).result_set
                )
                anchors = [
                    {
                        "name": row[0],
                        "type": row[1],
                        "chunk_count": int(row[2]) if row[2] is not None else 0,
                    }
                    for row in anchor_rows
                ]
                anchor_names = [a["name"] for a in anchors]
            except Exception as e:
                logger.warning(f"Anchor lookup failed: {e}")

            if anchor_names:
                try:
                    expand_params: Dict[str, Any] = {
                        "names": anchor_names,
                        "original_cids": vector_chunk_ids,
                        "extra_k": expand_k,
                    }
                    expand_query = """
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                        WHERE e.name IN $names
                          AND NOT c.id IN $original_cids
                        """
                    if document_ids:
                        expand_query += "AND c.document_id IN $docs\n"
                        expand_params["docs"] = list(document_ids)
                    expand_query += """
                        WITH c, count(DISTINCT e) AS shared_entities
                        ORDER BY shared_entities DESC LIMIT $extra_k
                        RETURN c.id, c.document_id, c.text, shared_entities
                    """
                    expanded_rows = await asyncio.to_thread(
                        lambda: g.query(expand_query, expand_params).result_set
                    )
                    for row in expanded_rows:
                        chunks.append({
                            "chunk_id": row[0],
                            "document_id": row[1],
                            "text": row[2],
                            "score": None,
                            "shared_entities": int(row[3]),
                            "via": "graph",
                        })
                except Exception as e:
                    logger.warning(f"Graph expand failed: {e}")

        # Phase 4: triples from chunk set
        triples: List[Dict[str, Any]] = []
        all_chunk_ids = [c["chunk_id"] for c in chunks]
        if all_chunk_ids:
            try:
                triple_rows = await asyncio.to_thread(
                    lambda: g.query(
                        """
                        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                        WHERE r.source_chunk IN $cids
                          AND coalesce(r.confidence, 0.0) >= $min_conf
                        RETURN a.name, a.type, r.predicate, b.name, b.type,
                               r.confidence, r.source_chunk
                        ORDER BY r.confidence DESC
                        LIMIT 100
                        """,
                        {"cids": all_chunk_ids, "min_conf": settings.MIN_TRIPLE_CONFIDENCE},
                    ).result_set
                )
                triples = [
                    {
                        "subject": row[0],
                        "subject_type": row[1],
                        "predicate": row[2],
                        "object": row[3],
                        "object_type": row[4],
                        "confidence": float(row[5]) if row[5] is not None else None,
                        "source_chunk": row[6],
                    }
                    for row in triple_rows
                ]
            except Exception as e:
                logger.warning(f"Triple fetch failed: {e}")

        if not chunks:
            return []

        logger.info(
            f"Search: graph={graph_name} query={query[:60]!r} "
            f"chunks={len(chunks)} ({sum(1 for c in chunks if c['via']=='vector')} vector + "
            f"{sum(1 for c in chunks if c['via']=='graph')} graph) "
            f"anchors={len(anchors)} triples={len(triples)}"
        )

        return [{
            "chunks": chunks,
            "anchors": anchors,
            "triples": triples,
            "count": len(chunks),
            "query": query.strip(),
        }]

    # ---- delete ---------------------------------------------------------------

    async def delete_document(
        self, document_id: str, organization_id: str
    ) -> bool:
        try:
            g = await asyncio.to_thread(_get_falkor_connection, _graph_name(organization_id))
            await asyncio.to_thread(
                lambda: g.query(
                    "MATCH (c:Chunk {document_id: $doc_id}) DETACH DELETE c",
                    {"doc_id": document_id},
                )
            )
            logger.info(f"Deleted chunks for doc={document_id}")
            return True
        except Exception as e:
            logger.warning(f"Delete failed for doc={document_id}: {e}")
            return False

    async def delete_org(self, organization_id: str) -> bool:
        graph_name = _graph_name(organization_id)
        try:
            g = await asyncio.to_thread(_get_falkor_connection, graph_name)
            await asyncio.to_thread(lambda: g.query("MATCH (n) DETACH DELETE n"))
            _connections.pop(graph_name, None)
            _schemas.pop(graph_name, None)
            logger.info(f"Deleted all nodes for graph={graph_name}")
            return True
        except Exception as e:
            logger.warning(f"Org delete failed for {organization_id}: {e}")
            return False

    async def shutdown(self) -> None:
        _connections.clear()
        _schemas.clear()


# Singleton
_graphrag_client: Optional[GraphRAGClient] = None


def get_graphrag_client() -> GraphRAGClient:
    global _graphrag_client
    if _graphrag_client is None:
        _graphrag_client = GraphRAGClient()
    return _graphrag_client
