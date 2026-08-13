"""
Two-layer retrieval: RBAC pre-filter + vector seed + relation traversal.

The old search rode only vector + MENTIONS and *fetched* triples as decoration —
it never traversed RELATES. This one:

  1. PRE-FILTER by the caller's accessible documents (RBAC). The vector search
     runs ONLY inside that scope, so results are always permitted and never
     post-filtered to zero.
  2. Vector-seed chunks inside the scope.
  3. Anchor entities = entities the seed chunks mention.
  4. TRAVERSE RELATES from the anchors (multi-hop) -> connected entities +
     the relation paths themselves (this is the part that actually uses the
     meaning layer).
  5. Expansion chunks = chunks mentioning the connected entities (within scope),
     plus NEXT_CHUNK neighbours of the seed chunks for adjacent context.
  6. Return {chunks, entities, relation_paths} for the agent.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.logger import logger
from clients.kg.store import get_graph
from clients.kg.rbac import RBACManager
from clients.kg.llm import embed_texts


async def search(
    query: str,
    organization_id: str,
    user_email: Optional[str] = None,
    top_k: int = 10,
    document_ids: Optional[List[str]] = None,
    expand_hops: int = 1,
    expand_k: int = 15,
    top_entities: int = 12,
) -> Dict[str, Any]:
    empty = {"query": query, "chunks": [], "entities": [], "relation_paths": [],
             "scope_size": 0}
    if not query or not query.strip():
        return empty

    # ---- 1. RBAC pre-filter -------------------------------------------------
    scope: Optional[List[str]] = None
    if user_email:
        allowed = await RBACManager(organization_id).accessible_document_ids(user_email)
        scope = allowed if not document_ids else [d for d in allowed if d in set(document_ids)]
        if not scope:
            logger.info(f"[retrieval] {user_email} has access to 0 docs — empty result")
            return empty
    elif document_ids:
        scope = list(document_ids)

    g = get_graph(organization_id)

    # ---- 2. embed query -----------------------------------------------------
    try:
        qv = (await embed_texts([query.strip()]))[0]
    except Exception as e:
        logger.error(f"[retrieval] query embed failed: {e}")
        return empty

    # ---- 3. vector seed (pre-filtered) --------------------------------------
    def _vector_scoped():
        return g.query(
            """
            MATCH (c:Chunk) WHERE c.document_id IN $scope
            WITH c, vec.cosineDistance(c.embedding, vecf32($qv)) AS dist
            ORDER BY dist ASC LIMIT $k
            RETURN c.id, c.document_id, c.text, dist
            """, {"qv": qv, "k": top_k, "scope": scope}).result_set

    def _vector_indexed():
        try:
            return g.query(
                """
                CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qv))
                YIELD node, score
                RETURN node.id, node.document_id, node.text, score
                ORDER BY score ASC
                """, {"qv": qv, "k": top_k}).result_set
        except Exception:
            return g.query(
                """
                MATCH (c:Chunk)
                WITH c, vec.cosineDistance(c.embedding, vecf32($qv)) AS dist
                ORDER BY dist ASC LIMIT $k
                RETURN c.id, c.document_id, c.text, dist
                """, {"qv": qv, "k": top_k}).result_set

    seed_rows = await asyncio.to_thread(_vector_scoped if scope is not None else _vector_indexed)
    chunks: List[Dict[str, Any]] = [
        {"chunk_id": r[0], "document_id": r[1], "text": r[2], "score": float(r[3]),
         "via": "vector"} for r in seed_rows
    ]
    seed_ids = [c["chunk_id"] for c in chunks]
    if not seed_ids:
        return {**empty, "scope_size": len(scope) if scope is not None else None}

    # ---- 4. anchor entities -------------------------------------------------
    anchor_rows = await asyncio.to_thread(lambda: g.query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $cids
        WITH e, count(DISTINCT c) AS mc
        ORDER BY mc DESC LIMIT $top_n
        RETURN e.name, e.type, mc
        """, {"cids": seed_ids, "top_n": top_entities}).result_set)
    entities = [{"name": r[0], "type": r[1], "mentions": int(r[2])} for r in anchor_rows]
    anchors = [e["name"] for e in entities]

    # ---- 5. RELATES traversal (the part that uses the meaning layer) --------
    relation_paths: List[Dict[str, Any]] = []
    connected: List[str] = []
    if anchors:
        hops = max(1, min(expand_hops, 2))
        rel_rows = await asyncio.to_thread(lambda: g.query(
            f"""
            MATCH path = (a:Entity)-[:RELATES*1..{hops}]-(b:Entity)
            WHERE a.name IN $anchors
            WITH a, b, relationships(path) AS rels
            UNWIND rels AS r
            RETURN DISTINCT startNode(r).name, r.predicate, endNode(r).name, r.confidence
            LIMIT 60
            """, {"anchors": anchors}).result_set)
        anchor_set = set(anchors)
        for s, p, o, c in rel_rows:
            relation_paths.append({"subject": s, "predicate": p, "object": o,
                                   "confidence": float(c) if c is not None else None})
            for endpoint in (s, o):
                if endpoint not in anchor_set:
                    connected.append(endpoint)
    connected = list(dict.fromkeys(connected))

    # ---- 6. expansion chunks (connected entities' chunks + NEXT_CHUNK) ------
    if connected:
        scope_clause = "AND c.document_id IN $scope" if scope is not None else ""
        exp_rows = await asyncio.to_thread(lambda: g.query(
            f"""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $names AND NOT c.id IN $seed {scope_clause}
            WITH c, count(DISTINCT e) AS shared
            ORDER BY shared DESC LIMIT $k
            RETURN c.id, c.document_id, c.text, shared
            """,
            {"names": connected, "seed": seed_ids, "k": expand_k,
             **({"scope": scope} if scope is not None else {})}).result_set)
        for r in exp_rows:
            chunks.append({"chunk_id": r[0], "document_id": r[1], "text": r[2],
                           "score": None, "shared_entities": int(r[3]), "via": "relation"})

    # NEXT_CHUNK neighbours of the seed chunks (adjacent context)
    have = {c["chunk_id"] for c in chunks}
    scope_clause = "AND n.document_id IN $scope" if scope is not None else ""
    nbr_rows = await asyncio.to_thread(lambda: g.query(
        f"""
        MATCH (c:Chunk)-[:NEXT_CHUNK]-(n:Chunk)
        WHERE c.id IN $seed AND NOT n.id IN $have {scope_clause}
        RETURN DISTINCT n.id, n.document_id, n.text
        LIMIT 10
        """,
        {"seed": seed_ids, "have": list(have),
         **({"scope": scope} if scope is not None else {})}).result_set)
    for r in nbr_rows:
        chunks.append({"chunk_id": r[0], "document_id": r[1], "text": r[2],
                       "score": None, "via": "next_chunk"})

    logger.info(
        f"[retrieval] org={organization_id} email={user_email} "
        f"scope={len(scope) if scope is not None else 'ALL'} "
        f"chunks={len(chunks)} (vector={sum(1 for c in chunks if c['via']=='vector')} "
        f"relation={sum(1 for c in chunks if c['via']=='relation')} "
        f"next={sum(1 for c in chunks if c['via']=='next_chunk')}) "
        f"anchors={len(anchors)} rel_paths={len(relation_paths)}"
    )
    return {
        "query": query.strip(),
        "chunks": chunks,
        "entities": entities,
        "relation_paths": relation_paths,
        "scope_size": len(scope) if scope is not None else None,
    }
