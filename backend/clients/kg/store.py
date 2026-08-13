"""
FalkorDB connection + index management for the KG revamp.

One graph per organization ("org_<id>"), matching the deployed convention.
Uses the DDL vector-index syntax that this FalkorDB actually supports (the
db.idx.vector.createNodeIndex procedure is NOT registered — see the earlier
graphrag_client fix).
"""
from __future__ import annotations

from typing import Any, Dict

import falkordb

from app.logger import logger
from app.settings import settings

_connections: Dict[str, Any] = {}


def graph_name(organization_id: str) -> str:
    return f"org_{organization_id}"


def get_graph(organization_id: str):
    name = graph_name(organization_id)
    if name not in _connections:
        db = falkordb.FalkorDB(
            host=settings.GRAPH_DATABASE_URL,
            port=settings.GRAPH_DATABASE_PORT,
            username=settings.GRAPH_DATABASE_USERNAME or None,
            password=settings.GRAPH_DATABASE_PASSWORD or None,
            ssl=settings.GRAPH_DATABASE_SSL,
        )
        _connections[name] = db.select_graph(name)
    return _connections[name]


def ensure_indexes(g) -> None:
    """Idempotently create the vector + range indexes the two-layer graph needs."""
    # Vector index on Chunk.embedding (DDL form; dimension inlined, params rejected).
    try:
        g.query(
            f"CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{dimension: {int(settings.EMBEDDING_DIM)}, "
            f"similarityFunction: 'cosine'}}"
        )
        logger.info("Created Chunk.embedding vector index")
    except Exception as e:
        msg = str(e).lower()
        if "already" not in msg and "exist" not in msg:
            logger.warning(f"Vector index creation failed: {e}")

    # Range indexes for fast lookups / joins / RBAC traversal.
    for label, prop in [
        ("Document", "document_id"),
        ("Chunk", "document_id"),
        ("Chunk", "id"),
        ("Entity", "name"),
        ("User", "email"),
    ]:
        try:
            g.query(f"CREATE INDEX ON :{label}({prop})")
        except Exception:
            pass
