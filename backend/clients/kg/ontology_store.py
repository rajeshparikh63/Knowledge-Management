"""
Per-org persistence for the OntologyManager.

Stored as a singleton node in the org's own graph so the learned ontology (seed
+ promoted types + pending counts) survives worker recycle, redeploys, and is
shared across replicas. Without this the controlled-extension ontology would
reset every restart and re-drift.
"""
from __future__ import annotations

import asyncio
import json

from app.logger import logger
from clients.kg.store import get_graph
from clients.kg.ontology_manager import OntologyManager

_NODE = "_KGOntology"


async def load_manager(organization_id: str) -> OntologyManager:
    """Load the org's persisted ontology, or a fresh seeded manager."""
    def _load() -> OntologyManager:
        try:
            g = get_graph(organization_id)
            rows = g.query(
                f"MATCH (o:{_NODE} {{id: 'singleton'}}) RETURN o.data"
            ).result_set
            if rows and rows[0][0]:
                return OntologyManager.from_dict(json.loads(rows[0][0]))
        except Exception as e:
            logger.warning(f"[ontology_store:{organization_id}] load failed: {e}")
        return OntologyManager.seeded(organization_id)

    return await asyncio.to_thread(_load)


async def save_manager(manager: OntologyManager) -> None:
    def _save() -> None:
        try:
            g = get_graph(manager.org_id)
            g.query(
                f"MERGE (o:{_NODE} {{id: 'singleton'}}) SET o.data = $data",
                {"data": json.dumps(manager.to_dict())},
            )
        except Exception as e:
            logger.warning(f"[ontology_store:{manager.org_id}] save failed: {e}")

    await asyncio.to_thread(_save)
