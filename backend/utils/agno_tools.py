"""
Agno tool factory — document-scoped retriever for the per-org FalkorDB graph.

Pattern: factory captures org_id + document_ids at creation time and returns
a closure whose only LLM-visible argument is `query`. The agent cannot change
which documents are searched.

Returns a structured payload (chunks / anchors / triples) — not markdown blobs.
"""
from __future__ import annotations

from typing import Any, List, Optional

from agno.agent import Agent

from app.logger import logger
# New two-layer KG engine — same search() result shape as graphrag_client.
from clients.kg.pipeline import get_kg_pipeline


def create_knowledge_retriever(
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,       # accepted for interface compat, unused
    document_ids: Optional[List[str]] = None,
    num_documents: int = 10,
):
    """Return an async Agno tool that retrieves chunks + triples from the org graph.

    `organization_id` and `document_ids` are captured at creation time — the
    LLM cannot change which org or which documents are searched.

    Args:
        organization_id: Org whose FalkorDB graph to search.
        user_id: Unused (kept for interface compatibility).
        document_ids: If set, search is scoped to these docs. If None, searches
            the entire org graph. If empty list, returns nothing immediately.
        num_documents: Default top-K to return.
    """
    num_documents_default = num_documents

    async def search_knowledge_base(
        query: str,
        agent: Optional[Agent] = None,
        num_documents: Optional[int] = None,
    ) -> Optional[List[dict[str, Any]]]:
        """Search the knowledge graph for chunks relevant to `query`.

        Args:
            query: Natural-language search string.
            agent: Agno agent reference (unused; kept for tool-protocol compat).
            num_documents: Override default top-K.

        Returns:
            list with one dict containing chunks, anchors, triples, count.
            Returns None if query is empty, no org is set, or no results found.
        """
        if not query or not query.strip():
            logger.warning("search_knowledge_base: empty query")
            return None

        # Empty list = caller explicitly said "no documents selected"
        if document_ids is not None and len(document_ids) == 0:
            logger.info("search_knowledge_base: no documents selected, returning empty")
            return None

        if not organization_id:
            logger.warning("search_knowledge_base: no organization_id")
            return None

        k = num_documents if isinstance(num_documents, int) and num_documents > 0 else num_documents_default

        logger.info(
            f"search_knowledge_base: org={organization_id} "
            f"docs={len(document_ids) if document_ids else 'all'} "
            f"query={query[:80]!r} k={k}"
        )

        try:
            client = get_kg_pipeline()
            results = await client.search(
                query=query.strip(),
                organization_id=organization_id,
                top_k=k,
                document_ids=document_ids,
            )
        except Exception as e:
            logger.error(f"search_knowledge_base: search failed: {e}")
            return None

        if not results:
            logger.info("search_knowledge_base: no results")
            return None

        return results

    return search_knowledge_base
