"""
KG pipeline — the orchestration entry point for the revamped two-layer system.

Deliberately exposes the SAME public surface as the old graphrag_client
(ingest_text / ingest_chunks / search / delete_document / delete_org) so the
cutover in ingestion_service and chat is a one-line client swap.

ingest_text ties the whole engine together:
   load ontology (persisted) -> ensure_for_document (detect+canonicalize+save)
   -> chunk -> embed (batched) -> HybridExtractor per chunk -> GraphWriter
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.logger import logger
from app.settings import settings

from clients.chunker_client import get_chunker_client
from clients.kg.llm import embed_texts
from clients.kg.ontology_store import load_manager, save_manager
from clients.kg.writer import GraphWriter, ChunkData
from clients.kg.store import get_graph
from clients.kg import retrieval
from clients.kg.extraction import extract_chunk as _llm_extract_chunk, Extraction


class _LLMExtractor:
    """Fallback extractor (same async interface as HybridExtractor) used when the
    GLiNER-Relex model can't be loaded in this process (missing package / OOM /
    no download). Keeps ingestion working rather than crashing."""

    def __init__(self, ontology: dict):
        self.ontology = ontology

    async def extract_chunk(self, text: str, sem: asyncio.Semaphore) -> Extraction:
        return await _llm_extract_chunk(text, self.ontology, sem)


def _build_extractor(ontology: dict):
    """Hybrid if the encoder loads; else graceful LLM-only."""
    try:
        from clients.kg.extraction_hybrid import HybridExtractor
        return HybridExtractor(ontology)
    except Exception as e:
        logger.warning(f"[kg-pipeline] hybrid extractor unavailable ({e}); "
                       f"falling back to LLM-only extraction")
        return _LLMExtractor(ontology)


class KGPipeline:
    async def ingest_text(
        self,
        text: str,
        organization_id: str,
        document_id: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"ingested": False, "reason": "empty_text"}

        async def emit(stage, desc):
            if progress_callback:
                try:
                    await progress_callback(stage, desc, None, None)
                except Exception:
                    pass

        # 1. ontology: load persisted -> detect+canonicalize this doc -> save
        await emit("schema_detect", "Detecting/extending ontology")
        manager = await load_manager(organization_id)
        ontology = await manager.ensure_for_document(text)
        await save_manager(manager)

        # 2. chunk
        await emit("chunking", "Chunking document")
        chunker = get_chunker_client()
        raw = chunker.chunk_text(text, chunker_type="large_chunk")
        chunk_texts = [c.text for c in raw if c.text.strip()]
        if not chunk_texts:
            return {"ingested": False, "reason": "no_chunks"}

        # 3. embed all chunks in ONE batched call
        await emit("embedding", f"Embedding {len(chunk_texts)} chunks")
        embeddings = await embed_texts(chunk_texts)

        # 4. extract (hybrid: encoder-first, LLM-fallback), concurrent
        await emit("extracting", f"Extracting from {len(chunk_texts)} chunks")
        extractor = _build_extractor(ontology)
        sem = asyncio.Semaphore(settings.LLM_CONCURRENCY)
        extractions = await asyncio.gather(
            *[extractor.extract_chunk(t, sem) for t in chunk_texts]
        )

        chunks = [
            ChunkData(id=f"{document_id}::{i}", text=t, embedding=embeddings[i], seq_index=i)
            for i, t in enumerate(chunk_texts)
        ]

        # 5. write the two-layer graph
        await emit("writing", "Writing two-layer knowledge graph")
        meta = dict(metadata or {})
        meta.setdefault("title", filename or "")
        writer = GraphWriter(organization_id)
        counts = await writer.write_document(document_id, chunks, extractions, meta)

        logger.info(f"[kg-pipeline] ingested doc={document_id} org={organization_id} {counts}")
        return {"ingested": True, "document_id": document_id, **counts}

    async def ingest_chunks(
        self,
        chunks: List[str],
        organization_id: str,
        document_id: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        # Preserve caller-provided chunk boundaries (e.g. video scenes) by
        # joining with a marker the chunker won't split mid-scene.
        non_empty = [c for c in chunks if c and c.strip()]
        if not non_empty:
            return {"ingested": False, "reason": "no_chunks"}
        return await self.ingest_text(
            text="\n\n".join(non_empty),
            organization_id=organization_id,
            document_id=document_id,
            progress_callback=progress_callback,
        )

    async def search(
        self,
        query: str,
        organization_id: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        user_email: Optional[str] = None,
        **_kw,
    ) -> List[Dict[str, Any]]:
        """Drop-in shape for the old graphrag_client.search (list with one
        wrapper dict), backed by the new pre-filter + relation-traversal search."""
        r = await retrieval.search(
            query, organization_id, user_email=user_email,
            top_k=top_k, document_ids=document_ids,
        )
        return [{
            "chunks": r["chunks"],
            "anchors": r["entities"],
            "triples": r["relation_paths"],
            "count": len(r["chunks"]),
            "query": r["query"],
        }]

    async def delete_document(self, document_id: str, organization_id: str) -> bool:
        def _del():
            g = get_graph(organization_id)
            # Delete the Document and its chunks; leave shared Entities in place.
            g.query(
                """
                MATCH (d:Document {document_id: $doc})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
                """, {"doc": document_id})
            return True
        try:
            return await asyncio.to_thread(_del)
        except Exception as e:
            logger.error(f"[kg-pipeline] delete_document failed: {e}")
            return False

    async def delete_org(self, organization_id: str) -> bool:
        def _del():
            get_graph(organization_id).delete()
            return True
        try:
            return await asyncio.to_thread(_del)
        except Exception as e:
            logger.error(f"[kg-pipeline] delete_org failed: {e}")
            return False


_pipeline: Optional[KGPipeline] = None


def get_kg_pipeline() -> KGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KGPipeline()
    return _pipeline
