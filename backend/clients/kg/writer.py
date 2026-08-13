"""
Two-layer knowledge-graph writer.

Turns a document's chunks + per-chunk extractions into the graph:

  LEXICAL  (:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
  MEANING  (:Entity)-[:RELATES {predicate,confidence,source_chunk}]->(:Entity)
  BRIDGE   (:Chunk)-[:MENTIONS]->(:Entity)

Creates the Document node (the thing RBAC will later attach HAS_ACCESS to) but
does NOT touch Users / HAS_ACCESS — that is the separate RBAC module's job.

Entity names are canonicalized here with a simple normalize-and-merge pass so
the writer is usable end-to-end; the richer semantic entity-resolution pass
(task #4) plugs in at the same seam (`_resolve_entities`).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.logger import logger
from clients.kg.store import get_graph, ensure_indexes
from clients.kg.extraction import Extraction

_BATCH = 500
_norm_re = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _norm_re.sub(" ", (s or "").strip().lower())


@dataclass
class ChunkData:
    id: str
    text: str
    embedding: List[float]
    seq_index: int


def _resolve_entities(extractions: List[Extraction]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Basic normalize-and-merge resolution.

    Returns (raw_name -> canonical_name, canonical_name -> type). Canonical form
    is the first surface form seen for a normalized key. The semantic resolution
    pass (task #4) replaces this function's body without changing the writer.
    """
    canon_by_norm: Dict[str, str] = {}
    type_by_canon: Dict[str, str] = {}
    raw_to_canon: Dict[str, str] = {}
    for ex in extractions:
        for e in ex.entities:
            raw = e.name.strip()
            if not raw:
                continue
            key = _norm(raw)
            if key not in canon_by_norm:
                canon_by_norm[key] = raw          # first surface form wins
                type_by_canon[raw] = e.type
            raw_to_canon[raw] = canon_by_norm[key]
    return raw_to_canon, type_by_canon


class GraphWriter:
    def __init__(self, organization_id: str):
        self.org_id = organization_id

    async def write_document(
        self,
        document_id: str,
        chunks: List[ChunkData],
        extractions: List[Extraction],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Write one document's lexical + meaning graph. Idempotent (MERGE-based)."""
        if len(chunks) != len(extractions):
            raise ValueError("chunks and extractions must be 1:1 and aligned")
        return await asyncio.to_thread(
            self._write_sync, document_id, chunks, extractions, metadata or {}
        )

    # -- sync body (FalkorDB client is synchronous) ------------------------

    def _write_sync(self, document_id, chunks, extractions, metadata) -> Dict[str, int]:
        g = get_graph(self.org_id)
        ensure_indexes(g)

        # ---- Document node (lexical root) ----
        g.query(
            """
            MERGE (d:Document {document_id: $doc_id})
            SET d.title = $title, d.source_link = $source_link,
                d.source_type = $source_type, d.classification = $classification,
                d.mime_type = $mime_type, d.updated_at = $ts
            """,
            {
                "doc_id": document_id,
                "title": metadata.get("title", ""),
                "source_link": metadata.get("source_link", ""),
                "source_type": metadata.get("source_type", ""),
                "classification": metadata.get("classification", ""),
                "mime_type": metadata.get("mime_type", ""),
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

        # ---- Chunks + HAS_CHUNK ----
        chunk_rows = [
            {"id": c.id, "document_id": document_id, "text": c.text,
             "embedding": [float(x) for x in c.embedding], "seq_index": c.seq_index}
            for c in chunks
        ]
        for i in range(0, len(chunk_rows), _BATCH):
            g.query(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.document_id = row.document_id, c.text = row.text,
                    c.embedding = vecf32(row.embedding), c.seq_index = row.seq_index
                WITH c, row
                MATCH (d:Document {document_id: row.document_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                {"rows": chunk_rows[i:i + _BATCH]},
            )

        # ---- NEXT_CHUNK (sequential, within this document) ----
        ordered = sorted(chunks, key=lambda c: c.seq_index)
        next_rows = [{"a": ordered[i].id, "b": ordered[i + 1].id}
                     for i in range(len(ordered) - 1)]
        for i in range(0, len(next_rows), _BATCH):
            g.query(
                """
                UNWIND $rows AS row
                MATCH (a:Chunk {id: row.a}), (b:Chunk {id: row.b})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                """,
                {"rows": next_rows[i:i + _BATCH]},
            )

        # ---- Resolve entities, then write Entity nodes ----
        raw_to_canon, type_by_canon = _resolve_entities(extractions)
        ent_rows = [{"name": name, "type": type_by_canon.get(name, "Unknown")}
                    for name in set(raw_to_canon.values())]
        for i in range(0, len(ent_rows), _BATCH):
            g.query(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {name: row.name})
                SET e.type = row.type
                """,
                {"rows": ent_rows[i:i + _BATCH]},
            )

        # ---- MENTIONS (chunk -> entity) and RELATES (entity -> entity) ----
        mention_rows: List[Dict[str, str]] = []
        triple_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for c, ex in zip(chunks, extractions):
            for e in ex.entities:
                canon = raw_to_canon.get(e.name.strip())
                if canon:
                    mention_rows.append({"cid": c.id, "name": canon})
            for t in ex.triples:
                s = raw_to_canon.get(t.subject.strip())
                o = raw_to_canon.get(t.object.strip())
                if not s or not o or s == o:
                    continue
                key = (s, t.predicate, o)
                # MERGE facts across chunks: keep max confidence, first source chunk.
                if key not in triple_map or t.confidence > triple_map[key]["confidence"]:
                    triple_map[key] = {"subj": s, "obj": o, "predicate": t.predicate,
                                       "confidence": t.confidence, "source_chunk": c.id}

        mention_rows = [dict(t) for t in {tuple(m.items()) for m in mention_rows}]
        for i in range(0, len(mention_rows), _BATCH):
            g.query(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.cid}), (e:Entity {name: row.name})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"rows": mention_rows[i:i + _BATCH]},
            )

        triple_rows = list(triple_map.values())
        for i in range(0, len(triple_rows), _BATCH):
            g.query(
                """
                UNWIND $rows AS row
                MATCH (s:Entity {name: row.subj}), (o:Entity {name: row.obj})
                MERGE (s)-[r:RELATES {predicate: row.predicate, target: row.obj}]->(o)
                SET r.confidence = row.confidence, r.source_chunk = row.source_chunk
                """,
                {"rows": triple_rows[i:i + _BATCH]},
            )

        counts = {
            "chunks": len(chunk_rows),
            "entities": len(ent_rows),
            "relations": len(triple_rows),
            "mentions": len(mention_rows),
        }
        logger.info(f"[kg-writer:{self.org_id}] doc={document_id} wrote {counts}")
        return counts
