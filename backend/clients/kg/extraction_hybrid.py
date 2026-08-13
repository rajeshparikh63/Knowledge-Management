"""
Hybrid extraction — the extraction path for the KG revamp.

Encoder-first, LLM-fallback:

  1. Run GLiNER-Relex (fast, free, offline-capable) and validate against the
     ontology.
  2. If the yield is healthy -> keep it. This is the common case (~80-90% of
     chunks): fast and free.
  3. If the yield is THIN -> the chunk is likely abstract prose where the encoder
     under-extracts; re-run with the LLM and keep the richer result.

"Thin" is measured on the validated output (the messy-JPMorgan runs calibrated
these): too few surviving triples, or the ontology dropped far more relations
than it kept (encoder mistyped entities -> relations failed src/tgt).

At the edge (no LLM reachable) the fallback simply fails and we return the
encoder result — so the same code degrades to encoder-only offline, no flag
needed.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Tuple

from app.logger import logger
from clients.kg.extraction import Extraction, extract_chunk as llm_extract_chunk
from clients.kg.extraction_gliner import GlinerRelexExtractor

# Fallback triggers (calibrated on the messy JPMorgan chunks):
#   #40 kept 2 triples          -> below MIN_KEPT -> fallback
#   #70 kept 3, dropped 20      -> drop ratio blown -> fallback
#   #110 kept 27, dropped 1     -> healthy -> keep
MIN_KEPT_TRIPLES = 3
DROP_RATIO = 1.5   # dropped relations > kept * ratio -> encoder mistyped, fall back


def _relation_drops(ex: Extraction) -> int:
    return sum(v for k, v in ex.dropped.items() if k.startswith("rel_"))


def _is_thin(ex: Extraction) -> bool:
    kept = len(ex.triples)
    if kept < MIN_KEPT_TRIPLES:
        return True
    if _relation_drops(ex) > kept * DROP_RATIO:
        return True
    return False


class HybridExtractor:
    """Built once per document with that document's active ontology. The heavy
    GLiNER-Relex model is process-cached, so construction is cheap."""

    def __init__(self, ontology: dict, relex_model: str = "knowledgator/gliner-relex-large-v1.0"):
        self.ontology = ontology
        self.encoder = GlinerRelexExtractor(ontology, model_name=relex_model)
        self.stats = {"encoder": 0, "llm": 0, "encoder_offline": 0}

    async def extract_chunk(self, text: str, sem: asyncio.Semaphore) -> Extraction:
        result, _ = await self.extract_chunk_traced(text, sem)
        return result

    async def extract_chunk_traced(
        self, text: str, sem: asyncio.Semaphore
    ) -> Tuple[Extraction, str]:
        """Returns (extraction, source) where source is 'encoder' | 'llm' |
        'encoder_offline' — for observability/telemetry."""
        if not text or not text.strip():
            return Extraction(), "encoder"

        enc = await asyncio.to_thread(self.encoder.extract_chunk, text)
        if not _is_thin(enc):
            self.stats["encoder"] += 1
            return enc, "encoder"

        # Thin yield → try the LLM for this chunk only.
        try:
            llm = await llm_extract_chunk(text, self.ontology, sem)
        except Exception as e:
            logger.warning(f"[hybrid] LLM fallback unavailable ({e}); using encoder result")
            self.stats["encoder_offline"] += 1
            return enc, "encoder_offline"

        # Keep whichever is richer; tie goes to the LLM (better typing/spans).
        if len(llm.triples) >= len(enc.triples):
            self.stats["llm"] += 1
            return llm, "llm"
        self.stats["encoder"] += 1
        return enc, "encoder"
