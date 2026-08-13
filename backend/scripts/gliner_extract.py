#!/usr/bin/env python3
"""
GLiNER + GLiREL extraction spike.

Reads a PDF/DOCX (or plain text), chunks it, and extracts entities + relation
triples using encoder models instead of an LLM. Output is shaped exactly like
`graphrag_client.Extraction` so results can be diffed against the current
LLM-based extractor.

Why this exists
---------------
`graphrag_client._extract_chunk` makes one LLM call per chunk. That is the
dominant cost and latency in ingestion, and it cannot run disconnected. GLiNER
(entities) and GLiREL (relations) are BERT-sized encoders that take the entity
and relation types at inference time — so the dynamic ontology this codebase
already detects feeds straight in — and run on CPU.

Install (not added to pyproject yet — this is a spike):

    uv pip install gliner glirel torch

Usage:

    # Defaults: generic ontology, CPU
    uv run python scripts/gliner_extract.py path/to/manual.pdf

    # Use the ontology this repo would detect for the document (needs API keys)
    uv run python scripts/gliner_extract.py manual.pdf --auto-ontology

    # Explicit ontology, Apple Silicon GPU, write results out
    uv run python scripts/gliner_extract.py manual.pdf \
        --entity-labels "MilitaryUnit,Equipment,Location,Person,Procedure" \
        --relation-labels "OPERATES,LOCATED_IN,REPORTS_TO,REQUIRES" \
        --device mps --json-out results.json

    # Run the current LLM extractor on the same chunks and print a diff
    uv run python scripts/gliner_extract.py manual.pdf --compare-llm
"""

from __future__ import annotations

import os

# --auto-ontology imports clients.graphrag_client, which pulls in FAISS; the
# extractors then pull in torch. On macOS both ship their own OpenMP runtime and
# the second one to load segfaults the process (exit 139). entity_vector_cache
# sets this too, but only once it is imported — too late if torch got there
# first. Must be set before ANY of those imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gliner_extract")


# ---------------------------------------------------------------------------
# Defaults — mirror the current pipeline so comparisons are apples-to-apples
# ---------------------------------------------------------------------------

# graphrag_client._build_extract_prompt falls back to these when no ontology
# has been detected yet.
DEFAULT_ENTITY_LABELS = [
    "Person", "Organization", "Product", "Concept", "Event", "Location", "Date",
]
DEFAULT_RELATION_LABELS = [
    "FOUNDED", "WORKS_FOR", "LOCATED_IN", "RELATED_TO", "IS_A",
]

# chonkie "large_chunk" settings from clients/chunker_client.py
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200

# Verified to exist on the Hub. Swap via --gliner-model / --glirel-model:
#   urchade/gliner_small-v2.1        fastest, lowest quality
#   urchade/gliner_medium-v2.1       default — good speed/quality balance
#   urchade/gliner_large-v2.1        best quality, ~3x slower
#   urchade/gliner_multi-v2.1        multilingual
#   gliner-community/gliner_*-v2.5   newer community retrain
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_GLIREL_MODEL = "jackboyla/glirel-large-v0"

# Joint NER+RE in a single forward pass, via the maintained `gliner` package.
# Selected with --joint. Apache 2.0, 0.5B params, ~1.9 GB on disk.
DEFAULT_RELEX_MODEL = "knowledgator/gliner-relex-large-v1.0"
RELEX_BATCH = 8

# GLiREL is DeBERTa-based (512-token limit). Chunks here are ~750 tokens, so
# each chunk is re-windowed before relation extraction or the tail is silently
# truncated and those relations are lost.
GLIREL_WINDOW_TOKENS = 380
GLIREL_WINDOW_STRIDE = 320

# GLiREL needs an explicit negative class to be able to predict "nothing here".
NO_RELATION = "no relation"


# ---------------------------------------------------------------------------
# Result types — same shape as graphrag_client.Extraction
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    name: str
    type: str
    score: float = 0.0


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0


@dataclass
class ChunkResult:
    index: int
    text: str
    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    entity_ms: float = 0.0
    relation_ms: float = 0.0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Pull raw text out of a PDF / DOCX / TXT / MD file."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    # markitdown[all] is already a dependency and handles pdf/docx/pptx/xlsx.
    try:
        from markitdown import MarkItDown

        log.info("Extracting text with markitdown: %s", path.name)
        result = MarkItDown().convert(str(path))
        text = (result.text_content or "").strip()
        if text:
            return text
        log.warning("markitdown returned empty content, trying fallback")
    except Exception as e:
        log.warning("markitdown failed (%s), trying fallback", e)

    if suffix == ".pdf":
        from pypdf import PdfReader

        log.info("Extracting text with pypdf: %s", path.name)
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            raise RuntimeError(
                "Could not read .docx — install python-docx or fix markitdown"
            )
        log.info("Extracting text with python-docx: %s", path.name)
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)

    raise RuntimeError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# PDF text reflow
#
# PDF extraction breaks lines wherever the page layout wrapped, and markitdown
# emits many of those as blank-line separated. In the JPMorgan report there are
# 24,657 "\n\n" breaks and 15,589 lines starting lowercase — most of those blank
# lines are mid-sentence wraps, not paragraphs.
#
# An LLM extractor hides this because it rewrites text as it reads. An encoder
# returns literal character spans, so a wrap inside a phrase produces junk
# entities like "$40\n\nbillion" or "#1\n\nin deposits and for small businesses".
# Reflowing first is what makes encoder extraction viable on real PDFs.
# ---------------------------------------------------------------------------

# Line-final tokens that cannot end a sentence — the next line continues them.
_CONTINUATION_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "than", "per", "into", "onto", "over",
    "under", "via", "vs", "page", "pages", "see", "refer", "including",
    "include", "includes", "such", "is", "are", "was", "were", "be", "been",
}

_UNICODE_SPACE = re.compile(r"[     ﻿]")
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n+[ \t]*(\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def _continues(prev: str, nxt: str) -> bool:
    """True if a line break between prev and nxt is a wrap, not a paragraph."""
    if not prev or not nxt:
        return False
    # Sentence clearly finished.
    if prev[-1] in ".!?:;":
        return False
    # Next line starts mid-sentence.
    if nxt[0].islower():
        return True
    # Dangling punctuation implies continuation.
    if prev[-1] in ",–—-":
        return True
    # Line ends on a word that cannot terminate a sentence.
    last = re.split(r"[^\w]+", prev)[-1].lower() if prev else ""
    return last in _CONTINUATION_WORDS


def clean_pdf_text(text: str) -> str:
    """Rejoin wrapped lines so entity spans stop straddling line breaks."""
    text = _UNICODE_SPACE.sub(" ", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "com-\npany" -> "company"

    # Split on any run of newlines, then decide per boundary whether to rejoin.
    parts = re.split(r"\n+", text)
    out: List[str] = []
    for part in parts:
        part = _MULTISPACE.sub(" ", part).strip()
        if not part:
            continue
        if out and _continues(out[-1], part):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)

    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Chunking — paragraph-aware, mirrors the repo's large_chunk sizing
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split on paragraph boundaries, packing up to chunk_size characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # A single oversized paragraph gets hard-split rather than dropped.
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i: i + chunk_size])
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Tokenization — GLiREL wants token indices, GLiNER gives character offsets
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Approximate spaCy tokenization, keeping (start, end) char spans."""
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def char_span_to_token_range(
    spans: Sequence[Tuple[int, int]],
    start: int,
    end: int,
) -> Optional[Tuple[int, int]]:
    """Map a character span to an INCLUSIVE token index range (GLiREL's format)."""
    hits = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    return (hits[0], hits[-1]) if hits else None


# ---------------------------------------------------------------------------
# Label handling
#
# The ontology detector emits PascalCase entity labels and SCREAMING_SNAKE_CASE
# relation labels. GLiNER/GLiREL were trained on natural-language type names and
# score noticeably better on "military unit" than on "MilitaryUnit", so labels
# are humanized on the way in and restored on the way out.
# ---------------------------------------------------------------------------

def humanize_entity_label(label: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", label.replace("_", " "))
    return re.sub(r"\s+", " ", spaced).strip().lower()


def humanize_relation_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("_", " ")).strip().lower()


def build_label_maps(
    entity_labels: Sequence[str],
    relation_labels: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (humanized -> original) maps for entities and relations."""
    ent_map: Dict[str, str] = {}
    for label in entity_labels:
        ent_map.setdefault(humanize_entity_label(label), label)

    rel_map: Dict[str, str] = {}
    for label in relation_labels:
        rel_map.setdefault(humanize_relation_label(label), label)
    return ent_map, rel_map


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class GlinerRelexExtractor:
    """Joint entity + relation extraction with a single GLiNER-Relex model.

    Same public interface as GlinerGlirelExtractor, but one model does both
    halves in a single forward pass — no tokenization, no span→token mapping,
    no windowing, and no second model.
    """

    def __init__(
        self,
        entity_labels: Sequence[str],
        relation_labels: Sequence[str],
        model_name: str = DEFAULT_RELEX_MODEL,
        device: str = "cpu",
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.7,
    ):
        try:
            from gliner import GLiNER
        except ImportError as e:
            raise SystemExit(f"Missing dependency ({e}).\n\n    uv pip install -U gliner\n")

        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.ent_map, self.rel_map = build_label_maps(entity_labels, relation_labels)
        self.labels = list(self.ent_map.keys())
        self.relations = list(self.rel_map.keys())

        log.info("Loading GLiNER-Relex: %s (device=%s)", model_name, device)
        t0 = time.perf_counter()
        self.model = GLiNER.from_pretrained(model_name)
        self.model.eval()
        if device != "cpu":
            self.model = self.model.to(device)
        log.info("Model loaded in %.1fs", time.perf_counter() - t0)
        log.info(
            "Ontology: %d entity types, %d relation types",
            len(self.labels), len(self.relations),
        )

    def _to_result(
        self,
        index: int,
        text: str,
        raw_entities: List[Dict[str, Any]],
        raw_relations: List[Dict[str, Any]],
        elapsed_ms: float,
    ) -> ChunkResult:
        entities = [
            Entity(
                name=(e.get("text") or "").strip(),
                type=self.ent_map.get(e.get("label", ""), e.get("label", "")),
                score=round(float(e.get("score", 0.0)), 4),
            )
            for e in raw_entities
            if (e.get("text") or "").strip()
        ]

        triples: List[Triple] = []
        seen: set = set()
        for r in raw_relations:
            head = (r.get("head") or {}).get("text", "").strip()
            tail = (r.get("tail") or {}).get("text", "").strip()
            rel = r.get("relation", "")
            if not head or not tail or head == tail or not rel:
                continue
            predicate = self.rel_map.get(rel, rel.upper().replace(" ", "_"))
            key = (head.lower(), predicate, tail.lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append(Triple(
                subject=head,
                predicate=predicate,
                object=tail,
                confidence=round(float(r.get("score", 0.0)), 4),
            ))

        return ChunkResult(
            index=index,
            text=text,
            entities=entities,
            triples=triples,
            entity_ms=elapsed_ms,
        )

    def extract_chunk(self, index: int, text: str) -> ChunkResult:
        return self.extract_document([text])[0]

    def extract_document(self, chunks: Sequence[str]) -> List[ChunkResult]:
        results: List[ChunkResult] = []

        for start in range(0, len(chunks), RELEX_BATCH):
            batch = list(chunks[start: start + RELEX_BATCH])
            t0 = time.perf_counter()
            try:
                entities, relations = self.model.inference(
                    texts=batch,
                    labels=self.labels,
                    relations=self.relations,
                    threshold=self.entity_threshold,
                    relation_threshold=self.relation_threshold,
                    return_relations=True,
                )
            except Exception as e:
                log.warning("GLiNER-Relex failed on batch at %d: %s", start, e)
                entities, relations = [[] for _ in batch], [[] for _ in batch]

            per_item_ms = (time.perf_counter() - t0) * 1000 / max(1, len(batch))

            for offset, chunk in enumerate(batch):
                result = self._to_result(
                    start + offset,
                    chunk,
                    entities[offset] if offset < len(entities) else [],
                    relations[offset] if offset < len(relations) else [],
                    per_item_ms,
                )
                results.append(result)

            log.info(
                "chunks %d-%d/%d — %d entities, %d triples (%.0fms/chunk)",
                start + 1, start + len(batch), len(chunks),
                sum(len(r.entities) for r in results[start:]),
                sum(len(r.triples) for r in results[start:]),
                per_item_ms,
            )

        return results


def _patch_glirel_hub_compat(glirel_cls: Any) -> None:
    """Make glirel 1.2.x loadable under huggingface_hub >= 1.x.

    GLiREL._from_pretrained declares `proxies` and `resume_download` as required
    keyword-only args. Newer huggingface_hub versions dropped both from the call
    it makes into the mixin, so loading dies with a TypeError before any weights
    are touched. Injecting the defaults restores the old contract.
    """
    original = glirel_cls._from_pretrained
    if getattr(original, "_hub_compat_patched", False):
        return

    def _patched(*args, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return original(*args, **kwargs)

    _patched._hub_compat_patched = True
    glirel_cls._from_pretrained = _patched


class GlinerGlirelExtractor:
    """Drop-in replacement for graphrag_client._extract_chunk, sans LLM."""

    def __init__(
        self,
        entity_labels: Sequence[str],
        relation_labels: Sequence[str],
        gliner_model: str = DEFAULT_GLINER_MODEL,
        glirel_model: str = DEFAULT_GLIREL_MODEL,
        device: str = "cpu",
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.5,
    ):
        try:
            from gliner import GLiNER
            from glirel import GLiREL
        except ImportError as e:
            raise SystemExit(
                f"Missing dependency ({e}).\n\n"
                "    uv pip install gliner glirel torch\n"
            )

        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold

        self.ent_map, self.rel_map = build_label_maps(entity_labels, relation_labels)
        self.gliner_labels = list(self.ent_map.keys())
        # GLiREL needs the negative class available as a prediction target.
        self.glirel_labels = list(self.rel_map.keys()) + [NO_RELATION]

        log.info("Loading GLiNER: %s (device=%s)", gliner_model, device)
        t0 = time.perf_counter()
        self.gliner = GLiNER.from_pretrained(gliner_model)
        self.gliner.eval()
        if device != "cpu":
            self.gliner = self.gliner.to(device)

        log.info("Loading GLiREL: %s", glirel_model)
        _patch_glirel_hub_compat(GLiREL)
        self.glirel = GLiREL.from_pretrained(glirel_model)
        self.glirel.eval()
        if device != "cpu":
            self.glirel = self.glirel.to(device)

        log.info("Models loaded in %.1fs", time.perf_counter() - t0)
        log.info(
            "Ontology: %d entity types, %d relation types",
            len(self.gliner_labels), len(self.rel_map),
        )

    # -- entities ----------------------------------------------------------

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        raw = self.gliner.predict_entities(
            text, self.gliner_labels, threshold=self.entity_threshold
        )
        # GLiNER can return overlapping spans; keep the highest-scoring per span.
        best: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for ent in raw:
            key = (ent["start"], ent["end"])
            if key not in best or ent.get("score", 0) > best[key].get("score", 0):
                best[key] = ent
        return sorted(best.values(), key=lambda e: e["start"])

    # -- relations ---------------------------------------------------------

    def _windows(
        self,
        tokens: List[str],
        ner: List[List[Any]],
    ) -> Iterable[Tuple[List[str], List[List[Any]]]]:
        """Yield (tokens, ner) windows that fit inside GLiREL's context limit.

        Entity token indices are rebased to the window, and entities straddling
        a window boundary are dropped from that window (they appear whole in the
        overlapping one).
        """
        if len(tokens) <= GLIREL_WINDOW_TOKENS:
            yield tokens, ner
            return

        for start in range(0, len(tokens), GLIREL_WINDOW_STRIDE):
            end = min(start + GLIREL_WINDOW_TOKENS, len(tokens))
            window_ner = [
                [e[0] - start, e[1] - start, e[2], e[3]]
                for e in ner
                if e[0] >= start and e[1] < end
            ]
            if len(window_ner) >= 2:
                yield tokens[start:end], window_ner
            if end == len(tokens):
                break

    def _extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Triple]:
        if len(entities) < 2:
            return []

        tokens, spans = tokenize_with_spans(text)
        ner: List[List[Any]] = []
        for ent in entities:
            rng = char_span_to_token_range(spans, ent["start"], ent["end"])
            if rng is None:
                continue
            ner.append([rng[0], rng[1], ent["label"], ent["text"]])

        if len(ner) < 2:
            return []

        seen: set = set()
        triples: List[Triple] = []

        for win_tokens, win_ner in self._windows(tokens, ner):
            try:
                relations = self.glirel.predict_relations(
                    win_tokens,
                    self.glirel_labels,
                    threshold=self.relation_threshold,
                    ner=win_ner,
                    top_k=1,
                )
            except Exception as e:
                log.warning("GLiREL failed on a window: %s", e)
                continue

            for rel in relations or []:
                label = rel.get("label", "")
                if not label or label == NO_RELATION:
                    continue

                subject = " ".join(rel.get("head_text") or []) \
                    if isinstance(rel.get("head_text"), list) else str(rel.get("head_text", ""))
                obj = " ".join(rel.get("tail_text") or []) \
                    if isinstance(rel.get("tail_text"), list) else str(rel.get("tail_text", ""))
                subject, obj = subject.strip(), obj.strip()
                if not subject or not obj or subject == obj:
                    continue

                predicate = self.rel_map.get(label, label.upper().replace(" ", "_"))
                key = (subject.lower(), predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)

                triples.append(Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=round(float(rel.get("score", 0.0)), 4),
                ))

        return triples

    # -- public ------------------------------------------------------------

    def extract_chunk(self, index: int, text: str) -> ChunkResult:
        t0 = time.perf_counter()
        raw_entities = self._extract_entities(text)
        t1 = time.perf_counter()
        triples = self._extract_relations(text, raw_entities)
        t2 = time.perf_counter()

        entities = [
            Entity(
                name=e["text"].strip(),
                type=self.ent_map.get(e["label"], e["label"]),
                score=round(float(e.get("score", 0.0)), 4),
            )
            for e in raw_entities
            if e["text"].strip()
        ]

        return ChunkResult(
            index=index,
            text=text,
            entities=entities,
            triples=triples,
            entity_ms=(t1 - t0) * 1000,
            relation_ms=(t2 - t1) * 1000,
        )

    def extract_document(self, chunks: Sequence[str]) -> List[ChunkResult]:
        results: List[ChunkResult] = []
        for i, chunk in enumerate(chunks):
            result = self.extract_chunk(i, chunk)
            results.append(result)
            log.info(
                "chunk %d/%d — %d entities, %d triples (%.0fms ner + %.0fms rel)",
                i + 1, len(chunks), len(result.entities), len(result.triples),
                result.entity_ms, result.relation_ms,
            )
        return results


# ---------------------------------------------------------------------------
# Optional: reuse the repo's LLM ontology detector / extractor for comparison
# ---------------------------------------------------------------------------

def detect_ontology_via_llm(text: str) -> Tuple[List[str], List[str]]:
    """Run the repo's own _detect_schema so the ontology matches production."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clients.graphrag_client import _detect_schema  # noqa: E402

    log.info("Detecting ontology via LLM (same path as ingestion)")
    schema = asyncio.run(_detect_schema(text, "spike"))
    if not schema.entity_labels:
        log.warning("Ontology detection returned nothing — using defaults")
        return DEFAULT_ENTITY_LABELS, DEFAULT_RELATION_LABELS
    return schema.entity_labels, schema.relation_labels


def extract_via_llm(
    chunks: Sequence[str],
    entity_labels: Sequence[str],
    relation_labels: Sequence[str],
) -> List[ChunkResult]:
    """Run the current LLM extractor on the same chunks, for diffing."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clients.graphrag_client import _extract_chunk, _OntologySchema  # noqa: E402

    schema = _OntologySchema()
    schema.entity_labels = list(entity_labels)
    schema.relation_labels = list(relation_labels)

    async def _run() -> List[ChunkResult]:
        sem = asyncio.Semaphore(10)
        t0 = time.perf_counter()
        extractions = await asyncio.gather(
            *[_extract_chunk(c, schema, sem) for c in chunks]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_chunk = elapsed_ms / max(1, len(chunks))

        return [
            ChunkResult(
                index=i,
                text=chunk,
                entities=[Entity(name=e.name, type=e.type) for e in ex.entities],
                triples=[
                    Triple(
                        subject=t.subject,
                        predicate=t.predicate,
                        object=t.object,
                        confidence=t.confidence,
                    )
                    for t in ex.triples
                ],
                entity_ms=per_chunk,
            )
            for i, (chunk, ex) in enumerate(zip(chunks, extractions))
        ]

    log.info("Running LLM extractor on %d chunks for comparison", len(chunks))
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _entity_keys(results: Sequence[ChunkResult]) -> set:
    return {e.name.strip().lower() for r in results for e in r.entities if e.name.strip()}


def _triple_keys(results: Sequence[ChunkResult]) -> set:
    return {
        (t.subject.strip().lower(), t.predicate, t.object.strip().lower())
        for r in results for t in r.triples
    }


def summarize(results: Sequence[ChunkResult], label: str) -> Dict[str, Any]:
    entities = _entity_keys(results)
    triples = _triple_keys(results)
    total_ms = sum(r.entity_ms + r.relation_ms for r in results)

    types: Dict[str, int] = {}
    for r in results:
        for e in r.entities:
            types[e.type] = types.get(e.type, 0) + 1

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  chunks             {len(results)}")
    print(f"  unique entities    {len(entities)}")
    print(f"  unique triples     {len(triples)}")
    print(f"  wall time          {total_ms / 1000:.1f}s")
    print(f"  per chunk          {total_ms / max(1, len(results)):.0f}ms")
    if types:
        print("  entity types:")
        for t, n in sorted(types.items(), key=lambda kv: -kv[1])[:12]:
            print(f"      {t:<28} {n}")

    return {
        "chunks": len(results),
        "unique_entities": len(entities),
        "unique_triples": len(triples),
        "total_ms": round(total_ms, 1),
        "ms_per_chunk": round(total_ms / max(1, len(results)), 1),
        "entity_types": types,
    }


def print_diff(encoder: Sequence[ChunkResult], llm: Sequence[ChunkResult]) -> Dict[str, Any]:
    e_ents, l_ents = _entity_keys(encoder), _entity_keys(llm)
    e_trip, l_trip = _triple_keys(encoder), _triple_keys(llm)

    ent_overlap = len(e_ents & l_ents)
    trip_overlap = len(e_trip & l_trip)

    def pct(n: int, d: int) -> str:
        return f"{(100 * n / d):.0f}%" if d else "n/a"

    print(f"\n{'=' * 62}")
    print("  ENCODER vs LLM")
    print(f"{'=' * 62}")
    print(f"  entities   both={ent_overlap}  "
          f"encoder-only={len(e_ents - l_ents)}  llm-only={len(l_ents - e_ents)}")
    print(f"             recall vs LLM: {pct(ent_overlap, len(l_ents))}")
    print(f"  triples    both={trip_overlap}  "
          f"encoder-only={len(e_trip - l_trip)}  llm-only={len(l_trip - e_trip)}")
    print(f"             recall vs LLM: {pct(trip_overlap, len(l_trip))}")

    missed = sorted(l_ents - e_ents)[:15]
    if missed:
        print("\n  entities the LLM found that the encoder missed:")
        for name in missed:
            print(f"      · {name}")

    return {
        "entity_overlap": ent_overlap,
        "entity_encoder_only": len(e_ents - l_ents),
        "entity_llm_only": len(l_ents - e_ents),
        "triple_overlap": trip_overlap,
        "triple_encoder_only": len(e_trip - l_trip),
        "triple_llm_only": len(l_trip - e_trip),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _split_labels(raw: Optional[str], fallback: List[str]) -> List[str]:
    if not raw:
        return fallback
    return [p.strip() for p in raw.split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a knowledge graph from a document using GLiNER + GLiREL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", type=Path, help="PDF, DOCX, TXT or MD file")
    ap.add_argument("--entity-labels", help="Comma-separated entity types")
    ap.add_argument("--relation-labels", help="Comma-separated relation types")
    ap.add_argument("--auto-ontology", action="store_true",
                    help="Detect the ontology with the repo's LLM detector (needs API keys)")
    ap.add_argument("--joint", action="store_true",
                    help="Use GLiNER-Relex (one model, joint NER+RE) instead of GLiNER+GLiREL")
    ap.add_argument("--gliner-model", default=DEFAULT_GLINER_MODEL)
    ap.add_argument("--glirel-model", default=DEFAULT_GLIREL_MODEL)
    ap.add_argument("--relex-model", default=DEFAULT_RELEX_MODEL)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--entity-threshold", type=float, default=0.5)
    ap.add_argument("--relation-threshold", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--no-clean", action="store_true",
                    help="Skip PDF line-reflow cleaning (on by default)")
    ap.add_argument("--max-chunks", type=int, help="Only process the first N chunks")
    ap.add_argument("--compare-llm", action="store_true",
                    help="Also run the current LLM extractor and diff the results")
    ap.add_argument("--json-out", type=Path, help="Write full results to this path")
    args = ap.parse_args()

    if not args.file.exists():
        log.error("File not found: %s", args.file)
        return 1

    # 1. text
    text = extract_text(args.file)
    if not text.strip():
        log.error("No text extracted from %s", args.file.name)
        return 1
    log.info("Extracted %d characters", len(text))

    if not args.no_clean:
        before_breaks = len(re.findall(r"\n+", text))
        text = clean_pdf_text(text)
        after_breaks = len(re.findall(r"\n+", text))
        log.info(
            "Reflowed wrapped lines: %d breaks -> %d (%d rejoined)",
            before_breaks, after_breaks, before_breaks - after_breaks,
        )

    # 2. chunks
    chunks = chunk_text(text, chunk_size=args.chunk_size)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    log.info("Split into %d chunks", len(chunks))

    # 3. ontology
    if args.auto_ontology:
        entity_labels, relation_labels = detect_ontology_via_llm(text)
    else:
        entity_labels = _split_labels(args.entity_labels, DEFAULT_ENTITY_LABELS)
        relation_labels = _split_labels(args.relation_labels, DEFAULT_RELATION_LABELS)
    log.info("Entity types:   %s", ", ".join(entity_labels))
    log.info("Relation types: %s", ", ".join(relation_labels))

    # 4. encoder extraction
    if args.joint:
        extractor = GlinerRelexExtractor(
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            model_name=args.relex_model,
            device=args.device,
            entity_threshold=args.entity_threshold,
            relation_threshold=args.relation_threshold,
        )
        encoder_name = "GLiNER-Relex (joint)"
    else:
        extractor = GlinerGlirelExtractor(
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            gliner_model=args.gliner_model,
            glirel_model=args.glirel_model,
            device=args.device,
            entity_threshold=args.entity_threshold,
            relation_threshold=args.relation_threshold,
        )
        encoder_name = "GLiNER + GLiREL"

    encoder_results = extractor.extract_document(chunks)
    payload: Dict[str, Any] = {
        "file": str(args.file),
        "chunks": len(chunks),
        "extractor": encoder_name,
        "entity_labels": entity_labels,
        "relation_labels": relation_labels,
        "encoder": {
            "stats": summarize(encoder_results, encoder_name),
            "results": [asdict(r) for r in encoder_results],
        },
    }

    # 5. optional LLM comparison
    if args.compare_llm:
        llm_results = extract_via_llm(chunks, entity_labels, relation_labels)
        payload["llm"] = {
            "stats": summarize(llm_results, "Current LLM extractor"),
            "results": [asdict(r) for r in llm_results],
        }
        payload["diff"] = print_diff(encoder_results, llm_results)

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("Wrote results to %s", args.json_out)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
