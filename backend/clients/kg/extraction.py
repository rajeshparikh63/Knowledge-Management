"""
Ontology-constrained entity + relation extraction.

The LLM extracts against the org's ACTIVE ontology (from OntologyManager), then
a hard validation layer enforces the constraints regardless of what the LLM
returns:
  * entity types must exist in the ontology
  * relation predicates must exist in the ontology
  * relation source/target entity types must satisfy the ontology's allowed pairs
  * junk is dropped: bare numbers, sentence fragments, generic nouns, self-loops
  * confidence floor

This is what stops the old failure modes (877 predicates, 30% untyped entities,
250 sentence-fragment "entities", numbers-as-entities) at the door.

LLM for now (per the plan); the same active-ontology schema can later drive a
fine-tuned GLiNER2 encoder without changing this module's interface.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.logger import logger
from app.settings import settings
from clients.kg.llm import llm_json

# ---- junk filters ----------------------------------------------------------
MAX_NAME_LEN = 60          # longer than this = a clause, not an entity
MIN_NAME_LEN = 2
MAX_NAME_WORDS = 8         # real entity names are rarely longer
_BARE_NUMBER = re.compile(r"^[\d\s.,%$€£+\-/()x×]+$", re.IGNORECASE)
# Monetary / quantity amounts are VALUES, not entities ("$57.0 billion",
# "45 billion dollars", "25 basis points", "$1.50 per share"). The named metric
# ("net revenue", "ROTCE") is the entity; the amount belongs in the chunk text.
_AMOUNT = re.compile(
    r"^[\$€£]?\s*[\d.,]+\s*"
    r"(billion|million|trillion|thousand|bn|mn|k|percent|%|basis\s+points|bps|"
    r"dollars?|euros?|pounds?|cents?|per\s+share|x|times)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")

# Fragment signals: a real entity name doesn't END on a function word, and
# doesn't contain a linking verb in the middle ("... data is the" is a clause).
_TRAILING_STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "and", "or", "to", "in",
    "for", "with", "on", "at", "by", "from", "as", "that", "this", "these",
    "those", "its", "their", "our", "his", "her",
}
_LINKING_VERBS = {"is", "are", "was", "were", "be", "been", "being"}

# Generic nouns that must never be entities (extend as needed).
_GENERIC_NOUNS = {
    "company", "companies", "organization", "organizations", "system", "systems",
    "tool", "tools", "user", "users", "people", "person", "team", "teams",
    "employee", "employees", "product", "products", "service", "services",
    "data", "information", "report", "reports", "document", "documents",
    "market", "markets", "revenue", "growth", "value", "assets", "capital",
    "management", "shareholders", "customers", "clients", "business",
    "businesses", "solution", "solutions", "firm", "firms", "platform",
    "platforms", "consumers", "franchises", "process", "processes", "thing",
    "things", "item", "items", "record", "records", "entity", "entities",
}

MIN_CONFIDENCE = settings.MIN_TRIPLE_CONFIDENCE  # 0.7


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass
class Extraction:
    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)   # reason -> count (observability)


def _clean_name(raw: str) -> str:
    return _WHITESPACE.sub(" ", (raw or "").replace("\x00", "")).strip()


def is_junk_entity(name: str) -> Optional[str]:
    """Return a reason string if the name is junk, else None."""
    if not name:
        return "empty"
    if len(name) < MIN_NAME_LEN:
        return "too_short"
    if len(name) > MAX_NAME_LEN:
        return "fragment"
    if _BARE_NUMBER.match(name):
        return "bare_number"
    if _AMOUNT.match(name):
        return "amount"
    if name.lower() in _GENERIC_NOUNS:
        return "generic_noun"
    words = name.split()
    if len(words) > MAX_NAME_WORDS:
        return "fragment"
    # ends on a function word, e.g. "... is the"
    if words and words[-1].lower().strip(".,;:!?") in _TRAILING_STOP:
        return "fragment"
    # linking verb somewhere in the middle → it's a clause, not a name
    if any(w.lower() in _LINKING_VERBS for w in words[1:]):
        return "fragment"
    return None


def _relation_allows(ontology: dict, predicate: str, src_type: str, tgt_type: str) -> bool:
    spec = ontology["relation_types"].get(predicate)
    if not spec:
        return False
    src, tgt = spec.get("src") or ["*"], spec.get("tgt") or ["*"]
    return ("*" in src or src_type in src) and ("*" in tgt or tgt_type in tgt)


def _build_prompt(ontology: dict) -> str:
    ent_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in ontology["entity_types"].items()
    )
    rel_lines = "\n".join(
        f"- {name} ({'/'.join(spec.get('src') or ['*'])} -> "
        f"{'/'.join(spec.get('tgt') or ['*'])}): {spec.get('desc','')}"
        for name, spec in ontology["relation_types"].items()
    )
    return f"""\
You extract a knowledge graph from a passage using ONLY the types listed below.

PRECISION OVER RECALL. A few correct, specific facts beat many vague ones.

# ENTITY TYPES (assign each entity exactly one; pick the closest fit)
{ent_lines}

# Step 1 — Entities
Extract SPECIFIC named entities: proper nouns or precise technical identifiers.
REJECT generic nouns ("company", "system", "user"), bare numbers/amounts, dates
written as plain numbers, and any sentence fragment. If it isn't a nameable
thing, don't extract it.

# RELATION TYPES (use ONLY these predicates; respect the source -> target types)
{rel_lines}

# Step 2 — Relations
Each relation is a CONCRETE FACT stated in the passage between two entities from
Step 1.
- predicate: exactly one from the RELATION TYPES list
- subject and object: must exactly match names from Step 1
- respect the allowed source -> target entity types
- confidence: 0.7-1.0 only (omit anything you are less sure of)

Return ONLY JSON, no prose:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]}}
"""


def validate(raw: dict, ontology: dict) -> Extraction:
    """Enforce the ontology + junk rules on a raw LLM extraction."""
    result = Extraction()
    dropped: Dict[str, int] = {}

    def drop(reason: str):
        dropped[reason] = dropped.get(reason, 0) + 1

    valid_types = set(ontology["entity_types"].keys())

    # ---- entities ----
    entity_type: Dict[str, str] = {}   # cleaned name -> type (first wins)
    for e in (raw.get("entities") or []):
        name = _clean_name(e.get("name", ""))
        etype = (e.get("type") or "").strip()
        reason = is_junk_entity(name)
        if reason:
            drop(f"entity_{reason}")
            continue
        if etype not in valid_types:
            drop("entity_off_ontology")
            continue
        if name not in entity_type:
            entity_type[name] = etype
            result.entities.append(Entity(name=name, type=etype))

    # ---- relations ----
    seen = set()
    for t in (raw.get("relations") or []):
        subj = _clean_name(t.get("subject", ""))
        obj = _clean_name(t.get("object", ""))
        pred = (t.get("predicate") or "").strip()
        try:
            conf = float(t.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        if not subj or not obj or subj == obj:
            drop("rel_empty_or_selfloop")
            continue
        if pred not in ontology["relation_types"]:
            drop("rel_off_ontology")
            continue
        if subj not in entity_type or obj not in entity_type:
            drop("rel_endpoint_not_entity")     # both ends must be extracted entities
            continue
        if conf < MIN_CONFIDENCE:
            drop("rel_low_confidence")
            continue
        if not _relation_allows(ontology, pred, entity_type[subj], entity_type[obj]):
            drop("rel_bad_src_tgt")
            continue
        key = (subj.lower(), pred, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        result.triples.append(Triple(subject=subj, predicate=pred, object=obj, confidence=conf))

    result.dropped = dropped
    return result


async def extract_chunk(text: str, ontology: dict, sem: asyncio.Semaphore) -> Extraction:
    """Extract one chunk against the active ontology, then validate."""
    if not text or not text.strip():
        return Extraction()
    prompt = _build_prompt(ontology) + f"\n\nPASSAGE:\n{text}"
    async with sem:
        try:
            raw = await llm_json(prompt, model=settings.EXTRACTION_MODEL, max_tokens=4096)
        except Exception as e:
            logger.warning(f"Extraction failed for a chunk: {e}")
            return Extraction()
    if not isinstance(raw, dict):
        return Extraction()
    return validate(raw, ontology)
