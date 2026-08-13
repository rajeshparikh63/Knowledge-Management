"""
GLiNER2-based extractor — a drop-in alternative to the LLM extractor.

Same contract as extraction.extract_chunk: produce raw entities/relations, then
run the SAME ontology `validate()`. The difference from the earlier raw GLiNER
spike is exactly that validation step — the ontology now drops GLiNER2's
off-ontology predicates, bad source/target pairs, and dangling endpoints, so we
see GLiNER2's *usable* yield, not its raw noise.

The active ontology (from OntologyManager) drives the GLiNER2 schema:
  * entity types -> humanized labels + their descriptions
  * relation types -> humanized predicate labels
Labels are humanized because encoders score natural-language labels better
("military unit" >> "MilitaryUnit"), then mapped back to the ontology names.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.logger import logger
from clients.kg.extraction import Extraction, validate, MIN_CONFIDENCE


# Encoder models are expensive to load (~1.9 GB / several seconds) — cache the
# loaded model per name so the hybrid extractor reuses it across documents/orgs.
_MODEL_CACHE: Dict[str, Any] = {}


def _load_gliner(model_name: str):
    if model_name not in _MODEL_CACHE:
        from gliner import GLiNER
        logger.info(f"Loading GLiNER model (cached): {model_name}")
        m = GLiNER.from_pretrained(model_name)
        m.eval()
        _MODEL_CACHE[model_name] = m
    return _MODEL_CACHE[model_name]


def _humanize_entity(label: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", label.replace("_", " "))
    return re.sub(r"\s+", " ", spaced).strip().lower()


def _humanize_relation(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("_", " ")).strip().lower()


class GlinerExtractor:
    def __init__(
        self,
        ontology: dict,
        model_name: str = "fastino/gliner2-base-v1",
        device: str = "cpu",
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.5,
    ):
        from gliner2 import GLiNER2  # lazy: heavy import

        self.ontology = ontology
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold

        # humanized label -> ontology name (for mapping back)
        self.ent_label_to_name = {_humanize_entity(n): n for n in ontology["entity_types"]}
        self.rel_label_to_name = {_humanize_relation(n): n for n in ontology["relation_types"]}
        # GLiNER2 entity schema: {humanized_label: description}
        self.ent_schema = {
            _humanize_entity(n): desc for n, desc in ontology["entity_types"].items()
        }
        self.rel_labels = list(self.rel_label_to_name.keys())

        logger.info(f"Loading GLiNER2: {model_name} ({device})")
        self.model = GLiNER2.from_pretrained(model_name)
        self.model.eval()
        if device != "cpu":
            self.model = self.model.to(device)

    def _raw(self, text: str) -> dict:
        schema = self.model.create_schema().entities(self.ent_schema).relations(self.rel_labels)
        out = self.model.extract(text, schema, threshold=self.entity_threshold,
                                 include_confidence=True)

        entities: List[Dict[str, str]] = []
        for label, vals in (out.get("entities") or {}).items():
            name_type = self.ent_label_to_name.get(label, label)
            for v in vals:
                txt = (v[0] if isinstance(v, (list, tuple)) else
                       v.get("text") if isinstance(v, dict) else str(v))
                if txt and str(txt).strip():
                    entities.append({"name": str(txt).strip(), "type": name_type})

        relations: List[Dict[str, Any]] = []
        rel_block = out.get("relations") or out.get("relation_extraction") or {}
        for rel_label, insts in rel_block.items():
            predicate = self.rel_label_to_name.get(rel_label, rel_label.upper().replace(" ", "_"))
            for it in insts:
                if isinstance(it, dict):
                    h, t = it.get("head"), it.get("tail")
                    head = h.get("text") if isinstance(h, dict) else h
                    tail = t.get("text") if isinstance(t, dict) else t
                    score = it.get("score", 0.9)
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    head, tail, score = it[0], it[1], 0.9
                else:
                    continue
                if head and tail:
                    relations.append({
                        "subject": str(head).strip(), "predicate": predicate,
                        "object": str(tail).strip(),
                        # encoders don't emit calibrated confidence; treat a
                        # returned relation as meeting the floor so validate()'s
                        # ontology checks (not confidence) do the filtering.
                        "confidence": max(float(score or 0.0), MIN_CONFIDENCE),
                    })
        return {"entities": entities, "relations": relations}

    def extract_chunk(self, text: str) -> Extraction:
        if not text or not text.strip():
            return Extraction()
        try:
            raw = self._raw(text)
        except Exception as e:
            logger.warning(f"GLiNER2 extraction failed: {e}")
            return Extraction()
        return validate(raw, self.ontology)


class GlinerRelexExtractor:
    """Joint entity+relation extraction with a single GLiNER-Relex model, through
    the same ontology validate(). Uses the `gliner` package's inference() API
    (flat entity + relation label lists, no per-type descriptions)."""

    def __init__(
        self,
        ontology: dict,
        model_name: str = "knowledgator/gliner-relex-large-v1.0",
        device: str = "cpu",
        entity_threshold: float = 0.5,
        relation_threshold: float = 0.4,
    ):
        self.ontology = ontology
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.ent_label_to_name = {_humanize_entity(n): n for n in ontology["entity_types"]}
        self.rel_label_to_name = {_humanize_relation(n): n for n in ontology["relation_types"]}
        self.ent_labels = list(self.ent_label_to_name.keys())
        self.rel_labels = list(self.rel_label_to_name.keys())

        self.model = _load_gliner(model_name)   # cached across instances
        if device != "cpu":
            self.model = self.model.to(device)

    def _raw(self, text: str) -> dict:
        entities, relations = self.model.inference(
            texts=[text],
            labels=self.ent_labels,
            relations=self.rel_labels,
            threshold=self.entity_threshold,
            relation_threshold=self.relation_threshold,
            return_relations=True,
        )
        ents = entities[0] if entities else []
        rels = relations[0] if relations else []

        raw_entities = []
        for e in ents:
            txt = (e.get("text") or "").strip() if isinstance(e, dict) else ""
            label = e.get("label", "") if isinstance(e, dict) else ""
            if txt:
                raw_entities.append({"name": txt, "type": self.ent_label_to_name.get(label, label)})

        raw_relations = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            head = (r.get("head") or {}).get("text", "").strip() if isinstance(r.get("head"), dict) else str(r.get("head", "")).strip()
            tail = (r.get("tail") or {}).get("text", "").strip() if isinstance(r.get("tail"), dict) else str(r.get("tail", "")).strip()
            rel = r.get("relation", "")
            score = r.get("score", 0.9)
            if head and tail and rel:
                raw_relations.append({
                    "subject": head,
                    "predicate": self.rel_label_to_name.get(rel, rel.upper().replace(" ", "_")),
                    "object": tail,
                    "confidence": max(float(score or 0.0), MIN_CONFIDENCE),
                })
        return {"entities": raw_entities, "relations": raw_relations}

    def extract_chunk(self, text: str) -> Extraction:
        if not text or not text.strip():
            return Extraction()
        try:
            raw = self._raw(text)
        except Exception as e:
            logger.warning(f"GLiNER-Relex extraction failed: {e}")
            return Extraction()
        return validate(raw, self.ontology)
