"""
Controlled-extension ontology manager — the heart of the KG revamp.

Resolves the fixed-vs-auto contradiction the way Graphiti / LlamaIndex-Dynamic /
Microsoft-GraphRAG do: start from a clean SEED ontology, let new types be
DISCOVERED from documents, but pass every proposal through canonicalization so
synonyms merge instead of piling up (the old pipeline's 877-predicate drift).

Per document:
  1. Detect proposed entity/relation types from a sample (LLM), shown the
     current active ontology as guidance ("reuse these; propose new only if
     genuinely absent").
  2. Canonicalize each proposal through 3 gates:
       a. exact name match            -> reuse
       b. description-embedding close -> merge into existing
       c. borderline                  -> one LLM "same concept?" judge
  3. Genuinely-new proposals are HELD as pending (not dropped). A pending type
     is PROMOTED to active only when it is distinct AND recurring
     (seen in >= promote_after documents) AND the active set is under the cap.
  4. Persist per org so the ontology grows and survives restarts.

Extraction then runs against `active_ontology()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from app.logger import logger
from clients.kg import ontology as seed
from clients.kg.llm import embed_texts, cosine_distance, llm_json, llm_yes_no

# Canonicalization thresholds (cosine distance over description embeddings).
# CALIBRATED on text-embedding-3-small over short type descriptions: synonyms
# (Enterprise~Organization, GeographicPlace~Location) land ~0.39-0.44; genuinely
# distinct concepts (Vessel, Vulnerability) land ~0.56-0.60. So the LLM judge
# must cover the wide ambiguous middle band, not a narrow sliver.
MERGE_DISTANCE = 0.35      # below -> clearly same concept, auto-merge
DISTINCT_DISTANCE = 0.55   # above -> clearly different, auto-distinct
# between the two -> ask the LLM judge (the synonym band lives here)

PROMOTE_AFTER_DOCS = 2     # a pending type must recur in >= N docs to promote
CAP_ENTITY_TYPES = 40
CAP_RELATION_TYPES = 50


@dataclass
class TypeEntry:
    name: str
    description: str
    status: str = "active"          # "active" | "pending"
    doc_count: int = 1              # how many documents proposed it (recurrence)
    src: List[str] = field(default_factory=list)   # relations only
    tgt: List[str] = field(default_factory=list)   # relations only
    embedding: Optional[List[float]] = None        # of description, cached (not persisted)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)    # re-embedded on load; keep persistence small
        return d


class OntologyManager:
    def __init__(self, organization_id: str):
        self.org_id = organization_id
        self.entities: Dict[str, TypeEntry] = {}
        self.relations: Dict[str, TypeEntry] = {}
        self.pending_entities: Dict[str, TypeEntry] = {}
        self.pending_relations: Dict[str, TypeEntry] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def seeded(cls, organization_id: str) -> "OntologyManager":
        """Fresh manager pre-loaded with the canonical seed ontology."""
        m = cls(organization_id)
        for name, desc in seed.ENTITY_TYPES.items():
            m.entities[name] = TypeEntry(name, desc, status="active", doc_count=0)
        for name, spec in seed.RELATION_TYPES.items():
            m.relations[name] = TypeEntry(
                name, str(spec["desc"]), status="active", doc_count=0,
                src=list(spec["src"]), tgt=list(spec["tgt"]),
            )
        return m

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "org_id": self.org_id,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "pending_entities": {k: v.to_dict() for k, v in self.pending_entities.items()},
            "pending_relations": {k: v.to_dict() for k, v in self.pending_relations.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OntologyManager":
        m = cls(d.get("org_id", ""))
        def load(section):
            return {k: TypeEntry(**v) for k, v in (d.get(section) or {}).items()}
        m.entities = load("entities")
        m.relations = load("relations")
        m.pending_entities = load("pending_entities")
        m.pending_relations = load("pending_relations")
        # An empty persisted ontology should still get the seed.
        if not m.entities and not m.relations:
            return cls.seeded(d.get("org_id", ""))
        return m

    # -- active ontology (what extraction uses) ----------------------------

    def active_ontology(self) -> dict:
        return {
            "entity_types": {k: v.description for k, v in self.entities.items()},
            "relation_types": {
                k: {"desc": v.description, "src": v.src, "tgt": v.tgt}
                for k, v in self.relations.items()
            },
        }

    # -- embedding cache ---------------------------------------------------

    async def _ensure_embeddings(self, table: Dict[str, TypeEntry]) -> None:
        missing = [e for e in table.values() if e.embedding is None]
        if not missing:
            return
        vecs = await embed_texts([e.description for e in missing])
        for entry, vec in zip(missing, vecs):
            entry.embedding = vec

    def _nearest(self, emb: List[float], table: Dict[str, TypeEntry]
                 ) -> Tuple[Optional[str], float]:
        best_name, best_dist = None, 2.0
        for name, entry in table.items():
            if entry.embedding is None:
                continue
            d = cosine_distance(emb, entry.embedding)
            if d < best_dist:
                best_name, best_dist = name, d
        return best_name, best_dist

    # -- the 3 gates -------------------------------------------------------

    async def _canonicalize(
        self, name: str, desc: str, emb: List[float], active: Dict[str, TypeEntry],
    ) -> Optional[str]:
        """Return the existing canonical name this proposal collapses into, or
        None if it is genuinely new."""
        # Gate 1 — exact name match
        if name in active:
            return name
        # Gate 2 — description embedding similarity
        nearest, dist = self._nearest(emb, active)
        if nearest is None:
            return None
        if dist < MERGE_DISTANCE:
            return nearest
        if dist > DISTINCT_DISTANCE:
            return None
        # Gate 3 — borderline: one LLM judge call.
        # Framed as a MERGE decision, not identity: "same kind of thing?" is too
        # strict because an existing type is often BROADER than the proposal
        # (Organization covers Enterprise). We ask whether the existing type
        # already covers the proposal, so subtypes fold into their parent.
        prompt = (
            "In a knowledge-graph ontology we avoid duplicate or overlapping types.\n"
            f"Existing type B: \"{active[nearest].description}\"\n"
            f"New proposed type A: \"{desc}\"\n"
            "Does existing type B already cover proposal A, so A should MERGE into "
            "B rather than become its own separate type?"
        )
        try:
            if await llm_yes_no(prompt):
                return nearest
        except Exception as e:
            logger.warning(f"Ontology judge failed ({e}); treating as distinct")
        return None

    # -- integration of one document's proposals ---------------------------

    async def _integrate(
        self, kind: str, proposals: List[dict],
        active: Dict[str, TypeEntry], pending: Dict[str, TypeEntry], cap: int,
    ) -> None:
        proposals = [p for p in proposals if (p.get("name") or "").strip()
                     and (p.get("description") or "").strip()]
        if not proposals:
            return
        await self._ensure_embeddings(active)
        embs = await embed_texts([p["description"] for p in proposals])

        promoted, merged = [], []
        for p, emb in zip(proposals, embs):
            name = p["name"].strip()
            desc = p["description"].strip()
            canon = await self._canonicalize(name, desc, emb, active)
            if canon is not None:
                merged.append((name, canon))
                continue
            # Genuinely new → hold as pending, count recurrence (keep-don't-drop).
            entry = pending.get(name)
            if entry is None:
                entry = TypeEntry(name, desc, status="pending", doc_count=1, embedding=emb,
                                  src=list(p.get("src") or []), tgt=list(p.get("tgt") or []))
                pending[name] = entry
            else:
                entry.doc_count += 1
                entry.embedding = emb
            # Promote if distinct AND recurring AND under cap.
            if entry.doc_count >= PROMOTE_AFTER_DOCS and len(active) < cap:
                entry.status = "active"
                active[name] = entry
                pending.pop(name, None)
                promoted.append(name)

        if promoted or merged:
            logger.info(
                f"[ontology:{self.org_id}] {kind}: +{len(promoted)} promoted "
                f"{promoted or ''}, {len(merged)} merged, "
                f"active={len(active)}/{cap}, pending={len(pending)}"
            )

    async def ensure_for_document(self, sample_text: str) -> dict:
        """Detect this doc's proposed types, integrate them, return active ontology."""
        proposed = await self._detect(sample_text)
        await self._integrate("entity", proposed.get("entities", []),
                              self.entities, self.pending_entities, CAP_ENTITY_TYPES)
        await self._integrate("relation", proposed.get("relations", []),
                              self.relations, self.pending_relations, CAP_RELATION_TYPES)
        return self.active_ontology()

    # -- detection (LLM proposes types, seed shown as guidance) ------------

    async def _detect(self, sample_text: str) -> dict:
        sample = sample_text[:12000].strip()
        ent_list = ", ".join(self.entities.keys())
        rel_list = ", ".join(self.relations.keys())
        prompt = f"""\
You maintain a knowledge-graph ontology. Below is the CURRENT ontology and a
SAMPLE of a new document. Propose the entity and relation types needed to
represent this document.

REUSE an existing type whenever it fits. Propose a NEW type ONLY if the concept
is genuinely absent — never a synonym or near-duplicate of an existing type.

CURRENT entity types: {ent_list}
CURRENT relation types: {rel_list}

Rules:
- Entity labels: PascalCase, singular.
- Relation labels: SCREAMING_SNAKE_CASE, a verb phrase.
- Give every proposed type a one-line description.
- Each relation must list allowed source/target entity-type pairs.

Return ONLY JSON:
{{"entities": [{{"name": "...", "description": "..."}}],
  "relations": [{{"name": "...", "description": "...", "src": ["..."], "tgt": ["..."]}}]}}

SAMPLE:
\"\"\"{sample}\"\"\"
"""
        try:
            data = await llm_json(prompt)
        except Exception as e:
            logger.warning(f"[ontology:{self.org_id}] detection failed: {e}")
            return {"entities": [], "relations": []}
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        return {
            "entities": data.get("entities") or [],
            "relations": data.get("relations") or [],
        }
