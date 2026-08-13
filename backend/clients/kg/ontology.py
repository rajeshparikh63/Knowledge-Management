"""
Canonical ontology for the knowledge-graph revamp.

ONTOLOGY-FIRST is the #1 KG best practice: define a bounded, canonical set of
entity and relation types UP FRONT, then extract against it. The old pipeline
let the LLM invent types per document, which drifted to 877 distinct relation
predicates and 20k+ entities (30% orphaned, 31% untyped). This file is the fix.

Rules that keep the graph clean:
  * Entity types: PascalCase, singular. ~14 general types cover the corpus
    (military doctrine, financial reports, IT/risk governance). Add a new type
    ONLY when a concept genuinely doesn't fit — not for a synonym.
  * Relation types: SCREAMING_SNAKE_CASE, each with a fixed meaning and a set of
    allowed (source_type -> target_type) pairs. RELATED_TO is the ONLY generic
    fallback and should be used sparingly.
  * Extraction is constrained to these labels. Nothing else gets written.

Each type carries a one-line description — used both in the LLM extraction prompt
and (later) as the schema fed to an encoder model like GLiNER2.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Entity types  (name -> description)
# ---------------------------------------------------------------------------
ENTITY_TYPES: Dict[str, str] = {
    "Person":       "A named individual human being (e.g. 'Jamie Dimon', 'SFC Diaz'). Never a title or a group.",
    "Organization": "A named company, agency, military unit, or institution (e.g. '3rd Battalion', 'JPMorganChase', 'FDIC').",
    "Location":     "A named geographic place: city, region, country, base, or site (e.g. 'Kandahar', 'Bagram Airfield').",
    "Equipment":    "A named physical system, vehicle, weapon, or product (e.g. 'M2 Bradley', 'AN/PRC-152').",
    "System":       "A named software, information, or mission system / platform (e.g. 'TAK', 'GCSS-Army').",
    "Concept":      "A named abstract concept, doctrine, or topic (e.g. 'blue force tracking', 'zero trust').",
    "Procedure":    "A named process, protocol, drill, or standard operating procedure (e.g. 'TCCC protocol', 'Battle Drill 6').",
    "Event":        "A named occurrence, operation, incident, or milestone (e.g. 'JPMorgan/Bank One merger').",
    "Date":         "A specific date, time, or time period (e.g. 'March 31, 2024', 'fiscal 2024').",
    "Metric":       "A named quantitative measure — financial, performance, or operational (e.g. 'net revenue', 'ROTCE').",
    "Role":         "A job title, position, or function (e.g. 'Chief Executive Officer', 'medic').",
    "Program":      "A named project, program, or initiative (e.g. 'American Dream Initiative').",
    "Regulation":   "A named law, policy, standard, or governing rule (e.g. 'GDPR', 'ROE').",
    "Risk":         "A named risk, threat, or hazard (e.g. 'database corruption risk').",
}

# Generic fallback relation — allowed, but extraction is told to prefer a
# specific type and use this only when nothing else fits.
FALLBACK_RELATION = "RELATED_TO"

# ---------------------------------------------------------------------------
# Relation types  (predicate -> {description, source_types, target_types})
# "*" in source/target means "any entity type".
# ---------------------------------------------------------------------------
RELATION_TYPES: Dict[str, Dict[str, object]] = {
    "PART_OF":         {"desc": "X is a component or subunit of Y.",
                        "src": ["Organization", "Location", "Equipment", "System", "Concept"],
                        "tgt": ["Organization", "Location", "Equipment", "System", "Concept"]},
    "LOCATED_IN":      {"desc": "X is physically located within place Y.",
                        "src": ["*"], "tgt": ["Location"]},
    "MEMBER_OF":       {"desc": "Person or unit X belongs to organization Y.",
                        "src": ["Person", "Organization"], "tgt": ["Organization"]},
    "HAS_ROLE":        {"desc": "Person X holds role/position Y.",
                        "src": ["Person"], "tgt": ["Role"]},
    "REPORTS_TO":      {"desc": "X reports to / is subordinate to Y in a hierarchy.",
                        "src": ["Person", "Organization"], "tgt": ["Person", "Organization"]},
    "RESPONSIBLE_FOR": {"desc": "Person or role X is accountable for Y.",
                        "src": ["Person", "Role", "Organization"], "tgt": ["*"]},
    "OPERATES":        {"desc": "X operates or runs equipment/system Y.",
                        "src": ["Person", "Organization"], "tgt": ["Equipment", "System"]},
    "PRODUCES":        {"desc": "Organization X produces or offers Y.",
                        "src": ["Organization"], "tgt": ["Equipment", "System", "Metric"]},
    "USES":            {"desc": "X uses or depends on Y.",
                        "src": ["*"], "tgt": ["Equipment", "System", "Procedure"]},
    "REQUIRES":        {"desc": "X requires Y to function.",
                        "src": ["Procedure", "System", "Program"], "tgt": ["Equipment", "System", "Procedure"]},
    "GOVERNED_BY":     {"desc": "X is subject to regulation/policy Y.",
                        "src": ["*"], "tgt": ["Regulation"]},
    "COMPLIES_WITH":   {"desc": "X meets or conforms to standard Y.",
                        "src": ["Organization", "System", "Procedure"], "tgt": ["Regulation"]},
    "ISSUED_BY":       {"desc": "X was issued, published, or authored by Y.",
                        "src": ["Regulation", "Metric", "Program", "Concept"], "tgt": ["Organization", "Person"]},
    "PARTICIPATES_IN": {"desc": "X takes part in event Y.",
                        "src": ["Person", "Organization", "System"], "tgt": ["Event"]},
    "OCCURRED_ON":     {"desc": "Event X happened on date Y.",
                        "src": ["Event"], "tgt": ["Date"]},
    "HAS_DATE":        {"desc": "X is associated with date Y.",
                        "src": ["*"], "tgt": ["Date"]},
    "HAS_METRIC":      {"desc": "X is quantified by metric Y.",
                        "src": ["Organization", "Program", "System"], "tgt": ["Metric"]},
    "MITIGATES":       {"desc": "Procedure/system X reduces or controls risk Y.",
                        "src": ["Procedure", "System", "Organization"], "tgt": ["Risk"]},
    "AFFECTS":         {"desc": "X has an impact or effect on Y.",
                        "src": ["*"], "tgt": ["*"]},
    "CAUSES":          {"desc": "X leads to or triggers Y.",
                        "src": ["*"], "tgt": ["Event", "Risk"]},
    "MANAGES":         {"desc": "X directs or manages Y.",
                        "src": ["Person", "Role", "Organization"], "tgt": ["Program", "Organization", "System"]},
    "PRECEDES":        {"desc": "X comes before Y in sequence or time.",
                        "src": ["Event", "Procedure"], "tgt": ["Event", "Procedure"]},
    "IS_A":            {"desc": "X is a kind of / instance of concept Y (taxonomy).",
                        "src": ["*"], "tgt": ["Concept"]},
    FALLBACK_RELATION: {"desc": "Generic association when no specific relation fits. Use sparingly.",
                        "src": ["*"], "tgt": ["*"]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def entity_type_names() -> List[str]:
    return list(ENTITY_TYPES.keys())


def relation_type_names() -> List[str]:
    return list(RELATION_TYPES.keys())


def is_valid_entity_type(t: str) -> bool:
    return t in ENTITY_TYPES


def is_valid_relation(predicate: str) -> bool:
    return predicate in RELATION_TYPES


def relation_allows(predicate: str, src_type: str, tgt_type: str) -> bool:
    """True if a predicate may connect src_type -> tgt_type per the ontology."""
    spec = RELATION_TYPES.get(predicate)
    if not spec:
        return False
    src = spec["src"]
    tgt = spec["tgt"]
    src_ok = "*" in src or src_type in src
    tgt_ok = "*" in tgt or tgt_type in tgt
    return src_ok and tgt_ok


def allowed_pairs() -> Set[Tuple[str, str, str]]:
    """Every (predicate, src_type, tgt_type) the ontology permits — for validation."""
    out: Set[Tuple[str, str, str]] = set()
    ents = entity_type_names()
    for pred, spec in RELATION_TYPES.items():
        srcs = ents if "*" in spec["src"] else spec["src"]
        tgts = ents if "*" in spec["tgt"] else spec["tgt"]
        for s in srcs:
            for t in tgts:
                out.add((pred, s, t))
    return out
