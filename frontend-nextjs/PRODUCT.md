# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences on **one platform** (confirmed):

1. **Operational users** (soldiers, analysts, S2/intel roles) who upload documents,
   audio, and video, then ask grounded questions and generate intelligence products
   from their own corpus.
2. **Data / system owners** who onboard mission-system data sources onto the central
   platform and control who may see each source (per-file access).

## Product Purpose

SoldierIQ — the "Operational Knowledge System." It turns a unit's documents, audio,
and video into a queryable **two-layer knowledge graph**, answers questions **grounded
in those sources with inline citations**, and generates **intelligence products**
(reports, mind maps, flashcards, audio overviews, voice sessions) from the same corpus.
Success = an operator gets a trustworthy, source-cited answer or product fast, from the
material they're allowed to see.

## Positioning

Grounded knowledge graph with **provenance + inline citations** (confirmed as the core
differentiator). Every answer traces to the exact source passage; the graph keeps a
reliable lexical layer (document → chunk) beneath the meaning layer, so retrieval is
trustworthy even when extraction is imperfect. Offline/edge-capable in addition to cloud.

## Operating Context

Military / operational use. Users work in a **Repository** (upload data, folders/knowledge
bases), a central **chat** surface with "general mode" vs document-scoped grounded answers,
**Missions** and **Sessions**, and a **Workflows** panel that generates products (Audio
overview, Mind map, Reports, Flashcards; Video/Quiz/Infographic/Slide-deck marked "soon").
Adjacent capabilities: **voice sessions**, **TAK** (markers/messages/routes), multi-tenant
per-organization isolation, and an emerging **per-file RBAC** (access by user email).

## Capabilities and Constraints

- Ingestion of documents / audio / video → two-layer knowledge graph (FalkorDB, one graph
  per organization).
- Retrieval-augmented **chat with inline citations**; document-scoped ("grounded") or general mode.
- **Intelligence-product generation**: reports, mind maps, flashcards, audio overviews; more planned.
- **Voice** mode; **TAK** integration; multi-tenant per-org isolation; per-file access control (emerging).
- Runs at the **edge (offline)** and in the **cloud** — same product, both environments.
- Terminology (preserve): "Repository", "Missions", "Sessions", "general mode", "grounded
  answers", "intelligence products".
- Undecided / in progress: XML-manifest onboarding of external data sources (being built in a
  separate SoldierIQ-SD repo); per-file RBAC wiring into the live retrieval path.

## Brand Commitments

**BINDING (confirmed):** the **SoldierIQ** name and its **operational / tactical identity**.
The product must read as serious, precise, mission-grade — not generic SaaS. The incumbent
"operational console" world (graphite surfaces, olive-drab + signal accents, classification
strip, mono data type, callsign/mission framing, "Operational Knowledge System" line) is the
identity to carry forward. A redesign may modernize its execution but must not discard this
character for a neutral/consumer look.

## Evidence on Hand

- Running app: `frontend-nextjs` (Next.js/React/Tailwind dashboard) + `backend` (FastAPI, FalkorDB).
- `ADOC_Approach_WhitePaper.md` — the ADOC/NGC2 technical approach (positioning, pillars).
- `soldieriq-sd/example_manifest.xml` — proposed data-source onboarding manifest.
- Do NOT fabricate customers, testimonials, benchmarks, pricing, or deployment claims — none are confirmed.

## Product Principles

1. **Grounding over generation.** Every answer traces to a cited source; provenance is
   non-negotiable, and the lexical layer is the reliable backbone.
2. **One platform, two audiences.** Field/operational users and data-source owners share the
   same isolated-per-tenant system; neither experience is an afterthought.
3. **Operational identity is the brand.** The tool looks and reads as mission-grade SoldierIQ,
   precise and purposeful — never generic consumer SaaS.
4. **Corpus → products.** The system doesn't only answer; it manufactures intelligence products
   from the knowledge it holds.
5. **Edge and cloud.** Capability degrades gracefully offline; connectivity is an enhancement,
   not a requirement.
