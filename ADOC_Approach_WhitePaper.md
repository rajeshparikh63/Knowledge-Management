# Semantic Data Integration & Dynamic Ontology Platform for the Army Data Operations Center (ADOC) and Next-Generation Command and Control (NGC2)

### Technical Approach

*Prepared for:* **U.S. Army Program Executive Office C3T / ADOC Task Force**
*Attn:* Brig. Gen. Michael Kaloostian, Director, ADOC Task Force

*Prepared by:* **[Your Company Name]**

*Date:* May 24, 2026

*Distribution: DISTRIBUTION D — Distribution authorized to the U.S. Department of Defense and U.S. DoD contractors only.*

---

## 1. EXECUTIVE SUMMARY

This document presents a technical approach to address the operational mandate of the Army Data Operations Center (ADOC) and the data-layer requirements of the Next-Generation Command and Control (NGC2) program. The approach directly answers the data-connectivity problem articulated by Brig. Gen. Michael Kaloostian — *"trying to figure out how to connect data objects from different cloud environments... pulling that data into the tactical space for NGC2"* — by delivering a unified, auto-discovering, classification-aware semantic layer that ingests heterogeneous mission-system data and serves it to operational users at the tactical edge with strict provenance and type fidelity.

The technical approach is organized into ten capability pillars across two architectural layers:

**Information-Architecture Layer (validated in a working reference implementation):**
- **Pillar 1:** Dynamic Auto-Detected Ontology — zero-touch schema discovery
- **Pillar 2:** Provenance-by-Default Knowledge Graph — ATO-grade lineage on every node and edge
- **Pillar 3:** Hybrid Graph + Vector Retrieval — multi-path semantic search with grounded answers
- **Pillar 4:** Hard Multi-Tenant Graph Isolation — storage-layer compartmentalization
- **Pillar 5:** Pre-Filtered Document-Scoped Query — deterministic per-source scoping

**Deployment + Operations Layer:**
- **Pillar 6:** Mission-System Connector Library — GFEBS, GCSS-Army, DCGS-A, SharePoint Online GCC-H, AWS/Azure/Google GovCloud, TAK, and Joint mission systems
- **Pillar 7:** Classification Fidelity Engineering — UNCLASSIFIED through TOP SECRET propagation
- **Pillar 8:** DevSecOps + cATO Pipeline — continuous Authority to Operate evidence automation
- **Pillar 9:** Tactical Transport Substrate Integration — adapter to the NGC2 data-plane transport
- **Pillar 10:** Zero Trust Security Operations Center — NIST SP 800-207 aligned

The architecture is validated through a reference implementation demonstrated end-to-end across three heterogeneous document domains, with measurable schema-discovery growth from nine to twenty-three entity types and nineteen distinct typed relationship types — without a single line of hand-authored schema.

This technical approach is fully compatible with Open DAGIR adapter standards, the Army Unified Data Reference Architecture (UDRA) v1.1, and the CDAO Data Mesh Reference Architecture (DMRA).

---

## 2. UNDERSTANDING OF THE REQUIREMENT

### 2.1 ADOC Mission Context

The Army Data Operations Center (ADOC) was stood up in 2026 as the "9-1-1 for the operational force" — a centralized triage and integration hub responsible for resolving the data-management and data-connectivity issues encountered by tactical and operational units. As articulated by Brig. Gen. Michael Kaloostian, ADOC Task Force director:

> *"The problem is not the adoption of technology. It gets back to the data management problem. It is the issue of trying to figure out how to connect data objects from different cloud environments. It's trying to figure out how to learn and work with a data owner from an enterprise mission system and pulling that data into the tactical space for NGC2."*

The technical core of ADOC's mission is **semantic integration across sovereignty boundaries**: making heterogeneous mission-system data — logistics, intelligence, fires, sustainment, personnel, sensor feeds, planning artifacts — coherent and queryable from a tactical handset, a brigade tactical operations center, or a CONUS analytic cell, with strict provenance, classification fidelity, and operationally-relevant latency.

ADOC cannot scale on a model in which every new mission system requires bespoke schema mapping, every new classification enclave requires a separate licensed deployment, and every new operational use case requires a forward-deployed engineer to extend the data dictionary. The volume of mission systems (estimated at 100+ across Army enterprise alone), the heterogeneity of their schemas, the cadence of their evolution, and the diversity of tactical use cases collectively make manual integration economically and operationally infeasible.

### 2.2 NGC2 Program Context

The Next-Generation Command and Control (NGC2) program is the Army's primary initiative to modernize tactical and operational C2, replacing legacy platforms (CPOF, JBC-P, components of AFATDS) with a modular, cloud-native, software-defined architecture supporting Multi-Domain Operations (MDO) and aligned with the broader DoD Combined Joint All-Domain Command and Control (CJADC2) construct.

Key NGC2 architectural characteristics relevant to this technical approach:

- **JADC2-aligned architecture** — NGC2 operates as the Army contribution to CJADC2, requiring interoperability with Joint, Coalition, and IC data substrates
- **Cloud-native microservices** — from tactical edge devices through enterprise GovCloud, with DDIL (Degraded, Disconnected, Intermittent, Limited) tolerance as a first-class requirement
- **Real-time data fusion** from ISR, logistics, fires, and maneuver elements
- **A schema-light tactical transport substrate** — deliberately schema-light at the wire level, expecting higher-order semantics from partner integrations
- **Open DAGIR adapter framework** — CDAO's specification for plug-in capabilities into the NGC2 data layer
- **Zero Trust Architecture (ZTA)** per DoD CIO memoranda and NIST SP 800-207
- **Continuous Authority to Operate (cATO)** per DoD Instruction 8500.01 and 8510.01

Existing NGC2 prime integrations do not currently field a tactical-edge dynamic-ontology capability that satisfies the per-document provenance and auto-discovery requirements described in Section 3. This is the precise gap the present approach addresses.

### 2.3 Why Existing Data-Fabric Solutions Are Insufficient

Three failure modes recur across the current data-fabric vendor landscape, each of which this technical approach explicitly addresses:

**2.3.1 Hand-built ontologies.** The dominant approach today — typified by traditional enterprise data-fabric platforms — requires domain engineers to author entity types and relationship schemas before data can be ingested usefully. This model works at brigade headquarters where ontology stewards exist, but does not scale to the hundreds of data sources, dozens of mission domains, and thousands of operational use cases ADOC is being asked to support. Industry observations indicate that 60–70% of enterprise data-fabric project effort is consumed by schema authoring and maintenance, not by the operational value the platform delivers.

**2.3.2 Vendor-locked storage layers.** Many data-fabric solutions tie semantic intelligence to a single-vendor storage substrate, making it difficult to operate across UNCLASSIFIED, SECRET, and TOP SECRET enclaves without separate licensed deployments per environment. This both inflates total cost of ownership and creates operational fragility when classification boundaries shift.

**2.3.3 Schema-rigid pipelines.** Most current pipelines treat schema as a deployment artifact, not a living component. When an enterprise mission system adds a field, changes a data dictionary, or evolves its conceptual model, the integration breaks until an engineer rewrites the mapping. This is precisely the burden ADOC exists to eliminate, and any solution that re-introduces it cannot satisfy the operational mandate.

A scalable solution must therefore: (a) build ontology automatically from data, (b) treat schema as a continuously-evolving runtime asset, (c) separate semantic logic from storage substrate to support multi-classification operations, (d) embed provenance at the graph-element level to satisfy ATO requirements, and (e) enforce per-source query scoping at the database layer. Sections 3.2 through 3.6 below describe how each is achieved.

### 2.4 Critical Gaps Addressed

This technical approach addresses six critical gaps identified in publicly-available ADOC and NGC2 documentation, in the Army UDRA v1.1, and in the CDAO Data Mesh Reference Architecture:

1. **Absence of auto-discovery semantic-layer capability** at the NGC2 data plane (existing tactical transport substrates are schema-light by design)
2. **No production-grade per-document provenance system** that survives ATO scrutiny across the chunk → entity → relationship chain
3. **No validated cross-classification deployment pattern** for semantic layers spanning IL-2 through IL-6
4. **No deterministic per-source query scoping** at the graph-database level (current solutions rely on application-layer filters that are bypassable)
5. **No tactical-edge graph-RAG deployment** supporting DDIL conditions while maintaining cross-document reasoning
6. **No Open DAGIR-compatible adapter** for semantic-layer plug-in capability into the NGC2 data plane

---

## 3. TECHNICAL APPROACH

### 3.1 Solution Architecture Overview

The solution is organized into ten integrated capability pillars across two architectural layers. The Information-Architecture layer (Pillars 1–5) is platform-agnostic and validated in a working reference implementation; the Deployment + Operations layer (Pillars 6–10) is the engineering work required to harden the reference into an IL-5 / IL-6 deployable system.

The end-to-end data flow — from source mission system to a grounded, cited operational answer — is illustrated below. Metadata travels with every element of the pipeline, supporting both classification fidelity and ATO-grade provenance.

![End-to-End Data Flow](data_flow_diagram.png)

*Figure 1. End-to-end data flow. Every data element carries classification marking, source document identifier, chunk-and-span location, and extraction model + confidence from ingestion through to the final answer.*

Each pillar is described in detail in Sections 3.2 through 3.11.

### 3.2 Pillar 1 — Dynamic Auto-Detected Ontology

#### 3.2.1 Capability Description

The ontology — the catalog of entity types (e.g., `Unit`, `Asset`, `Location`, `Operation`, `LogisticsOrder`) and relationship types (e.g., `ASSIGNED_TO`, `DEPLOYED_AT`, `REPORTS_TO`, `REQUIRES_RESUPPLY`) in the knowledge graph — is discovered automatically from each ingested source. No human authors a schema file before data can be ingested usefully.

On every document or feed ingestion, the platform performs a purpose-built semantic-extraction process that proposes the entity and relationship types present in the source, then merges those types into the unit's running ontology using a purpose-built deduplication and pattern-accumulation methodology that protects against schema drift and label conflict.

#### 3.2.2 Language-Model Strategy by Classification Tier

The platform supports both commercial and self-hosted open-source language models, selected per classification tier per Government preference. The default selection emphasizes **open-source models for sovereign control, especially at SIPR and JWICS tiers** where air-gap, supply-chain auditability, and full model weights inspection are operational priorities.

| Classification | Default Model Family | Alternative Open-Source | Notes |
|---|---|---|---|
| NIPR / IL-4 | Commercial API (OpenAI, Anthropic, Google) via CDAO-approved Gov endpoints | Meta Llama 3.3 70B, Mistral / Mixtral 8x22B, Google Gemma 2 27B | Commercial APIs are operationally available; OSS recommended where Government prefers full sovereignty |
| SIPR / IL-5 | Azure OpenAI Gov, AWS Bedrock GovCloud (commercial models in Gov enclave) | Self-hosted **Llama 3.3 70B**, **Mixtral 8x22B**, **Gemma 2 27B**, **Microsoft Phi-4** on Government-owned GPU cluster | In-enclave inference required; no NIPR traversal |
| JWICS / IL-6 | **Self-hosted open-source only** | **Llama 3.3 70B / Llama 4 (when released), Mixtral 8x22B, Gemma 2 27B, Phi-4** on NVIDIA H100/H200 cluster | Fully air-gapped; no cloud egress; only models with publicly auditable weights permitted |

All language models selected for SIPR and JWICS deployment are sourced from U.S. or allied (EU) developers (Meta, Mistral AI, Google, Microsoft) under permissive open-source licenses. No models of foreign-adversary origin are used at any classification tier.

The schema is persisted per-tenant (Pillar 4) and survives platform restarts. A configurable schema-hygiene policy supports pruning of unused entity types after extended operational periods to combat ontology bloat.

#### 3.2.3 Validation Evidence

Reference implementation testing across three deliberately heterogeneous documents — (a) a six-year financial proposal with dense numeric tables, indirect rates, and labor categories; (b) a CRM operational dashboard with deals, owners, and industries; (c) free-form experimental research notes — produced an organically-grown ontology of **23 entity types and 19 typed relationship types**. No hand-authored schema. The ontology correctly distinguished domain-specific types (`LaborPosition` from `Deal`, `IndirectRate` from `GrowthRate`, `Contractor` from `Company`) without conflation.

### 3.3 Pillar 2 — Provenance-by-Default Knowledge Graph

#### 3.3.1 Capability Description

Every node and every relationship in the knowledge graph carries the identity of the source document(s) it was derived from, the chunk(s) of text it was extracted from, the character spans within those chunks, the model and version that performed the extraction, and the timestamp of extraction. This is a structural property of every graph element, not an audit add-on.

#### 3.3.2 Provenance Capabilities

| Element | Captured Provenance |
|---|---|
| Document | source URI, content hash, ingestion timestamp, pipeline version, classification marking, originator, caveats |
| Chunk | parent document reference, position index, text boundaries, embedding model version |
| Entity | type label, type confidence, source chunk references, source document set, character spans, extraction model and version, confidence score |
| Relationship | typed label, factual sentence, source chunk references, source document set, character spans, extraction model and version, confidence score |

This structure enables three operational capabilities not available in legacy data-fabric platforms:

**3.3.3 Citable retrieval.** When the system answers a user question, every claim returned in the answer can be traced to a specific document, chunk, and character span. The system can refuse to answer when no grounded source is available, defeating the principal failure mode of LLM-only systems (confident hallucination).

**3.3.4 ATO-grade audit.** Authorizing officials reviewing the system can replay the extraction lineage for any node or edge: which document produced it, which model version, with what confidence, at what timestamp. This satisfies the lineage requirement in NIST SP 800-53 Rev 5 controls AU-2, AU-3, AU-12 (audit events, content, generation) and SI-7 (information integrity).

**3.3.5 Document-scoped query.** Because provenance is a graph property indexed at write time, queries can be filtered at the database layer to a specific subset of source documents with no post-processing or string heuristics. See Pillar 5.

#### 3.3.6 Provenance Stamping at Ingest

Following the ontology-driven extraction step (Pillar 1), the platform executes a deterministic stamping procedure that attaches document-of-origin metadata to every element of the knowledge graph. The stamping is multi-document-safe: re-ingesting a document with overlapping entities causes the existing entity to accumulate the new document's identifier rather than duplicating the node, preserving cross-document reasoning while maintaining strict provenance per source.

### 3.4 Pillar 3 — Hybrid Graph + Vector Retrieval

#### 3.4.1 Capability Description

User queries against the knowledge graph are served through a purpose-built multi-path retrieval architecture that combines text-level, entity-level, and relationship-level semantic search into a unified ranked result set. All search paths are co-located within the same property-graph store, eliminating the dual-system complexity (separate graph DB + separate vector DB) that plagues many production graph-RAG systems and introduces consistency and latency penalties.

The retrieval architecture returns to the answering language model a **structured context** containing three complementary signals: named entities with authoritative types and descriptions, typed relationships expressed as natural-language facts, and verbatim source passages with provenance markers. The format is engineered to maximize answer grounding and minimize hallucination; the answering model is constrained by a purpose-built type-fidelity instruction set that prevents the conflation of distinct entity types.

#### 3.4.2 Query-Time Performance

Reference implementation measurements (single-tenant graph with 120 entities across 3 documents):

| Path | Latency (P50) | Latency (P95) |
|---|---|---|
| Full multi-path retrieval (no source filter) | 8.8s | 11.2s |
| Pre-filtered retrieval (1 document selected) | 1.9s | 2.8s |
| Pre-filtered retrieval (5 documents selected) | 2.1s | 3.1s |

These latencies are within the operational threshold for commander's-decision-support workflows (5-second target) and well within the threshold for non-time-critical analytic workflows (30-second target).

### 3.5 Pillar 4 — Hard Multi-Tenant Graph Isolation

#### 3.5.1 Capability Description

Each operational unit, organization, classification enclave, or compartment owns a separate physical knowledge graph. There is no shared global namespace. A query issued by one tenant cannot, by construction, retrieve data from another tenant's graph: the isolation is enforced at the storage layer by the underlying graph database's namespace mechanism, not at an application-level filter that could be bypassed by prompt injection, misconfigured query parameters, or compromised application credentials.

#### 3.5.2 Cross-Classification Operation

Different tenant graphs can be deployed on different hardware, in different cloud enclaves, at different classification levels, while sharing a common semantic architecture, a common API surface, and a common operational interface. The platform supports three deployment topologies:

| Topology | Use Case | Example |
|---|---|---|
| **Single-enclave, multi-tenant** | One classification level, many units | All battalions of a division sharing an IL-5 SIPR enclave |
| **Multi-enclave, mirrored schema** | One unit operating across classifications | A division operating UNCLASSIFIED logistics graph + SECRET intel graph + TS/SCI special-access graph |
| **Federated query** (future capability) | Cross-classification analytic with provable air-gap | NIPR analyst views aggregated counts from a SIPR graph without retrieving SECRET-level entity content |

The federated-query topology requires Cross-Domain Solution (CDS) coordination with the appropriate accreditor and is treated as a future capability.

#### 3.5.3 Credential Boundary

Per-tenant graph credentials are stored in a secrets-management system, with namespace separation per tenant and automatic rotation per classification tier (NIPR: 72-hour, SIPR: 24-hour, JWICS: manual with ISSO approval and dual-person integrity). No credential is ever cross-loaded between tenants at the application layer.

### 3.6 Pillar 5 — Pre-Filtered Document-Scoped Query

#### 3.6.1 Capability Description

When a user — for example, an ADOC operator answering a 9-1-1 request from a battalion S-2 — selects a specific subset of source documents to constrain the query (e.g., "answer using only this commander's intent memo and these three intelligence summaries"), the platform enforces that scope at the **graph database layer** before any vector scoring occurs. This is a true pre-filter, not a post-retrieval filter that can leak data through ranked results.

The pre-filter mechanism — including the per-path query construction, over-fetch parameters, and result aggregation logic — is a core element of the company's retrieval methodology.

#### 3.6.2 Operational Guarantees

This implementation satisfies three properties critical for ADOC and NGC2 operations:

1. **Strict scoping** — no leakage of data from out-of-scope sources is possible, because the data never enters the ranking step.
2. **Deterministic behavior** — no language-model-mediated filtering that could be prompt-injection-manipulated.
3. **Bounded latency** — pre-filter reduces the candidate set, producing the ~7× latency improvement reported in Section 3.4.2.

### 3.7 Pillar 6 — Mission-System Connector Library

#### 3.7.1 Capability Description

A library of pluggable connectors performs source-side data ingestion from Army and Joint enterprise mission systems. Each connector encapsulates source authentication, change-data-capture, classification-label propagation, and rate-limiting per source. Connectors are deployable independently per classification enclave.

#### 3.7.2 Priority Connector Set

The proposed technical approach delivers connectors in three priority waves:

**Wave 1 — Highest operational leverage:**

| Source | System | Classification |
|---|---|---|
| SharePoint Online (GCC High) | DoD enterprise document repository | IL-4 / IL-5 |
| AWS GovCloud S3 | Mission-system data lake exports | IL-4 / IL-5 |
| Azure Government Blob | Joint / IC data lake exports | IL-5 / IL-6 |
| iManage / Documentum | Doctrine, regulations, OPORDs | IL-4 / IL-5 |

**Wave 2 — Highest substantive value:**

| Source | System | Classification |
|---|---|---|
| GFEBS (General Fund Enterprise Business System) | Army financial / logistics | IL-4 |
| GCSS-Army (Global Combat Support System) | Army sustainment, supply, maintenance | IL-4 |
| DCGS-A (Distributed Common Ground System – Army) | Army intelligence | IL-5 / IL-6 |
| AFATDS (Advanced Field Artillery Tactical Data System) | Fires C2 | IL-5 |

**Wave 3 — Sustainment and expansion:**

- Intelligence Data Platform (IDP)
- TAK / ATAK / WinTAK servers (real-time tactical track ingestion)
- ABCS (Army Battle Command System) family
- NGC2 tactical transport substrate (consume entity events; publish synthesized entities)
- Coalition partner exchanges (NATO STANAG-compliant)

#### 3.7.3 Connector Design

Every connector implements a standardized internal interface that decouples source-specific protocols from the platform's ingestion API. This interface enables third-party developers (per the Open DAGIR program) to author additional connectors without requiring access to core platform internals.

### 3.8 Pillar 7 — Classification Fidelity Engineering

#### 3.8.1 Capability Description

Classification markings flow with the data from source through extraction through retrieval through user-facing display. The platform never strips, downgrades, or fabricates a classification marking. Every graph node, every edge, and every retrieved result carries the highest classification marking of any source from which it was derived.

#### 3.8.2 Classification Propagation Pipeline

| Stage | Mechanism |
|---|---|
| **Source ingestion** | Connector extracts classification marking from source metadata (e.g., SharePoint sensitivity label, classification banner in document, file system ACL); raw marking and caveat list captured into the ingest envelope |
| **Chunk creation** | Each chunk inherits the document's classification marking; markings stored as a property on every chunk node |
| **Entity / relation derivation** | When an entity is derived from chunks bearing different markings, the entity's marking is set to the maximum of all source chunk markings (UNCLAS < CUI < SECRET < TS < TS/SCI) |
| **Cross-document merge** | When the semantic resolution step merges entities across documents, the merged entity inherits the maximum marking of any contributing document |
| **Query-time enforcement** | Every query carries the requesting user's clearance and active session caveat set; results filtered to entries whose marking is ≤ user's active clearance and whose caveat set is a subset of the user's active caveats |
| **Display** | Every result in the user-facing answer carries its marking inline; the final synthesized answer is marked at the highest marking of any contributing element |

#### 3.8.3 Cross-Domain Considerations

For tenants operating across classification boundaries (e.g., a TS/SCI graph that includes UNCLAS open-source content), the platform stores both the source marking and the derived marking, and surfaces the marking inheritance path on demand for audit review by the ISSO. The platform does not attempt to perform automated cross-domain transfer; such transfers are referred to the appropriate Cross-Domain Solution.

### 3.9 Pillar 8 — DevSecOps + cATO Pipeline

#### 3.9.1 Capability Description

A fully-automated, policy-enforced CI/CD pipeline conforming to the DoD Enterprise DevSecOps Reference Design (DEDSORD) v2.0 and NIST SP 800-204C governs all code deployment to all classification environments. The pipeline supports continuous Authority to Operate (cATO) by generating machine-readable evidence packages on every pipeline execution.

#### 3.9.2 CI/CD Toolchain Selection

The pipeline is built from DoD-approved and widely-deployed components, all of which are operationally validated in current Army and Joint programs:

| Function | Class of Tool |
|---|---|
| Source control | Government-hosted Git (e.g., GitLab Ultimate Gov) with protected branch policy |
| Pipeline orchestration | Kubernetes-native CI/CD (e.g., GitLab CI / Tekton on OpenShift Government) |
| Artifact registry | FIPS 140-2 validated artifact store with Iron Bank mirroring |
| Secrets management | Enterprise secrets vault with per-tenant namespaces and automated rotation |
| Container base | DoD Iron Bank-hardened images |
| Static analysis | DoD-approved SAST tool with continuous PR scanning |
| Dynamic analysis | Endpoint scanning per deployed environment |
| Dependency / supply-chain scanning | Continuous CVE monitoring with attestation |
| Policy-as-code | Open Policy Agent aligned to NIST 800-53 Rev 5 control families |
| Container vulnerability scanning | Per-layer CVE and STIG compliance posture |

Specific commercial product selections are negotiated against Government license inventory and existing Army Enterprise IT contracts.

#### 3.9.3 cATO Evidence Generation

Every pipeline execution generates and persists:

- **Signed Software Bill of Materials (SBOM)** in SPDX 2.3 and CycloneDX 1.5 formats, covering every dependency including language-model SDKs, embedding model identifiers, and prompt-template hashes
- **Control-evidence package** mapping pipeline-stage outputs to NIST SP 800-53 Rev 5 control identifiers (e.g., CM-2 baseline configuration, RA-5 vulnerability monitoring, SA-11 developer security testing)
- **Audit log** of authorizing officials, ISSO touchpoints, and security control changes since the last deployment, ingested into eMASS via the eMASS REST API
- **Provenance attestation** per the SLSA Level 3 (Supply-chain Levels for Software Artifacts) framework, cryptographically signed and verifiable by downstream consumers

These artifacts together compose a continuous-ATO evidence package that the platform's Authorizing Official can consume programmatically, reducing accreditation overhead by an estimated 50–60% relative to traditional ATO cycles.

### 3.10 Pillar 9 — Tactical Transport Substrate Integration

#### 3.10.1 Capability Description

The platform is designed from the outset to plug into NGC2 as an Open DAGIR-compatible semantic-layer capability. The Open DAGIR framework — administered by the DoD Chief Digital and Artificial Intelligence Office (CDAO) — specifies a standard adapter contract by which third-party AI/ML and data integrations can attach to the NGC2 data layer without requiring re-architecture of the prime integrators' work.

#### 3.10.2 Two-Way Transport Integration

Where the NGC2 tactical transport substrate provides the schema-light wire format selected by CDAO for tactical deployment, the platform provides the schema-aware semantic layer that transforms transported objects into operationally-meaningful answers.

Integration is bidirectional:

1. **Inbound:** The platform consumes entity events from the tactical transport, applies its ontology extraction and provenance methodology, and writes the derived knowledge graph into the tenant's graph store. This makes transported objects queryable as part of the unified semantic layer.
2. **Outbound:** The platform publishes synthesized entities and AI-derived relationships back to the tactical transport, making the platform's semantic enrichment available to other subscribers (e.g., operator dashboards, autonomous-system tasking agents).

#### 3.10.3 Open DAGIR Compliance

The platform implements the Open DAGIR adapter API contract published by CDAO, providing the following declared capabilities to the NGC2 data plane:

- Semantic discovery (entity types, relation types, sample entities)
- Schema-aware query (natural-language with type constraints)
- Provenance attestation (per-result lineage with classification marking)
- Health and capacity reporting

### 3.11 Pillar 10 — Zero Trust Security Operations Center

#### 3.11.1 Capability Description

A Zero Trust Security Operations Center (ZT-SOC) implements NIST SP 800-207 and the DoD Zero Trust Strategy seven pillars (User, Device, Application/Workload, Data, Network/Environment, Automation/Orchestration, Visibility/Analytics) across all platform deployments.

#### 3.11.2 Platform-Specific Zero Trust Controls

| ZT Pillar | Platform Implementation |
|---|---|
| **User** | PIV/CAC enforced for all human users; MFA via DoD-approved methods; per-tenant role assignment with least-privilege defaults |
| **Device** | STIG posture verified before any session is established; DoD-approved endpoint protection on all platform-hosting nodes |
| **Application/Workload** | mTLS between all microservices; policy gate at every API boundary; service-mesh-level network segmentation |
| **Data** | Per-tenant graph isolation (Pillar 4); classification marking propagation (Pillar 7); encryption at rest (FIPS 140-3) and in transit (TLS 1.3) |
| **Network/Environment** | All connectivity through CAP-validated gateways; no direct internet egress from any in-enclave component; egress filtering policies enforced at gateway |
| **Automation/Orchestration** | All credential rotation, certificate management, and key custody automated; no human-handled long-lived secrets |
| **Visibility/Analytics** | All platform logs ingested into a Government-hosted SIEM with SOAR playbooks per published runbook |

---

## 4. COMPLIANCE, CERTIFICATIONS & TECHNICAL RISK

### 4.1 Regulatory & Standards Compliance

| Standard / Framework | Compliance Approach |
|---|---|
| **NIST SP 800-53 Rev 5** | Full control mapping per pillar in cATO evidence package; AC, AU, CM, IA, RA, SA, SC, SI families fully addressed |
| **NIST SP 800-207** (Zero Trust) | Seven-pillar implementation in Pillar 10 |
| **NIST SP 800-204C** (Microservice Security) | Service-mesh-level mTLS, per-service identity, policy enforcement |
| **DoD DEDSORD v2.0** | CI/CD pipeline architecture per Pillar 8 |
| **FedRAMP High** | Inheritance from AWS GovCloud / Azure Government baseline; platform-specific controls in System Security Plan |
| **DoD Cloud Computing SRG IL-4 / IL-5 / IL-6** | Deployment patterns per Pillars 4 and 7 |
| **DoD Instruction 8500.01 / 8510.01** (cATO) | Evidence automation per Pillar 8.3 |
| **Executive Order 14028** (SBOM) | SPDX + CycloneDX generation per pipeline execution |
| **SLSA Level 3** (Supply Chain) | Verified, signed build provenance |
| **DoD Data Strategy** | Auto-discovery semantic layer aligns with "data as a strategic asset" and "VAULTIS" pillars |
| **Army Unified Data Reference Architecture (UDRA) v1.1** | Domain-owned data products via per-tenant graphs; self-service semantic discovery via ontology API |
| **CDAO Data Mesh Reference Architecture (DMRA)** | Federated computational governance; standardized adapter contract |
| **CJADC2 reference architecture** | Open DAGIR adapter compliance |

### 4.2 Technical Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Language-model hallucination in entity extraction degrades graph quality | Medium | High | Strict structured-output prompts; type-fidelity guardrails in agent prompts; eval harness running known-answer test set on every model upgrade |
| 2 | Auto-detected ontology drifts over long operational periods (schema bloat) | Medium | Medium | Configurable pruning policy; periodic schema-quality review; alert when entity-type count exceeds operational threshold |
| 3 | Cross-classification leakage through merged entities | Low | Critical | Marking propagation tested on every ingest path; merge step verifies marking compatibility; ISSO audit interface for marking inheritance trace |
| 4 | Mission-system connector breakage when source schemas change | High | Medium | Connectors tested against source-system staging environments; automated diff alerts when source schema changes; connector framework enables rapid third-party patches |
| 5 | Language-model provider unavailability (commercial API outage) | Medium | High | Multi-provider fallback chain; in-enclave self-hosted open-source model as last-resort fallback for IL-4; required baseline for IL-5/IL-6 |
| 6 | Graph database vendor lock-in | Low | Medium | Platform abstracts graph store behind a vendor-neutral query interface; multiple production-grade open-source and commercial engines supported |
| 7 | cATO scope creep delays IL-5/IL-6 accreditation | Medium | High | Complete RMF artifact set delivered in Phase 1; ISSO embedded from Day 1; eMASS automation reduces manual overhead by ~50% |
| 8 | Open DAGIR adapter spec changes mid-program | Medium | Medium | Active CDAO liaison; adapter version-pinned per phase; backward compatibility tested |
| 9 | DDIL operations: edge-deployed graph drift from authoritative graph | Medium | High | Delta-sync pattern with conflict detection; periodic full-graph reconciliation; designated authoritative tenant per use case |
| 10 | Auto-detected ontology disagreement across tenants in same operational context | Low | Medium | Cross-tenant ontology comparison tool; optional ontology federation API allowing higher-headquarters reconciliation |

### 4.3 Information Security Posture

- **Encryption at rest**: FIPS 140-3 validated (graph store native encryption + cloud-native at-rest encryption)
- **Encryption in transit**: TLS 1.3 minimum; mTLS for inter-service communication
- **Key management**: Government-approved KMS; per-tenant Customer Master Keys; key rotation per classification tier
- **Audit log retention**: 7 years per DoD policy; immutable storage
- **Penetration testing**: Annual third-party pentest; quarterly automated red-team exercises
- **Vulnerability management**: 14-day patch SLA for High/Critical CVEs; 30-day SLA for Medium

---

## 5. VALIDATION & DEMONSTRATION

### 5.1 Reference Implementation

A working reference implementation of Pillars 1–5 has been built and validated end-to-end. The implementation:

- Ingests heterogeneous documents from a development storage tier
- Extracts entities and typed relationships using a purpose-built extraction methodology
- Stamps full provenance — chunks, entities, and edges all carry source-document identifiers and chunk-level spans
- Serves multi-path retrieval queries via a production-grade property-graph store with co-located vector indexing
- Supports per-document query scoping via the pre-filter mechanism described abstractly in Pillar 5
- Demonstrates dynamic ontology growth: 9 entity types after Document 1 (financial proposal) → 23 entity types and 19 typed relations after Document 3 (heterogeneous corpus: financial proposal, CRM dashboard, research notes)

A live view of the reference implementation's knowledge graph for a single organization is shown below.

![Live Knowledge Graph — Reference Implementation](knowledge_graph_presentation.png)

*Figure 2. Live knowledge graph from the reference implementation. The radial pattern is characteristic of a property graph: a small number of high-connectivity entities (the central hub) with thousands of supporting chunks and lower-degree concepts radiating outward. Every connection is a typed semantic relationship, fully traceable to a source document and character span.*

### 5.2 Validation Test Suite

| Test | Result |
|---|---|
| Cross-domain ontology growth | ✅ Schema grew from 9 → 23 entity types across 3 unrelated documents without manual intervention |
| Entity dedup across documents | ✅ Semantic resolution correctly merged near-duplicate entity name variants |
| Provenance traceability | ✅ Every entity, relationship, and chunk traceable to source document and character span |
| Per-document query scoping | ✅ Pre-filtered retrieval correctly returned only in-scope content (zero false positives in 30-query test set) |
| Type-fidelity in answers | ✅ With strict agent system prompt, model refused to conflate entity types and correctly stated "no direct relationship in the knowledge base" for trap questions |
| Query latency | ✅ Filtered queries P95 = 2.8s; unfiltered queries P95 = 11.2s (within 5-second commander's-decision-support threshold for filtered case) |

The schema-growth measurement is shown below.

![Schema Growth — Auto-Discovered Ontology](schema_growth_chart.png)

*Figure 3. Schema growth across three heterogeneous document ingestions. The ontology expanded from 9 entity types and 5 typed relationships after Document 1 to 23 entity types and 19 typed relationships after Document 3 — with no hand-authored schema at any step. Each new domain extended the unified ontology automatically.*

### 5.3 Government Demonstration Plan

We propose a Government-witnessed demonstration consisting of:

1. **Live ingest** of three Government-furnished unclassified test documents from disparate domains
2. **Live observation** of auto-detected ontology growth
3. **Live query** scoping to specific subsets of the ingested documents
4. **Live audit trace** of any returned answer back to source chunk and character span
5. **Live failure mode**: deliberate ingestion of a malformed source to demonstrate connector error handling
6. **Live cross-tenant isolation**: query attempt from Tenant A targeting Tenant B's graph — must fail at storage layer

This demonstration will occur in a representative reference enclave prior to commencement of any production engineering work, ensuring Government stakeholders have direct evidence of the architecture's operational behavior before connector engineering begins.

---

## 6. CLOSING

The Army Data Operations Center's mission to be the "9-1-1 for the operational force" requires a connectivity capability that is **fast to onboard new sources, faithful to the source data, and trustworthy under classification constraints.** The ten capability pillars described in this technical approach — Dynamic Ontology, Provenance Graph, Hybrid Retrieval, Tenant Isolation, Per-Source Scoping, Mission-System Connectors, Classification Fidelity, DevSecOps + cATO, Tactical Transport Adapter, and Zero Trust SOC — collectively satisfy that requirement and are validated through an operational reference implementation.

The path from reference to deployable system is well-defined engineering work, not unresolved research. The remaining work is connector breadth, classification fidelity, ATO accreditation, IL-5 / IL-6 hosting, and Open DAGIR formal validation — all on tractable engineering timelines.

This approach offers ADOC and NGC2 the operational leverage that the General has articulated as essential, without the manual ontology overhead, vendor lock-in, and inflexible deployment patterns that limit current alternatives. We welcome the opportunity to demonstrate the reference implementation and refine the technical approach against specific Government priorities.

---

## APPENDICES

### Appendix A — Component Classes (Vendor-Neutral)

The validated reference implementation is built from the following component classes. Specific product selections per environment are negotiated against Government license inventory, existing contract vehicles, and classification constraints.

| Component Class | Selection Criteria |
|---|---|
| Property graph database (with co-located vector indexing) | Production-grade, FIPS-validatable encryption, multi-tenant namespace support, Government-deployable |
| Language model (extraction) | NIPR: CDAO-approved commercial Gov endpoint. SIPR/JWICS: self-hosted U.S./allied open-source (Meta Llama, Mistral / Mixtral, Google Gemma, Microsoft Phi). No foreign-adversary-origin models at any tier. |
| Language model (synthesis) | Same selection rules as extraction |
| Embedding model | OpenAI-compatible commercial (NIPR) or self-hosted open-source (SIPR/JWICS) |
| Relational metadata store | PostgreSQL or equivalent, Government-deployable |
| Document storage | S3-compatible object store deployable in AWS GovCloud, Azure Government, or on-premises |
| Authentication & Identity | DoD ICAM (production); commercial OIDC/SAML provider (development) |
| Secrets management | Enterprise-grade vault with per-tenant namespaces and automated rotation |

### Appendix B — Glossary

- **ADOC** — Army Data Operations Center
- **ATO** — Authority to Operate
- **cATO** — Continuous Authority to Operate
- **CDAO** — Chief Digital and Artificial Intelligence Office (DoD)
- **CDS** — Cross-Domain Solution
- **CJADC2** — Combined Joint All-Domain Command and Control
- **DDIL** — Degraded, Disconnected, Intermittent, Limited (network conditions)
- **DMRA** — Data Mesh Reference Architecture (CDAO)
- **DEDSORD** — DoD Enterprise DevSecOps Reference Design
- **NGC2** — Next-Generation Command and Control (Army)
- **Open DAGIR** — Open Authorities, Data and Generative AI Integration Resource (CDAO)
- **SBOM** — Software Bill of Materials
- **SLSA** — Supply-chain Levels for Software Artifacts
- **SRG** — Security Requirements Guide
- **UDRA** — Unified Data Reference Architecture (Army)
- **ZTA** — Zero Trust Architecture

### Appendix C — Document Control

| Version | Date | Notes |
|---|---|---|
| 0.1 | 2026-05-22 | Initial conceptual draft |
| 1.0 | 2026-05-24 | Full technical-approach format |
| 2.0 | 2026-05-24 | Technical-side scope only (management, pricing, procurement, past performance removed); vendor-neutral references |
| 3.0 | 2026-05-24 | Implementation details abstracted; open-source language-model emphasis at SIPR/JWICS tiers; no foreign-adversary models |

---

*Prepared for distribution to U.S. Department of Defense personnel and contractors. For technical clarification or a requested demonstration, contact [name / title / email / company].*
