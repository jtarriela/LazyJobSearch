 # Technical Review Packet – LazyJobSearch

> Version: 1.0  
> Date: 2025-09-23  
> Source of Truth Reference: `docs/ARCHITECTURE.md`

---

## 1. Executive One-Pager

**Problem**  
Manually hunting company-site jobs, tailoring resumes, and re‑entering the same profile data into ATS portals is slow, inconsistent, and error-prone. Users spend hours discovering roles, prioritizing them, and crafting slight resume variations.

**Outcome**  
Automate the funnel: crawl target companies, persist job descriptions, rank them against the user’s resume (FTS + vector + LLM), generate a per‑posting critique, optionally produce a tailored resume v2 (iterative loop), then auto‑apply using a stored application profile—capturing receipts and artifacts for interview prep.

**Success Metrics (Initial Targets)**
* Time to shortlist 20 roles: < 15 min (baseline > 2 hr)
* % of “Apply” recommendations yielding interview within 14 days: ≥ 12%
* Auto‑apply success rate (receipt captured): ≥ 95%
* Iterative resume improvement adoption: ≥ 60% of reviewed jobs trigger ≥1 rewrite

**MVP Scope**  
Company-site crawl (2–3 portals), Postgres+pgvector store, FTS + vector + LLM scoring pipeline, resume v2 improvement loop (iteration cap=3), auto‑apply (Greenhouse + Lever), application tracking, “Prep Packet” (strengths, gaps, suggested interview angles).

**Constraints**  
Single region; per-domain politeness; PII stored only with consent; sensitive fields (veteran, ethnicity, disability) never sent to LLM; single DB cluster; cost ceiling <$150/mo infra + variable LLM.

**Timeline (Indicative)**
* Weeks 0–2/3: Core schema, crawling, embedding, matching, first LLM review, basic UI/CLI.
* Weeks 3–5: Auto‑apply (Greenhouse/Lever), resume iteration loop, dashboard hardening, metrics.
* Post Week 5: Additional portals (Workday), cover letter generator, richer analytics.

**Open Asks / Sponsor Decisions**
| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Primary data store | Postgres + pgvector | Unified consistency & simpler ops for first 10^5–10^6 vectors |
| Crawler engine | Selenium primary, Playwright fallback | Faster adapter development; fallback for stubborn JS-heavy sites |
| LLM provider | OpenAI embeddings + GPT‑4.1/4o-mini | Quality vs. cost; abstraction layer for swap later |
| Auto‑apply portals (MVP) | Greenhouse + Lever only | Highest initial coverage w/ predictable DOM |
| Resume iteration cap | 3 by default | Controls cost & user fatigue |

---

## 2. Architecture Overview (C4 L1–L2)

### 2.1 Context & Container Diagram
```mermaid
graph TD
  subgraph External
    User[User]
    Web[Public Career Sites]
    LLM_API[LLM Provider]
  end

  subgraph Backend (Private Zone)
    API[FastAPI / API Gateway]
    SCHED[Cron / Beat]
    Q[(Queue)]
    SCR[Scraper Workers]
    RESUME[Resume Ingest Worker]
    JPROC[JD Processor]
    MATCHER[Matcher Service]
    REVIEW[Review/Rewrite Worker]
    APPLY[Apply Orchestrator]
  end

  subgraph Data Stores
    PG[(Postgres + pgvector)]
    OBJ[(Object Store)]
  end

  User --> API
  SCHED --> Q
  Q --> SCR
  SCR --> PG
  SCR --> OBJ
  RESUME --> PG
  JPROC --> PG
  MATCHER --> PG
  MATCHER --> REVIEW
  REVIEW --> PG
  APPLY --> PG
  APPLY --> OBJ
  API --> PG
  API --> APPLY
  APPLY --> Q
  REVIEW --> LLM_API
  MATCHER --> LLM_API
```

**Trust Zones / Boundaries**
* External: Untrusted web + third-party LLM API (minimized PII exposure).
* Backend Private: Controlled services + internal network policies.
* Data Stores: Encrypted at rest; restricted connections.

### 2.2 Key Quality Attributes
| Attribute | Target | Notes |
|-----------|--------|-------|
| Availability | 99.5% | Single region acceptable early |
| Latency – Match listing | < 15 min post crawl | Batch ingestion cadence |
| Latency – LLM review P95 | < 3s | Model + reduced prompt context |
| Crawl politeness | ≤5 pages/min/domain | Adaptive backoff |
| Cost – DB | <$50/mo | Managed Postgres tier |
| Cost – LLM/job | <$0.15/job | Prompt trimming + thresholding |
| Privacy | No sensitive PII to LLM | Redaction + consent gates |

---

## 3. System Design Specification (SDS)

### 3.1 Primary Workflows (Summaries)
1. **Resume Ingest & Embedding**: Upload → parse → chunk → embed → store vectors and metadata; triggers baseline YOE + skill extraction.
2. **Crawl & JD Storage**: Scheduler enqueues companies → scraper adapters collect job pages → dedupe by URL → store JD fulltext + compressed artifact.
3. **JD Embedding & Matching**: New jobs chunked & embedded → FTS + vector prefilter → top pairs forwarded to LLM scoring → persisted `MATCHES`.
4. **Review → Rewrite Loop → Auto‑Apply**: For a selected job, generate structured review (score, strengths, weaknesses, plan) → optional AI or manual revision → iterate (cap) → user satisfaction → auto‑apply using stored profile + capturing receipt.

### 3.2 Components & Responsibilities
| Component | Responsibility |
|----------|----------------|
| scraper | Site adapters, rate limiting, dedupe, artifact capture |
| resume_ingest | File parsing (PDF/docx), sectioning, chunking, embeddings |
| jd_processor | Detect new/changed JDs, chunk, embed |
| matcher | FTS + vector retrieval orchestration, LLM scoring requests |
| review/rewriter | Review generation + transformation of resume (partial diffs) |
| apply_orchestrator | Portal template execution, form fill, file upload, receipt capture |
| api (FastAPI) | REST endpoints, auth, validation, WebSockets |
| pg | Authoritative relational store + vector & FTS indices |
| obj | Artifact storage (compressed JDs, DOM snapshots, receipts) |

### 3.3 Data Contracts (At a Glance)
| Table | Purpose (Key Fields) |
|-------|----------------------|
| [JOBS](schema/JOBS.md) | id, company_id, url, jd_fulltext, jd_tsv, scraped_at |
| [JOB_CHUNKS](schema/JOB_CHUNKS.md) | job_id, chunk_text, embedding |
| [RESUMES](schema/RESUMES.md) | version chain, fulltext, sections_json, file_url |
| [RESUME_CHUNKS](schema/RESUME_CHUNKS.md) | resume_id, chunk_text, embedding |
| [MATCHES](schema/MATCHES.md) | job_id, resume_id, vector_score, llm_score, reasoning |
| [REVIEWS](schema/REVIEWS.md) | job_id, resume_id, iteration, improvement_plan_json, satisfaction |
| [APPLICATION_PROFILES](schema/APPLICATION_PROFILES.md) | user_id, answers_json, files_map_json |
| [APPLICATIONS](schema/APPLICATIONS.md) | job_id, resume_id, profile_id, status, receipt_url |
| [APPLICATION_EVENTS](schema/APPLICATION_EVENTS.md) | application_id, event_type, payload_json, occurred_at |
| [APPLICATION_ARTIFACTS](schema/APPLICATION_ARTIFACTS.md) | application_id, kind, file_url |
| [PORTALS](schema/PORTALS.md) | portal name/type |
| [PORTAL_TEMPLATES](schema/PORTAL_TEMPLATES.md) | portal_id, template_json (DSL) |
| [COMPANY_PORTAL_CONFIGS](schema/COMPANY_PORTAL_CONFIGS.md) | company_id, portal config overrides |
| [PORTAL_FIELD_DICTIONARY](schema/PORTAL_FIELD_DICTIONARY.md) | canonical field definitions |

### 3.4 SLA / SLO Targets
| Workflow | SLO (P95) |
|----------|-----------|
| Crawl job → JD persisted | < 90s/posting |
| Match computation (200 jobs) | < 5s for FTS+vector stage |
| Review completion (single job) | < 3s model latency target |
| Auto‑apply receipt persisted | ≤ 60s after submit |

### 3.5 Capacity Assumptions
* Users: 1–5 (initial) – design remains linear; minimal contention.
* Companies: 50–200 tracked.
* JDs/day: 200–800 typical.
* Stored embeddings: O(10^5) vectors (IVF indexes comfortable).

### 3.6 Matching Strategy (Why It Works)
Three-tier pipeline (FTS → vector → LLM) reduces LLM token spend by pruning >90% of candidates before semantic/LLM cost layers, keeping per-job inference bounded.

---

## 4. Top 5 Architecture Decision Records (ADRs)

### ADR 1 – Unified DB (Postgres + pgvector)
**Decision**: Store relational + vector workloads in Postgres.  
**Status**: Accepted (MVP).  
**Alternatives**: Pinecone, Qdrant, ElasticSearch hybrid.  
**Consequences**: Simplifies ops; acceptable until >10^6 vectors or multi-tenant scale; potential future migration path to external vector store if recall/latency degrade.

### ADR 2 – Selenium Primary, Playwright Fallback
**Decision**: Use Selenium + undetected-chromedriver as baseline; stub interface to allow Playwright for problematic sites.  
**Rationale**: Team familiarity + broad ecosystem; fallback reduces brittle site risk.  
**Tradeoff**: Need dual test harness; slightly more maintenance.

### ADR 3 – Retrieval Pipeline (FTS → Vector → LLM)
**Decision**: Keep LLM last for precision after cheap filters.  
**Rationale**: Cost control, incremental explainability (each stage scored).  
**Risk**: Overly aggressive thresholds could drop good jobs; mitigated by calibration set.

### ADR 4 – OpenAI Provider Abstraction
**Decision**: Use OpenAI embeddings + GPT‑4.1/4o-mini behind a client interface.  
**Swap Path**: Implement provider contract: embed(text[]) -> vectors; score(payload) -> structured JSON.  
**Mitigation**: Avoid provider-specific prompt tokens (e.g., extended system roles).

### ADR 5 – Limited Auto‑Apply Scope (Greenhouse + Lever)
### ADR 6 – Embedding Versioning & Migration (0006)
Ensures backward compatibility & progressive re‑embedding with evaluation gating & backlog metrics.

### ADR 7 – Adaptive Matching Feedback Loop (0007)
Closed-loop learning: logistic regression over match feature vector with rollback on AUC regression.

### ADR 8 – Production Anti‑Bot Posture (0008)
Layered proxies, fingerprint randomization, human behavior simulation, adaptive backoff, optional captcha solver, ethical guardrails.
**Decision**: Deliver reliable automation for two major portals first; define portal template DSL for scalable expansion.  
**Consequence**: Early user value with controlled complexity; reduces initial risk from Workday/Taleo variance.

---

## 5. Security & Compliance Overview

### 5.1 Data Classification
| Class | Examples | Handling |
|-------|----------|----------|
| PII | name, phone, address | Encrypted columns (pgcrypto / KMS) |
| Sensitive | veteran, ethnicity, disability | Consent-gated; never sent to LLM |
| Operational | crawl logs, job metadata | Retained 30 days (cost control) |
| Artifacts | DOM snapshots, receipts | Signed URLs; retention policy |

### 5.2 Threats & Mitigations
| Threat | Mitigation |
|--------|-----------|
| Credential theft | Short-lived sessions + refresh; hashed passwords (argon2/bcrypt); no secrets in logs |
| Portal anti-bot / captchas | Rate caps, random jitter, manual review switch, domain backoff |
| LLM data leakage | Redaction layer, omit sensitive fields, minimal chunk context |
| DOM drift / selector change | Snapshot & template diff alerts, fast patch release |
| SQL injection / API abuse | Pydantic validation, parameterized queries, rate limiting per IP |

### 5.3 Privacy Controls
* Explicit consent flags per sensitive attribute; absence → null storage.
* Never include sensitive columns in LLM prompt assembly.
* Future: Right-to-delete endpoint scrubs resume fulltext + embeddings.

### 5.4 Key Management
* KMS-managed keys for pgcrypto.
* Vault or parameter store for portal credentials; not persisted in plain DB except hashed references.

---

## 6. Reliability & Observability

### 6.1 Golden Signals & Metrics
| Domain | Metric | Purpose |
|--------|--------|---------|
| Crawl | success_rate, duplicated_urls, pages_per_min | Health + politeness tuning |
| Crawl (Anti-bot) | challenge_rate, captcha_rate, block_rate | Detection pressure signals |
| Embedding | latency_ms, batch_size, cached_hits | Cost + performance insight |
| Embedding Migration | reembed_backlog, active_version, reembed_rate | Track migration progress |
| Matching | fts_pruned_count, vector_pruned_count, lmm_calls | Efficiency tracking |
| Matching Adaptive | model_version, training_samples, training_auc | Feedback loop quality |
| Review Loop | avg_iterations, score_delta, rewrite_failures | UX & cost optimization |
| Apply | apply_success_rate, receipt_latency, portal_error_rate | Reliability for automation |
| Cost | tokens_review, tokens_rewrite, cost_per_job | Budget guardrails |

### 6.2 Instrumentation
* Structured JSON logs (correlation ids: job_id, resume_id, application_id).
* Metrics exporter (Prometheus/OpenTelemetry) → dashboards.
* Sampling tracing (OpenTelemetry) around crawl, match pipeline, rewrite, apply.

### 6.3 Resilience Patterns
* Queue-based isolation between crawl, ingest, process.
* Idempotent UPSERT on unique job URL.
* Backoff + dead-letter queues for irrecoverable failures.
* Iteration cap prevents runaway LLM loops.

### 6.4 Backup & DR
| Asset | Strategy | RPO | RTO |
|-------|----------|-----|-----|
| Postgres | Nightly snapshot + WAL retention 24h | 24h | 4h |
| Object store | Versioning enabled; lifecycle transitions | 24h | 4h |
| Config (templates) | Git + infra as code | <1h | 2h |

---

## 7. Deployment & Operations

### 7.1 Environments
* **dev**: Docker Compose (Postgres+pgvector, MinIO, Redis, workers, frontend).
* **prod**: Single region (Fly.io / ECS) with managed Postgres + object store (S3/compatible).

### 7.2 CI/CD Flow
1. Lint + type check (Python & TS).  
2. Run unit + fast integration tests.  
3. Generate OpenAPI types → fail if diff uncommitted.  
4. Build images (multi-stage).  
5. Deploy via GitHub Actions (tag-driven).  

### 7.3 Rollouts & Feature Flags
* Canary worker group for new scraper adapters.
* Flags: `enable_workday_portal`, `enable_resume_iteration_loop`, `enable_cover_letter_gen`.

### 7.4 Runbook Snippets
| Incident | Action |
|----------|--------|
| Crawl 429 spike | Lower QPS threshold; requeue after exponential delay |
| Selector failure | Review snapshot, update portal template, redeploy adapter |
| LLM cost surge | Raise cosine threshold, reduce jobs/day, switch to smaller model |
| High apply error rate | Disable affected portal flag; inspect DOM updates |

---

## 8. Data & API Snapshot

### 8.1 Core REST Endpoints (Illustrative)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /v1/matches?resume_id=... | List ranked jobs |
| POST | /v1/reviews/{job_id}/start | Initiate review iteration=1 |
| POST | /v1/reviews/{review_id}/rewrite | AI rewrite request |
| POST | /v1/reviews/{review_id}/upload | Manual revision upload |
| POST | /v1/reviews/{review_id}/next | Next iteration after acceptance |
| POST | /v1/reviews/{review_id}/satisfied | Mark satisfied (enable apply) |
| POST | /v1/applications/{job_id}/apply | Trigger auto‑apply |
| GET | /v1/applications?company_id=... | Retrieve applications + receipts |
| POST | /v1/uploads/presign | Get presigned upload URL |
| POST | /v1/resumes/ingest | Register & parse uploaded resume |

### 8.2 Retention & TTL Policies
| Data | TTL | Notes |
|------|-----|------|
| Raw JD HTML snapshots | 90 days | Compression + potential dedupe |
| DOM/PDF receipts | 1 year | Compliance & audit value |
| Logs | 30 days | Cost control; aggregated metrics kept longer |
| Review iteration artifacts | 180 days | May purge early if user satisfied |

---

## 9. Risk Register (Top 5)
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Portal DOM change | High | Medium | Template DSL, snapshots, fast patch release, feature flags |
| Rate limiting / Captcha | High | Medium | Anti-bot posture (ADR 0008), adaptive backoff, captcha solver flag |
| Embedding model deprecation | Medium | High | Version tracking & progressive migration (ADR 0006) |
| Feedback model drift | Medium | Medium | A/B + AUC monitoring, rollback weights (ADR 0007) |
| LLM cost overruns | Medium | Medium | Threshold tuning, iteration cap, switch model, budget alerts |
| PII mishandling | Low | High | Encryption, consent gating, redaction, audit logging |
| pgvector perf degradation | Low | Medium | IVF tuning, ANALYZE, partition plan, external engine fallback |

---

## 10. “Hows & Wows” (Differentiators)
| Area | How (Implementation Detail) | Wow (Value/Edge) |
|------|------------------------------|------------------|
| Cost Control | 3-stage pruning + iteration cap | Sub-dollar evaluation for dozens of jobs |
| Resume Iteration | Structured improvement_plan_json diffed + partial embedding reuse | Faster feedback cycles; avoids full re-embedding cost |
| Portal Automation | Declarative template DSL + artifact capture (DOM snapshots) | Rapid portal patching + auditable actions |
| Observability | Correlated IDs across crawl→match→apply | Trace root cause quickly (selector or data shift) |
| Privacy | Explicit consent gating + selective prompt assembly | Trustworthy handling of sensitive attributes |

---

## 11. Next Steps Checklist (Execution Ready)
| Priority | Task | Owner (TBD) |
|----------|------|-------------|
| P0 | Finalize SQLAlchemy models & Alembic migrations (REVIEWS extensions) |  |
| P0 | Implement crawl adapters (2 portals) + politeness config |  |
| P0 | Embed + match pipeline (batching + thresholds) |  |
| P0 | Review generation + rewrite worker skeleton |  |
| P0 | Auto‑apply orchestrator (Greenhouse, Lever) |  |
| P1 | Frontend Matches & Resume Versions pages |  |
| P1 | Iterative review loop UI integration |  |
| P1 | Metrics & dashboards (Prometheus/Grafana) |  |
| P1 | Security pass (encryption, consent flags) |  |
| P2 | Cover letter generation |  |
| P2 | Additional portals (Workday) |  |

---

## 12. Appendix
* Full ER diagrams & extended flows: see `docs/ARCHITECTURE.md`.
* Future ADR candidates: Cover letter generation strategy, Workday adapter sandboxing, multi-tenant scaling posture.

---

End of Technical Review Packet.
