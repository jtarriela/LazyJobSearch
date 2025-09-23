# Roadmap (Living)

This roadmap highlights major engineering workstreams including newly added production-hardening pillars (embedding versioning, adaptive feedback loop, anti-bot posture). Dates are indicative; adjust as capacity & validation data evolve.

## Legend
P0 = Critical foundation / unblockers  
P1 = High impact after foundations  
P2 = Nice-to-have / differentiators

## Phase 1 (MVP Foundations)
| Priority | Epic | Key Deliverables |
|----------|------|------------------|
| P0 | Core Schema & Migrations | Initial tables + Alembic baseline |
| P0 | Resume Ingest & Chunking | Parser, chunker, embedding client (v1 model) |
| P0 | Crawl & JD Storage | 2–3 portal adapters, dedupe, artifact compression |
| P0 | Matching Pipeline v1 | FTS → vector → LLM scoring, MATCHES persistence |
| P0 | Review & Resume v2 Loop | LLM critique, iterative version chain |
| P0 | Auto-Apply MVP | Greenhouse + Lever template DSL (ADR 0005) |

## Phase 2 (Stabilization & Hardening)
| Priority | Epic | Key Deliverables |
|----------|------|------------------|
| P0 | Anti-Bot Posture v1 (ADR 0008) | Proxy pool, fingerprint randomization, behavior sim, adaptive backoff, metrics |
| P0 | Embedding Version Infrastructure (ADR 0006) | Added columns, registry table, backlog metrics, migration CLI |
| P1 | Feedback Data Capture (ADR 0007) | match_outcomes table, manual CLI capture, baseline analytics |
| P1 | Admin Observability | Dashboards: crawl, embedding, matching, apply, cost |
| P1 | Change Detection | JD fingerprint diff → selective reprocessing |
| P1 | Cover Letter Generation | Prompt + traceability mapping |

## Phase 3 (Adaptive Intelligence & Scale)
| Priority | Epic | Key Deliverables |
|----------|------|------------------|
| P0 | Adaptive Ranking Activation (ADR 0007) | Logistic regression training job, weight rollout & rollback gates |
| P0 | Embedding Migration Execution (ADR 0006) | Progressive re-embed, evaluation harness (precision/recall) |
| P1 | Advanced Anti-Bot v2 | Captcha solver flag, session outcome heuristics, session pool auto-tune |
| P1 | Multi-Resume Strategy | A/B resume variant scoring & selection |
| P2 | Social / Warm Intro Module | Contact graph ingestion, intro recommendation |
| P2 | External Search Engine Option | OpenSearch integration (hybrid rank) |

## Cross-Cutting Concerns
| Concern | Strategy |
|---------|----------|
| Cost Control | Token budgets, adaptive thresholds, backlog pacing |
| Privacy & Ethics | Consent gates, no sensitive PII in prompts, robots compliance |
| Rollback Safety | Feature flags, versioned weights, previous embedding retention until success |
| Observability | Metrics prefix per domain: scrape.*, embed.*, match.*, apply.*, feedback.* |
| Security | Vault/KMS for credentials & proxy keys; rotate secrets quarterly |

## Metrics Gates (Promotion Criteria)
| Gate | Metric | Threshold |
|------|--------|-----------|
| Activate Adaptive Ranking | Labeled samples | ≥500 match_outcomes with interview labels |
| Adaptive Ranking Rollout | AUC vs baseline | ΔAUC ≥ +0.03 or maintain with lower cost |
| New Embedding Version | Recall@50 regression | ≤2% relative drop after partial migration |
| Anti-Bot Posture Healthy | scrape.block_rate | <2% sustained |
| Migration Backlog Health | reembed_backlog burn | 95% cleared within target window |

## Open Questions
| Topic | Notes |
|-------|-------|
| Proxy Pool Sizing | Start N=10 rotating; auto-scale based on block_rate & QPS target |
| Feature Set for Ranking v2 | Add skill_overlap, recency_factor, portal_type weighting? |
| Embedding Eval Dataset | Curate 200+ labeled good/bad pairs (semi-manual) |
| Captcha Solver Use Policy | Only enable on explicit operator approval; audit tokens |

## References
* ADR 0006 – Embedding Versioning & Migration
* ADR 0007 – Adaptive Matching Feedback Loop
* ADR 0008 – Production Anti-Bot & Humanization Posture
* Technical Review Packet sections 4, 6, 9

---
This file is informational; update via PR alongside ADR or architecture changes.
