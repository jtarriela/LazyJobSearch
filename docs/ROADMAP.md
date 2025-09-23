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
| P1 | Adaptive Ranking v1 | RLHF pipeline, manual feedback loop, A/B test framework |
| P1 | Multi-User & Teams | User management, workspace isolation, collaboration features |
| P2 | Advanced Portals | Workday, SAP SuccessFactors, custom ATS integration |
| P2 | Resume Coaching AI | Personalized improvement suggestions, industry benchmarking |
| P2 | Interview Prep | Mock interviews, company-specific preparation |

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
- **Resume Version Storage**: Delta-only vs. full snapshots for v2+ iterations?
- **Embedding Backfill**: Batch size for re-embedding when model changes?
- **Portal Rate Limits**: Dynamic backoff vs. fixed delays per site?
- **LLM Provider Failover**: OpenAI → Anthropic → local model fallback chain?
- **Multi-resume Strategy**: Per-job tailoring vs. industry-specific versions?

## References
- [Architecture Document](./ARCHITECTURE.md) - Complete system design
- [Technical Review Packet](./TECHNICAL_REVIEW_PACKET.md) - Implementation details
- [ADRs](./adrs/) - Architecture decision records
- [Schema Documentation](./schema/) - Database schema reference