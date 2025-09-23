# Architecture Decision Records (ADRs)

This directory contains architecturally significant decisions for LazyJobSearch. Each ADR follows a lightweight, immutable (once Accepted) format: Context → Decision → Rationale → Consequences → Alternatives → Follow-Up.

## Index

| ID | Title | Status | Date | Summary |
|----|-------|--------|------|---------|
| 0001 | Use Postgres + pgvector as Unified Store | Accepted | 2025-09-23 | Single DB for relational + vector workloads. |
| 0002 | Selenium Primary, Playwright Fallback | Accepted | 2025-09-23 | Selenium baseline with optional Playwright for hard sites. |
| 0003 | FTS → Vector → LLM Matching Pipeline | Accepted | 2025-09-23 | Cheap lexical + semantic pruning before LLM judge. |
| 0004 | OpenAI Provider Abstraction | Accepted | 2025-09-23 | Thin interface to allow future model swap. |
| 0005 | Auto-Apply MVP Scope = Greenhouse + Lever | Accepted | 2025-09-23 | Constrain automation to two portals first. |
| 0006 | Embedding Versioning & Migration Strategy | Accepted | 2025-09-23 | Track versions & progressive re-embedding. |
| 0007 | Adaptive Matching Feedback Loop & Learned Weights | Accepted | 2025-09-23 | Closed-loop ranking via outcomes. |
| 0008 | Production-Grade Anti-Bot & Humanization Posture | Accepted | 2025-09-23 | Layered proxies, fingerprint & behavior sim. |
| 0009 | Multi LLM Provider Expansion | Accepted | 2025-09-23 | Add Gemini & Claude under unified provider registry. |

## Workflow
1. Draft new ADR using template.
2. PR review: discuss trade-offs; may revise.
3. Merge with `Status: Accepted` or `Status: Superseded`/`Rejected` as needed.
4. Never edit historical Accepted ADR text (except typo); create a new ADR to supersede.

## Naming & Numbering
- Increment numeric prefix (zero-padded to 4 digits).
- Kebab-case succinct slug after number.
- Keep title under ~60 chars.

## Status Values
- Proposed: Under review.
- Accepted: Decision stands and is in effect.
- Rejected: Considered but not adopted.
- Superseded: Replaced by a newer ADR; include reference.
- Deprecated: Still in codebase but scheduled for removal.

## Template
See `TEMPLATE.md` in this directory.

## Cross-Links
- System architecture reference: `../ARCHITECTURE.md` (Section 4 summarizes ADRs)
- Technical review packet: `../TECHNICAL_REVIEW_PACKET.md` (Section 4 lists top ADRs)

---
_Last updated: 2025-09-23_
