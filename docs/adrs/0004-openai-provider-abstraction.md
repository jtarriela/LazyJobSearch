# ADR 0004: OpenAI Provider Abstraction

Date: 2025-09-23
Status: Accepted

## Context
Embedding + scoring quality needed quickly; want future portability to alternative models.

## Decision
Use OpenAI (text-embedding-3-large + GPT-4.1/4o-mini) behind a thin provider interface.

## Rationale
- High quality out-of-box
- Reduced prompt engineering time
- Stable latency and availability

## Consequences
- Cost variability; need budget guardrails
- Vendor lock-in risk (mitigated via abstraction)

## Alternatives
- Local embedding models (inference infra overhead now)
- Other SaaS LLM APIs (similar lock-in w/out current quality parity)

## Follow-Up
- Define interface: `EmbeddingProvider`, `LLMScoringProvider`
- Add unit tests using a mock provider for deterministic outputs
