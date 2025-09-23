## ADR 0009: Multi LLM Provider Expansion (OpenAI + Gemini + Claude)

Date: 2025-09-23  
Status: Accepted

### Context
ADR 0004 created an abstraction around OpenAI for embeddings & LLM scoring. Single‑provider reliance raises outage, pricing, and policy risk and prevents comparative regression detection. Newer vendor offerings (Anthropic Claude, Google Gemini) present distinct latency/quality/cost tradeoffs enabling strategic routing (e.g., cheap fast model for bulk scoring; premium model for borderline decisions).

### Decision
Introduce a provider registry implementing a unified contract: `embed(texts) -> list[vectors]`, `score_match(payload) -> ScoringResult`, `generate_review(payload) -> ReviewResult`. Register OpenAI (primary), Anthropic Claude, Google Gemini. YAML configuration defines ordered fallbacks per logical role (scoring, review). A policy component observes rolling latency & error windows; threshold breach triggers failover and emits metrics; periodic half‑open probe attempts restore primary.

Embeddings remain OpenAI initially; alternative embedding evaluation (Gemini/Claude or local) deferred behind evaluation harness (ADR 0006).

### Rationale
- Resilience: automatic failover reduces MTTR for upstream incidents.
- Cost optimization: route high‑confidence or bulk tasks to cheaper fast models (Gemini Flash / Claude Haiku) while retaining premium model for ambiguous edge cases.
- Quality monitoring: cross‑provider evaluation set surfaces silent regressions.
- Lock‑in mitigation: diversified posture strengthens negotiation & roadmap agility.

### Implementation Outline
1. Define `LLMProvider` protocol / ABC with capability metadata (supports_embeddings, max_input_tokens, pricing_hints).
2. Implement `OpenAIProvider`, `ClaudeProvider`, `GeminiProvider` (initial: scoring/review only for non‑OpenAI embeddings).
3. Policy manager: sliding window (e.g., last 50 calls) tracks `error_rate`, `latency_p95`; on breach → promote next provider & increment `failover_events`.
4. Response validator: strict JSON schema; layered retry (provider JSON mode / forced format) → escalate after N failures.
5. Evaluation harness: nightly job runs curated gold prompt set across providers; compute score deltas & parse failure rate; alert anomalies.
6. CLI commands: `ljs llm providers list`, `ljs llm failover status`, `ljs llm eval run`.
7. Config keys: `llm.scoring.primary`, `llm.scoring.fallbacks[]`, `llm.review.primary`, `llm.review.fallbacks[]`, per‑provider rate caps.

### Metrics
- `llm.provider.usage{provider=}`
- `llm.provider.latency_p95{provider=}`
- `llm.provider.error_rate{provider=}`
- `llm.provider.failover_events`
- `llm.provider.parse_fail_rate{provider=}`
- `llm.provider.cost_estimate{provider=}` (derived)

### Success Criteria
- Failover MTTR < 2 minutes.
- JSON parse failure rate < 2% after normalization.
- Mean absolute score delta vs primary on gold set < 5 points (0–100 scale) unless flagged.

### Consequences
- More secrets (API keys) to manage → enforce rotation schedule.
- Normalization layer required (safety refusal semantics, JSON adherence differences).
- Greater test surface (evaluation harness) adds maintenance overhead.

### Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Divergent safety filtering | Track refusal spike metric; adapt prompt; switch provider |
| Inconsistent JSON adherence | Multi‑attempt validator + forced JSON mode fallback |
| Key leakage | Central secrets store; least‑priv env vars; quarterly rotation |

### Follow-Up (Deferred)
- Alternative embedding provider experiment & migration playbook.
- Capability-aware dynamic model selection (context size vs token limit).
- Cost attribution dashboard (per provider & feature) & anomaly alerts.

Status: Accepted.

---