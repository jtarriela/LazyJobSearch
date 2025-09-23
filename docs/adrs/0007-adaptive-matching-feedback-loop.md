## ADR 0007: Adaptive Matching Feedback Loop & Learned Weights

Date: 2025-09-23  
Status: Accepted

### Context
Static heuristic ranking ignores real-world outcomes (responses, interviews, offers). Without feedback, accuracy plateaus and blind spots persist.

### Decision
Capture application outcomes and train a lightweight logistic regression over engineered features (vector_score, llm_score, fts_rank_norm, yoe_gap, skill_overlap, recency_factor). Store weights and apply them for final score once a data sufficiency threshold reached.

### Rationale
Interpretable, low-latency improvement path; incremental adoption; supports audit & rollback.

### Consequences
- Requires new tables (`match_outcomes`, `matching_feature_weights`).
- Training job scheduling & monitoring.
- Early sparsity; activation gated on minimum samples.

### Alternatives
| Option | Drawback |
|--------|----------|
| Manual tuning | Slow, subjective |
| Deep model early | Overkill, cost, opacity |
| External ML platform | Operational drag |

### Follow-Up
1. Alembic migration create tables.
2. CLI capture command for manual data seeding.
3. Metrics: `matching.training.samples`, `matching.training.auc`.
4. Bias detection query surfaces false negatives.
5. Rollback mechanism if AUC drops.

---