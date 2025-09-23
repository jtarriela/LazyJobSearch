## ADR 0006: Embedding Versioning & Migration Strategy

Date: 2025-09-23  
Status: Accepted

### Context
Current design stores raw embedding vectors without model/version metadata. Provider model changes (name, dimensions, quality) risk incompatibility and silent drift. Need controlled progressive re-embedding and evaluation before promotion.

### Decision
Add version metadata columns, a registry table, and a batch migration workflow:
1. Columns: `embedding_version`, `embedding_model`, `needs_reembedding BOOLEAN DEFAULT false` on `job_chunks`, `resume_chunks`.
2. Registry table `embedding_versions(version_id PK, model_name, dimensions, created_at, deprecated_at, compatible_with[])`.
3. All new vectors stamped with active version (queried from registry).
4. Migration pipeline marks legacy rows `needs_reembedding=true` and processes in rate-limited batches.
5. Evaluation harness compares old vs new (precision/recall, recall@K) before activating.
6. Backlog metrics exposed for ops.

### Rationale
Prevents data invalidation, supports rollback, enables objective quality gating, and smooths cost.

### Consequences
- Slight storage increase.
- Added complexity in embedding service & migrations.
- Mixed-version period requires consistent dimensionality; dimension change requires full index rebuild after migration.

### Alternatives
| Option | Drawback |
|--------|----------|
| No versioning | Undetected drift, broken similarity |
| Only store model name | No semantic versioning / compatibility sets |
| Stop-the-world re-embed | Downtime, spike cost |

### Follow-Up
1. Alembic migration for columns + table.
2. Implement `EmbeddingVersionManager`.
3. CLI: `ljs embeddings migrate --target-version vX.Y`.
4. Add metrics: `embeddings.reembed_backlog`, `embeddings.active_version`.
5. Add tests ensuring version metadata persisted.

---