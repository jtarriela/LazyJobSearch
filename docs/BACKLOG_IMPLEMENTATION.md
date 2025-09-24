# Implementation Backlog (Post-MVP Audit)

This document lists concrete follow-up tasks identified during the comprehensive MVP audit, organized by priority and implementation wave. Updated based on audit findings from September 2024.

## CRITICAL PRIORITY - System Blockers

### Resume Processing (CRITICAL)
- [ ] **Implement PDF parsing** - Currently only placeholder text (libs/resume/parser.py)
- [ ] **Implement DOCX parsing** - Currently only placeholder text (libs/resume/parser.py)  
- [ ] **Add PII encryption** - Sensitive resume data stored in plaintext (libs/db/models.py)
- [ ] **Implement resume deduplication** - Same resume creates multiple records (libs/resume/ingestion.py)

### Matching Engine (CRITICAL)
- [ ] **Implement FTS prefiltering** - Missing PostgreSQL TSVECTOR queries for O(log n) performance
- [ ] **Implement vector similarity search** - Missing pgvector cosine distance queries
- [ ] **Replace LLM scoring stub** - Currently returns mock values (libs/matching/pipeline.py:139)
- [ ] **Add database indexes** - Missing GIN indexes for FTS, vector indexes for embeddings

### Job Crawling (CRITICAL)
- [ ] **Add duplicate detection** - Jobs inserted without checking for existing records (libs/scraper/crawl_worker.py)
- [ ] **Implement pagination** - Multi-page job listings not supported (libs/scraper/anduril_adapter.py)

### Review Workflow (CRITICAL)
- [ ] **Integrate LLM for resume critique** - No actual AI review implementation (libs/resume/review.py)
- [ ] **Implement resume rewriting** - CLI commands are placeholders (cli/ljs.py:1217+)

## HIGH PRIORITY - Core Functionality Gaps

### CLI Layer (HIGH)
- [ ] **Implement user management commands** - `ljs user show` and `ljs user sync` missing (cli/ljs.py)
- [ ] **Add apply bulk operations** - `ljs apply bulk` and `ljs apply status` commands missing
- [ ] **Standardize review command naming** - CLI has `review list` but docs expect `review show`

### Portal Templates (HIGH) 
- [ ] **Create Lever portal template** - Only Greenhouse template exists (docs/examples/portal_templates/)
- [ ] **Implement template execution sandbox** - Security risk without execution isolation
- [ ] **Add input sanitization** - Template variables not sanitized for XSS protection

### Schema & Documentation (HIGH)
- [ ] **Document missing tables** - 7 tables lack markdown docs (companies, embedding_versions, etc.)
- [ ] **Fix model-documentation mismatches** - 20+ documented fields missing from models
- [ ] **Add migration testing** - No forward/backward compatibility tests

## MEDIUM PRIORITY - Infrastructure & Performance

### Observability (MEDIUM)
- [ ] **Define log schema** - Structured logging fields not standardized
- [ ] **Add metrics inventory** - Missing counters for jobs_crawled_total, match_runtime_seconds
- [ ] **Implement error budget tracking** - No SLO definition for pipeline success rates

### Performance & Scalability (MEDIUM)
- [ ] **Add performance benchmarks** - No runtime measurements for matching (N=1k jobs)
- [ ] **Implement parallel processing** - Single-threaded company crawling
- [ ] **Add resource profiling** - No CPU/memory measurement for pipeline runs

### Security & Compliance (MEDIUM)
- [ ] **Add input validation** - Resume parsing lacks size limits and format validation
- [ ] **Implement PII redaction** - Sensitive data may leak in logs
- [ ] **Add dependency audit** - No SBOM or CVE scanning

## EXISTING ADR TASKS (Unchanged)
- [ ] Populate `embedding_versions` with initial active version row.
- [ ] Implement DB query logic in `EmbeddingVersionManager.get_active_version`.
- [ ] Add CLI commands:
  - `ljs embeddings status`
  - `ljs embeddings migrate --target-version vX.Y --batch-size N`
- [ ] Implement batch selector + update for `needs_reembedding` rows.
- [ ] Add metric emitters: `embeddings.reembed_backlog`, `embeddings.active_version`.
- [ ] Add integration test verifying progressive migration updates counts.

## Adaptive Matching Feedback Loop (ADR 0007)
- [ ] Implement persistence layer for `FeedbackCapture.capture`.
- [ ] Implement `FeedbackTrainer.fetch_training_data` SQL query joining matches + outcomes.
- [ ] Add scikit-learn logistic regression training with train/validation split & AUC metric.
- [ ] Persist new weight row in `matching_feature_weights` with model_version semantic versioning.
- [ ] Feature gating: environment flag to enable adaptive scoring.
- [ ] Add evaluation harness CLI: `ljs feedback evaluate --lookback 30`.
- [ ] Add metrics: `matching.training.samples`, `matching.training.auc`, `matching.model.version`.
- [ ] Add unit tests for weight application + rollback logic.

## Anti-Bot Posture (ADR 0008)
- [ ] Integrate `ProxyPool` with actual provider (config file / env secrets).
- [ ] Implement fingerprint injection (Chrome options & script patches) in scraper worker.
- [ ] Replace linear mouse path with Bezier curve interpolation.
- [ ] Add scroll behavior simulation & dwell time randomization.
- [ ] Add detection signal hooks (e.g., checking for challenge elements) -> state machine transitions.
- [ ] Persist `scrape_sessions` rows at start/finish with metrics JSON.
- [ ] Metrics emission: `scrape.block_rate`, `scrape.challenge_rate`, `scrape.captcha_rate`.
- [ ] Add optional captcha solver integration behind `ENABLE_CAPTCHA_SOLVER` flag.
- [ ] Add integration test using a mock site fixture to validate session capture.

## Cross-Cutting
- [ ] Update CI to run Alembic `--autogenerate` check ensuring migration coverage.
- [ ] Document new CLI commands in README or CLI_DESIGN.
- [ ] Add type hints & mypy target (if introduced later) for new modules.
- [ ] Security review: store proxy & captcha creds in secrets manager (placeholder doc update).

## Deferred / Nice-to-Have
- [ ] Weight model calibration dashboard (simple FastAPI endpoint returning current weights & AUC history).
- [ ] Proxy health scoring (success rate, average session length) and rotation heuristic.
- [ ] Outcome labeling helper UI (manual annotation tool to accelerate feedback data collection).

---
Generated alongside migration + stub scaffolding on 2025-09-23.
