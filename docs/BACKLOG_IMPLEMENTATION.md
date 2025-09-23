# Implementation Backlog (Post-Scaffolding)

This document lists concrete follow-up tasks required to move from stubs to production implementation for ADRs 0006–0008.

## Embedding Versioning (ADR 0006)
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
