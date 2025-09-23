# ADR 0003: Matching Pipeline = FTS Prefilter → Vector Similarity → LLM Judge

Date: 2025-09-23
Status: Accepted

## Context
Need to rank large sets of job postings against resumes without incurring excessive LLM costs.

## Decision
Three-stage retrieval & scoring pipeline: PostgreSQL FTS prefilter, pgvector ANN similarity, final LLM scoring for top-K.

## Rationale
- Cheapest lexical filter removes obviously irrelevant postings
- Vector similarity catches semantic matches missed by keywords
- LLM reserved for small candidate set to produce explanations & final ranking

## Consequences
- Requires calibration of thresholds (FTS query expansion + cosine cut)
- Potential false negatives if thresholds too strict

## Alternatives
- LLM-only scoring: prohibitively expensive, slower
- Vector-only ranking: lacks explainability and fine-grained reasoning
- Hybrid BM25 + vector rerank (no LLM): less qualitative reasoning for user insight

## Follow-Up
- Build evaluation harness with labeled good/bad pairs
- Track recall@K while tuning thresholds
