# ADR 0001: Use Postgres + pgvector as Unified Store

Date: 2025-09-23
Status: Accepted

## Context
Need a single persistence layer for relational entities (jobs, resumes, matches) and similarity search over embeddings.

## Decision
Adopt Postgres 16 with `pgvector` extension instead of a dedicated vector DB (Pinecone/Qdrant) for MVP.

## Rationale
- Operational simplicity (one backup + monitoring surface)
- Adequate performance for O(10^5–10^6) vectors with IVF indexes
- Native transactional consistency (schema + embeddings updated atomically)
- Built-in FTS pairs naturally with semantic search.

## Consequences
- Potential reindex / maintenance overhead for very large (>10^7) vectors
- May later require external engine for advanced ANN features or multi-region low latency.

## Alternatives Considered
- Pinecone: faster vector iteration, higher cost & separate ops.
- Qdrant: self-host overhead + added operational surface.
- Elastic w/ dense_vector: less mature for hybrid scoring compared to dedicated PG approach.

## Follow-Up Actions
- Define maintenance task: `VACUUM ANALYZE` schedule for vector-heavy tables.
- Capture empirical latency benchmarks once >200k vectors.
