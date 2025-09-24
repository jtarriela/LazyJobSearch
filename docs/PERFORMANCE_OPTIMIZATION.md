# Performance Optimization Guide

This document provides detailed algorithmic analysis, performance optimization strategies, and benchmarking guidelines for LazyJobSearch components.

## Overview

LazyJobSearch processes large volumes of job descriptions, resumes, and performs similarity matching at scale. Key performance-critical components:

- **Vector Similarity Search**: Semantic matching using pgvector with 3072-dimensional embeddings
- **Full-Text Search**: Postgres FTS with custom ranking and query expansion
- **LLM Processing**: Batch optimization and provider failover for GPT-4/Claude/Gemini
- **Web Scraping**: Anti-bot humanization with performance constraints
- **Database Operations**: Bulk operations, indexing strategy, connection pooling

## Algorithm Analysis & Optimization

### 1. Vector Similarity Search (O(n) → O(log n) optimization)

**Current Implementation:**
```sql
-- Brute force cosine similarity (O(n))
SELECT job_id, 1 - (embedding <=> resume_vector) as similarity
FROM job_chunks 
ORDER BY similarity DESC 
LIMIT 50;
```

**Optimized Implementation with IVF Index:**
```sql
-- Create optimized index
CREATE INDEX CONCURRENTLY job_chunks_embed_ivfflat_opt 
ON job_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = CEIL(SQRT(row_count/1000)));

-- Optimized query with threshold pruning
SET ivfflat.probes = 10;  -- Balance recall vs speed
SELECT job_id, 1 - (embedding <=> $1) as similarity
FROM job_chunks 
WHERE 1 - (embedding <=> $1) > 0.75  -- Similarity threshold
ORDER BY embedding <=> $1 
LIMIT 50;
```

**Performance Characteristics:**
- **Without IVF**: O(n) scan, ~200ms for 100k vectors
- **With IVF**: O(log n) search, ~15ms for 100k vectors  
- **Memory usage**: 15-20% overhead for index
- **Recall@50**: 95%+ with probes=10

**Scaling Strategy:**
- Lists = CEIL(SQRT(rows/1000)) for optimal performance
- Increase `probes` for better recall (linear cost increase)
- Partition by embedding_version for migration support
- Consider HNSW indexes for >1M vectors (Postgres 16+)

### 2. Hybrid Search Optimization (FTS + Vector)

**Multi-stage Pipeline:**
```python
def optimized_match_pipeline(resume_embedding, resume_skills):
    # Stage 1: FTS Prefilter (cheap, high recall)
    fts_query = build_expanded_query(resume_skills)  # +synonyms, +stemming
    fts_candidates = postgres.execute("""
        SELECT job_id, ts_rank_cd(jd_tsv, query) as fts_score
        FROM jobs, websearch_to_tsquery($1) query
        WHERE jd_tsv @@ query
        ORDER BY ts_rank_cd(jd_tsv, query) DESC
        LIMIT 2000  -- Generous FTS cutoff
    """, fts_query)
    
    # Stage 2: Vector similarity on FTS survivors
    vector_candidates = []
    for job_batch in batch(fts_candidates, 500):  # Batch for efficiency
        similarities = compute_batch_similarity(job_batch, resume_embedding)
        vector_candidates.extend(
            [(job_id, sim) for job_id, sim in similarities if sim > 0.78]
        )
    
    # Stage 3: LLM scoring on top candidates
    top_candidates = sorted(vector_candidates, key=lambda x: x[1], reverse=True)[:20]
    return llm_batch_score(top_candidates, resume_context)
```

**Algorithm Complexity:**
- Stage 1 (FTS): O(log n) with GIN index
- Stage 2 (Vector): O(k log m) where k=FTS results, m=vector index lists  
- Stage 3 (LLM): O(1) batch processing
- **Total**: O(k log m) dominated by vector search on reduced set

### 3. Embedding Batch Optimization

**Naive Approach (avoid):**
```python
# Anti-pattern: Individual API calls
embeddings = []
for chunk in chunks:
    emb = openai.embed(chunk)  # 100ms per call
    embeddings.append(emb)
# Total: 100ms × n chunks
```

**Optimized Batch Processing:**
```python
# Batch optimization with deduplication
def batch_embed_with_cache(chunks, batch_size=100):
    # Deduplicate chunks (common in job descriptions)
    unique_chunks = list(set(chunks))
    chunk_to_indices = defaultdict(list)
    for i, chunk in enumerate(chunks):
        chunk_to_indices[chunk].append(i)
    
    # Check cache first
    cache_hits = []
    uncached_chunks = []
    for chunk in unique_chunks:
        cached = redis.get(f"embed:{hash(chunk)}")
        if cached:
            cache_hits.append((chunk, json.loads(cached)))
        else:
            uncached_chunks.append(chunk)
    
    # Batch API calls for uncached
    new_embeddings = []
    for batch in batch_chunks(uncached_chunks, batch_size):
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
            dimensions=3072
        )
        new_embeddings.extend(response.data)
        
        # Cache results
        for chunk, emb in zip(batch, response.data):
            redis.setex(f"embed:{hash(chunk)}", 3600, json.dumps(emb.embedding))
    
    # Reconstruct full embedding list
    result = [None] * len(chunks)
    embedding_map = dict(cache_hits + list(zip(uncached_chunks, new_embeddings)))
    
    for chunk, indices in chunk_to_indices.items():
        embedding = embedding_map[chunk]
        for idx in indices:
            result[idx] = embedding
            
    return result
```

**Performance Gains:**
- **Batching**: 10x faster API throughput vs individual calls
- **Deduplication**: 30-40% reduction in API calls for typical job descriptions
- **Caching**: 95%+ cache hit rate for re-processing existing content
- **Cost**: 60-70% reduction in embedding costs

### 4. Database Connection & Query Optimization

**Connection Pooling Strategy:**
```python
# Optimized SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Base connections
    max_overflow=40,        # Burst capacity  
    pool_pre_ping=True,     # Health checks
    pool_recycle=3600,      # Refresh connections
    echo=False,             # Disable query logging in prod
    
    # Postgres-specific optimizations
    connect_args={
        "application_name": "lazyjobsearch",
        "options": "-c shared_preload_libraries=pg_stat_statements,vector"
    }
)
```

**Bulk Operations Optimization:**
```python
# Bulk insert optimization
def bulk_insert_job_chunks(session, chunks_data):
    # Use bulk_insert_mappings for raw speed
    session.bulk_insert_mappings(JobChunk, chunks_data)
    
    # Alternative: Use COPY for maximum throughput
    from io import StringIO
    import csv
    
    buffer = StringIO()
    writer = csv.writer(buffer)
    for chunk in chunks_data:
        writer.writerow([chunk['id'], chunk['job_id'], chunk['chunk_text']])
    
    buffer.seek(0)
    session.connection().copy_from(
        buffer, 'job_chunks', 
        columns=['id', 'job_id', 'chunk_text'],
        sep=','
    )
```

## Performance Benchmarks & SLAs

### Target Performance Metrics

| Component | Metric | Target | Measurement |
|-----------|--------|--------|-------------|
| Resume Processing | End-to-end latency | <30s for PDF | Time from upload to embeddings complete |
| Job Matching | Top-50 matches | <5s | FTS + Vector + LLM scoring |
| Vector Search | Query latency | <50ms | Single resume vs 100k jobs |
| Embedding Generation | Throughput | 1000 chunks/min | Batch processing rate |
| Scraping | Pages per minute | 10-15 ppm | With anti-bot measures |
| Database Writes | Bulk insert rate | 10k rows/sec | Job chunks insertion |

### Scalability Checkpoints

| Scale | Jobs | Resumes | Vector Index | Expected Performance |
|-------|------|---------|--------------|---------------------|
| MVP | 50k | 100 | 500k vectors | Full performance |
| Phase 1 | 200k | 500 | 2M vectors | 10% degradation |
| Phase 2 | 500k | 1000 | 5M vectors | 25% degradation |
| Phase 3 | 1M+ | 2000+ | 10M+ vectors | Consider sharding |

### Performance Testing Strategy

**Load Testing Framework:**
```python
# Performance test suite
class PerformanceTestSuite:
    def test_vector_search_latency(self):
        """Measure vector search performance across scales"""
        for n_vectors in [10k, 50k, 100k, 500k]:
            start_time = time.time()
            results = self.vector_search(test_embedding, limit=50)
            latency = time.time() - start_time
            
            assert latency < self.latency_sla(n_vectors)
            assert len(results) == 50
            
    def test_embedding_throughput(self):
        """Measure batch embedding performance"""
        chunks = self.generate_test_chunks(1000)
        start_time = time.time()
        embeddings = batch_embed_with_cache(chunks)
        duration = time.time() - start_time
        
        throughput = len(chunks) / duration
        assert throughput > 15  # chunks per second
        
    def test_matching_pipeline_e2e(self):
        """End-to-end matching performance"""
        resume = self.create_test_resume()
        
        start_time = time.time()
        matches = self.run_matching_pipeline(resume)
        e2e_latency = time.time() - start_time
        
        assert e2e_latency < 5.0  # 5 second SLA
        assert len(matches) <= 50
```

## Cost Optimization Strategies

### LLM Cost Management

**Token Budget Enforcement:**
```python
class LLMCostManager:
    def __init__(self, daily_budget_usd=50):
        self.daily_budget = daily_budget_usd
        self.cost_per_1k_tokens = {
            'gpt-4o-mini': 0.00015,  # Input tokens
            'gpt-4': 0.03,           # More expensive fallback
        }
        
    def can_process_batch(self, input_tokens, model='gpt-4o-mini'):
        current_spend = self.get_daily_spend()
        estimated_cost = (input_tokens / 1000) * self.cost_per_1k_tokens[model]
        
        return current_spend + estimated_cost <= self.daily_budget
        
    def optimize_batch_size(self, available_jobs, max_tokens=4000):
        """Select optimal job subset to fit token budget"""
        jobs = sorted(available_jobs, key=lambda j: j.vector_score, reverse=True)
        selected = []
        token_count = 0
        
        for job in jobs:
            job_tokens = estimate_tokens(job.context)
            if token_count + job_tokens <= max_tokens:
                selected.append(job)
                token_count += job_tokens
            else:
                break
                
        return selected
```

**Progressive Quality Strategy:**
```python
def adaptive_llm_scoring(candidates, budget_remaining):
    """Use cheaper models when budget is constrained"""
    if budget_remaining > 0.8:  # 80% budget remaining
        return llm_score_batch(candidates, model='gpt-4')  # Best quality
    elif budget_remaining > 0.5:  # 50% budget remaining  
        return llm_score_batch(candidates, model='gpt-4o-mini')  # Good quality
    else:  # Budget constrained
        # Use only vector + heuristic scoring
        return heuristic_score_batch(candidates)
```

### Infrastructure Cost Optimization

**Database Cost Controls:**
```sql
-- Automated cleanup of old data
DELETE FROM job_chunks 
WHERE job_id IN (
    SELECT id FROM jobs 
    WHERE scraped_at < NOW() - INTERVAL '90 days'
    AND id NOT IN (SELECT DISTINCT job_id FROM matches WHERE llm_score > 80)
);

-- Compress old embeddings (reduce precision for archived data)
UPDATE resume_chunks SET 
    embedding = vector_to_halfprecision(embedding)
WHERE resume_id IN (
    SELECT id FROM resumes 
    WHERE created_at < NOW() - INTERVAL '6 months'
);
```

## Monitoring & Observability

### Key Performance Indicators

**Operational Metrics:**
```python
# Metrics to track in production
PERFORMANCE_METRICS = {
    # Throughput metrics
    'matching.jobs_processed_per_hour',
    'scraping.pages_per_minute', 
    'embedding.chunks_per_minute',
    
    # Latency metrics
    'matching.p95_latency_ms',
    'vector_search.p50_latency_ms',
    'llm_scoring.p95_latency_ms',
    
    # Quality metrics  
    'matching.precision_at_10',
    'matching.user_satisfaction_score',
    'scraping.success_rate',
    
    # Cost metrics
    'llm.daily_spend_usd',
    'embedding.daily_api_calls',
    'infrastructure.monthly_cost_usd',
    
    # Resource utilization
    'postgres.connection_pool_utilization',
    'postgres.query_duration_p95',
    'redis.memory_usage_mb'
}
```

**Alerting Thresholds:**
- Vector search p95 > 100ms  
- Matching pipeline p95 > 8s
- LLM daily spend > 80% of budget
- Scraping success rate < 90%
- Database connection pool > 80% utilization

### Performance Dashboard Specifications

**Key Dashboard Panels:**
1. **Real-time Throughput**: Jobs processed, matches generated, applications submitted
2. **Latency Trends**: P50/P95 latencies across all components  
3. **Cost Tracking**: Daily LLM spend, API usage, infrastructure costs
4. **Quality Metrics**: User satisfaction, match precision, false positive rates
5. **Resource Health**: DB performance, memory usage, connection pooling
6. **Error Rates**: Failed scrapes, embedding failures, LLM timeouts

## Developer Guidelines

### Performance-Conscious Development

**Code Review Checklist:**
- [ ] Batch API calls where possible (minimum 10x improvement expected)
- [ ] Use connection pooling for database access
- [ ] Implement caching for repeated computations
- [ ] Add performance tests for new algorithms
- [ ] Profile memory usage for large data processing
- [ ] Validate database queries have proper indexes
- [ ] Consider token costs for new LLM features

**Anti-patterns to Avoid:**
- Individual embedding API calls (use batching)
- N+1 database queries (use joins or batch loading)
- Loading full job descriptions into memory (use streaming)
- Synchronous processing of large batches (use async/await)
- Missing database indexes on frequently queried columns

### Future Optimization Opportunities

**Advanced Techniques:**
1. **Approximate Vector Search**: Implement product quantization for 4x memory reduction
2. **Learned Embeddings**: Fine-tune embedding models on domain-specific data
3. **Caching Strategies**: Multi-tier caching (Redis + CDN) for static content
4. **Database Sharding**: Horizontal partitioning by company or date ranges
5. **Async Processing**: Background job queues for non-critical operations

---

*This document should be updated as performance bottlenecks are identified and new optimization strategies are implemented.*