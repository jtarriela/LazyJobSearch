# Scalability Architecture Guide

This document outlines scaling strategies, architectural patterns, and capacity planning for LazyJobSearch as it grows from MVP to enterprise scale.

## Scaling Dimensions

LazyJobSearch faces scaling challenges across multiple dimensions:

- **Data Volume**: Job postings (10k → 10M+), resumes (100 → 100k+), embeddings (1M → 1B+ vectors)
- **User Concurrency**: Simultaneous matching requests, review sessions, auto-apply operations  
- **Processing Throughput**: Crawling rate, embedding generation, LLM API calls
- **Geographic Distribution**: Multi-region deployment for latency and compliance
- **Cost Efficiency**: Linear cost scaling vs exponential data growth

## Architecture Evolution Roadmap

### Phase 1: Monolithic Foundation (0-50k jobs, 0-500 users)

**Current Architecture:**
```
┌─────────────────────────────────────────────────┐
│                Single Region                     │
│                                                 │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐   │
│  │   CLI   │───▶│  App Server │───▶│ Postgres│   │
│  │  Users  │    │   (Python)  │    │+pgvector│   │
│  └─────────┘    └─────────────┘    └─────────┘   │
│                        │                         │
│                        ▼                         │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐   │
│  │  Redis  │◀───│  Background │───▶│  S3/    │   │
│  │ (Cache) │    │   Workers   │    │ MinIO   │   │
│  └─────────┘    └─────────────┘    └─────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Scaling Characteristics:**
- Single Postgres instance with pgvector
- Shared Redis for caching and job queues
- Monolithic Python application
- Simple horizontal scaling of workers

**Bottlenecks:**
- Database becomes read/write bottleneck at ~100k concurrent vectors
- Single Redis instance limits concurrent job processing
- LLM API rate limits affect user experience

### Phase 2: Service Decomposition (50k-500k jobs, 500-5k users)

**Service-Oriented Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
└─────────────────┬─────────────────┬─────────────────────────┘
                  │                 │
        ┌─────────▼─────────┐   ┌───▼─────────────────────┐
        │   API Gateway     │   │    Web Frontend        │
        │   (FastAPI)       │   │    (Next.js)          │
        └─────────┬─────────┘   └─────────────────────────┘
                  │
        ┌─────────▼──────────────────────────────────────────┐
        │              Service Mesh                          │
        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
        │  │ Matching │ │ Scraping │ │Embedding │ │ LLM    │ │
        │  │ Service  │ │ Service  │ │ Service  │ │Service │ │
        │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
        └─────────┬──────────────────────────────────────────┘
                  │
        ┌─────────▼─────────────────────────────────────────┐
        │              Data Layer                         │
        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
        │  │PostgreSQL│ │  Redis   │ │   S3     │ │ Vector│ │
        │  │(Primary) │ │ Cluster  │ │          │ │  DB   │ │
        │  └──────────┘ └──────────┘ └──────────┘ └──────┘ │
        └─────────────────────────────────────────────────┘
```

**Key Changes:**
- Decompose monolith into domain services
- Introduce dedicated vector database (Qdrant/Pinecone)
- Redis cluster for high-availability caching
- API gateway for service orchestration and rate limiting

**Service Definitions:**

**Matching Service:**
```python
class MatchingService:
    """Handles resume-job matching pipeline"""
    
    async def generate_matches(self, resume_id: str) -> List[Match]:
        # Stage 1: FTS prefilter 
        fts_candidates = await self.fts_search(resume_id)
        
        # Stage 2: Vector similarity (parallel)
        vector_tasks = [
            self.vector_search(resume_embedding, job_batch) 
            for job_batch in batch_split(fts_candidates, 1000)
        ]
        vector_results = await asyncio.gather(*vector_tasks)
        
        # Stage 3: LLM scoring (with circuit breaker)
        top_candidates = self.merge_and_rank(vector_results)[:20]
        return await self.llm_score_with_fallback(top_candidates)
```

**Embedding Service:**
```python
class EmbeddingService:
    """Centralized embedding generation with caching"""
    
    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'cohere': CohereProvider(),  # Fallback
        }
        self.cache = VectorCache(redis_cluster)
        
    async def embed_batch(self, texts: List[str]) -> List[Vector]:
        # Deduplication and cache lookup
        unique_texts, index_map = self.deduplicate(texts)
        cached, uncached = await self.cache.get_batch(unique_texts)
        
        # Batch embedding for uncached texts
        if uncached:
            embeddings = await self.providers['openai'].embed_batch(
                uncached, batch_size=100
            )
            await self.cache.set_batch(uncached, embeddings)
            
        return self.reconstruct_results(cached, embeddings, index_map)
```

### Phase 3: Multi-Region & Microservices (500k+ jobs, 5k+ users)

**Distributed Architecture:**
```
                            ┌─────────────────┐
                            │  Global DNS     │
                            │  (Route 53)     │
                            └─────────┬───────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
         ┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
         │   US-East-1    │  │   US-West-2     │  │   EU-West-1    │
         │                │  │                 │  │                │
         │ ┌─────────────┐│  │ ┌─────────────┐ │  │ ┌─────────────┐│
         │ │API Gateway  ││  │ │API Gateway  │ │  │ │API Gateway  ││
         │ └─────┬───────┘│  │ └─────┬───────┘ │  │ └─────┬───────┘│
         │       │        │  │       │         │  │       │        │
         │ ┌─────▼───────┐││  │ ┌─────▼───────┐ │  │ ┌─────▼───────┐│
         │ │Microservices│││  │ │Microservices│ │  │ │Microservices││
         │ │   Cluster   │││  │ │   Cluster   │ │  │ │   Cluster   ││
         │ └─────────────┘││  │ └─────────────┘ │  │ └─────────────┘│
         │                │  │                 │  │                │
         │ ┌─────────────┐││  │ ┌─────────────┐ │  │ ┌─────────────┐│
         │ │Primary DB   │││  │ │Read Replica │ │  │ │Read Replica ││
         │ │(Write)      │││  │ │             │ │  │ │             ││
         │ └─────────────┘││  │ └─────────────┘ │  │ └─────────────┘│
         └────────────────┘  └─────────────────┘  └────────────────┘
                 │                    │                    │
         ┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
         │Vector DB Shard │  │Vector DB Shard  │  │Vector DB Shard │
         │    (US-E1)     │  │    (US-W2)      │  │    (EU-W1)     │
         └────────────────┘  └─────────────────┘  └────────────────┘
```

**Data Sharding Strategy:**
```python
class ShardingStrategy:
    """Geographic and functional data partitioning"""
    
    def route_user_data(self, user_id: str) -> Region:
        """Route user data to nearest region"""
        user_region = self.get_user_region(user_id)
        return REGIONS[user_region]
    
    def route_job_data(self, company_location: str) -> Region:
        """Shard job data by company location"""
        if company_location in ['US', 'CA']:
            return 'us-east-1' if self.hash(company_location) % 2 == 0 else 'us-west-2'
        elif company_location in ['EU', 'UK']:
            return 'eu-west-1'
        else:
            return 'us-east-1'  # Default
            
    def route_vector_search(self, query_region: str) -> List[Region]:
        """Multi-region vector search with latency optimization"""
        primary = query_region
        fallback_regions = [r for r in REGIONS if r != primary]
        
        return [primary] + fallback_regions[:2]  # Search max 3 regions
```

## Database Scaling Strategies

### Postgres Scaling Evolution

**Phase 1: Single Instance Optimization**
```sql
-- Optimized configuration for high-throughput
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB'; 
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Vector-specific optimizations  
ALTER SYSTEM SET shared_preload_libraries = 'vector';
ALTER SYSTEM SET vector.max_build_memory = '2GB';
SELECT pg_reload_conf();
```

**Phase 2: Read Replicas + Connection Pooling**
```python
class DatabaseRouter:
    """Route queries based on read/write patterns"""
    
    def __init__(self):
        self.write_db = create_engine(WRITE_DATABASE_URL, pool_size=50)
        self.read_replicas = [
            create_engine(url, pool_size=30) 
            for url in READ_REPLICA_URLS
        ]
        
    def get_session(self, read_only=True):
        if read_only:
            replica = random.choice(self.read_replicas)
            return Session(replica)
        else:
            return Session(self.write_db)
            
    async def execute_read_query(self, query, params=None):
        """Load balance across read replicas"""
        replica = min(self.read_replicas, key=lambda r: r.pool.size())
        return await replica.execute(query, params)
```

**Phase 3: Horizontal Sharding**
```python
class ShardedDatabase:
    """Horizontal partitioning by entity type"""
    
    def __init__(self):
        self.user_shards = {
            'shard_0': DatabaseCluster('user-shard-0'),
            'shard_1': DatabaseCluster('user-shard-1'),
        }
        self.job_shards = {
            'us_jobs': DatabaseCluster('job-shard-us'),
            'eu_jobs': DatabaseCluster('job-shard-eu'),
        }
        
    def get_user_shard(self, user_id: str) -> str:
        return f'shard_{hash(user_id) % len(self.user_shards)}'
        
    def get_job_shard(self, company_location: str) -> str:
        if company_location.upper() in ['US', 'CA']:
            return 'us_jobs'
        else:
            return 'eu_jobs'
            
    async def cross_shard_query(self, query_func):
        """Execute query across all shards and merge results"""
        tasks = []
        for shard_name, shard in self.all_shards.items():
            tasks.append(query_func(shard))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self.merge_results(results)
```

### Vector Database Scaling

**Phase 1: pgvector Optimization**
```sql
-- Partition by embedding version
CREATE TABLE job_chunks_v1 (LIKE job_chunks INCLUDING ALL)
    PARTITION OF job_chunks FOR VALUES IN ('v1.0');
    
CREATE TABLE job_chunks_v2 (LIKE job_chunks INCLUDING ALL)  
    PARTITION OF job_chunks FOR VALUES IN ('v2.0');

-- Optimize index parameters per partition
CREATE INDEX CONCURRENTLY job_chunks_v1_embed_idx 
ON job_chunks_v1 USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 200);

CREATE INDEX CONCURRENTLY job_chunks_v2_embed_idx
ON job_chunks_v2 USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 500);  -- More lists for larger dataset
```

**Phase 2: Dedicated Vector Database**
```python
class HybridVectorStore:
    """Combination of pgvector and specialized vector DB"""
    
    def __init__(self):
        self.postgres = PostgresVectorStore()
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
        
    async def search(self, query_vector, limit=50):
        # Use pgvector for metadata filtering + vector search
        pg_results = await self.postgres.search_with_filters(
            query_vector, 
            filters={'scraped_at': {'gte': datetime.now() - timedelta(days=30)}},
            limit=limit*2
        )
        
        # Use Qdrant for pure vector similarity on larger corpus  
        qdrant_results = await self.qdrant.search(
            collection_name="job_embeddings",
            query_vector=query_vector,
            limit=limit*2,
            score_threshold=0.75
        )
        
        # Merge and deduplicate results
        return self.merge_results(pg_results, qdrant_results, limit)
```

**Phase 3: Distributed Vector Search**
```python  
class DistributedVectorSearch:
    """Shard vectors across multiple regions/clusters"""
    
    def __init__(self):
        self.clusters = {
            'us-east': VectorCluster('qdrant-us-east'),
            'us-west': VectorCluster('qdrant-us-west'), 
            'eu-west': VectorCluster('qdrant-eu-west'),
        }
        
    async def global_search(self, query_vector, limit=50):
        """Search across all clusters with latency-based ranking"""
        search_tasks = []
        
        for region, cluster in self.clusters.items():
            task = asyncio.create_task(
                cluster.search(query_vector, limit=limit*2)
            )
            search_tasks.append((region, task))
            
        # Wait for fastest responses first
        done, pending = await asyncio.wait(
            [task for _, task in search_tasks],
            timeout=0.5,  # 500ms timeout
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel slow requests
        for task in pending:
            task.cancel()
            
        # Merge results from completed searches
        results = []
        for task in done:
            try:
                region_results = await task
                results.extend(region_results)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                
        return sorted(results, key=lambda x: x.score, reverse=True)[:limit]
```

## Service Communication Patterns

### Event-Driven Architecture

**Domain Events:**
```python
class JobScrapedEvent:
    job_id: str
    company_id: str
    scraped_at: datetime
    content_hash: str

class ResumeUploadedEvent:
    resume_id: str
    user_id: str
    version: int
    content_hash: str

class MatchGeneratedEvent:
    match_id: str
    resume_id: str
    job_id: str
    score: float
    reasoning: str
```

**Event Bus Implementation:**
```python
class EventBus:
    """Reliable event distribution with ordering guarantees"""
    
    def __init__(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKERS,
            key_serializer=str.encode,
            value_serializer=lambda v: json.dumps(v).encode(),
            # Reliability settings
            acks='all',  # Wait for all replicas
            retries=5,
            batch_size=16384,
            linger_ms=10,  # Small batching delay
        )
        
    async def publish_event(self, topic: str, event: dict, partition_key: str = None):
        """Publish event with ordering guarantee"""
        try:
            future = self.kafka_producer.send(
                topic=topic,
                key=partition_key,  # Ensures ordering per key
                value=event
            )
            # Don't wait for acknowledgment (async)
            return await asyncio.wrap_future(future)
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            # Fallback to database outbox pattern
            await self.store_in_outbox(topic, event, partition_key)
            
class EventHandler:
    """Consumer with retry and dead letter queue"""
    
    async def handle_job_scraped(self, event: JobScrapedEvent):
        try:
            # Trigger embedding pipeline
            await self.embedding_service.process_job(event.job_id)
            
            # Update search index
            await self.search_service.index_job(event.job_id)
            
        except RetryableError as e:
            # Retry with exponential backoff
            raise  # Kafka will retry automatically
        except NonRetryableError as e:
            # Send to dead letter queue
            await self.dlq.send(event, error=str(e))
```

## Caching Strategies

### Multi-Layer Caching Architecture

```python
class CacheHierarchy:
    """Multi-tier caching with different eviction policies"""
    
    def __init__(self):
        # L1: Local in-memory cache (fastest)
        self.l1_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL
        
        # L2: Redis cluster (shared, persistent)  
        self.l2_cache = RedisCluster(
            startup_nodes=REDIS_NODES,
            decode_responses=True,
            skip_full_coverage_check=True
        )
        
        # L3: Object store (bulk data)
        self.l3_cache = S3Cache(bucket=CACHE_BUCKET)
        
    async def get(self, key: str) -> Any:
        # Check L1 first (microsecond latency)
        if key in self.l1_cache:
            return self.l1_cache[key]
            
        # Check L2 (millisecond latency)  
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value  # Populate L1
            return value
            
        # Check L3 for large objects (100ms+ latency)
        try:
            value = await self.l3_cache.get(key) 
            if value:
                # Selective population based on size
                if len(str(value)) < 10000:  # Small objects only
                    self.l1_cache[key] = value
                await self.l2_cache.setex(key, 3600, value)
                return value
        except Exception:
            pass  # L3 cache miss acceptable
            
        return None
        
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Write-through caching strategy"""
        # Always write to L2 (authoritative)
        await self.l2_cache.setex(key, ttl, value)
        
        # Write to L1 if small enough
        if len(str(value)) < 1000:
            self.l1_cache[key] = value
            
        # Write large objects to L3
        if len(str(value)) > 10000:
            await self.l3_cache.put(key, value, ttl=ttl*24)  # Longer TTL
```

### Cache Invalidation Strategy

```python
class CacheInvalidation:
    """Event-driven cache invalidation"""
    
    @event_handler('job_updated')
    async def invalidate_job_caches(self, event: JobUpdatedEvent):
        keys_to_invalidate = [
            f"job:{event.job_id}",
            f"job_chunks:{event.job_id}",
            f"company_jobs:{event.company_id}",
            f"search:*",  # Wildcard pattern
        ]
        
        await self.cache.delete_batch(keys_to_invalidate)
        
    @event_handler('resume_updated')  
    async def invalidate_resume_caches(self, event: ResumeUpdatedEvent):
        # Invalidate user-specific caches
        user_pattern = f"user:{event.user_id}:*"
        matching_pattern = f"matches:{event.resume_id}:*"
        
        await self.cache.delete_pattern(user_pattern)
        await self.cache.delete_pattern(matching_pattern)
```

## Capacity Planning & Resource Scaling

### Auto-Scaling Policies

**Horizontal Pod Autoscaler (HPA) Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: matching-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: matching-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: matching_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"  # 10 RPS per pod
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
      - type: Percent
        value: 50  # Max 50% scale down per cycle
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60   # 1 min warmup
      policies:
      - type: Percent
        value: 100  # Double capacity per cycle  
        periodSeconds: 60
```

**Database Auto-Scaling:**
```python
class DatabaseAutoScaler:
    """Automatic read replica scaling based on load"""
    
    def __init__(self):
        self.cloudwatch = CloudWatchClient()
        self.rds = RDSClient()
        self.min_replicas = 2
        self.max_replicas = 10
        
    async def check_and_scale(self):
        """Monitor metrics and scale read replicas"""
        metrics = await self.get_db_metrics()
        
        cpu_usage = metrics['CPUUtilization']
        read_latency = metrics['ReadLatency'] 
        active_connections = metrics['DatabaseConnections']
        
        current_replicas = await self.get_replica_count()
        
        # Scale up conditions
        if (cpu_usage > 80 or read_latency > 100 or active_connections > 150):
            if current_replicas < self.max_replicas:
                await self.add_read_replica()
                logger.info(f"Scaled up to {current_replicas + 1} replicas")
                
        # Scale down conditions  
        elif (cpu_usage < 50 and read_latency < 20 and active_connections < 50):
            if current_replicas > self.min_replicas:
                await self.remove_read_replica()
                logger.info(f"Scaled down to {current_replicas - 1} replicas")
```

### Cost Management at Scale

**Resource Optimization:**
```python
class CostOptimizer:
    """Automatic resource optimization to control costs"""
    
    async def optimize_vector_storage(self):
        """Compress old embeddings using quantization"""
        old_embeddings = await self.db.query("""
            SELECT id, embedding FROM resume_chunks 
            WHERE created_at < NOW() - INTERVAL '6 months'
            AND needs_reembedding = FALSE
        """)
        
        for chunk in old_embeddings:
            # Quantize to half precision (50% storage savings)
            quantized = self.quantize_embedding(chunk.embedding)
            await self.db.execute(
                "UPDATE resume_chunks SET embedding = %s WHERE id = %s",
                quantized, chunk.id
            )
            
    async def archive_cold_data(self):
        """Move old job data to cheaper storage tier"""
        old_jobs = await self.db.query("""
            SELECT id FROM jobs 
            WHERE scraped_at < NOW() - INTERVAL '1 year'
            AND id NOT IN (
                SELECT DISTINCT job_id FROM matches 
                WHERE created_at > NOW() - INTERVAL '6 months'
            )
        """)
        
        # Move to S3 Glacier
        for job in old_jobs:
            job_data = await self.export_job_data(job.id)
            await self.s3.put_object(
                Bucket=ARCHIVE_BUCKET,
                Key=f"archived_jobs/{job.id}.json.gz",
                Body=gzip.compress(json.dumps(job_data).encode()),
                StorageClass='GLACIER'
            )
            
            # Remove from active database
            await self.db.execute("DELETE FROM jobs WHERE id = %s", job.id)
```

## Monitoring & Observability at Scale

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Initialize tracing
tracer = trace.get_tracer("lazyjobsearch")

class TracedMatchingService:
    """Matching service with comprehensive tracing"""
    
    @tracer.start_as_current_span("generate_matches")
    async def generate_matches(self, resume_id: str) -> List[Match]:
        span = trace.get_current_span()
        span.set_attribute("resume_id", resume_id)
        
        with tracer.start_as_current_span("fts_prefilter") as fts_span:
            fts_candidates = await self.fts_search(resume_id)
            fts_span.set_attribute("candidates_found", len(fts_candidates))
            
        with tracer.start_as_current_span("vector_search") as vec_span:
            vector_candidates = await self.vector_search(fts_candidates)
            vec_span.set_attribute("vector_candidates", len(vector_candidates))
            
        with tracer.start_as_current_span("llm_scoring") as llm_span:
            matches = await self.llm_score(vector_candidates[:20])
            llm_span.set_attribute("final_matches", len(matches))
            
        span.set_attribute("total_matches", len(matches))
        return matches
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
MATCHES_GENERATED = Counter(
    'matches_generated_total',
    'Total matches generated',
    ['resume_version', 'user_tier']
)

MATCHING_LATENCY = Histogram(
    'matching_duration_seconds',
    'Time spent generating matches',
    ['stage'],  # fts, vector, llm
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

VECTOR_DB_SIZE = Gauge(
    'vector_database_vectors_total',
    'Total vectors in database',
    ['collection']
)

# Infrastructure metrics  
DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database', 'type']  # type: read/write
)

CACHE_HIT_RATE = Counter(
    'cache_operations_total',
    'Cache operations',
    ['layer', 'operation']  # layer: l1/l2/l3, operation: hit/miss
)
```

### Alerting Strategy

```yaml
# Alert definitions for production monitoring
groups:
- name: lazyjobsearch.performance  
  rules:
  - alert: HighMatchingLatency
    expr: histogram_quantile(0.95, matching_duration_seconds) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Matching service latency is high"
      
  - alert: VectorDatabaseDown
    expr: up{job="vector-database"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Vector database is unreachable"
      
  - alert: LLMBudgetExceeded  
    expr: llm_daily_spend_usd > llm_daily_budget_usd * 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "LLM spending approaching daily budget"
      
  - alert: DatabaseConnectionsHigh
    expr: database_connections_active / database_max_connections > 0.8
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Database connection pool nearly exhausted"
```

## Disaster Recovery & High Availability

### Multi-Region Failover

```python
class RegionFailoverManager:
    """Automatic failover between regions"""
    
    def __init__(self):
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        self.current_primary = 'us-east-1'
        self.health_checks = {}
        
    async def monitor_region_health(self):
        """Continuous health monitoring with automatic failover"""
        while True:
            for region in self.regions:
                health = await self.check_region_health(region)
                self.health_checks[region] = health
                
                if region == self.current_primary and not health.is_healthy:
                    await self.initiate_failover(region)
                    
            await asyncio.sleep(30)  # Check every 30s
            
    async def initiate_failover(self, failed_region: str):
        """Failover to healthy region"""
        healthy_regions = [
            r for r in self.regions 
            if r != failed_region and self.health_checks[r].is_healthy
        ]
        
        if not healthy_regions:
            logger.critical("No healthy regions available!")
            return
            
        new_primary = healthy_regions[0]  # Choose first healthy
        
        logger.warning(f"Failing over from {failed_region} to {new_primary}")
        
        # Update DNS to point to new region
        await self.update_dns_records(new_primary)
        
        # Update internal service discovery
        await self.update_service_registry(new_primary)
        
        # Notify monitoring systems
        await self.send_failover_alert(failed_region, new_primary)
        
        self.current_primary = new_primary
```

### Backup & Recovery Strategy

```python
class BackupManager:
    """Automated backup with point-in-time recovery"""
    
    async def create_full_backup(self):
        """Create consistent backup across all data stores"""
        timestamp = datetime.utcnow().isoformat()
        backup_id = f"backup_{timestamp}"
        
        # Create database backup
        db_backup = await self.backup_postgresql(backup_id)
        
        # Create vector database backup
        vector_backup = await self.backup_vector_store(backup_id)
        
        # Create object store backup
        s3_backup = await self.backup_object_storage(backup_id)
        
        # Store backup metadata
        backup_manifest = {
            'backup_id': backup_id,
            'timestamp': timestamp,
            'components': {
                'postgresql': db_backup,
                'vector_store': vector_backup, 
                'object_storage': s3_backup,
            },
            'size_bytes': sum(b['size'] for b in [db_backup, vector_backup, s3_backup]),
        }
        
        await self.store_backup_manifest(backup_manifest)
        
        # Verify backup integrity
        await self.verify_backup(backup_id)
        
        return backup_id
```

---

*This scalability guide should be updated as the system grows and new scaling bottlenecks are identified. Regular architecture reviews are recommended at each major scaling milestone.*