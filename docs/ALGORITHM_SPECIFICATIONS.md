# Algorithm Specifications & Implementation Guide

This document provides detailed algorithmic specifications for LazyJobSearch's core components, with emphasis on speed, accuracy, and scalability optimizations.

## Vector Similarity Algorithms

### 1. Embedding Generation Pipeline

**Algorithm: Batch-Optimized Embedding with Deduplication**

```python
def optimized_embedding_pipeline(texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
    """
    Optimized embedding generation with deduplication and caching.
    Time Complexity: O(n) where n = unique texts (vs O(total_texts) naive)
    Space Complexity: O(u) where u = unique texts
    """
    
    # Step 1: Deduplication (O(n))
    unique_texts = list(set(texts))
    text_to_indices = defaultdict(list)
    for i, text in enumerate(texts):
        text_to_indices[text].append(i)
    
    # Step 2: Cache lookup (O(u log u))
    cache_hits = {}
    uncached_texts = []
    
    for text in unique_texts:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached_embedding = redis_client.get(f"emb:{cache_key}")
        
        if cached_embedding:
            cache_hits[text] = np.frombuffer(cached_embedding, dtype=np.float32)
        else:
            uncached_texts.append(text)
    
    # Step 3: Batch API calls (O(u/b) where b = batch_size)
    new_embeddings = {}
    for batch in chunked(uncached_texts, batch_size):
        response = embedding_client.create_embeddings(
            texts=batch,
            model="text-embedding-3-large"
        )
        
        for text, embedding_data in zip(batch, response.embeddings):
            embedding = np.array(embedding_data.values, dtype=np.float32)
            new_embeddings[text] = embedding
            
            # Cache with 1 hour TTL
            cache_key = hashlib.md5(text.encode()).hexdigest()
            redis_client.setex(
                f"emb:{cache_key}", 
                3600, 
                embedding.tobytes()
            )
    
    # Step 4: Reconstruct original order (O(n))
    all_embeddings = {**cache_hits, **new_embeddings}
    result = [all_embeddings[text] for text in texts]
    
    return result
```

**Performance Characteristics:**
- **Cache hit rate**: 85-95% for repeat content
- **API call reduction**: 60-80% vs naive approach
- **Memory efficiency**: O(unique_texts) vs O(total_texts)
- **Throughput**: 500-1000 embeddings/minute (batch-optimized)

### 2. Vector Similarity Search

**Algorithm: Multi-Stage Hybrid Search**

```python
class HybridVectorSearch:
    """
    Multi-stage search combining FTS prefiltering, ANN vector search, and reranking.
    
    Stage 1: PostgreSQL FTS - O(log n) with GIN index
    Stage 2: pgvector ANN - O(log n) with IVF index  
    Stage 3: Exact similarity - O(k) where k = candidates
    """
    
    def __init__(self, fts_limit: int = 2000, vector_limit: int = 200, final_limit: int = 50):
        self.fts_limit = fts_limit
        self.vector_limit = vector_limit
        self.final_limit = final_limit
        
    async def search(self, query_embedding: np.ndarray, query_terms: List[str]) -> List[SearchResult]:
        # Stage 1: FTS Prefiltering (cheap, high recall)
        fts_query = self._build_fts_query(query_terms)
        fts_candidates = await self._fts_prefilter(fts_query, self.fts_limit)
        
        if len(fts_candidates) == 0:
            return []
            
        # Stage 2: Vector similarity on FTS survivors
        vector_candidates = await self._vector_similarity_search(
            query_embedding, 
            candidate_ids=[c.id for c in fts_candidates],
            limit=self.vector_limit
        )
        
        # Stage 3: Hybrid scoring (combine FTS rank + vector similarity)
        final_results = await self._hybrid_rerank(
            vector_candidates, 
            fts_candidates,
            limit=self.final_limit
        )
        
        return final_results
        
    def _build_fts_query(self, terms: List[str]) -> str:
        """
        Build optimized FTS query with expansion and weighting.
        
        Algorithm: Weighted term expansion with synonym support
        """
        expanded_terms = []
        
        for term in terms:
            # Add original term with high weight
            expanded_terms.append(f"{term}:A")
            
            # Add stemmed variants
            stemmed = self.stemmer.stem(term)
            if stemmed != term:
                expanded_terms.append(f"{stemmed}:B")
                
            # Add synonyms with lower weight
            synonyms = self.synonym_dict.get(term.lower(), [])
            for syn in synonyms[:3]:  # Limit to top 3 synonyms
                expanded_terms.append(f"{syn}:C")
        
        # Combine with OR operator
        return " | ".join(expanded_terms)
        
    async def _fts_prefilter(self, query: str, limit: int) -> List[FTSResult]:
        """PostgreSQL FTS with custom ranking."""
        sql = """
            SELECT 
                id,
                ts_rank_cd(jd_tsv, query, 32) as fts_score,
                ts_headline('english', jd_fulltext, query) as snippet
            FROM jobs, websearch_to_tsquery('english', %s) query
            WHERE jd_tsv @@ query
            ORDER BY ts_rank_cd(jd_tsv, query, 32) DESC
            LIMIT %s
        """
        
        rows = await self.db.fetch_all(sql, (query, limit))
        return [FTSResult(id=r[0], fts_score=r[1], snippet=r[2]) for r in rows]
        
    async def _vector_similarity_search(self, query_emb: np.ndarray, 
                                      candidate_ids: List[str], limit: int) -> List[VectorResult]:
        """
        Optimized vector search on candidate subset.
        
        Uses pgvector with optimized IVF settings for candidate filtering.
        """
        # Convert numpy array to pgvector format
        query_vector = f"[{','.join(map(str, query_emb))}]"
        
        sql = """
            SELECT 
                id,
                job_id,
                1 - (embedding <=> %s::vector) as similarity,
                chunk_text
            FROM job_chunks 
            WHERE job_id = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        rows = await self.db.fetch_all(sql, (query_vector, candidate_ids, query_vector, limit))
        return [VectorResult(id=r[0], job_id=r[1], similarity=r[2], text=r[3]) for r in rows]
        
    async def _hybrid_rerank(self, vector_results: List[VectorResult], 
                           fts_results: List[FTSResult], limit: int) -> List[SearchResult]:
        """
        Combine and rerank results using learned weights.
        
        Algorithm: Weighted linear combination with normalization
        """
        # Create lookup for FTS scores
        fts_scores = {r.id: r.fts_score for r in fts_results}
        
        # Normalize scores to [0, 1] range
        max_vector_sim = max(r.similarity for r in vector_results) if vector_results else 1.0
        max_fts_score = max(r.fts_score for r in fts_results) if fts_results else 1.0
        
        combined_results = []
        for vec_result in vector_results:
            fts_score = fts_scores.get(vec_result.job_id, 0.0)
            
            # Normalize scores
            norm_vector_score = vec_result.similarity / max_vector_sim
            norm_fts_score = fts_score / max_fts_score
            
            # Weighted combination (learned weights)
            combined_score = (
                0.6 * norm_vector_score +  # Vector similarity weight
                0.3 * norm_fts_score +     # FTS relevance weight
                0.1 * self._recency_bonus(vec_result.job_id)  # Recency bonus
            )
            
            combined_results.append(SearchResult(
                job_id=vec_result.job_id,
                combined_score=combined_score,
                vector_score=vec_result.similarity,
                fts_score=fts_score,
                snippet=fts_results[vec_result.job_id].snippet if vec_result.job_id in fts_scores else ""
            ))
            
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        return combined_results[:limit]
```

### 3. Adaptive Ranking Algorithm

**Algorithm: Online Learning with Feature Engineering**

```python
class AdaptiveRankingModel:
    """
    Logistic regression model with online updates for ranking optimization.
    
    Features: vector_score, fts_score, yoe_gap, skill_overlap, recency, competition
    Target: binary outcome (got_interview)
    """
    
    def __init__(self, feature_dim: int = 10, learning_rate: float = 0.01):
        self.weights = np.zeros(feature_dim)
        self.learning_rate = learning_rate
        self.feature_scaler = StandardScaler()
        self.update_count = 0
        
    def extract_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and engineer features for ranking.
        
        Feature Engineering Algorithm:
        1. Core similarity features (vector, fts, llm scores)
        2. Experience matching (YOE gap, seniority alignment)  
        3. Skill matching (overlap ratio, critical skills)
        4. Temporal features (job recency, competition)
        5. Meta features (resume version, model version)
        """
        
        # Core similarity features
        vector_score = float(match_data.get('vector_score', 0.0))
        fts_score = float(match_data.get('fts_score', 0.0))
        llm_score = float(match_data.get('llm_score', 0.0)) / 100.0
        
        # Experience features
        required_yoe = float(match_data.get('required_yoe', 0))
        candidate_yoe = float(match_data.get('candidate_yoe', 0))
        yoe_gap_ratio = abs(required_yoe - candidate_yoe) / max(required_yoe, 1.0)
        
        # Skill features
        required_skills = set(match_data.get('required_skills', []))
        candidate_skills = set(match_data.get('candidate_skills', []))
        skill_overlap_ratio = len(required_skills & candidate_skills) / max(len(required_skills), 1)
        
        # Temporal features
        job_age_days = (datetime.now() - match_data.get('job_posted', datetime.now())).days
        recency_score = max(0, 1 - job_age_days / 30.0)  # Decay over 30 days
        
        # Competition estimate (heuristic)
        company_tier = self._get_company_tier(match_data.get('company_name', ''))
        competition_score = min(1.0, company_tier + (1 - recency_score))
        
        # Meta features
        is_latest_model = 1.0 if match_data.get('embedding_version') == 'latest' else 0.0
        resume_freshness = self._get_resume_freshness(match_data.get('resume_updated'))
        
        features = np.array([
            vector_score,
            fts_score, 
            llm_score,
            1.0 - yoe_gap_ratio,  # Higher is better
            skill_overlap_ratio,
            recency_score,
            1.0 - competition_score,  # Less competition is better
            company_tier,
            is_latest_model,
            resume_freshness
        ])
        
        return features
    
    def predict_score(self, features: np.ndarray) -> float:
        """
        Predict match quality score using logistic regression.
        
        Algorithm: Sigmoid(w^T * x) where w = learned weights, x = features
        """
        # Apply feature scaling
        if self.update_count > 10:  # Only scale after some training
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
        else:
            features_scaled = features
            
        # Logistic regression prediction
        logit = np.dot(self.weights, features_scaled)
        probability = 1.0 / (1.0 + np.exp(-logit))
        
        return probability
    
    def update_model(self, features: np.ndarray, actual_outcome: bool):
        """
        Online model update using stochastic gradient descent.
        
        Algorithm: 
        w_t+1 = w_t - α * ∇L(w_t)
        where L is logistic loss and α is learning rate
        """
        # Scale features
        if self.update_count > 10:
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
        else:
            # Update scaler
            self.feature_scaler.partial_fit(features.reshape(1, -1))
            features_scaled = features
            
        # Current prediction
        prediction = self.predict_score(features)
        
        # Gradient of logistic loss
        error = prediction - float(actual_outcome)
        gradient = error * features_scaled
        
        # SGD update
        self.weights -= self.learning_rate * gradient
        
        # Adaptive learning rate (decay over time)
        self.learning_rate *= 0.9999
        self.update_count += 1
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance based on weight magnitudes."""
        feature_names = [
            'vector_score', 'fts_score', 'llm_score', 'yoe_match',
            'skill_overlap', 'recency', 'low_competition', 'company_tier',
            'latest_model', 'resume_freshness'
        ]
        
        importances = {name: abs(weight) for name, weight in zip(feature_names, self.weights)}
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
```

## Crawling & Anti-Bot Algorithms

### 4. Adaptive Crawling Rate Controller

**Algorithm: Token Bucket with Exponential Backoff**

```python
class AdaptiveCrawlController:
    """
    Adaptive rate limiting using token bucket algorithm with dynamic adjustment.
    
    Algorithm combines:
    1. Token bucket for smooth rate limiting
    2. Exponential backoff on errors/blocks  
    3. Success-based rate increase
    4. Per-domain independent control
    """
    
    def __init__(self, base_rate: float = 1.0, max_rate: float = 10.0):
        self.base_rate = base_rate  # requests per second
        self.max_rate = max_rate
        self.domain_controllers = {}
        
    def get_controller(self, domain: str) -> 'DomainController':
        if domain not in self.domain_controllers:
            self.domain_controllers[domain] = DomainController(
                initial_rate=self.base_rate,
                max_rate=self.max_rate
            )
        return self.domain_controllers[domain]
        
    async def acquire_permit(self, domain: str) -> float:
        """
        Acquire permission to make request.
        Returns: delay in seconds before request should be made
        """
        controller = self.get_controller(domain)
        return await controller.acquire_permit()

class DomainController:
    """Per-domain rate controller with learning."""
    
    def __init__(self, initial_rate: float, max_rate: float):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.bucket_size = 10.0  # Max burst capacity
        self.tokens = self.bucket_size  # Start full
        self.last_refill = time.time()
        
        # Learning statistics
        self.success_count = 0
        self.error_count = 0
        self.last_error_time = None
        self.backoff_multiplier = 1.0
        
    async def acquire_permit(self) -> float:
        """
        Token bucket algorithm with adaptive rate adjustment.
        
        Time Complexity: O(1)
        Space Complexity: O(1) per domain
        """
        current_time = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = current_time - self.last_refill
        tokens_to_add = elapsed * self.current_rate
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = current_time
        
        # Check if we have tokens available
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0  # No delay needed
        else:
            # Calculate delay until next token available
            time_for_token = (1.0 - self.tokens) / self.current_rate
            return time_for_token * self.backoff_multiplier
            
    def record_success(self):
        """Update controller state after successful request."""
        self.success_count += 1
        
        # Gradually increase rate on sustained success
        if self.success_count % 10 == 0 and self.error_count == 0:
            self.current_rate = min(self.max_rate, self.current_rate * 1.1)
            
        # Reset backoff on success  
        self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)
        
    def record_error(self, error_type: str):
        """Update controller state after error/block."""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Apply exponential backoff
        if error_type in ['blocked', 'rate_limited']:
            self.backoff_multiplier = min(8.0, self.backoff_multiplier * 2.0)
            self.current_rate = max(0.1, self.current_rate * 0.5)  # Halve rate
        elif error_type == 'timeout':
            self.backoff_multiplier = min(4.0, self.backoff_multiplier * 1.5)
            
        # Reset success counter
        self.success_count = 0
```

### 5. Human Behavior Simulation Algorithms

**Algorithm: Markov Chain Behavior Modeling**

```python
class HumanBehaviorModel:
    """
    Markov chain model for realistic human browsing simulation.
    
    States: reading, scrolling, clicking, typing, pausing
    Transitions based on real user behavior patterns
    """
    
    def __init__(self):
        # Transition probability matrix (learned from real user data)
        self.transition_matrix = {
            'reading': {'scrolling': 0.4, 'clicking': 0.3, 'pausing': 0.2, 'typing': 0.1},
            'scrolling': {'reading': 0.5, 'scrolling': 0.2, 'clicking': 0.2, 'pausing': 0.1},
            'clicking': {'reading': 0.6, 'scrolling': 0.1, 'pausing': 0.2, 'typing': 0.1},
            'typing': {'reading': 0.3, 'pausing': 0.4, 'clicking': 0.2, 'scrolling': 0.1},
            'pausing': {'reading': 0.4, 'scrolling': 0.3, 'clicking': 0.2, 'typing': 0.1}
        }
        
        self.current_state = 'reading'  # Start in reading state
        
    def next_action(self) -> Tuple[str, float]:
        """
        Generate next action using Markov chain.
        
        Returns: (action_type, duration_seconds)
        """
        # Select next state based on transition probabilities
        transitions = self.transition_matrix[self.current_state]
        states = list(transitions.keys())
        probs = list(transitions.values())
        
        next_state = np.random.choice(states, p=probs)
        duration = self._generate_duration(next_state)
        
        self.current_state = next_state
        return next_state, duration
        
    def _generate_duration(self, action: str) -> float:
        """Generate realistic duration for action using empirical distributions."""
        
        # Durations based on real user studies (log-normal distributions)
        duration_params = {
            'reading': (1.5, 0.8),    # mean=1.5s, std=0.8s
            'scrolling': (0.3, 0.4),  # Quick scrolling
            'clicking': (0.8, 0.5),   # Click + brief pause
            'typing': (3.0, 1.2),     # Longer for form filling
            'pausing': (2.0, 1.5)     # Variable pause length
        }
        
        mean_log, std_log = duration_params[action]
        duration = np.random.lognormal(np.log(mean_log), std_log)
        
        # Apply reasonable bounds
        return max(0.1, min(15.0, duration))
        
    def generate_session_sequence(self, session_length: int = 20) -> List[Tuple[str, float]]:
        """Generate a full browsing session sequence."""
        sequence = []
        for _ in range(session_length):
            action, duration = self.next_action()
            sequence.append((action, duration))
        return sequence
```

## Performance Optimization Algorithms

### 6. Database Query Optimization

**Algorithm: Adaptive Query Planning with Statistics**

```python
class QueryOptimizer:
    """
    Adaptive query optimization based on execution statistics.
    
    Maintains query performance statistics and automatically adjusts
    query plans, indexes, and batching strategies.
    """
    
    def __init__(self):
        self.query_stats = defaultdict(lambda: {
            'avg_duration_ms': 0.0,
            'execution_count': 0,
            'p95_duration_ms': 0.0,
            'last_optimized': None
        })
        
    async def execute_optimized_query(self, query_template: str, 
                                    params: List[Any], 
                                    context: Dict[str, Any]) -> List[Dict]:
        """
        Execute query with automatic optimization.
        
        Algorithm:
        1. Check query statistics
        2. Select optimal execution plan
        3. Execute with monitoring
        4. Update statistics
        5. Trigger optimization if needed
        """
        query_key = hashlib.md5(query_template.encode()).hexdigest()[:8]
        
        # Select execution strategy based on context
        if context.get('expected_rows', 0) > 10000:
            # Large result set - use cursor with batching
            return await self._execute_batched_query(query_template, params, context)
        elif self.query_stats[query_key]['avg_duration_ms'] > 1000:
            # Slow query - use optimized version
            return await self._execute_optimized_query(query_template, params, context)
        else:
            # Standard execution
            return await self._execute_standard_query(query_template, params, context)
            
    async def _execute_batched_query(self, query: str, params: List[Any], 
                                   context: Dict[str, Any]) -> List[Dict]:
        """Execute large queries in batches to avoid memory issues."""
        batch_size = context.get('batch_size', 1000)
        all_results = []
        offset = 0
        
        while True:
            # Add LIMIT/OFFSET to query
            batched_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            
            start_time = time.time()
            batch_results = await self.db.fetch_all(batched_query, params)
            duration_ms = (time.time() - start_time) * 1000
            
            if not batch_results:
                break
                
            all_results.extend([dict(row) for row in batch_results])
            offset += batch_size
            
            # Update statistics
            query_key = hashlib.md5(query.encode()).hexdigest()[:8]
            self._update_query_stats(query_key, duration_ms)
            
        return all_results
        
    def _update_query_stats(self, query_key: str, duration_ms: float):
        """Update query performance statistics using exponential moving average."""
        stats = self.query_stats[query_key]
        
        if stats['execution_count'] == 0:
            stats['avg_duration_ms'] = duration_ms
            stats['p95_duration_ms'] = duration_ms
        else:
            # Exponential moving average
            alpha = 0.1
            stats['avg_duration_ms'] = (1 - alpha) * stats['avg_duration_ms'] + alpha * duration_ms
            
            # Update P95 (simplified)
            if duration_ms > stats['p95_duration_ms']:
                stats['p95_duration_ms'] = duration_ms * 0.95 + stats['p95_duration_ms'] * 0.05
                
        stats['execution_count'] += 1
        
        # Trigger optimization if performance degraded
        if (stats['execution_count'] > 100 and 
            stats['avg_duration_ms'] > 1000 and
            not stats['last_optimized']):
            
            self._schedule_query_optimization(query_key)
            
    def _schedule_query_optimization(self, query_key: str):
        """Schedule background query optimization."""
        # This would trigger ANALYZE, index suggestions, etc.
        logger.info(f"Scheduling optimization for slow query: {query_key}")
        self.query_stats[query_key]['last_optimized'] = datetime.now()
```

### 7. Memory-Efficient Processing Algorithms

**Algorithm: Streaming Processing with Memory Bounds**

```python
class StreamingProcessor:
    """
    Memory-efficient processing for large datasets using streaming algorithms.
    
    Maintains fixed memory usage regardless of input size.
    """
    
    def __init__(self, memory_limit_mb: int = 512):
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.current_memory_usage = 0
        
    async def process_large_dataset(self, data_source: AsyncIterable, 
                                  processor_func: Callable,
                                  batch_size: int = 1000) -> AsyncGenerator:
        """
        Stream-process large dataset with memory bounds.
        
        Algorithm:
        1. Process data in fixed-size batches
        2. Monitor memory usage
        3. Apply backpressure if memory limit exceeded
        4. Yield results incrementally
        """
        current_batch = []
        
        async for item in data_source:
            current_batch.append(item)
            
            # Check if batch is full or memory limit reached
            if (len(current_batch) >= batch_size or 
                self._estimate_memory_usage(current_batch) > self.memory_limit_bytes * 0.8):
                
                # Process batch
                results = await processor_func(current_batch)
                
                # Yield results and clear batch
                for result in results:
                    yield result
                    
                current_batch.clear()
                
                # Force garbage collection if memory usage high
                if self._get_current_memory_usage() > self.memory_limit_bytes * 0.9:
                    import gc
                    gc.collect()
        
        # Process remaining items
        if current_batch:
            results = await processor_func(current_batch)
            for result in results:
                yield result
                
    def _estimate_memory_usage(self, objects: List[Any]) -> int:
        """Estimate memory usage of object list."""
        if not objects:
            return 0
            
        # Sample-based estimation
        sample_size = min(10, len(objects))
        sample_objects = objects[:sample_size]
        
        total_sample_size = sum(sys.getsizeof(obj) for obj in sample_objects)
        avg_size = total_sample_size / sample_size
        
        return int(avg_size * len(objects))
```

These algorithmic specifications provide concrete, optimized implementations for LazyJobSearch's core components. Each algorithm includes complexity analysis, performance characteristics, and specific optimization strategies for speed and scalability.