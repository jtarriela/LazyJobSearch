# Performance Optimization Examples

This document provides practical examples of implementing the performance optimization strategies described in `PERFORMANCE_OPTIMIZATION.md`.

## Example 1: Optimized Matching Pipeline

```python
from libs.matching.feedback import AdvancedFeatureEngineer, FeedbackTrainer, FeatureWeightModel
from libs.embed.versioning import EmbeddingVersionManager, ProgressiveMigrationManager
import asyncio
import time

class OptimizedMatchingService:
    """Production-ready matching service with all optimizations applied."""
    
    def __init__(self, session, embedding_service, llm_service):
        self.session = session
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        
        # Initialize optimization components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.version_manager = EmbeddingVersionManager(session, embedding_service)
        self.adaptive_model = None  # Loaded lazily
        
    async def generate_matches(self, resume_id: str, limit: int = 50) -> List[Dict]:
        """
        Optimized matching pipeline with all performance enhancements.
        
        Pipeline:
        1. Load resume with caching
        2. FTS prefiltering (O(log n))
        3. Vector similarity on filtered set (O(k log m))
        4. Adaptive ranking with learned weights
        5. LLM scoring with batch optimization
        """
        start_time = time.time()
        
        # Step 1: Load resume with embedding (cached)
        resume_data = await self._load_resume_cached(resume_id)
        resume_embedding = resume_data['embedding']
        resume_skills = resume_data['skills']
        
        # Step 2: FTS Prefiltering (fast, high recall)
        fts_start = time.time()
        fts_query = self._build_optimized_fts_query(resume_skills)
        fts_candidates = await self.session.execute("""
            SELECT job_id, title, company_name, required_yoe, required_skills,
                   ts_rank_cd(jd_tsv, query, 32) as fts_score
            FROM jobs j
            JOIN companies c ON j.company_id = c.id,
                 websearch_to_tsquery('english', %s) query
            WHERE j.jd_tsv @@ query
            ORDER BY ts_rank_cd(jd_tsv, query, 32) DESC
            LIMIT 2000
        """, (fts_query,))
        
        fts_duration = time.time() - fts_start
        print(f"FTS prefilter: {len(fts_candidates)} candidates in {fts_duration:.2f}s")
        
        if not fts_candidates:
            return []
            
        # Step 3: Vector similarity on FTS survivors (batch optimized)
        vector_start = time.time()
        job_ids = [row['job_id'] for row in fts_candidates]
        
        # Batch vector similarity calculation
        vector_candidates = await self._batch_vector_similarity(
            resume_embedding, job_ids, limit=min(200, len(job_ids))
        )
        
        vector_duration = time.time() - vector_start
        print(f"Vector similarity: {len(vector_candidates)} candidates in {vector_duration:.2f}s")
        
        # Step 4: Adaptive ranking (if model available)
        ranking_start = time.time()
        if not self.adaptive_model:
            self.adaptive_model = await self._load_adaptive_model()
            
        if self.adaptive_model:
            scored_candidates = await self._apply_adaptive_ranking(
                vector_candidates, fts_candidates, resume_data
            )
        else:
            # Fallback to heuristic scoring
            scored_candidates = self._apply_heuristic_ranking(vector_candidates, fts_candidates)
            
        ranking_duration = time.time() - ranking_start
        print(f"Adaptive ranking: {ranking_duration:.2f}s")
        
        # Step 5: LLM scoring for top candidates (batch optimized)
        llm_start = time.time()
        top_candidates = scored_candidates[:20]  # Only LLM score top 20
        
        final_matches = await self._batch_llm_scoring(
            top_candidates, resume_data, limit=limit
        )
        
        llm_duration = time.time() - llm_start
        total_duration = time.time() - start_time
        
        print(f"LLM scoring: {len(final_matches)} matches in {llm_duration:.2f}s")
        print(f"Total pipeline: {total_duration:.2f}s")
        
        return final_matches
        
    async def _load_resume_cached(self, resume_id: str) -> Dict:
        """Load resume with Redis caching."""
        cache_key = f"resume:{resume_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
            
        # Load from database
        resume = await self.session.execute("""
            SELECT r.id, r.fulltext, r.skills_csv, r.yoe_adjusted,
                   array_agg(rc.embedding) as chunk_embeddings
            FROM resumes r
            LEFT JOIN resume_chunks rc ON r.id = rc.resume_id  
            WHERE r.id = %s
            GROUP BY r.id
        """, (resume_id,))
        
        resume_data = dict(resume.fetchone())
        
        # Compute aggregated embedding (mean of chunks)
        chunk_embeddings = resume_data['chunk_embeddings']
        if chunk_embeddings and chunk_embeddings[0]:
            resume_embedding = np.mean(chunk_embeddings, axis=0)
            resume_data['embedding'] = resume_embedding.tolist()
        else:
            resume_data['embedding'] = [0.0] * 3072  # Fallback
            
        resume_data['skills'] = resume_data['skills_csv'].split(',') if resume_data['skills_csv'] else []
        
        # Cache for 1 hour
        await self.redis.setex(cache_key, 3600, json.dumps(resume_data, default=str))
        return resume_data
        
    async def _batch_vector_similarity(self, resume_embedding: List[float], 
                                     job_ids: List[str], limit: int) -> List[Dict]:
        """Optimized batch vector similarity calculation."""
        
        # Convert to pgvector format
        embedding_str = f"[{','.join(map(str, resume_embedding))}]"
        
        # Batch query with optimized index usage
        results = await self.session.execute("""
            WITH job_similarities AS (
                SELECT 
                    jc.job_id,
                    1 - (jc.embedding <=> %s::vector) as similarity,
                    row_number() OVER (PARTITION BY jc.job_id ORDER BY 1 - (jc.embedding <=> %s::vector) DESC) as rn
                FROM job_chunks jc
                WHERE jc.job_id = ANY(%s)
                AND jc.embedding IS NOT NULL
            )
            SELECT job_id, MAX(similarity) as max_similarity
            FROM job_similarities  
            WHERE rn <= 3  -- Top 3 chunks per job
            GROUP BY job_id
            ORDER BY MAX(similarity) DESC
            LIMIT %s
        """, (embedding_str, embedding_str, job_ids, limit))
        
        return [{'job_id': row[0], 'vector_score': row[1]} for row in results.fetchall()]
        
    async def _apply_adaptive_ranking(self, vector_candidates: List[Dict], 
                                    fts_candidates: List[Dict], 
                                    resume_data: Dict) -> List[Dict]:
        """Apply learned ranking model."""
        
        # Create lookup for FTS data
        fts_lookup = {row['job_id']: row for row in fts_candidates}
        
        ranked_candidates = []
        for candidate in vector_candidates:
            job_id = candidate['job_id']
            fts_data = fts_lookup.get(job_id, {})
            
            # Engineer features
            match_data = {
                'vector_score': candidate['vector_score'],
                'fts_score': fts_data.get('fts_score', 0.0),
                'llm_score': 50,  # Default before LLM scoring
                'required_yoe': fts_data.get('required_yoe', 0),
                'candidate_yoe': resume_data.get('yoe_adjusted', 0),
                'required_skills': fts_data.get('required_skills', '').split(','),
                'candidate_skills': resume_data.get('skills', []),
                'job_title': fts_data.get('title', ''),
                'company_name': fts_data.get('company_name', ''),
            }
            
            features = self.feature_engineer.engineer_features(match_data)
            feature_dict = {k: v for k, v in asdict(features).items() if isinstance(v, (int, float))}
            
            # Get adaptive score
            adaptive_score = self.adaptive_model.score(feature_dict)
            
            ranked_candidates.append({
                **candidate,
                'adaptive_score': adaptive_score,
                'match_data': match_data
            })
            
        # Sort by adaptive score
        return sorted(ranked_candidates, key=lambda x: x['adaptive_score'], reverse=True)
        
    async def _batch_llm_scoring(self, candidates: List[Dict], resume_data: Dict, 
                               limit: int) -> List[Dict]:
        """Batch LLM scoring with cost optimization."""
        
        if not candidates:
            return []
            
        # Prepare batch prompt
        batch_contexts = []
        for candidate in candidates:
            context = {
                'job_title': candidate['match_data']['job_title'],
                'company': candidate['match_data']['company_name'],
                'required_skills': candidate['match_data']['required_skills'][:10],  # Limit for token cost
                'candidate_skills': resume_data['skills'][:15],
                'yoe_required': candidate['match_data']['required_yoe'],
                'yoe_candidate': resume_data['yoe_adjusted']
            }
            batch_contexts.append(context)
            
        # Batch LLM call
        try:
            llm_results = await self.llm_service.batch_score_matches(
                batch_contexts, 
                model='gpt-4o-mini'  # Cost-optimized model
            )
            
            # Combine results
            final_matches = []
            for candidate, llm_result in zip(candidates, llm_results):
                final_matches.append({
                    'job_id': candidate['job_id'],
                    'vector_score': candidate['vector_score'],
                    'adaptive_score': candidate['adaptive_score'],
                    'llm_score': llm_result.get('score', 50),
                    'reasoning': llm_result.get('reasoning', ''),
                    'action': llm_result.get('recommendation', 'review')
                })
                
            return final_matches[:limit]
            
        except Exception as e:
            print(f"LLM scoring failed: {e}")
            # Fallback to adaptive scores only
            return [{
                'job_id': c['job_id'],
                'vector_score': c['vector_score'],
                'adaptive_score': c['adaptive_score'],
                'llm_score': int(c['adaptive_score'] * 100),
                'reasoning': 'Adaptive model prediction',
                'action': 'review'
            } for c in candidates[:limit]]


# Usage Example
async def main():
    """Example of using the optimized matching service."""
    
    # Initialize service
    matching_service = OptimizedMatchingService(session, embedding_service, llm_service)
    
    # Generate matches for a user
    matches = await matching_service.generate_matches(
        resume_id="550e8400-e29b-41d4-a716-446655440000",
        limit=25
    )
    
    print(f"Generated {len(matches)} optimized matches")
    
    # Display top matches
    for i, match in enumerate(matches[:5], 1):
        print(f"{i}. Job {match['job_id']}: "
              f"Vector={match['vector_score']:.3f}, "
              f"Adaptive={match['adaptive_score']:.3f}, "
              f"LLM={match['llm_score']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 2: Embedding Migration with Performance Monitoring

```python
async def perform_embedding_migration():
    """Example of progressive embedding migration with monitoring."""
    
    from libs.embed.versioning import EmbeddingVersionManager, ProgressiveMigrationManager
    
    # Initialize managers
    version_manager = EmbeddingVersionManager(session, embedding_service)
    migration_manager = ProgressiveMigrationManager(
        session, embedding_service,
        batch_size=500,
        max_concurrent_batches=3,
        cost_limit_usd=50.0
    )
    
    # Create new embedding version
    new_version = await version_manager.create_new_version(
        model_name="text-embedding-3-large",
        dimensions=3072
    )
    
    print(f"Created new version: {new_version.version_id}")
    print(f"Compatibility score: {new_version.compatibility_score:.3f}")
    
    # Mark old versions for re-embedding
    old_versions = ['v1.0', 'v1.1']
    marked_count = await version_manager.mark_legacy_for_reembedding(old_versions)
    print(f"Marked {marked_count} embeddings for re-embedding")
    
    # Start migration
    migration_id = await migration_manager.start_migration('v1.0', new_version.version_id)
    print(f"Started migration: {migration_id}")
    
    # Monitor progress
    while True:
        status = await migration_manager.get_migration_status(migration_id)
        if not status or status['status'] in ['completed', 'failed']:
            break
            
        print(f"Progress: {status['completed_items']}/{status['total_items']} "
              f"({status['completed_items']/status['total_items']*100:.1f}%)")
        
        await asyncio.sleep(30)  # Check every 30 seconds
        
    print("Migration completed!")

# Run the migration example
asyncio.run(perform_embedding_migration())
```

## Example 3: Anti-Bot Session with Adaptive Behavior

```python
from libs.scraper.anti_bot import (
    ProxyPool, ProxyConfig, FingerprintGenerator, 
    ScrapeSessionManager, HumanBehaviorSimulator
)

async def scrape_with_anti_bot_measures():
    """Example of scraping with full anti-bot protection."""
    
    # Configure proxy pool
    proxy_configs = [
        ProxyConfig("proxy1.example.com", 8080, success_rate=0.95),
        ProxyConfig("proxy2.example.com", 8080, success_rate=0.88),
        ProxyConfig("proxy3.example.com", 8080, success_rate=0.92),
    ]
    proxy_pool = ProxyPool(proxy_configs)
    
    # Initialize components
    fp_generator = FingerprintGenerator()
    session_manager = ScrapeSessionManager(proxy_pool, fp_generator)
    behavior_sim = HumanBehaviorSimulator()
    
    # Start scraping session
    session = session_manager.start(domain="jobs.example.com")
    
    print(f"Started session with proxy: {session.proxy.host if session.proxy else 'none'}")
    print(f"Fingerprint: {session.profile.user_agent}")
    print(f"Viewport: {session.profile.viewport}")
    
    try:
        # Simulate realistic browsing behavior
        pages_scraped = 0
        for page_num in range(10):
            # Human-like delay before each page
            delay = behavior_sim.sleep_interval(base_duration=2.0)
            await asyncio.sleep(delay)
            
            # Simulate page request
            start_time = time.time()
            try:
                # Your actual scraping logic would go here
                await simulate_page_scrape(f"https://jobs.example.com/page/{page_num}")
                
                response_time = time.time() - start_time
                session_manager.record_page_result(
                    session.id, "jobs.example.com", response_time
                )
                pages_scraped += 1
                
                # Simulate reading behavior
                reading_actions = behavior_sim.generate_reading_pattern(5)
                for element_idx, focus_time in reading_actions:
                    await asyncio.sleep(focus_time * 0.1)  # Scale down for demo
                
            except BlockedException:
                session_manager.record_page_result(
                    session.id, "jobs.example.com", 0, was_blocked=True
                )
                print("Blocked! Switching tactics...")
                break
                
            except ChallengeException:
                session_manager.record_page_result(
                    session.id, "jobs.example.com", 0, was_challenge=True  
                )
                print("Challenge detected")
                # Extended delay after challenge
                await asyncio.sleep(behavior_sim.sleep_interval(base_duration=10.0))
        
        # Finish session
        outcome = SessionOutcome.SUCCESS if pages_scraped > 0 else SessionOutcome.BLOCKED
        session_manager.finish(session, outcome, pages_scraped)
        
        print(f"Session completed: {pages_scraped} pages scraped")
        print(f"Session stats: {session.to_metrics_dict()}")
        
    except Exception as e:
        session_manager.finish(session, SessionOutcome.ERROR, pages_scraped, [str(e)])
        raise

# Run the scraping example  
asyncio.run(scrape_with_anti_bot_measures())
```

## Example 4: Performance Monitoring Dashboard

```python
import time
from collections import defaultdict
from datetime import datetime, timedelta

class PerformanceMonitor:
    """Real-time performance monitoring for LazyJobSearch components."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        
    def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a performance metric."""
        if not timestamp:
            timestamp = datetime.now()
            
        self.metrics[metric_name].append((timestamp, value))
        
        # Keep only last hour of data for real-time monitoring
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics[metric_name] = [
            (ts, val) for ts, val in self.metrics[metric_name] if ts > cutoff
        ]
        
        # Check for alerts
        self._check_alerts(metric_name, value)
        
    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts."""
        alert_thresholds = {
            'matching.duration_seconds': 10.0,  # Matching too slow
            'vector_search.p95_latency_ms': 100.0,  # Vector search slow
            'llm.daily_spend_usd': 80.0,  # Approaching budget limit
            'scraping.block_rate': 0.1,  # High block rate
            'database.connection_pool_utilization': 0.8  # Pool nearly full
        }
        
        if metric_name in alert_thresholds and value > alert_thresholds[metric_name]:
            alert = {
                'metric': metric_name,
                'value': value,
                'threshold': alert_thresholds[metric_name],
                'timestamp': datetime.now(),
                'severity': 'warning' if value < alert_thresholds[metric_name] * 1.5 else 'critical'
            }
            self.alerts.append(alert)
            print(f"ALERT: {alert['severity'].upper()} - {metric_name} = {value}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
                
            recent_values = [v for ts, v in values]
            
            summary[metric_name] = {
                'current': recent_values[-1] if recent_values else 0,
                'avg': sum(recent_values) / len(recent_values),
                'p95': sorted(recent_values)[int(len(recent_values) * 0.95)] if len(recent_values) > 5 else recent_values[-1] if recent_values else 0,
                'min': min(recent_values),
                'max': max(recent_values),
                'count': len(recent_values)
            }
            
        return summary
        
    def generate_dashboard(self) -> str:
        """Generate ASCII dashboard for terminal display."""
        summary = self.get_summary()
        
        dashboard = "\n" + "="*80 + "\n"
        dashboard += "LazyJobSearch Performance Dashboard\n"
        dashboard += "="*80 + "\n"
        
        # Key metrics
        key_metrics = [
            'matching.duration_seconds',
            'vector_search.p95_latency_ms', 
            'llm.daily_spend_usd',
            'scraping.success_rate'
        ]
        
        for metric in key_metrics:
            if metric in summary:
                data = summary[metric]
                dashboard += f"{metric:30} Current: {data['current']:8.2f} "
                dashboard += f"Avg: {data['avg']:8.2f} P95: {data['p95']:8.2f}\n"
                
        # Recent alerts
        if self.alerts:
            dashboard += "\nRecent Alerts:\n"
            dashboard += "-" * 40 + "\n"
            
            for alert in self.alerts[-5:]:  # Last 5 alerts
                dashboard += f"[{alert['severity'].upper()}] {alert['metric']}: "
                dashboard += f"{alert['value']:.2f} > {alert['threshold']:.2f}\n"
                
        return dashboard

# Usage example with integrated monitoring
monitor = PerformanceMonitor()

class MonitoredMatchingService(OptimizedMatchingService):
    """Matching service with integrated performance monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        
    async def generate_matches(self, resume_id: str, limit: int = 50):
        """Generate matches with performance monitoring."""
        start_time = time.time()
        
        try:
            matches = await super().generate_matches(resume_id, limit)
            
            # Record success metrics
            duration = time.time() - start_time
            self.monitor.record_metric('matching.duration_seconds', duration)
            self.monitor.record_metric('matching.matches_generated', len(matches))
            self.monitor.record_metric('matching.success_rate', 1.0)
            
            return matches
            
        except Exception as e:
            # Record failure metrics
            duration = time.time() - start_time
            self.monitor.record_metric('matching.duration_seconds', duration)
            self.monitor.record_metric('matching.success_rate', 0.0)
            self.monitor.record_metric('matching.error_rate', 1.0)
            
            raise

# Example dashboard display
async def show_dashboard():
    """Display real-time performance dashboard."""
    
    while True:
        # Clear screen
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Show dashboard
        print(monitor.generate_dashboard())
        
        # Wait before refresh
        await asyncio.sleep(5)
        
# Run dashboard in background
# asyncio.create_task(show_dashboard())
```

These examples demonstrate practical implementation of the optimization strategies, showing how to integrate performance monitoring, adaptive algorithms, and anti-bot measures into production code.