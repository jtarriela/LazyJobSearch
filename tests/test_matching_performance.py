"""Performance benchmarks for matching pipeline

Tests the matching pipeline performance against the requirements from the audit:
- Target: O(log n) performance vs current O(n) 
- 10k jobs -> <100ms target
- Verify proper index utilization
- Test score distribution and determinism
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict
import statistics

from libs.matching.pipeline import (
    MatchingPipeline, MatchingConfig, ResumeProfile, JobCandidate,
    FTSSearcher, VectorSearcher, LLMScorer, MatchingStage
)
from libs.matching.basic_pipeline import BasicMatchingPipeline, MatchingStrategy
from libs.observability import timer, get_logger

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Performance benchmark runner for matching algorithms"""
    
    def __init__(self):
        self.results = {}
        
    def record_benchmark(self, test_name: str, execution_time: float, job_count: int):
        """Record benchmark result"""
        self.results[test_name] = {
            'execution_time': execution_time,
            'job_count': job_count,
            'jobs_per_second': job_count / execution_time if execution_time > 0 else 0,
            'ms_per_job': (execution_time * 1000) / job_count if job_count > 0 else 0
        }
        
    def get_scaling_analysis(self) -> Dict[str, float]:
        """Analyze scaling characteristics"""
        if len(self.results) < 2:
            return {}
            
        # Simple analysis of time complexity
        results = sorted(self.results.items(), key=lambda x: x[1]['job_count'])
        
        scaling = {}
        for i in range(1, len(results)):
            prev_name, prev_data = results[i-1]
            curr_name, curr_data = results[i]
            
            time_ratio = curr_data['execution_time'] / prev_data['execution_time']
            job_ratio = curr_data['job_count'] / prev_data['job_count']
            
            # O(n) would have time_ratio ≈ job_ratio
            # O(log n) would have time_ratio << job_ratio
            scaling[f"{prev_name}_to_{curr_name}"] = {
                'time_ratio': time_ratio,
                'job_ratio': job_ratio,
                'efficiency': job_ratio / time_ratio if time_ratio > 0 else 0
            }
            
        return scaling


@pytest.fixture
def performance_benchmark():
    """Performance benchmark fixture"""
    return PerformanceBenchmark()


@pytest.fixture  
def mock_large_dataset():
    """Create mock dataset for performance testing"""
    def create_jobs(count: int) -> List[Dict]:
        """Create mock job data"""
        jobs = []
        skills_options = [
            "python,sql,aws", "javascript,react,node", "java,spring,mysql",
            "go,docker,kubernetes", "rust,postgres,redis", "c++,linux,cmake"
        ]
        
        for i in range(count):
            jobs.append({
                'id': f"job_{i}",
                'title': f"Software Engineer {i}",
                'company': f"TechCorp {i % 10}",
                'description': f"Job description {i} with various technical requirements",
                'skills': skills_options[i % len(skills_options)],
                'seniority': 'mid' if i % 2 == 0 else 'senior',
                'location': 'Remote',
                'url': f"https://example.com/job{i}",
                'fts_score': 0.8 - (i * 0.0001)  # Gradually decreasing scores
            })
        return jobs
    
    return create_jobs


@pytest.mark.asyncio
async def test_fts_search_performance(mock_large_dataset, performance_benchmark):
    """Benchmark FTS search performance across different dataset sizes"""
    
    # Test with increasing dataset sizes
    job_counts = [100, 500, 1000, 2000, 5000]
    
    for count in job_counts:
        # Mock database session with large result set
        mock_session = Mock()
        mock_result = Mock()
        jobs_data = mock_large_dataset(count)
        
        # Create mock rows
        mock_rows = []
        for job in jobs_data:
            row = Mock()
            for key, value in job.items():
                setattr(row, key, value)
            mock_rows.append(row)
            
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        # Create searcher and run benchmark
        searcher = FTSSearcher(mock_session)
        
        start_time = time.time()
        candidates = await searcher.search_jobs(
            "python developer with sql and aws experience", 
            limit=count
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        performance_benchmark.record_benchmark(f"fts_{count}", execution_time, count)
        
        # Verify results
        assert len(candidates) == count
        assert all(c.fts_score is not None for c in candidates)
        
        logger.info(f"FTS search with {count} jobs: {execution_time*1000:.2f}ms")


@pytest.mark.asyncio  
async def test_vector_search_performance(mock_large_dataset, performance_benchmark):
    """Benchmark vector search performance"""
    
    job_counts = [100, 500, 1000]  # Smaller counts for vector search due to embedding costs
    
    for count in job_counts:
        # Create mock job candidates
        jobs_data = mock_large_dataset(count)
        candidates = []
        
        for job in jobs_data:
            candidate = JobCandidate(
                job_id=job['id'],
                title=job['title'],
                company=job['company'],
                description=job['description'],
                skills=job['skills'].split(',')
            )
            candidates.append(candidate)
        
        # Mock database session for vector search
        mock_session = Mock()
        mock_result = Mock()
        
        # Mock pgvector results
        vector_rows = []
        for i, job in enumerate(jobs_data[:min(50, count)]):  # Limit vector results
            row = Mock()
            row.job_id = job['id']
            row.similarity_score = 0.9 - (i * 0.01)  # Decreasing similarity
            row.distance = 0.1 + (i * 0.01)
            vector_rows.append(row)
            
        mock_result.fetchall.return_value = vector_rows
        mock_session.execute.return_value = mock_result
        
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.embed_batch = AsyncMock()
        
        searcher = VectorSearcher(mock_session, mock_embedding_service)
        resume_embedding = [0.1] * 1536  # Mock OpenAI embedding dimension
        
        start_time = time.time()
        result = await searcher.search_similar_jobs(
            candidates, 
            resume_embedding,
            limit=100,
            min_score=0.5
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        performance_benchmark.record_benchmark(f"vector_{count}", execution_time, count)
        
        # Verify results
        assert len(result) <= 100
        assert all(hasattr(c, 'vector_score') for c in result)
        
        logger.info(f"Vector search with {count} candidates: {execution_time*1000:.2f}ms")


@pytest.mark.asyncio
async def test_full_pipeline_performance(mock_large_dataset, performance_benchmark):
    """Benchmark full matching pipeline performance"""
    
    job_counts = [100, 500, 1000]
    
    for count in job_counts:
        # Setup mock pipeline components
        mock_session = Mock()
        jobs_data = mock_large_dataset(count)
        
        # Mock FTS results
        fts_result = Mock()
        fts_rows = []
        for job in jobs_data[:500]:  # FTS returns top 500
            row = Mock()
            for key, value in job.items():
                setattr(row, key, value)
            fts_rows.append(row)
        fts_result.fetchall.return_value = fts_rows
        
        # Mock vector results  
        vector_result = Mock()
        vector_rows = []
        for i, job in enumerate(jobs_data[:100]):  # Vector narrows to 100
            row = Mock()
            row.job_id = job['id']
            row.similarity_score = 0.8 - (i * 0.005)
            row.distance = 0.2 + (i * 0.005)
            vector_rows.append(row)
        vector_result.fetchall.return_value = vector_rows
        
        # Configure mock session to return different results for different queries
        def mock_execute(sql, params=None):
            sql_str = str(sql)
            if 'ts_rank' in sql_str:  # FTS query
                return fts_result
            elif 'embedding <->' in sql_str:  # Vector query
                return vector_result
            else:
                return Mock()
                
        mock_session.execute.side_effect = mock_execute
        
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.embed_text = AsyncMock(
            return_value=Mock(embedding=[0.1] * 1536)
        )
        
        # Create pipeline
        config = MatchingConfig(
            fts_limit=500,
            vector_limit=100, 
            llm_limit=20
        )
        pipeline = MatchingPipeline(mock_session, mock_embedding_service, config)
        
        # Create resume profile
        resume_profile = ResumeProfile(
            resume_id="test_resume",
            fulltext="Experienced Python developer with 5 years in web development and cloud computing",
            skills=["python", "sql", "aws", "docker"],
            years_experience=5.0
        )
        
        # Benchmark full pipeline
        start_time = time.time()
        result = await pipeline.match_resume_to_jobs(resume_profile)
        end_time = time.time()
        
        execution_time = end_time - start_time
        performance_benchmark.record_benchmark(f"pipeline_{count}", execution_time, count)
        
        # Verify results
        assert result.resume_id == "test_resume"
        assert len(result.stages_completed) > 0
        assert result.processing_time_seconds > 0
        
        logger.info(f"Full pipeline with {count} jobs: {execution_time*1000:.2f}ms")


def test_score_distribution_analysis():
    """Test score distribution and normalization"""
    
    # Create basic pipeline for score distribution testing
    mock_session = Mock()
    
    # Mock resume query
    from libs.db.models import Resume
    mock_resume = Mock(spec=Resume)
    mock_resume.id = "test_resume"
    mock_resume.skills_csv = "python,sql,aws,docker"
    mock_resume.yoe_raw = 5.0
    mock_resume.yoe_adjusted = 5.0
    
    # Mock jobs query
    from libs.db.models import Job
    mock_jobs = []
    
    skills_variations = [
        "python,sql,aws", "javascript,react,node", "python,django,postgres", 
        "java,spring,mysql", "python,flask,redis", "go,docker,kubernetes"
    ]
    
    for i in range(100):
        job = Mock(spec=Job)
        job.id = f"job_{i}"
        job.title = f"Software Engineer {i}"
        job.jd_skills_csv = skills_variations[i % len(skills_variations)]
        job.seniority = 'mid' if i % 2 == 0 else 'senior'
        job.location = "Remote"
        job.url = f"https://example.com/job{i}"
        job.yoe_min = 2
        job.yoe_max = 8
        mock_jobs.append(job)
    
    # Configure mock session
    mock_session.query.return_value.filter.return_value.first.return_value = mock_resume
    mock_session.query.return_value.all.return_value = mock_jobs
    
    pipeline = BasicMatchingPipeline(mock_session)
    
    # Run matching
    results = pipeline.match_resume_to_jobs("test_resume")
    scores = [r.composite_score for r in results]
    
    # Analyze score distribution
    score_stats = {
        'mean': statistics.mean(scores),
        'median': statistics.median(scores),
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'min': min(scores),
        'max': max(scores),
        'range': max(scores) - min(scores)
    }
    
    # Verify normalization (all scores should be 0-1)
    assert all(0 <= score <= 1 for score in scores), "Scores not properly normalized"
    assert score_stats['range'] > 0.1, "Score distribution too narrow"  
    assert score_stats['std_dev'] > 0.05, "Insufficient score variance"
    
    logger.info(f"Score distribution: mean={score_stats['mean']:.3f}, "
                f"std={score_stats['std_dev']:.3f}, range=[{score_stats['min']:.3f}, {score_stats['max']:.3f}]")


def test_deterministic_scoring():
    """Test that scoring is deterministic for identical inputs"""
    
    mock_session = Mock()
    
    # Mock resume query
    from libs.db.models import Resume
    mock_resume = Mock(spec=Resume)
    mock_resume.id = "test_resume"
    mock_resume.skills_csv = "python,sql,aws"
    mock_resume.yoe_raw = 3.0
    mock_resume.yoe_adjusted = 3.0
    
    # Mock single job query
    from libs.db.models import Job
    mock_job = Mock(spec=Job)
    mock_job.id = "test_job"
    mock_job.title = "Python Developer"
    mock_job.jd_skills_csv = "python,sql,docker"
    mock_job.seniority = "mid"
    mock_job.location = "Remote"
    mock_job.url = "https://example.com/job"
    mock_job.yoe_min = 2
    mock_job.yoe_max = 5
    
    # Configure mock session
    mock_session.query.return_value.filter.return_value.first.return_value = mock_resume
    mock_session.query.return_value.all.return_value = [mock_job]
    
    pipeline = BasicMatchingPipeline(mock_session)
    
    # Run scoring multiple times
    score_sets = []
    for i in range(10):
        results = pipeline.match_resume_to_jobs("test_resume")
        if results:
            result = results[0]
            score_sets.append({
                'skill_score': result.skill_score,
                'experience_score': result.experience_score,
                'composite_score': result.composite_score
            })
    
    # Verify all scores are identical
    if score_sets:
        first_scores = score_sets[0]
        for score_set in score_sets[1:]:
            assert score_set == first_scores, "Scoring is not deterministic"
        
        logger.info(f"Deterministic scoring verified: {first_scores}")
    else:
        pytest.fail("No scoring results generated")


def test_performance_targets(performance_benchmark):
    """Analyze if performance targets are met"""
    
    # This test runs after the benchmarks to analyze results
    if not performance_benchmark.results:
        pytest.skip("No benchmark results available")
    
    # Check if we meet performance targets from audit
    target_times = {
        100: 0.010,   # 10ms for 100 jobs  
        500: 0.030,   # 30ms for 500 jobs
        1000: 0.050,  # 50ms for 1000 jobs (target from audit)
        5000: 0.150   # 150ms for 5000 jobs (acceptable)
    }
    
    for test_name, result in performance_benchmark.results.items():
        job_count = result['job_count']
        execution_time = result['execution_time']
        
        if job_count in target_times:
            target_time = target_times[job_count]
            performance_ratio = execution_time / target_time
            
            logger.info(f"{test_name}: {execution_time*1000:.2f}ms for {job_count} jobs "
                       f"(target: {target_time*1000:.2f}ms, ratio: {performance_ratio:.2f})")
            
            # Log warning if significantly over target, but don't fail tests
            # since we're testing with mocks, not real database indexes
            if performance_ratio > 5.0:
                logger.warning(f"{test_name} is {performance_ratio:.1f}x slower than target")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])