"""Tests for matching pipeline components"""
import pytest
from unittest.mock import Mock, AsyncMock
from libs.matching.pipeline import (
    MatchingPipeline, MatchingConfig, ResumeProfile, JobCandidate,
    FTSSearcher, VectorSearcher, LLMScorer
)
from libs.matching.persistence import MatchPersistenceService

@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    service = Mock()
    service.embed_text = AsyncMock()
    service.embed_batch = AsyncMock()
    return service

def test_matching_config_defaults():
    """Test matching configuration defaults"""
    config = MatchingConfig()
    
    assert config.fts_limit == 1000
    assert config.vector_limit == 100
    assert config.llm_limit == 20
    assert config.fts_weight + config.vector_weight + config.llm_weight == 1.0

def test_job_candidate_creation():
    """Test job candidate data structure"""
    candidate = JobCandidate(
        job_id="123",
        title="Software Engineer",
        company="TechCorp",
        description="Python development role",
        skills=["python", "sql", "aws"]
    )
    
    assert candidate.job_id == "123"
    assert candidate.title == "Software Engineer"
    assert len(candidate.skills) == 3
    assert candidate.metadata == {}

def test_resume_profile_creation():
    """Test resume profile data structure"""
    profile = ResumeProfile(
        resume_id="456",
        fulltext="John Doe, Software Engineer with Python experience",
        skills=["python", "javascript", "sql"],
        years_experience=3.0
    )
    
    assert profile.resume_id == "456"
    assert len(profile.skills) == 3
    assert profile.years_experience == 3.0

@pytest.mark.asyncio
async def test_fts_searcher_basic(mock_session):
    """Test FTS searcher basic functionality"""
    # Mock SQL result
    mock_result = Mock()
    mock_result.fetchall.return_value = [
        Mock(
            id="job1",
            title="Python Developer",
            company="TechCorp",
            description="Python development role",
            skills="python,sql,aws",
            seniority="mid",
            location="Remote",
            url="https://example.com/job1",
            fts_score=0.8
        )
    ]
    mock_session.execute.return_value = mock_result
    
    searcher = FTSSearcher(mock_session)
    
    candidates = await searcher.search_jobs("python developer sql", limit=10)
    
    assert len(candidates) == 1
    assert candidates[0].job_id == "job1"
    assert candidates[0].fts_score == 0.8
    assert "python" in candidates[0].skills

@pytest.mark.asyncio 
async def test_vector_searcher_basic(mock_session, mock_embedding_service):
    """Test vector searcher basic functionality"""
    # Mock embedding response
    from libs.resume.embedding_service import EmbeddingResponse
    from datetime import datetime
    
    mock_embedding_service.embed_batch.return_value = [
        EmbeddingResponse(
            text_id="job_test",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
            tokens_used=10,
            cost_cents=0.01,
            created_at=datetime.now()
        )
    ]
    
    searcher = VectorSearcher(mock_session, mock_embedding_service)
    
    candidates = [
        JobCandidate(
            job_id="job1",
            title="Python Developer", 
            company="TechCorp",
            description="Python development role",
            skills=["python"]
        )
    ]
    
    resume_embedding = [0.1, 0.2, 0.3]
    
    result = await searcher.search_similar_jobs(candidates, resume_embedding)
    
    assert len(result) <= len(candidates)
    mock_embedding_service.embed_batch.assert_called_once()

@pytest.mark.asyncio
async def test_llm_scorer_basic():
    """Test LLM scorer basic functionality"""
    scorer = LLMScorer()
    
    candidates = [
        JobCandidate(
            job_id="job1",
            title="Python Developer",
            company="TechCorp", 
            description="Python development role",
            skills=["python", "sql"]
        )
    ]
    
    resume_profile = ResumeProfile(
        resume_id="resume1",
        fulltext="Python developer with 3 years experience",
        skills=["python", "javascript"],
        years_experience=3.0
    )
    
    scored = await scorer.score_matches(candidates, resume_profile, max_cost_cents=10.0)
    
    assert len(scored) == 1
    assert scored[0].llm_score is not None
    assert scored[0].llm_reasoning is not None
    assert 0 <= scored[0].llm_score <= 100

def test_match_persistence_service_basic(mock_session):
    """Test match persistence service basic functionality"""
    from libs.matching.pipeline import MatchingResult, MatchingStage
    
    service = MatchPersistenceService(mock_session)
    
    # Mock existing match query
    mock_session.query.return_value.filter.return_value.first.return_value = None
    
    candidate = JobCandidate(
        job_id="job1",
        title="Python Developer",
        company="TechCorp",
        description="Python role",
        skills=["python"],
        llm_score=85,
        vector_score=0.7,
        llm_reasoning="Good match"
    )
    
    result = MatchingResult(
        resume_id="resume1",
        matches=[candidate],
        total_candidates=1,
        stages_completed=[MatchingStage.LLM_SCORING],
        total_cost_cents=1.0,
        processing_time_seconds=2.5
    )
    
    match_ids = service.save_matching_result(result)
    
    # Verify session operations were called
    mock_session.begin.assert_called_once()
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()
    mock_session.commit.assert_called_once()

def test_cosine_similarity():
    """Test cosine similarity calculation"""
    from libs.matching.pipeline import VectorSearcher
    
    searcher = VectorSearcher(Mock(), Mock())
    
    # Test identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = searcher._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6
    
    # Test orthogonal vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = searcher._cosine_similarity(vec1, vec2)
    assert abs(similarity) < 1e-6
    
    # Test opposite vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    similarity = searcher._cosine_similarity(vec1, vec2)
    assert abs(similarity - (-1.0)) < 1e-6

if __name__ == "__main__":
    pytest.main([__file__])