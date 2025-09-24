"""Integration tests for resume ingestion pipeline"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from libs.resume.ingestion import ResumeIngestionService, IngestionError
from libs.resume.parser import ParsedResume


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = None
    return session


@pytest.fixture
def sample_parsed_resume():
    """Sample parsed resume for testing"""
    return ParsedResume(
        fulltext="John Doe\nSoftware Engineer\nPython, Java, SQL",
        sections={"experience": "Software Engineer at TechCorp", "education": "BS Computer Science"},
        skills=["python", "java", "sql"],
        years_of_experience=3.0,
        education_level="bachelors",
        contact_info={"email": "john@example.com", "phone": "555-1234"},
        word_count=100,
        char_count=500,
        full_name="John Doe",
        experience=[{"title": "Software Engineer", "company": "TechCorp", "duration": "2020-2023"}],
        education=[{"degree": "BS Computer Science", "institution": "University", "year": "2020"}],
        certifications=[],
        summary="Experienced software engineer",
        parsing_method="llm"
    )


def test_ingestion_error_is_exception():
    """Test that IngestionError is a proper Exception"""
    error = IngestionError("parsing", "Test error", "test_file.pdf")
    
    # Should be an instance of Exception
    assert isinstance(error, Exception)
    
    # Should have proper attributes
    assert error.stage == "parsing"
    assert error.error_message == "Test error"
    assert error.file_path == "test_file.pdf"
    
    # Should be raisable
    with pytest.raises(IngestionError) as exc_info:
        raise error
    
    assert exc_info.value.stage == "parsing"
    assert "Test error" in str(exc_info.value)


def test_persistence_failure_handling(mock_db_session, sample_parsed_resume):
    """Test that persistence failure is properly caught and rethrown as IngestionError"""
    # Setup
    service = ResumeIngestionService(db_session=mock_db_session)
    
    # Mock the session to raise an exception during commit
    mock_db_session.commit.side_effect = Exception("Database connection failed")
    
    # Test data - include all required fields for chunks
    chunks = [{
        "text": "sample chunk",
        "token_count": 10,
        "embedding": [0.1, 0.2, 0.3],
        "embedding_version": "v1.0",
        "embedding_model": "text-embedding-ada-002"
    }]
    
    # Test
    with pytest.raises(IngestionError) as exc_info:
        service._persist_resume_data(sample_parsed_resume, chunks, None, "test_file.pdf")
    
    # Verify
    assert exc_info.value.stage == "persistence"
    assert "Failed to persist to database" in exc_info.value.error_message
    assert "Database connection failed" in exc_info.value.error_message
    

@patch('libs.resume.ingestion.create_resume_parser')
@patch('libs.resume.ingestion.create_resume_chunker')
@patch('libs.resume.ingestion.create_embedding_service')
def test_happy_path_ingestion_mock(mock_embedding_service, mock_chunker, mock_parser, mock_db_session, sample_parsed_resume):
    """Test happy path ingestion with mocked persistence"""
    # Create service
    service = ResumeIngestionService(db_session=mock_db_session)
    
    # Setup mocks - patch the service's components directly
    service.parser = Mock()
    service.chunker = Mock()
    service.embedding_service = Mock()
    
    service.parser.parse_file.return_value = sample_parsed_resume
    service.chunker.chunk_resume.return_value = [
        {"text": "chunk 1", "token_count": 50},
        {"text": "chunk 2", "token_count": 40}
    ]
    service.embedding_service.generate_embeddings.return_value = [
        {"embedding": [0.1, 0.2, 0.3], "cost_cents": 0.1},
        {"embedding": [0.4, 0.5, 0.6], "cost_cents": 0.1}
    ]
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(b"dummy pdf content")
        tmp_file_path = Path(tmp_file.name)
    
    try:
        # Test ingestion
        result = service.ingest_resume_file(tmp_file_path)
        
        # Verify result
        assert result.resume_id
        assert result.parsed_resume == sample_parsed_resume
        assert len(result.chunks) == 2
        assert result.processing_time_ms > 0
        
        # Verify database operations were called
        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()
        
    finally:
        # Cleanup
        tmp_file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])