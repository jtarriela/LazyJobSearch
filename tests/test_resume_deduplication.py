"""Tests for resume deduplication functionality"""
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from libs.resume.ingestion import ResumeIngestionService
from libs.resume.parser import ParsedResume


class TestResumeDeduplication:
    """Test resume content deduplication"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_db_session = Mock()
        self.service = ResumeIngestionService(self.mock_db_session)
    
    def test_compute_content_hash(self):
        """Test content hash computation"""
        # Create identical resume content
        resume1 = ParsedResume(
            fulltext="John Doe Software Engineer",
            sections={"experience": "3 years Python", "skills": "Python, SQL"},
            skills=["python", "sql"],
            years_of_experience=3,
            education_level="bachelors",
            contact_info={"email": "john@example.com"},
            word_count=100,
            char_count=500
        )
        
        resume2 = ParsedResume(
            fulltext="John Doe Software Engineer",
            sections={"experience": "3 years Python", "skills": "Python, SQL"},
            skills=["python", "sql"],
            years_of_experience=3,
            education_level="bachelors",
            contact_info={"email": "john@example.com"},
            word_count=100,
            char_count=500
        )
        
        hash1 = self.service._compute_content_hash(resume1)
        hash2 = self.service._compute_content_hash(resume2)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
    
    def test_compute_content_hash_different(self):
        """Test content hash computation for different content"""
        resume1 = ParsedResume(
            fulltext="John Doe Software Engineer",
            sections={"experience": "3 years Python"},
            skills=["python"],
            years_of_experience=3,
            education_level="bachelors",
            contact_info={"email": "john@example.com"},
            word_count=100,
            char_count=500
        )
        
        resume2 = ParsedResume(
            fulltext="Jane Smith Data Scientist",
            sections={"experience": "2 years R"},
            skills=["r"],
            years_of_experience=2,
            education_level="masters",
            contact_info={"email": "jane@example.com"},
            word_count=90,
            char_count=450
        )
        
        hash1 = self.service._compute_content_hash(resume1)
        hash2 = self.service._compute_content_hash(resume2)
        
        assert hash1 != hash2
    
    def test_check_resume_duplicate_found(self):
        """Test duplicate detection when duplicate exists"""
        content_hash = "abc123def456"
        existing_resume = Mock()
        existing_resume.id = "existing-resume-123"
        
        # Mock database query to return existing resume
        self.mock_db_session.query.return_value.filter.return_value.first.return_value = existing_resume
        
        result = self.service._check_resume_duplicate(content_hash)
        
        assert result == "existing-resume-123"
        self.mock_db_session.query.assert_called_once()
    
    def test_check_resume_duplicate_not_found(self):
        """Test duplicate detection when no duplicate exists"""
        content_hash = "abc123def456"
        
        # Mock database query to return None
        self.mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        result = self.service._check_resume_duplicate(content_hash)
        
        assert result is None
        self.mock_db_session.query.assert_called_once()
    
    def test_check_resume_duplicate_error(self):
        """Test duplicate detection error handling"""
        content_hash = "abc123def456"
        
        # Mock database query to raise exception
        self.mock_db_session.query.side_effect = Exception("Database error")
        
        result = self.service._check_resume_duplicate(content_hash)
        
        assert result is None
    
    @patch.object(ResumeIngestionService, '_parse_resume_file')
    @patch.object(ResumeIngestionService, '_check_resume_duplicate')
    def test_ingestion_skips_duplicate(self, mock_check_duplicate, mock_parse_resume):
        """Test that ingestion skips duplicate resumes"""
        # Set up mocks
        parsed_resume = ParsedResume(
            fulltext="Test resume content",
            sections={},
            skills=[],
            years_of_experience=None,
            education_level=None,
            contact_info={},
            word_count=10,
            char_count=50
        )
        mock_parse_resume.return_value = parsed_resume
        mock_check_duplicate.return_value = "existing-resume-123"  # Duplicate found
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)
        
        try:
            result = self.service.ingest_resume_file(temp_path)
            
            # Should return existing resume ID
            assert result.resume_id == "existing-resume-123"
            assert result.embedding_stats.get("duplicate") is True
            assert len(result.chunks) == 0
            
            # Should not proceed to chunking/embedding
            mock_check_duplicate.assert_called_once()
            
        finally:
            temp_path.unlink()
    
    @patch.object(ResumeIngestionService, '_parse_resume_file')
    @patch.object(ResumeIngestionService, '_check_resume_duplicate')  
    @patch.object(ResumeIngestionService, '_chunk_resume_content')
    @patch.object(ResumeIngestionService, '_generate_embeddings')
    @patch.object(ResumeIngestionService, '_persist_resume_data')
    def test_ingestion_processes_unique(self, mock_persist, mock_embeddings, 
                                       mock_chunk, mock_check_duplicate, mock_parse_resume):
        """Test that ingestion processes unique resumes"""
        # Set up mocks
        parsed_resume = ParsedResume(
            fulltext="Test resume content",
            sections={},
            skills=[],
            years_of_experience=None,
            education_level=None,
            contact_info={},
            word_count=10,
            char_count=50
        )
        mock_parse_resume.return_value = parsed_resume
        mock_check_duplicate.return_value = None  # No duplicate
        mock_chunk.return_value = [{"text": "chunk1", "token_count": 10}]
        mock_embeddings.return_value = {"chunks_processed": 1}
        mock_persist.return_value = "new-resume-123"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)
        
        try:
            result = self.service.ingest_resume_file(temp_path)
            
            # Should create new resume
            assert result.resume_id == "new-resume-123"
            assert result.embedding_stats.get("duplicate") is not True
            assert len(result.chunks) > 0
            
            # Should proceed through full pipeline
            mock_check_duplicate.assert_called_once()
            mock_chunk.assert_called_once()
            mock_embeddings.assert_called_once()
            mock_persist.assert_called_once()
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])