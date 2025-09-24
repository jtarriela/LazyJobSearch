"""Tests for resume processing components"""
import pytest
from unittest.mock import Mock, AsyncMock
from libs.resume.parser import ResumeParser, ParsedResume
from libs.resume.chunker import ResumeChunker, ChunkingConfig, ChunkStrategy
from libs.resume.embedding_service import EmbeddingService, EmbeddingProvider

def test_resume_parser_basic():
    """Test basic resume parsing functionality"""
    parser = ResumeParser()
    
    # Test text parsing
    resume_text = """
    John Doe
    Software Engineer
    
    Experience:
    - Python development (3 years)
    - Web applications with Django
    - Database design
    
    Skills: Python, SQL, AWS, Docker
    
    Education:
    Bachelor's degree in Computer Science
    
    Contact: john@example.com
    """
    
    result = parser.parse_text(resume_text)
    
    assert isinstance(result, ParsedResume)
    assert result.fulltext
    assert len(result.skills) > 0
    assert 'python' in [s.lower() for s in result.skills]
    assert result.contact_info.get('email') == 'john@example.com'
    assert result.education_level == 'bachelors'

def test_resume_chunker_basic():
    """Test basic resume chunking functionality"""
    config = ChunkingConfig(
        max_tokens=100,
        overlap_tokens=20,
        strategy=ChunkStrategy.SLIDING_WINDOW
    )
    chunker = ResumeChunker(config)
    
    resume_text = """
    John Doe is a software engineer with 5 years of experience.
    He has worked on Python applications, web development, and database design.
    His skills include Python, JavaScript, SQL, AWS, and Docker.
    He holds a Bachelor's degree in Computer Science.
    """
    
    chunks = chunker.chunk_resume(resume_text)
    
    assert len(chunks) > 0
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)  # No empty chunks

@pytest.mark.asyncio
async def test_embedding_service_basic():
    """Test basic embedding service functionality"""
    service = EmbeddingService(provider=EmbeddingProvider.MOCK)
    
    text = "Python developer with 3 years of experience"
    
    response = await service.embed_text(text)
    
    assert response.embedding
    assert len(response.embedding) > 0
    assert response.dimensions > 0
    assert response.tokens_used > 0
    assert response.model

@pytest.mark.asyncio 
async def test_embedding_service_caching():
    """Test embedding service caching"""
    service = EmbeddingService(provider=EmbeddingProvider.MOCK, cache_enabled=True)
    
    text = "Test text for caching"
    
    # First request
    response1 = await service.embed_text(text)
    assert not response1.cached
    
    # Second request should be cached
    response2 = await service.embed_text(text)
    assert response2.cached
    assert response1.embedding == response2.embedding

def test_chunking_strategies():
    """Test different chunking strategies"""
    resume_text = """
    Experience:
    Software Engineer at TechCorp (2020-2023)
    - Developed Python applications
    - Worked with AWS and Docker
    
    Junior Developer at StartupInc (2018-2020)  
    - Built web applications
    - Used JavaScript and React
    
    Skills:
    Python, JavaScript, AWS, Docker, React
    """
    
    sections = {
        'experience': 'Software Engineer at TechCorp...',
        'skills': 'Python, JavaScript, AWS, Docker, React'
    }
    
    # Test section-based chunking
    config = ChunkingConfig(strategy=ChunkStrategy.SECTION_BASED)
    chunker = ResumeChunker(config)
    chunks = chunker.chunk_resume(resume_text, sections)
    
    assert len(chunks) > 0
    assert any(chunk.section == 'experience' for chunk in chunks)
    assert any(chunk.section == 'skills' for chunk in chunks)

if __name__ == "__main__":
    pytest.main([__file__])