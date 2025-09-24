"""Unit tests for LLM resume parsing logic"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from libs.resume.llm_service import (
    LLMResumeParser, LLMConfig, ParsedResumeData, 
    create_llm_client, MockLLMClient, REQUIRED_FIELDS
)


def test_llm_config_defaults():
    """Test LLM configuration defaults"""
    config = LLMConfig()
    
    assert config.provider == 'mock'
    assert config.model == 'gpt-3.5-turbo'
    assert config.timeout == 30
    assert config.max_tokens == 2000


def test_required_fields_definition():
    """Test that required fields are properly defined"""
    expected_fields = ["full_name", "email", "phone", "skills", "experience", "education", "full_text"]
    assert REQUIRED_FIELDS == expected_fields


def test_parsed_resume_data_validation():
    """Test pydantic validation in ParsedResumeData"""
    # Test with minimal valid data
    data = ParsedResumeData(
        full_name="John Doe",
        email="john@example.com",
        phone="555-1234",
        skills=["Python", "Java"],
        experience=[{"title": "Engineer", "company": "TechCorp", "duration": "2020-2023", "description": "Software development"}],
        education=[{"degree": "BS", "field": "Computer Science", "institution": "University", "year": "2020"}],
        full_text="John Doe resume text"
    )
    
    assert data.full_name == "John Doe"
    assert data.email == "john@example.com"
    assert len(data.skills) == 2
    assert data.is_complete() == True
    assert len(data.get_missing_fields()) == 0


def test_parsed_resume_data_missing_fields():
    """Test missing fields detection"""
    data = ParsedResumeData(
        full_name="John Doe",
        skills=["Python"],
        # Missing email, phone, experience, education, full_text
    )
    
    missing = data.get_missing_fields()
    assert "email" in missing
    assert "phone" in missing
    assert "experience" in missing
    assert "education" in missing
    assert "full_text" in missing
    assert data.is_complete() == False


def test_llm_client_creation():
    """Test LLM client factory function"""
    config = LLMConfig()
    client = create_llm_client(config)
    
    assert isinstance(client, MockLLMClient)


@pytest.mark.asyncio
async def test_mock_llm_client():
    """Test mock LLM client functionality"""
    client = MockLLMClient()
    
    response = await client.chat(
        model="gpt-3.5-turbo",
        system="You are a resume parser",
        user="Parse this resume",
        text="John Doe\nSoftware Engineer\nPython, Java"
    )
    
    assert response.content
    assert response.model == "gpt-3.5-turbo"
    assert response.tokens_used > 0
    assert response.cost_cents >= 0


@pytest.mark.asyncio 
async def test_llm_resume_parser_basic():
    """Test basic LLM resume parser functionality"""
    parser = LLMResumeParser()
    
    result, responses = await parser.parse_resume(
        fallback_text="John Doe\nSoftware Engineer\nEmail: john@example.com\nSkills: Python, Java"
    )
    
    # Verify result
    assert isinstance(result, ParsedResumeData)
    assert result.full_text == "John Doe\nSoftware Engineer\nEmail: john@example.com\nSkills: Python, Java"
    assert len(responses) > 0
    
    # Verify parser state
    assert parser.requests_made > 0
    assert parser.total_tokens_used > 0


def test_json_extraction_utility():
    """Test JSON extraction from LLM responses with extra text"""
    parser = LLMResumeParser()
    
    # Test with clean JSON
    clean_json = '{"name": "John", "email": "john@example.com"}'
    extracted = parser._extract_json(clean_json)
    assert extracted == clean_json
    
    # Test with extra text before and after
    messy_response = 'Here is the JSON:\n{"name": "John", "email": "john@example.com"}\nThat\'s all!'
    extracted = parser._extract_json(messy_response)
    assert extracted == '{"name": "John", "email": "john@example.com"}'
    
    # Test with nested JSON
    nested_json = '{"person": {"name": "John"}, "contact": {"email": "john@example.com"}}'
    extracted = parser._extract_json(nested_json)
    assert extracted == nested_json


def test_data_merge_functionality():
    """Test merging of parsed data across attempts"""
    parser = LLMResumeParser()
    
    existing = ParsedResumeData(
        full_name="John Doe",
        email="john@example.com",
        skills=["Python"],
        full_text="Resume text"
    )
    
    new = ParsedResumeData(
        phone="555-1234",
        experience=[{"title": "Engineer", "company": "TechCorp", "duration": "2020-2023", "description": "Development"}],
        full_text="Resume text"
    )
    
    merged = parser._merge_parsed_data(existing, new)
    
    assert merged.full_name == "John Doe"  # From existing
    assert merged.email == "john@example.com"  # From existing
    assert merged.phone == "555-1234"  # From new
    assert len(merged.experience) == 1  # From new
    assert len(merged.skills) == 1  # From existing


if __name__ == "__main__":
    pytest.main([__file__])