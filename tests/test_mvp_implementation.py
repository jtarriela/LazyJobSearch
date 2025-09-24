"""Integration test to validate MVP implementation against gap analysis

This test validates that the major MVP components identified in the problem 
statement have been implemented and are working together.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

def test_mvp_gap_analysis_implementation():
    """Test that MVP gaps from problem statement are addressed"""
    
    # Test resume processing pipeline exists and works
    from libs.resume.parser import create_resume_parser
    from libs.resume.chunker import create_resume_chunker, ChunkingConfig
    from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
    
    parser = create_resume_parser()
    chunker = create_resume_chunker()
    embedding_service = create_embedding_service(provider=EmbeddingProvider.MOCK)
    
    assert parser is not None
    assert chunker is not None
    assert embedding_service is not None
    
    # Test matching pipeline exists and works
    from libs.matching.pipeline import create_matching_pipeline, MatchingConfig
    from libs.matching.persistence import create_match_persistence_service
    
    mock_session = Mock()
    config = MatchingConfig()
    pipeline = create_matching_pipeline(mock_session, embedding_service, config)
    persistence = create_match_persistence_service(mock_session)
    
    assert pipeline is not None
    assert persistence is not None
    assert config.fts_limit == 1000  # Verify default config
    assert config.vector_limit == 100
    assert config.llm_limit == 20
    
    # Test review and iteration loop exists
    from libs.resume.review import create_review_iteration_manager
    
    review_manager = create_review_iteration_manager(mock_session)
    assert review_manager is not None
    
    # Test auto-apply DSL exists
    from libs.autoapply.template_dsl import create_template_builder, create_field_mapper
    
    template_builder = create_template_builder()
    field_mapper = create_field_mapper()
    
    assert template_builder is not None
    assert field_mapper is not None
    
    # Test notifications/digest exists
    from libs.notifications.digest import create_digest_service
    
    digest_service = create_digest_service(mock_session)
    assert digest_service is not None

@pytest.mark.asyncio
async def test_end_to_end_resume_processing():
    """Test end-to-end resume processing pipeline"""
    
    # Create a sample resume file
    resume_content = """
    John Doe
    Software Engineer
    
    Experience:
    - Python development (3 years)
    - Web applications with Django
    - Database design with PostgreSQL
    
    Skills: Python, Django, PostgreSQL, AWS
    
    Education: Bachelor's degree in Computer Science
    Contact: john@example.com
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(resume_content)
        resume_file = Path(f.name)
    
    try:
        # Test parsing
        from libs.resume.parser import create_resume_parser
        parser = create_resume_parser()
        parsed_resume = parser.parse_file(resume_file)
        
        assert parsed_resume.fulltext
        assert len(parsed_resume.skills) > 0
        assert 'python' in [s.lower() for s in parsed_resume.skills]
        assert parsed_resume.years_of_experience == 3.0
        assert parsed_resume.education_level == 'bachelors'
        
        # Test chunking
        from libs.resume.chunker import create_resume_chunker
        chunker = create_resume_chunker()
        chunks = chunker.chunk_resume(parsed_resume.fulltext, parsed_resume.sections)
        
        assert len(chunks) > 0
        assert all(chunk.text.strip() for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)
        
        # Test embedding
        from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
        embedding_service = create_embedding_service(provider=EmbeddingProvider.MOCK)
        
        response = await embedding_service.embed_text(parsed_resume.fulltext)
        
        assert response.embedding
        assert len(response.embedding) > 0
        assert response.tokens_used > 0
        
    finally:
        resume_file.unlink()

@pytest.mark.asyncio 
async def test_matching_pipeline_integration():
    """Test that matching pipeline components work together"""
    
    from libs.matching.pipeline import (
        create_matching_pipeline, ResumeProfile, MatchingConfig
    )
    from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
    
    # Mock database session with job data
    mock_session = Mock()
    mock_result = Mock()
    mock_result.fetchall.return_value = [
        Mock(
            id="job1",
            title="Python Developer",
            company="TechCorp",
            description="Python development role",
            skills="python,django,postgresql",
            seniority="mid",
            location="Remote",
            url="https://example.com/job1",
            fts_score=0.8
        )
    ]
    mock_session.execute.return_value = mock_result
    
    # Create services
    embedding_service = create_embedding_service(provider=EmbeddingProvider.MOCK)
    config = MatchingConfig(fts_limit=10, vector_limit=5, llm_limit=3)
    pipeline = create_matching_pipeline(mock_session, embedding_service, config)
    
    # Create resume profile
    resume_profile = ResumeProfile(
        resume_id="test_resume",
        fulltext="Python developer with Django and PostgreSQL experience",
        skills=["python", "django", "postgresql"],
        years_experience=3.0
    )
    
    # Run matching pipeline
    result = await pipeline.match_resume_to_jobs(resume_profile)
    
    assert result.resume_id == "test_resume"
    assert result.processing_time_seconds > 0
    assert len(result.stages_completed) > 0
    # Note: Matches may be empty due to mock data, but pipeline should complete

@pytest.mark.asyncio
async def test_review_system_integration():
    """Test resume review and critique system"""
    
    from libs.resume.review import ResumeCritic, ResumeRewriter
    
    critic = ResumeCritic()
    rewriter = ResumeRewriter()
    
    resume_content = "Python developer with 2 years experience building web applications."
    job_description = "Looking for a senior Python developer with 5+ years experience."
    
    # Test critique
    critique, cost = await critic.critique_resume(
        resume_content, job_description, "Senior Python Developer", "TechCorp"
    )
    
    assert critique.overall_score >= 0
    assert critique.overall_score <= 100
    assert len(critique.strengths) > 0
    assert len(critique.weaknesses) > 0
    assert len(critique.improvement_suggestions) > 0
    assert cost > 0
    
    # Test rewrite
    rewrite, rewrite_cost = await rewriter.rewrite_resume(
        resume_content, critique, job_description
    )
    
    assert rewrite.new_content
    assert rewrite.changes_summary
    assert len(rewrite.sections_changed) > 0
    assert rewrite_cost > 0

def test_auto_apply_dsl_functionality():
    """Test auto-apply DSL template system"""
    
    from libs.autoapply.template_dsl import (
        create_template_builder, create_field_mapper,
        FieldType, FieldMapping, ValidationRule
    )
    
    # Test template building
    builder = create_template_builder()
    
    template = (builder
        .create_template("test_portal", "TestPortal", "1.0")
        .add_section("personal_info", "Personal Information")
        .add_field(
            "first_name", "first_name", FieldType.TEXT, "#first_name",
            is_required=True,
            validation_rules=[ValidationRule.REQUIRED],
            mapping=FieldMapping("personal.first_name")
        )
        .add_field(
            "email", "email", FieldType.EMAIL, "#email",
            is_required=True,
            validation_rules=[ValidationRule.REQUIRED, ValidationRule.EMAIL_FORMAT],
            mapping=FieldMapping("contact.email")
        )
        .set_submission_config({"submit_button": "#submit", "confirmation_selector": ".success"})
        .build()
    )
    
    assert template.template_id == "test_portal"
    assert template.portal_name == "TestPortal"
    assert len(template.form_sections) == 1
    assert len(template.form_sections[0].fields) == 2
    
    # Test field mapping
    mapper = create_field_mapper()
    
    candidate_data = {
        "personal": {"first_name": "John", "last_name": "Doe"},
        "contact": {"email": "john.doe@example.com", "phone": "555-1234"}
    }
    
    mapped_data = mapper.map_candidate_to_form(candidate_data, template)
    
    assert "first_name" in mapped_data
    assert "email" in mapped_data
    assert mapped_data["first_name"] == "John"
    assert mapped_data["email"] == "john.doe@example.com"

@pytest.mark.asyncio
async def test_digest_generation():
    """Test daily digest generation and email templating"""
    
    from libs.notifications.digest import create_digest_service
    
    mock_session = Mock()
    digest_service = create_digest_service(mock_session)
    
    # Test digest generation and sending
    success = await digest_service.send_daily_digest("test_user", "test@example.com")
    
    # Should succeed with mock data
    assert success == True

def test_mvp_component_factory_functions():
    """Test that all factory functions work and return proper instances"""
    
    # Resume processing factories
    from libs.resume.parser import create_resume_parser
    from libs.resume.chunker import create_resume_chunker
    from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
    from libs.resume.review import create_review_iteration_manager
    
    # Matching pipeline factories  
    from libs.matching.pipeline import create_matching_pipeline
    from libs.matching.persistence import create_match_persistence_service
    
    # Auto-apply factories
    from libs.autoapply.template_dsl import create_template_builder, create_field_mapper
    
    # Notifications factories
    from libs.notifications.digest import create_digest_service
    
    mock_session = Mock()
    embedding_service = create_embedding_service(provider=EmbeddingProvider.MOCK)
    
    # Test all factory functions
    factories_to_test = [
        (create_resume_parser, []),
        (create_resume_chunker, []),
        (create_embedding_service, [EmbeddingProvider.MOCK]),
        (create_review_iteration_manager, [mock_session]),
        (create_matching_pipeline, [mock_session, embedding_service]),
        (create_match_persistence_service, [mock_session]),
        (create_template_builder, []),
        (create_field_mapper, []),
        (create_digest_service, [mock_session])
    ]
    
    for factory_func, args in factories_to_test:
        instance = factory_func(*args)
        assert instance is not None, f"Factory {factory_func.__name__} returned None"

if __name__ == "__main__":
    pytest.main([__file__])