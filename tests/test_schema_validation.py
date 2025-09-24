#!/usr/bin/env python
"""Basic migration and schema validation tests.

Tests to ensure:
1. Schema validation passes without errors
2. Models can be imported without errors
3. All documented tables have corresponding models
"""
import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_schema_validation():
    """Test that schema validation passes."""
    result = subprocess.run([
        sys.executable, 
        'scripts/validate_schema_docs.py'
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    # Should pass (exit code 0)
    assert result.returncode == 0, f"Schema validation failed: {result.stderr}"
    print("✅ Schema validation passed")

def test_models_import():
    """Test that all models can be imported without errors."""
    try:
        from libs.db import models
        # Verify critical models exist
        assert hasattr(models, 'Company')
        assert hasattr(models, 'Job') 
        assert hasattr(models, 'Resume')
        assert hasattr(models, 'Review')
        assert hasattr(models, 'Application')
        assert hasattr(models, 'Portal')
        assert hasattr(models, 'PortalTemplate')
        print("✅ All models import successfully")
    except Exception as e:
        raise AssertionError(f"Failed to import models: {e}")

def test_unique_constraints():
    """Test that critical unique constraints are defined."""
    from libs.db.models import Resume, Application, Job
    
    # Check resume content hash constraint
    resume_columns = [col.name for col in Resume.__table__.columns]
    assert 'content_hash' in resume_columns, "Resume content_hash column missing"
    
    # Check job URL unique constraint  
    job_url_col = next((col for col in Job.__table__.columns if col.name == 'url'), None)
    assert job_url_col is not None, "Job URL column missing"
    assert job_url_col.unique, "Job URL should have unique constraint"
    
    # Check application unique constraint
    app_constraints = getattr(Application, '__table_args__', ())
    constraint_names = [getattr(c, 'name', '') for c in app_constraints if hasattr(c, 'name')]
    assert 'uq_application_job_profile' in constraint_names, "Application deduplication constraint missing"
    
    print("✅ Critical unique constraints verified")

def test_check_constraints():
    """Test that check constraints are defined for status fields."""
    from libs.db.models import Application, Review
    
    # Check application status constraint
    app_constraints = getattr(Application, '__table_args__', ())
    constraint_names = [getattr(c, 'name', '') for c in app_constraints if hasattr(c, 'name')]
    assert 'ck_application_status' in constraint_names, "Application status check constraint missing"
    
    # Check review status constraint
    review_constraints = getattr(Review, '__table_args__', ())
    constraint_names = [getattr(c, 'name', '') for c in review_constraints if hasattr(c, 'name')]
    assert 'ck_review_status' in constraint_names, "Review status check constraint missing"
    
    print("✅ Check constraints verified")

if __name__ == "__main__":
    print("Running schema and migration validation tests...")
    test_models_import()
    test_schema_validation()
    test_unique_constraints()
    test_check_constraints()
    print("🎉 All tests passed!")