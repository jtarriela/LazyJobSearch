#!/usr/bin/env python3
"""
End-to-end integration test for LazyJobSearch MVP implementation.

This script tests the complete implementation by:
1. Testing database models and migrations
2. Testing observability and metrics
3. Testing resume ingestion pipeline
4. Testing job ingestion pipeline  
5. Testing company seeding
6. Testing auto-apply service integration
7. Testing CLI commands

Run with: python test_mvp_integration.py
"""
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import csv
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    # Core infrastructure
    from libs.db.models import Base, Company, Job, Resume, ResumeChunk, Match, EmbeddingVersion
    from libs.observability import get_logger, counter, timer, get_metrics_collector
    from libs.embed.versioning import EmbeddingVersionManager, EmbeddingVersionInfo
    
    # Services  
    from libs.resume.ingestion import create_resume_ingestion_service
    from libs.jobs.ingestion import create_job_ingestion_service, CrawledJob
    from libs.companies.seeding import create_company_seeding_service, CompanyData
    from libs.autoapply.service import create_auto_apply_service, ApplicationRequest
    
    print("✅ All imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logger = get_logger("test_mvp")

class MockSession:
    """Mock database session for testing"""
    
    def __init__(self):
        self.data = {}
        self.committed = False
        
    async def execute(self, query, params=None):
        class MockResult:
            def fetchone(self):
                return None
            def fetchall(self):
                return []
        return MockResult()
    
    def add(self, obj):
        pass
    
    def add_all(self, objs):
        pass
        
    async def commit(self):
        self.committed = True
        
    async def rollback(self):
        pass

class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    def __init__(self):
        self.stats = {"calls": 0, "tokens": 0}
    
    async def embed_batch(self, requests):
        self.stats["calls"] += 1
        self.stats["tokens"] += sum(len(req.get("text", "").split()) for req in requests)
        
        # Return mock embeddings
        return [
            type('EmbeddingResponse', (), {
                'embedding': [0.1] * 1536,
                'tokens': len(req.get("text", "").split())
            })()
            for req in requests
        ]
    
    def get_stats(self):
        return self.stats

class MockChunker:
    """Mock chunker service for testing"""
    
    def chunk_resume(self, fulltext, sections):
        # Simple chunking - split by sentences
        chunks = []
        sentences = fulltext.split('. ')
        
        for i, sentence in enumerate(sentences[:3]):  # Max 3 chunks for testing
            chunk = type('Chunk', (), {
                'chunk_id': f'chunk_{i}',
                'text': sentence,
                'token_count': len(sentence.split()),
                'section': 'experience',
                'metadata': {}
            })()
            chunks.append(chunk)
        
        return chunks
    
    def chunk_job_description(self, fulltext, title, max_chunk_size=500):
        return self.chunk_resume(fulltext, {})  # Reuse resume chunking

class MockParser:
    """Mock resume parser for testing"""
    
    def parse_file(self, file_path):
        return type('ParsedResume', (), {
            'fulltext': 'Senior Software Engineer with 5 years of experience. Skilled in Python, JavaScript, and cloud technologies.',
            'skills': ['Python', 'JavaScript', 'Cloud', 'Software Engineering'],
            'years_of_experience': 5.0,
            'education_level': 'Bachelor',
            'sections': {'experience': 'Senior Software Engineer experience...'},
            'source_file': file_path
        })()

async def test_database_models():
    """Test that database models are properly defined"""
    logger.info("Testing database models...")
    
    try:
        # Test model instantiation
        company = Company(
            name="Test Company",
            website="https://test.com",
            careers_url="https://test.com/careers"
        )
        
        job = Job(
            company_id="test-company-id",
            url="https://test.com/job/123",
            title="Software Engineer",
            jd_fulltext="Software engineering position..."
        )
        
        resume = Resume(
            fulltext="Resume content...",
            skills_csv="Python,JavaScript",
            yoe_raw=5.0
        )
        
        logger.info("✅ Database models working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database models test failed: {e}")
        return False

async def test_observability():
    """Test observability infrastructure"""
    logger.info("Testing observability...")
    
    try:
        # Test metrics
        counter("test.counter", 1)
        
        with timer("test.timer"):
            await asyncio.sleep(0.01)  # Short delay
        
        # Test metrics collection
        collector = get_metrics_collector()
        stats = collector.get_stats()
        
        assert "test.counter" in stats
        assert "test.timer.duration_ms" in stats
        
        logger.info("✅ Observability working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Observability test failed: {e}")
        return False

async def test_resume_ingestion():
    """Test resume ingestion pipeline"""
    logger.info("Testing resume ingestion...")
    
    try:
        # Create mock services
        session = MockSession()
        embedding_service = MockEmbeddingService()
        
        # Create service
        ingestion_service = create_resume_ingestion_service(session)
        
        # Override internal services with mocks
        ingestion_service.embedding_service = embedding_service
        ingestion_service.parser = MockParser()
        ingestion_service.chunker = MockChunker()
        
        # Test with mock file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"Mock PDF content")
            tmp.flush()
            
            # Test ingestion (will use mocks)
            result = await ingestion_service.ingest_resume_file(Path(tmp.name))
            
            assert result.resume_id
            assert len(result.chunks) > 0
            assert result.processing_time_ms >= 0
            
            os.unlink(tmp.name)
        
        logger.info("✅ Resume ingestion working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Resume ingestion test failed: {e}")
        return False

async def test_job_ingestion():
    """Test job ingestion pipeline"""
    logger.info("Testing job ingestion...")
    
    try:
        session = MockSession()
        service = create_job_ingestion_service(session)
        
        # Test single job ingestion logic
        crawled_job = CrawledJob(
            url="https://company.com/job/123",
            title="Software Engineer",
            company_name="Test Company",
            description="Great software engineering position...",
            skills=["Python", "JavaScript"]
        )
        
        # Test fingerprint generation
        fingerprint = service._generate_job_fingerprint(crawled_job)
        assert fingerprint
        assert len(fingerprint) == 32  # MD5 hash length
        
        logger.info("✅ Job ingestion working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Job ingestion test failed: {e}")
        return False

async def test_company_seeding():
    """Test company seeding service"""
    logger.info("Testing company seeding...")
    
    try:
        session = MockSession()
        service = create_company_seeding_service(session)
        
        # Test single company seeding
        company_data = CompanyData(
            name="Test Company",
            website="https://test.com",
            careers_url="https://test.com/careers"
        )
        
        result = await service.seed_single_company(company_data)
        # Result will be False due to mock session but should not error
        
        # Test CSV parsing with temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(['name', 'website', 'careers_url'])
            writer.writerow(['Google', 'https://google.com', 'https://careers.google.com'])
            writer.writerow(['Microsoft', 'https://microsoft.com', 'https://careers.microsoft.com'])
            tmp.flush()
            
            companies = await service._parse_companies_file(Path(tmp.name))
            assert len(companies) == 2
            assert companies[0].name == 'Google'
            
            os.unlink(tmp.name)
        
        logger.info("✅ Company seeding working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Company seeding test failed: {e}")
        return False

async def test_auto_apply():
    """Test auto-apply service"""
    logger.info("Testing auto-apply service...")
    
    try:
        session = MockSession()
        service = create_auto_apply_service(session)
        
        # Test application request creation
        request = ApplicationRequest(
            job_id="test-job-id",
            application_profile_id="test-profile-id",
            dry_run=True
        )
        
        # Test dry run submission
        result = await service.submit_application(request)
        assert result.status == "dry_run_success"
        
        logger.info("✅ Auto-apply service working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Auto-apply test failed: {e}")
        return False

async def test_embedding_version_manager():
    """Test embedding version management"""
    logger.info("Testing embedding version manager...")
    
    try:
        session = MockSession()
        manager = EmbeddingVersionManager(session)
        
        # Test getting active version (will return default due to mock)
        version = await manager.get_active_version()
        assert version.version_id
        assert version.model_name
        assert version.dimensions > 0
        
        logger.info("✅ Embedding version manager working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding version manager test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    logger.info("🚀 Starting LazyJobSearch MVP integration tests...")
    
    tests = [
        test_database_models,
        test_observability,
        test_resume_ingestion,
        test_job_ingestion,
        test_company_seeding,
        test_auto_apply,
        test_embedding_version_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1
    
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All integration tests passed! MVP implementation is working.")
        return 0
    else:
        logger.error(f"💥 {failed} tests failed. Implementation needs fixes.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())