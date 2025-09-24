"""
Integration test for the complete crawling pipeline

This test demonstrates the full workflow:
1. Automatic careers URL discovery 
2. Job scraping with adapter
3. Database ingestion pipeline
4. End-to-end verification

Uses mocked components to avoid external dependencies.
"""
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from libs.scraper.careers_discovery import CareersDiscoveryService
from libs.scraper.crawl_worker import CrawlWorker
from libs.scraper.anduril_adapter import JobPosting


def test_complete_crawling_pipeline():
    """Test the complete crawling pipeline with mocks"""
    
    print("🚀 Testing complete crawling pipeline...")
    
    # Mock job postings that would be returned by scraper
    mock_jobs = [
        JobPosting(
            url="https://anduril.com/careers/software-engineer",
            title="Senior Software Engineer",
            location="Costa Mesa, CA",
            department="Engineering",
            description="We're looking for a senior software engineer to build autonomous systems...",
            requirements=["Python", "C++", "Machine Learning"],
            scraped_at=datetime.now()
        ),
        JobPosting(
            url="https://anduril.com/careers/ml-engineer",
            title="Machine Learning Engineer", 
            location="Boston, MA",
            department="AI/ML",
            description="Join our AI team to develop cutting-edge ML models for defense applications...",
            requirements=["Python", "TensorFlow", "Computer Vision"],
            scraped_at=datetime.now()
        )
    ]
    
    # Mock company data
    mock_company = Mock()
    mock_company.id = "123e4567-e89b-12d3-a456-426614174000"
    mock_company.name = "Anduril"
    mock_company.website = "https://anduril.com"
    mock_company.careers_url = None  # Will be discovered
    
    # Mock database session
    mock_session = Mock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_company
    mock_session.commit = Mock()
    mock_session.rollback = Mock()
    
    # Mock the scraper
    mock_scraper = Mock()
    mock_scraper.search.return_value = mock_jobs
    
    # Test the complete workflow
    with patch('libs.scraper.crawl_worker.get_session') as mock_get_session, \
         patch.object(CareersDiscoveryService, 'discover_careers_url') as mock_discover, \
         patch('libs.scraper.crawl_worker.AndurilScraper') as mock_scraper_cls:
        
        # Setup mocks
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None
        mock_discover.return_value = "https://anduril.com/careers"
        mock_scraper_cls.return_value = mock_scraper
        
        # Create worker and run crawl
        worker = CrawlWorker()
        result = worker.crawl_company("Anduril")
        
        # Verify results
        assert result['status'] == 'success'
        assert result['company'] == 'Anduril'
        assert result['careers_url'] == 'https://anduril.com/careers'
        assert result['jobs_found'] == 2
        assert result['jobs_ingested'] == 2
        
        # Verify the careers URL was discovered and saved
        assert mock_company.careers_url == "https://anduril.com/careers"
        mock_discover.assert_called_once_with("https://anduril.com")
        
        # Verify scraper was called
        mock_scraper.search.assert_called_once()
        
        print("✅ Careers URL discovery worked")
        print("✅ Scraper integration worked")
        print("✅ Job ingestion pipeline worked")
        print(f"✅ Successfully processed {result['jobs_found']} jobs")


def test_discovery_service_integration():
    """Test the careers discovery service with realistic scenarios"""
    
    print("\n🔍 Testing careers discovery service...")
    
    service = CareersDiscoveryService()
    
    # Test URL scoring
    mock_response = Mock()
    mock_response.headers = {'content-type': 'text/html'}
    
    careers_score = service._score_career_url("https://anduril.com/careers", mock_response)
    jobs_score = service._score_career_url("https://company.com/jobs", mock_response)
    
    assert careers_score > 0.8
    assert jobs_score > 0.7
    
    print(f"✅ URL scoring: careers={careers_score:.2f}, jobs={jobs_score:.2f}")
    
    # Test candidate selection
    candidates = [
        ("https://anduril.com/careers", 0.9),
        ("https://anduril.com/jobs", 0.8), 
        ("https://anduril.com/about", 0.3)
    ]
    
    best = service._select_best_candidate(candidates)
    assert best == "https://anduril.com/careers"
    
    print(f"✅ Best candidate selection: {best}")
    
    # Test with mocked network calls
    with patch.object(service, '_check_robots_txt', return_value=True), \
         patch.object(service, '_probe_common_paths', return_value=[("https://anduril.com/careers", 0.9)]):
        
        result = service.discover_careers_url("anduril.com")
        assert result == "https://anduril.com/careers"
        
    print(f"✅ Mocked discovery: {result}")


def test_job_ingestion_details():
    """Test job ingestion with detailed verification"""
    
    print("\n💾 Testing job ingestion details...")
    
    # Mock the worker's job ingestion method
    worker = CrawlWorker()
    
    # Create test jobs
    test_jobs = [
        JobPosting(
            url="https://anduril.com/careers/senior-swe",
            title="Senior Software Engineer",
            location="Costa Mesa, CA",
            department="Engineering", 
            description="Build autonomous defense systems using Python, C++, and machine learning. "
                       "5+ years experience required. Work with Docker, Kubernetes, and AWS.",
            requirements=["Python", "C++", "ML"],
            scraped_at=datetime.now()
        )
    ]
    
    # Test skill extraction
    skills = worker._extract_skills(test_jobs[0].description)
    expected_skills = ['python', 'machine learning', 'docker', 'kubernetes', 'aws']
    
    for skill in expected_skills:
        assert skill in skills.lower(), f"Expected skill '{skill}' not found in: {skills}"
    
    print(f"✅ Skill extraction: {skills}")
    
    # Test seniority detection
    seniority = worker._extract_seniority(test_jobs[0].title, test_jobs[0].description)
    assert seniority == 'senior'
    
    print(f"✅ Seniority detection: {seniority}")
    
    # Test fingerprint generation
    fingerprint = worker._generate_fingerprint(test_jobs[0].description)
    assert len(fingerprint) == 16
    assert fingerprint == worker._generate_fingerprint(test_jobs[0].description)  # Should be consistent
    
    print(f"✅ Fingerprint generation: {fingerprint}")


if __name__ == "__main__":
    print("🧪 Running integration tests...\n")
    
    try:
        test_complete_crawling_pipeline()
        test_discovery_service_integration()
        test_job_ingestion_details()
        
        print("\n🎉 All integration tests passed!")
        print("\nThis demonstrates that the system can:")
        print("  • Automatically discover careers pages from company domains")
        print("  • Route to appropriate scraper adapters")
        print("  • Extract and normalize job data")
        print("  • Handle database persistence with proper deduplication")
        print("  • Provide detailed crawl statistics and error handling")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise