"""
Simple tests for careers discovery service (no pytest dependency)
"""
from unittest.mock import Mock, patch

from libs.scraper.careers_discovery import CareersDiscoveryService


def test_score_career_url():
    """Test URL scoring functionality"""
    service = CareersDiscoveryService()
    
    # Mock response
    mock_response = Mock()
    mock_response.headers = {'content-type': 'text/html'}
    
    # Test scoring for different URL patterns
    careers_score = service._score_career_url("https://company.com/careers", mock_response)
    jobs_score = service._score_career_url("https://company.com/jobs", mock_response)
    hiring_score = service._score_career_url("https://company.com/hiring", mock_response)
    about_score = service._score_career_url("https://company.com/about", mock_response)
    
    assert careers_score > 0.8, f"careers score too low: {careers_score}"
    assert jobs_score > 0.7, f"jobs score too low: {jobs_score}"
    assert hiring_score > 0.6, f"hiring score too low: {hiring_score}"
    assert about_score < 0.5, f"about score too high: {about_score}"
    
    print(f"URL scores - careers: {careers_score}, jobs: {jobs_score}, hiring: {hiring_score}, about: {about_score}")


def test_score_career_link():
    """Test link scoring functionality"""
    service = CareersDiscoveryService()
    
    careers_score = service._score_career_link("careers", "/careers", "/careers")
    jobs_score = service._score_career_link("jobs", "/jobs", "/jobs")
    work_score = service._score_career_link("work with us", "/join", "/join")
    about_score = service._score_career_link("about us", "/about", "/about")
    
    assert careers_score > 0.5, f"careers link score too low: {careers_score}"
    assert jobs_score > 0.5, f"jobs link score too low: {jobs_score}"
    assert work_score > 0.3, f"work link score too low: {work_score}"
    assert about_score == 0.0, f"about link score should be 0: {about_score}"
    
    print(f"Link scores - careers: {careers_score}, jobs: {jobs_score}, work: {work_score}, about: {about_score}")


def test_select_best_candidate():
    """Test candidate selection"""
    service = CareersDiscoveryService()
    
    candidates = [
        ("https://company.com/careers", 0.9),
        ("https://company.com/jobs", 0.8),
        ("https://company.com/about", 0.3)
    ]
    
    best = service._select_best_candidate(candidates)
    assert best == "https://company.com/careers", f"Expected careers URL, got: {best}"
    
    # Test with no good candidates
    poor_candidates = [("https://company.com/about", 0.3)]
    poor_best = service._select_best_candidate(poor_candidates)
    assert poor_best is None, f"Expected None for poor candidates, got: {poor_best}"
    
    print(f"Best candidate: {best}")


def test_is_same_domain():
    """Test domain comparison"""
    service = CareersDiscoveryService()
    
    assert service._is_same_domain("https://company.com", "https://company.com/careers")
    assert service._is_same_domain("https://www.company.com", "https://company.com/careers")
    assert not service._is_same_domain("https://company.com", "https://other.com/careers")
    
    print("Domain comparison tests passed")


def test_careers_discovery_mock():
    """Integration test with mocked network calls"""
    service = CareersDiscoveryService()
    
    with patch.object(service, '_check_robots_txt', return_value=True), \
         patch.object(service, '_probe_common_paths', return_value=[("https://company.com/careers", 0.9)]):
        
        result = service.discover_careers_url("company.com")
        assert result == "https://company.com/careers", f"Expected careers URL, got: {result}"
    
    print(f"Mock discovery result: {result}")


if __name__ == "__main__":
    print("Running careers discovery tests...")
    
    try:
        test_score_career_url()
        print("✅ URL scoring test passed")
        
        test_score_career_link()
        print("✅ Link scoring test passed")
        
        test_select_best_candidate()
        print("✅ Candidate selection test passed")
        
        test_is_same_domain()
        print("✅ Domain comparison test passed")
        
        test_careers_discovery_mock()
        print("✅ Mock integration test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise