"""
Tests for careers discovery service
"""
import pytest
from unittest.mock import Mock, patch
import requests

from libs.scraper.careers_discovery import CareersDiscoveryService


class TestCareersDiscoveryService:
    """Test suite for CareersDiscoveryService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = CareersDiscoveryService()
    
    def test_score_career_url(self):
        """Test URL scoring functionality"""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/html'}
        
        # Test scoring for different URL patterns
        assert self.service._score_career_url("https://company.com/careers", mock_response) > 0.8
        assert self.service._score_career_url("https://company.com/jobs", mock_response) > 0.7
        assert self.service._score_career_url("https://company.com/hiring", mock_response) > 0.6
        assert self.service._score_career_url("https://company.com/about", mock_response) < 0.5
    
    def test_score_career_link(self):
        """Test link scoring functionality"""
        # Test link text scoring
        assert self.service._score_career_link("careers", "/careers", "/careers") > 0.5
        assert self.service._score_career_link("jobs", "/jobs", "/jobs") > 0.5
        assert self.service._score_career_link("work with us", "/join", "/join") > 0.3
        assert self.service._score_career_link("about us", "/about", "/about") == 0.0
    
    def test_select_best_candidate(self):
        """Test candidate selection"""
        candidates = [
            ("https://company.com/careers", 0.9),
            ("https://company.com/jobs", 0.8),
            ("https://company.com/about", 0.3)
        ]
        
        best = self.service._select_best_candidate(candidates)
        assert best == "https://company.com/careers"
        
        # Test with no good candidates
        poor_candidates = [("https://company.com/about", 0.3)]
        assert self.service._select_best_candidate(poor_candidates) is None
    
    def test_is_same_domain(self):
        """Test domain comparison"""
        assert self.service._is_same_domain(
            "https://company.com", 
            "https://company.com/careers"
        )
        assert self.service._is_same_domain(
            "https://www.company.com", 
            "https://company.com/careers"
        )
        assert not self.service._is_same_domain(
            "https://company.com", 
            "https://other.com/careers"
        )
    
    @patch('requests.Session.head')
    def test_probe_common_paths(self, mock_head):
        """Test probing common career paths"""
        # Mock successful response for /careers path
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_head.return_value = mock_response
        
        candidates = self.service._probe_common_paths("https://company.com")
        
        # Should find candidates for common paths
        assert len(candidates) > 0
        assert any("careers" in url for url, score in candidates)
    
    @patch('requests.Session.get')
    def test_parse_homepage_links(self, mock_get):
        """Test parsing homepage for career links"""
        # Mock HTML content with career links
        mock_response = Mock()
        mock_response.content = """
        <html>
            <body>
                <a href="/careers">Careers</a>
                <a href="/jobs">Jobs</a>
                <a href="https://external.com/jobs">External Jobs</a>
                <a href="/about">About Us</a>
            </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        candidates = self.service._parse_homepage_links("https://company.com")
        
        # Should find internal career links but not external ones
        career_urls = [url for url, score in candidates]
        assert "https://company.com/careers" in career_urls
        assert "https://company.com/jobs" in career_urls
        assert "https://external.com/jobs" not in career_urls
        assert "https://company.com/about" not in career_urls


def test_careers_discovery_mock():
    """Integration test with mocked network calls"""
    service = CareersDiscoveryService()
    
    with patch.object(service, '_check_robots_txt', return_value=True), \
         patch.object(service, '_probe_common_paths', return_value=[("https://company.com/careers", 0.9)]):
        
        result = service.discover_careers_url("company.com")
        assert result == "https://company.com/careers"


if __name__ == "__main__":
    # Simple test runner for development
    test_service = TestCareersDiscoveryService()
    test_service.setup_method()
    
    print("Running careers discovery tests...")
    test_service.test_score_career_url()
    print("✅ URL scoring test passed")
    
    test_service.test_score_career_link()
    print("✅ Link scoring test passed")
    
    test_service.test_select_best_candidate()
    print("✅ Candidate selection test passed")
    
    test_service.test_is_same_domain()
    print("✅ Domain comparison test passed")
    
    test_careers_discovery_mock()
    print("✅ Mock integration test passed")
    
    print("\n🎉 All tests passed!")