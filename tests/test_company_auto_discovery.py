"""
Unit tests for company auto-discovery functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from libs.companies.models import CompanySeed, PortalType, generate_slug
from libs.companies.portal_detection import PortalDetectionService
from libs.companies.domain_resolver import DomainResolverService
from libs.companies.yaml_writer import YamlWriterService
from libs.companies.auto_discovery import CompanyAutoDiscoveryService


class TestCompanyModels:
    """Test Pydantic models and validation"""
    
    def test_generate_slug(self):
        """Test slug generation from company names"""
        assert generate_slug("Anduril Industries") == "anduril-industries"
        assert generate_slug("Meta Inc") == "meta"
        assert generate_slug("Google LLC") == "google"
        assert generate_slug("Microsoft Corporation") == "microsoft-corporation"  # Corporation doesn't get removed
        assert generate_slug("Big-Tech Corp") == "big-tech"
    
    def test_company_seed_validation(self):
        """Test CompanySeed model validation"""
        seed = CompanySeed(
            id="test-company",
            name="Test Company",
            domain="test.com",
            careers={
                "primary_url": "https://test.com/careers",
                "discovered_alternatives": []
            },
            portal={
                "type": PortalType.GREENHOUSE,
                "adapter": "greenhouse_v1",
                "portal_config": {"company_id": "test"}
            }
        )
        
        assert seed.id == "test-company"
        assert seed.domain == "test.com"  # Should normalize
        assert seed.portal.type == PortalType.GREENHOUSE
    
    def test_domain_normalization(self):
        """Test domain normalization in validation"""
        seed = CompanySeed(
            id="test",
            name="Test",
            domain="https://www.test.com/",
            careers={"primary_url": "https://test.com/careers"},
            portal={"type": PortalType.CUSTOM}
        )
        
        assert seed.domain == "test.com"


class TestPortalDetection:
    """Test portal detection service"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = PortalDetectionService()
    
    def test_greenhouse_detection(self):
        """Test Greenhouse portal detection"""
        html_content = """
        <html>
            <body>
                <a href="https://boards.greenhouse.io/testcompany">Apply Now</a>
                <div>Join our team at TestCompany</div>
            </body>
        </html>
        """
        
        portal_type, config, confidence = self.service.detect_portal(html_content, "https://test.com/careers")
        
        assert portal_type == PortalType.GREENHOUSE
        assert config.company_id == "testcompany"
        assert confidence > 0.7
    
    def test_lever_detection(self):
        """Test Lever portal detection"""
        html_content = """
        <html>
            <body>
                <iframe src="https://jobs.lever.co/mycompany"></iframe>
                <p>Career opportunities</p>
            </body>
        </html>
        """
        
        portal_type, config, confidence = self.service.detect_portal(html_content, "https://test.com/jobs")
        
        assert portal_type == PortalType.LEVER
        assert config.company_id == "mycompany"
        assert confidence > 0.7
    
    def test_custom_fallback(self):
        """Test fallback to custom portal type"""
        html_content = """
        <html>
            <body>
                <h1>Join Our Team</h1>
                <p>We are hiring for various positions</p>
            </body>
        </html>
        """
        
        portal_type, config, confidence = self.service.detect_portal(html_content, "https://test.com/careers")
        
        assert portal_type == PortalType.CUSTOM
        assert confidence < 0.5


class TestDomainResolver:
    """Test domain resolution service"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = DomainResolverService()
    
    def test_generate_heuristic_domains(self):
        """Test heuristic domain generation"""
        candidates = self.service._generate_heuristic_domains("Anduril Industries")
        
        assert "anduril-industries.com" in candidates
        assert "andurilindustries.com" in candidates
        assert "ai.com" in candidates  # Abbreviation
        # Note: anduril.com should be generated after removing "industries" suffix
    
    def test_clean_company_name(self):
        """Test company name cleaning"""
        assert self.service._clean_company_name("Anduril Industries Inc.") == "anduril-industries-inc"
        assert self.service._clean_company_name("Meta, Inc") == "meta-inc"
        assert self.service._clean_company_name("Google LLC") == "google-llc"
    
    @patch('requests.Session.head')
    def test_validate_domain_success(self, mock_head):
        """Test successful domain validation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_head.return_value = mock_response
        
        result = self.service._validate_domain("test.com")
        assert result is True
    
    @patch('requests.Session.head')
    def test_validate_domain_failure(self, mock_head):
        """Test domain validation failure"""
        mock_head.side_effect = Exception("Connection error")
        
        result = self.service._validate_domain("nonexistent.com")
        assert result is False


class TestYamlWriter:
    """Test YAML writer service"""
    
    def setup_method(self):
        """Set up test fixtures with temp directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = YamlWriterService(Path(self.temp_dir))
        
        # Create a sample seed
        self.sample_seed = CompanySeed(
            id="test-company",
            name="Test Company",
            domain="test.com",
            careers={"primary_url": "https://test.com/careers"},
            portal={"type": PortalType.GREENHOUSE, "adapter": "greenhouse_v1"}
        )
    
    def test_write_and_read_seed(self):
        """Test writing and reading company seed"""
        # Write seed
        file_path = self.service.write_company_seed(self.sample_seed)
        
        assert file_path.exists()
        assert file_path.name == "test-company.yaml"
        
        # Read seed back
        read_seed = self.service.read_company_seed("test-company")
        
        assert read_seed is not None
        assert read_seed.id == self.sample_seed.id
        assert read_seed.name == self.sample_seed.name
        assert read_seed.domain == self.sample_seed.domain
    
    def test_list_company_seeds(self):
        """Test listing company seeds"""
        # Write a seed first
        self.service.write_company_seed(self.sample_seed)
        
        # List seeds
        companies = self.service.list_company_seeds()
        
        assert "test-company" in companies
        assert companies["test-company"]["name"] == "Test Company"
    
    def test_dry_run_yaml_generation(self):
        """Test dry run YAML generation"""
        yaml_content = self.service.generate_dry_run_yaml(self.sample_seed)
        
        # Parse the YAML to ensure it's valid
        parsed = yaml.safe_load(yaml_content)
        
        assert parsed["id"] == "test-company"
        assert parsed["name"] == "Test Company"
        assert parsed["domain"] == "test.com"
        assert parsed["portal"]["type"] == "greenhouse"


class TestAutoDiscovery:
    """Test main auto-discovery service"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = CompanyAutoDiscoveryService(self.temp_dir)
    
    @patch('libs.companies.auto_discovery.CompanyAutoDiscoveryService._detect_portal_with_retry')
    @patch('libs.scraper.careers_discovery.CareersDiscoveryService.discover_careers_url')
    def test_discover_company_with_domain(self, mock_careers_discovery, mock_portal_detection):
        """Test company discovery with provided domain"""
        # Mock careers discovery
        mock_careers_discovery.return_value = "https://test.com/careers"
        
        # Mock portal detection
        from libs.companies.models import PortalType, PortalConfig
        mock_portal_detection.return_value = (
            PortalType.GREENHOUSE, 
            PortalConfig(company_id="test"),
            0.9
        )
        
        # Run discovery
        seed, metadata = self.service.discover_company("Test Company", "test.com")
        
        # Verify results
        assert seed is not None
        assert seed.id == "test"  # generate_slug("Test Company") -> "test"
        assert seed.name == "Test Company"
        assert seed.domain == "test.com"
        assert seed.portal.type == PortalType.GREENHOUSE
        assert str(seed.careers.primary_url) == "https://test.com/careers"
        
        # Check metadata
        assert metadata["resolved_domain"] == "test.com"
        assert "careers_discovery" in metadata["steps_completed"]
        assert "portal_detection" in metadata["steps_completed"]
    
    def test_dry_run_generation(self):
        """Test dry run seed generation"""
        with patch.object(self.service, 'discover_company') as mock_discover:
            # Mock successful discovery
            sample_seed = CompanySeed(
                id="test",
                name="Test Company", 
                domain="test.com",
                careers={"primary_url": "https://test.com/careers"},
                portal={"type": PortalType.CUSTOM}
            )
            
            mock_discover.return_value = (sample_seed, {"steps_completed": ["all"]})
            
            # Test dry run
            success, message, seed = self.service.create_company_seed(
                "Test Company", 
                domain="test.com",
                dry_run=True
            )
            
            assert success is True
            assert "Generated YAML configuration:" in message
            assert seed is not None
            assert seed.id == "test"