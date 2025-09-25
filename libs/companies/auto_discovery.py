# libs/companies/auto_discovery.py
from .domain_resolver import DomainResolverService
from ..scraper.careers_discovery import CareersDiscoveryService
from .portal_detection import PortalDetectionService
from .yaml_writer import YamlWriterService

class CompanyAutoDiscoveryService:
    """Main service for automatic company seed generation"""

    def __init__(self):
        self.domain_resolver = DomainResolverService()
        self.careers_discovery = CareersDiscoveryService() # Now uses Selenium
        self.portal_detection = PortalDetectionService()
        self.yaml_writer = YamlWriterService()

    def discover_and_create_seed(self, company_name: str, dry_run: bool = True):
        # 1. Resolve domain from name (e.g., "Meta" -> "meta.com")
        domain = self.domain_resolver.resolve_domain(company_name)
        if not domain:
            print(f"Could not resolve domain for {company_name}")
            return

        # 2. Use the browser-powered service to find the careers URL
        careers_url, careers_page_html = self.careers_discovery.discover_with_browser(domain)
        if not careers_url:
            print(f"Could not find careers page for {domain}")
            return

        # 3. Analyze the page HTML to detect the correct portal (ATS)
        portal_type, confidence = self.portal_detection.detect_portal(careers_page_html, careers_url)
        print(f"Detected Portal: {portal_type.value} with {confidence:.0%} confidence")

        # 4. Generate and save the correct YAML configuration file
        seed = self._build_company_seed(company_name, domain, careers_url, portal_type)
        if not dry_run:
            self.yaml_writer.write_company_seed(seed)
            print(f"Successfully created configuration for {company_name}")
        else:
            print(f"DRY RUN: Would write the following config:\n{seed}")