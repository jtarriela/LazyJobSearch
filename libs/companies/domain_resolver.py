"""
Domain Resolution Service

This module implements heuristic-based domain resolution for companies
when only the company name is provided.
"""
from __future__ import annotations
import re
import logging
import time
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DomainResolverService:
    """Service for resolving company domains from company names"""
    
    def __init__(self, timeout: int = 10):
        """Initialize domain resolver service"""
        self.timeout = timeout
        self.session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set a reasonable user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def resolve_domain(self, company_name: str) -> Optional[str]:
        """
        Resolve domain for a company using heuristic approaches
        
        Args:
            company_name: Name of the company
            
        Returns:
            Resolved domain or None if not found
        """
        logger.info(f"Resolving domain for company: {company_name}")
        
        # Try direct heuristic approaches first (faster)
        heuristic_domains = self._generate_heuristic_domains(company_name)
        for domain in heuristic_domains:
            if self._validate_domain(domain):
                logger.info(f"Found valid domain via heuristics: {domain}")
                return domain
        
        # If heuristics fail, try search-based approach (slower, optional)
        search_domain = self._search_based_resolution(company_name)
        if search_domain:
            logger.info(f"Found domain via search: {search_domain}")
            return search_domain
        
        logger.warning(f"Could not resolve domain for {company_name}")
        return None
    
    def _generate_heuristic_domains(self, company_name: str) -> List[str]:
        """Generate candidate domains using heuristic rules"""
        candidates = []
        
        # Clean company name
        clean_name = self._clean_company_name(company_name)
        
        # Pattern 1: company-name.com
        candidates.append(f"{clean_name}.com")
        
        # Pattern 2: companyname.com (no hyphens)  
        no_hyphens = clean_name.replace('-', '')
        if no_hyphens != clean_name:
            candidates.append(f"{no_hyphens}.com")
        
        # Pattern 3: Abbreviations for long names
        if len(clean_name) > 15:
            # Take first letters of each word
            words = clean_name.split('-')
            if len(words) > 1:
                abbrev = ''.join(word[0] for word in words if word)
                candidates.append(f"{abbrev}.com")
        
        # Pattern 4: Common variations
        # Remove common suffixes and try
        suffixes_to_remove = ['inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'company', 'co']
        for suffix in suffixes_to_remove:
            if clean_name.endswith(f'-{suffix}'):
                short_name = clean_name[:-len(f'-{suffix}')]
                candidates.append(f"{short_name}.com")
        
        # Pattern 5: Alternative TLDs for well-known patterns
        candidates.extend([
            f"{clean_name}.io",
            f"{clean_name}.ai", 
            f"{no_hyphens}.io",
            f"{no_hyphens}.ai",
        ])
        
        return candidates
    
    def _clean_company_name(self, name: str) -> str:
        """Clean company name for domain generation"""
        # Convert to lowercase
        clean = name.lower()
        
        # Remove special characters except spaces and hyphens
        clean = re.sub(r'[^\w\s-]', '', clean)
        
        # Replace spaces with hyphens
        clean = re.sub(r'\s+', '-', clean)
        
        # Remove multiple consecutive hyphens
        clean = re.sub(r'-+', '-', clean)
        
        # Remove leading/trailing hyphens
        clean = clean.strip('-')
        
        return clean
    
    def _validate_domain(self, domain: str) -> bool:
        """Validate that a domain exists and serves web content"""
        try:
            # Try HTTPS first
            for protocol in ['https', 'http']:
                url = f"{protocol}://{domain}"
                try:
                    response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
                    
                    # Check if we got a reasonable response
                    if response.status_code < 400:
                        # Additional check: ensure it's not a parking page or error
                        if self._is_valid_website(response, url):
                            return True
                            
                except requests.RequestException:
                    continue  # Try next protocol
                    
        except Exception as e:
            logger.debug(f"Error validating domain {domain}: {e}")
        
        return False
    
    def _is_valid_website(self, response: requests.Response, url: str) -> bool:
        """Check if response indicates a valid company website"""
        # Check for common parking page indicators in headers
        server = response.headers.get('server', '').lower()
        if any(indicator in server for indicator in ['sedo', 'parking', 'domain']):
            return False
        
        # Check content-type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            # If it's not HTML, do a quick GET to check content
            try:
                get_response = self.session.get(url, timeout=self.timeout)
                content = get_response.text.lower()
                
                # Check for parking page indicators
                parking_indicators = [
                    'domain for sale',
                    'this domain may be for sale',
                    'parked domain',
                    'coming soon',
                    'under construction',
                    'godaddy',
                    'namecheap'
                ]
                
                if any(indicator in content for indicator in parking_indicators):
                    return False
                    
                # Check for company-like content
                company_indicators = [
                    'about us',
                    'contact',
                    'careers',
                    'products',
                    'services',
                    'team'
                ]
                
                if any(indicator in content for indicator in company_indicators):
                    return True
                    
            except requests.RequestException:
                pass
        
        return True  # Assume valid if we can't determine otherwise
    
    def _search_based_resolution(self, company_name: str) -> Optional[str]:
        """
        Attempt to resolve domain using search engines
        
        Note: This is a simplified implementation. In production,
        you might want to use search APIs like DuckDuckGo or Bing.
        """
        # For now, we'll skip the search-based approach to avoid
        # dependencies on external search APIs and potential rate limiting
        logger.debug(f"Search-based resolution not implemented for {company_name}")
        return None
    
    def extract_domain_from_url(self, url: str) -> str:
        """Extract clean domain from URL"""
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain