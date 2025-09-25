from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urljoin

from ..scraper.careers_discovery import CareersDiscoveryService
from .domain_resolver import DomainResolverService
from .portal_detection import PortalDetectionService
from .yaml_writer import YamlWriterService
from .models import CompanySeed, Careers, Portal, CrawlerConfig, Confidence, generate_slug, PortalType, PortalConfig

logger = logging.getLogger(__name__)


class CompanyAutoDiscoveryService:
    """Main service for automatic company seed generation"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize auto-discovery service with all components"""
        self.domain_resolver = DomainResolverService()
        self.careers_discovery = CareersDiscoveryService()
        self.portal_detection = PortalDetectionService()
        self.yaml_writer = YamlWriterService(config_dir)
    
    def discover_company(
        self, 
        company_name: str, 
        domain: Optional[str] = None
    ) -> Tuple[Optional[CompanySeed], Dict[str, Any]]:
        """
        Automatically discover and generate company seed configuration
        
        Args:
            company_name: Name of the company
            domain: Optional domain (if not provided, will be resolved)
            
        Returns:
            Tuple of (CompanySeed or None, discovery_metadata)
        """
        logger.info(f"Starting auto-discovery for company: {company_name}")
        
        discovery_metadata = {
            'company_name': company_name,
            'provided_domain': domain,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Resolve domain if not provided
            if not domain:
                logger.info("Resolving company domain...")
                domain = self.domain_resolver.resolve_domain(company_name)
                discovery_metadata['steps_completed'].append('domain_resolution')
                
                if not domain:
                    error_msg = f"Could not resolve domain for company: {company_name}"
                    logger.error(error_msg)
                    discovery_metadata['errors'].append(error_msg)
                    return None, discovery_metadata
                    
                discovery_metadata['resolved_domain'] = domain
            else:
                discovery_metadata['resolved_domain'] = domain
            
            # Step 2: Discover careers page
            logger.info(f"Discovering careers page for domain: {domain}")
            careers_url = self.careers_discovery.discover_careers_url(f"https://{domain}")
            discovery_metadata['steps_completed'].append('careers_discovery')
            
            if not careers_url:
                error_msg = f"Could not find careers page for domain: {domain}"
                logger.error(error_msg)
                discovery_metadata['errors'].append(error_msg)
                return None, discovery_metadata
            
            discovery_metadata['careers_url'] = careers_url
            
            # Step 3: Detect portal type
            logger.info(f"Detecting portal type for careers page: {careers_url}")
            portal_type, portal_config, portal_confidence = self._detect_portal_with_retry(careers_url)
            discovery_metadata['steps_completed'].append('portal_detection')
            discovery_metadata['portal_detection'] = {
                'type': portal_type.value if portal_type else 'unknown',
                'confidence': portal_confidence,
                'config': portal_config.model_dump() if portal_config else None
            }
            
            # Step 4: Generate company seed
            logger.info("Generating company seed configuration...")
            seed = self._build_company_seed(
                company_name=company_name,
                domain=domain,
                careers_url=careers_url,
                portal_type=portal_type,
                portal_config=portal_config,
                confidence_scores={
                    'careers_url': 0.8,  # Default confidence for discovered URLs
                    'portal_detection': portal_confidence
                }
            )
            discovery_metadata['steps_completed'].append('seed_generation')
            
            logger.info(f"Successfully generated company seed for {company_name}")
            return seed, discovery_metadata
            
        except Exception as e:
            error_msg = f"Error during auto-discovery: {str(e)}"
            logger.error(error_msg, exc_info=True)
            discovery_metadata['errors'].append(error_msg)
            return None, discovery_metadata
    
    def create_company_seed(
        self, 
        company_name: str, 
        domain: Optional[str] = None,
        dry_run: bool = False,
        overwrite: bool = False
    ) -> Tuple[bool, str, Optional[CompanySeed]]:
        """
        Create and optionally save company seed
        
        Args:
            company_name: Name of the company
            domain: Optional domain
            dry_run: If True, generate but don't save
            overwrite: If True, overwrite existing files
            
        Returns:
            Tuple of (success, message, seed)
        """
        logger.info(f"Creating company seed for {company_name} (dry_run={dry_run})")
        
        # Discover company configuration
        seed, metadata = self.discover_company(company_name, domain)
        
        if not seed:
            errors = '; '.join(metadata.get('errors', []))
            return False, f"Failed to discover company configuration: {errors}", None
        
        if dry_run:
            yaml_content = self.yaml_writer.generate_dry_run_yaml(seed)
            return True, f"Generated YAML configuration:\n\n{yaml_content}", seed
        
        # Save the seed
        try:
            file_path = self.yaml_writer.write_company_seed(seed, overwrite=overwrite)
            return True, f"Company seed saved to: {file_path}", seed
            
        except FileExistsError:
            return False, f"Company seed already exists for {seed.id}. Use --update to overwrite.", seed
        except Exception as e:
            return False, f"Failed to save company seed: {str(e)}", seed
    
    def _detect_portal_with_retry(self, careers_url: str) -> Tuple[Optional[PortalType], Optional[PortalConfig], float]:
        """Detect portal type with HTML content fetching"""
        try:
            # Fetch careers page content
            response = self.careers_discovery.session.get(careers_url, timeout=10)
            response.raise_for_status()
            
            # Detect portal from HTML content
            return self.portal_detection.detect_portal(response.text, careers_url)
            
        except Exception as e:
            logger.warning(f"Could not fetch careers page content for portal detection: {e}")
            # Return default values if we can't fetch content
            return PortalType.CUSTOM, PortalConfig(), 0.1
    
    def _build_company_seed(
        self,
        company_name: str,
        domain: str,
        careers_url: str,
        portal_type: PortalType,
        portal_config: PortalConfig,
        confidence_scores: Dict[str, float]
    ) -> CompanySeed:
        """Build CompanySeed object from discovery results"""
        
        # Generate company slug
        company_slug = generate_slug(company_name)
        
        # Build careers configuration
        careers = Careers(
            primary_url=careers_url,
            discovered_alternatives=[]
        )
        
        # Build portal configuration
        portal = Portal(
            type=portal_type,
            adapter=self.portal_detection.get_adapter_name(portal_type),
            portal_config=portal_config
        )
        
        # Build crawler configuration
        crawler = CrawlerConfig(
            enabled=True,
            start_urls=[careers_url],
            heuristics={
                'job_link_keywords': ['engineer', 'developer', 'software', 'technical'],
                'pagination_keywords': ['next', 'more jobs', 'load more']
            }
        )
        
        # Build confidence metadata
        confidence = Confidence(
            careers_url=confidence_scores['careers_url'],
            portal_detection=confidence_scores['portal_detection']
        )
        
        # Create the seed
        seed = CompanySeed(
            id=company_slug,
            name=company_name,
            domain=domain,
            careers=careers,
            portal=portal,
            crawler=crawler,
            metadata={
                'confidence': confidence.model_dump(),
                'discovery_method': 'auto',
                'notes': f'Auto-generated via company add --auto'
            },
            notes=f'Auto-generated via company add --auto'
        )
        
        return seed