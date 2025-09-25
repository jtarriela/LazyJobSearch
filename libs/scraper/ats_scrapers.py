"""
Generic ATS (Applicant Tracking System) Scrapers

This module provides generic scraper implementations for popular ATS platforms
like Greenhouse, Lever, and Workday. These scrapers can be configured for different
companies using the same underlying ATS platform.
"""
from __future__ import annotations
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import re

# Optional selenium imports - gracefully handle missing dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.firefox.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Selenium not available: {e}. Install with 'pip install selenium'")
    SELENIUM_AVAILABLE = False
    # Create stub classes to prevent import errors
    class webdriver:
        class Chrome: pass
    class TimeoutException(Exception): pass
    class NoSuchElementException(Exception): pass

from libs.scraper.anti_bot import (
    ScrapeSessionManager, 
    ProxyPool, 
    FingerprintGenerator,
    HumanBehaviorSimulator
)
from libs.scraper.retry_logic import retry_with_backoff, RATE_LIMIT_RETRY_CONFIG
from libs.scraper.pagination import paginate_scraper, PaginationConfig, StandardPaginationHandler
from libs.scraper.failure_metrics import get_failure_tracker, classify_exception, CrawlErrorType

logger = logging.getLogger(__name__)


@dataclass
class JobPosting:
    """Standard job posting data structure for ATS scrapers"""
    url: str
    title: str
    location: str
    description: str
    scraped_at: datetime
    department: Optional[str] = None
    job_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    seniority: Optional[str] = None
    salary_range: Optional[str] = None
    company_name: Optional[str] = None


class ATSScraper(ABC):
    """Abstract base class for ATS scrapers"""
    
    def __init__(self, proxy_pool: Optional[ProxyPool] = None, rate_limit_ppm: int = 10):
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "ATS scrapers require selenium. Install with: pip install selenium beautifulsoup4 requests"
            )
        
        self.rate_limit_ppm = rate_limit_ppm
        self.session_manager = ScrapeSessionManager(
            proxy_pool or ProxyPool([]),
            FingerprintGenerator()
        )
        self.behavior_sim = HumanBehaviorSimulator()
        self.failure_tracker = get_failure_tracker()
        self.pagination_config = PaginationConfig(
            max_pages=10,  # Most job sites have reasonable pagination
            page_delay=2.5,
            retry_failed_pages=True
        )
    
    @abstractmethod
    def get_base_url(self, company_domain: str) -> str:
        """Get the base careers URL for a company domain"""
        pass
    
    @abstractmethod
    def get_job_listing_selectors(self) -> Dict[str, str]:
        """Get CSS selectors for job listing elements"""
        pass
    
    @abstractmethod
    def get_job_detail_selectors(self) -> Dict[str, str]:
        """Get CSS selectors for job detail elements"""
        pass
    
    def search(self, company_domain: str, query_terms: Optional[List[str]] = None) -> List[JobPosting]:
        """Search for jobs on the ATS platform"""
        if not company_domain:
            raise ValueError("company_domain is required for ATS scrapers")
        
        base_url = self.get_base_url(company_domain)
        session = self.session_manager.start()
        all_jobs = []
        
        try:
            driver = self._setup_driver(session.profile)
            
            logger.info(f"Starting {self.__class__.__name__} scrape session for {company_domain}")
            
            # Navigate to careers page
            self._navigate_to_page(driver, base_url)
            
            # Use pagination to scrape all pages
            page_jobs_generator = paginate_scraper(
                lambda d: self._scrape_current_page(d, company_domain),
                driver,
                self.pagination_config,
                StandardPaginationHandler()
            )
            
            pages_scraped = 0
            pages_failed = 0
            
            for page_jobs in page_jobs_generator:
                if page_jobs:
                    all_jobs.extend(page_jobs)
                    pages_scraped += 1
                    logger.info(f"Scraped {len(page_jobs)} jobs from page {pages_scraped}")
                else:
                    pages_failed += 1
                    logger.warning(f"No jobs found on page {pages_scraped + pages_failed}")
            
            # Record pagination metrics
            self.failure_tracker.record_pagination_metrics(
                pages_scraped, pages_failed, company_domain
            )
            
            logger.info(f"Completed {self.__class__.__name__} scrape: {len(all_jobs)} jobs from {pages_scraped} pages")
            
        except Exception as e:
            error_type = classify_exception(e)
            self.failure_tracker.record_error(
                error_type, str(e), company_domain, self.__class__.__name__.lower(), None
            )
            
            # Check if it's a rate limiting error
            if error_type == CrawlErrorType.RATE_LIMITED:
                self.failure_tracker.record_rate_limit_hit(company_domain, self.__class__.__name__.lower())
            
            logger.error(f"{self.__class__.__name__} scrape failed: {e}")
            self.session_manager.finish(session, "error")
            raise
        
        finally:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
            
            self.session_manager.finish(session, "success" if all_jobs else "error")
        
        return all_jobs
    
    def _setup_driver(self, fingerprint_profile) -> webdriver.Firefox:
        """Setup Firefox driver with anti-bot measures"""
        options = Options()
        options.add_argument("--headless")  # <-- ADD THIS LINE
        
        # Basic stealth options
        options.add_argument('--disable-blink-features=AutomationControlled')
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--no-first-run')
        options.add_argument('--no-service-autorun')
        options.add_argument('--password-store=basic')
        
        # Fingerprint customization
        options.add_argument(f'--user-agent={fingerprint_profile.user_agent}')
        options.add_argument(f'--window-size={fingerprint_profile.viewport[0]},{fingerprint_profile.viewport[1]}')
        
        # Create driver
        driver = webdriver.Firefox(options=options)

        
        # Execute stealth JavaScript
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            delete navigator.__proto__.webdriver;
        """)
        
        return driver
    
    @retry_with_backoff()
    def _navigate_to_page(self, driver: webdriver.Firefox, url: str):
        """Navigate to page with retry logic"""
        try:
            driver.get(url)
            
            # Wait for page to load and simulate human behavior
            time.sleep(self.behavior_sim.sleep_interval())
            
            # Wait for page to be ready
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
        except TimeoutException as e:
            self.failure_tracker.record_error(
                CrawlErrorType.TIMEOUT, f"Page load timeout: {e}", 
                url, self.__class__.__name__.lower()
            )
            raise
    
    def _scrape_current_page(self, driver: webdriver.Firefox, company_domain: str) -> List[JobPosting]:
        """Scrape jobs from the current page"""
        jobs = []
        selectors = self.get_job_listing_selectors()
        
        try:
            # Wait for job listings to load
            wait = WebDriverWait(driver, 10)
            
            job_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, selectors['job_container']))
            )
            
            logger.info(f"Found {len(job_elements)} job postings on current page")
            
            for i, job_element in enumerate(job_elements):
                try:
                    # Simulate human scroll behavior
                    if i > 0:
                        time.sleep(self.behavior_sim.sleep_interval())
                    
                    job = self._extract_job_info(driver, job_element, company_domain)
                    if job:
                        jobs.append(job)
                    
                    # Rate limiting
                    if i > 0 and i % 5 == 0:
                        rate_limit_delay = 60 / self.rate_limit_ppm
                        logger.debug(f"Rate limiting: sleeping {rate_limit_delay:.1f}s")
                        time.sleep(rate_limit_delay)
                
                except Exception as e:
                    error_type = classify_exception(e)
                    self.failure_tracker.record_error(
                        error_type, f"Failed to extract job {i}: {e}",
                        company_domain, self.__class__.__name__.lower(), f"job_{i}"
                    )
                    logger.warning(f"Failed to extract job {i}: {e}")
                    continue
        
        except TimeoutException:
            self.failure_tracker.record_error(
                CrawlErrorType.TIMEOUT, "Timeout waiting for job listings to load",
                company_domain, self.__class__.__name__.lower()
            )
            logger.warning("Timeout waiting for job listings to load")
        except Exception as e:
            error_type = classify_exception(e)
            self.failure_tracker.record_error(
                error_type, f"Error scraping page: {e}",
                company_domain, self.__class__.__name__.lower()
            )
            raise
        
        return jobs
    
    def _extract_job_info(self, driver: webdriver.Firefox, job_element, company_domain: str) -> Optional[JobPosting]:
        """Extract job information from a job listing element"""
        try:
            selectors = self.get_job_listing_selectors()
            
            # Extract basic info
            title = self._safe_extract_text(job_element, selectors['title'])
            location = self._safe_extract_text(job_element, selectors['location'])
            
            # Get job URL
            job_url = self._extract_job_url(job_element, selectors, company_domain)
            if not job_url:
                logger.warning("Could not extract job URL, skipping")
                return None
            
            # Extract optional fields
            department = self._safe_extract_text(job_element, selectors.get('department'))
            job_type = self._safe_extract_text(job_element, selectors.get('job_type'))
            
            # Get job description (might require clicking through to detail page)
            description = self._extract_job_description(driver, job_element, job_url, selectors)
            
            return JobPosting(
                url=job_url,
                title=title or "Unknown Position",
                location=location or "Unknown Location",
                description=description or "",
                department=department,
                job_type=job_type,
                scraped_at=datetime.now(),
                company_name=company_domain
            )
        
        except Exception as e:
            logger.warning(f"Error extracting job info: {e}")
            return None
    
    def _safe_extract_text(self, element, selector: Optional[str]) -> Optional[str]:
        """Safely extract text from an element using a selector"""
        if not selector:
            return None
        
        try:
            sub_element = element.find_element(By.CSS_SELECTOR, selector)
            return sub_element.text.strip() if sub_element.text else None
        except NoSuchElementException:
            return None
        except Exception as e:
            logger.debug(f"Error extracting text with selector '{selector}': {e}")
            return None
    
    def _extract_job_url(self, job_element, selectors: Dict[str, str], company_domain: str) -> Optional[str]:
        """Extract job URL from job element"""
        try:
            if 'link' in selectors:
                link_elem = job_element.find_element(By.CSS_SELECTOR, selectors['link'])
                href = link_elem.get_attribute('href')
                if href:
                    # Ensure absolute URL
                    if href.startswith('http'):
                        return href
                    elif href.startswith('/'):
                        base_url = self.get_base_url(company_domain)
                        base_domain = re.match(r'https?://[^/]+', base_url)
                        if base_domain:
                            return base_domain.group(0) + href
                    return href
        except NoSuchElementException:
            pass
        except Exception as e:
            logger.debug(f"Error extracting job URL: {e}")
        
        return None
    
    def _extract_job_description(self, driver: webdriver.Firefox, job_element, job_url: str, 
                                selectors: Dict[str, str]) -> str:
        """Extract job description (may require navigation to detail page)"""
        # Try to get description from listing page first
        if 'description' in selectors:
            description = self._safe_extract_text(job_element, selectors['description'])
            if description and len(description) > 100:  # Substantial description
                return description
        
        # If no substantial description on listing page, try to navigate to detail page
        # This is optional and may be disabled for performance
        return self._get_description_from_listing(job_element, selectors)
    
    def _get_description_from_listing(self, job_element, selectors: Dict[str, str]) -> str:
        """Get description from job listing without navigation"""
        description_parts = []
        
        # Try multiple selectors for description content
        for selector_name in ['description', 'summary', 'snippet']:
            if selector_name in selectors:
                text = self._safe_extract_text(job_element, selectors[selector_name])
                if text:
                    description_parts.append(text)
        
        return '\n'.join(description_parts) if description_parts else ""


class GreenhouseScraper(ATSScraper):
    """Scraper for Greenhouse ATS platform"""
    
    def get_base_url(self, company_domain: str) -> str:
        """Greenhouse URLs typically follow the pattern: https://boards.greenhouse.io/company"""
        # Handle both full domain and company name
        if company_domain.startswith('http'):
            return company_domain
        elif '.' in company_domain:
            # Extract company name from domain
            company_name = company_domain.split('.')[0]
            return f"https://boards.greenhouse.io/{company_name}"
        else:
            # Assume it's already the company name
            return f"https://boards.greenhouse.io/{company_domain}"
    
    def get_job_listing_selectors(self) -> Dict[str, str]:
        """CSS selectors for Greenhouse job listings"""
        return {
            'job_container': '.opening',
            'title': 'a',
            'location': '.location',
            'department': '.department',
            'link': 'a',
            'description': '.content'
        }
    
    def get_job_detail_selectors(self) -> Dict[str, str]:
        """CSS selectors for Greenhouse job detail pages"""
        return {
            'title': '#header h1',
            'location': '.location',
            'department': '.department',
            'description': '#content .section-wrapper'
        }


class LeverScraper(ATSScraper):
    """Scraper for Lever ATS platform"""
    
    def get_base_url(self, company_domain: str) -> str:
        """Lever URLs typically follow the pattern: https://jobs.lever.co/company"""
        if company_domain.startswith('http'):
            return company_domain
        elif '.' in company_domain:
            # Extract company name from domain
            company_name = company_domain.split('.')[0]
            return f"https://jobs.lever.co/{company_name}"
        else:
            # Assume it's already the company name
            return f"https://jobs.lever.co/{company_domain}"
    
    def get_job_listing_selectors(self) -> Dict[str, str]:
        """CSS selectors for Lever job listings"""
        return {
            'job_container': '.posting',
            'title': '.posting-name h5',
            'location': '.posting-categories .location',
            'department': '.posting-categories .department',  
            'job_type': '.posting-categories .commitment',
            'link': 'a.posting-btn-submit',
            'description': '.posting-description'
        }
    
    def get_job_detail_selectors(self) -> Dict[str, str]:
        """CSS selectors for Lever job detail pages"""
        return {
            'title': '.posting-headline h2',
            'location': '.posting-headline .location',
            'department': '.posting-headline .department',
            'description': '.section-wrapper .posting-description'
        }


class WorkdayScraper(ATSScraper):
    """Scraper for Workday ATS platform"""
    
    def get_base_url(self, company_domain: str) -> str:
        """Workday URLs are more complex and company-specific"""
        if company_domain.startswith('http'):
            return company_domain
        else:
            # Workday URLs vary significantly, may need company-specific configuration
            logger.warning(f"Workday URLs are company-specific. Please provide full URL for {company_domain}")
            return f"https://{company_domain}/jobs"
    
    def get_job_listing_selectors(self) -> Dict[str, str]:
        """CSS selectors for Workday job listings"""
        return {
            'job_container': '[data-automation-id="jobPostingItem"]',
            'title': '[data-automation-id="jobPostingTitle"]',
            'location': '[data-automation-id="jobPostingLocation"]', 
            'link': '[data-automation-id="jobPostingTitle"] a',
            'description': '[data-automation-id="jobPostingDescription"]'
        }
    
    def get_job_detail_selectors(self) -> Dict[str, str]:
        """CSS selectors for Workday job detail pages"""
        return {
            'title': '[data-automation-id="jobPostingHeader"]',
            'location': '[data-automation-id="locations"]',
            'description': '[data-automation-id="jobPostingDescription"]'
        }