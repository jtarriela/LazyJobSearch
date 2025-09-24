"""
Anduril Industries Career Scraper Adapter

This module provides a scraper adapter specifically designed for Anduril's careers page.
It follows the anti-bot patterns established in the system and implements the 
per-site adapter interface described in the architecture.

NOTE: This scraper requires selenium and beautifulsoup4 dependencies:
    pip install selenium beautifulsoup4 requests
"""
from __future__ import annotations
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Optional selenium imports - gracefully handle missing dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
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
    """Represents a job posting scraped from Anduril"""
    url: str
    title: str
    location: str
    department: str
    description: str
    requirements: List[str]
    scraped_at: datetime
    source: str = "anduril.com"

class AndurilScraper:
    """Scraper adapter for Anduril Industries careers page"""
    
    def __init__(self, proxy_pool: Optional[ProxyPool] = None, rate_limit_ppm: int = 10):
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "AndurilScraper requires selenium. Install with: pip install selenium beautifulsoup4 requests"
            )
            
        self.base_url = "https://www.anduril.com/careers/"
        self.rate_limit_ppm = rate_limit_ppm
        self.session_manager = ScrapeSessionManager(
            proxy_pool or ProxyPool([]),
            FingerprintGenerator()
        )
        self.behavior_sim = HumanBehaviorSimulator()
        self.failure_tracker = get_failure_tracker()
        self.pagination_config = PaginationConfig(
            max_pages=5,  # Reasonable default for job sites
            page_delay=3.0,  # Longer delays for politeness
            retry_failed_pages=True
        )
        
    def _setup_driver(self, fingerprint_profile) -> webdriver.Chrome:
        """Setup Chrome driver with anti-bot measures"""
        options = Options()
        
        # Basic stealth options
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--no-first-run')
        options.add_argument('--no-service-autorun')
        options.add_argument('--password-store=basic')
        
        # Fingerprint customization
        options.add_argument(f'--user-agent={fingerprint_profile.user_agent}')
        options.add_argument(f'--window-size={fingerprint_profile.viewport[0]},{fingerprint_profile.viewport[1]}')
        
        # Create driver
        driver = webdriver.Chrome(options=options)
        
        # Execute stealth JavaScript
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            delete navigator.__proto__.webdriver;
        """)
        
        return driver
        
    @retry_with_backoff(RATE_LIMIT_RETRY_CONFIG)
    def search(self, query_terms: Optional[List[str]] = None) -> List[JobPosting]:
        """
        Search for jobs on Anduril careers page with pagination and retry support
        
        Args:
            query_terms: Optional list of search terms (not used for Anduril as they 
                        typically show all open positions on main page)
        
        Returns:
            List of JobPosting objects
        """
        session = self.session_manager.start()
        all_jobs = []
        
        try:
            driver = self._setup_driver(session.profile)
            
            logger.info(f"Starting Anduril scrape session with proxy: {session.proxy}")
            
            # Navigate to careers page  
            self._navigate_to_page(driver)
            
            # Use pagination to scrape all pages
            page_jobs_generator = paginate_scraper(
                self._scrape_current_page,
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
                pages_scraped, pages_failed, "anduril"
            )
            
            logger.info(f"Completed Anduril scrape: {len(all_jobs)} jobs from {pages_scraped} pages")
            
        except Exception as e:
            error_type = classify_exception(e)
            self.failure_tracker.record_error(
                error_type, str(e), "anduril", "anduril_careers", None
            )
            
            # Check if it's a rate limiting error
            if error_type == CrawlErrorType.RATE_LIMITED:
                self.failure_tracker.record_rate_limit_hit("anduril", "anduril_careers")
            
            logger.error(f"Anduril scrape failed: {e}")
            self.session_manager.finish(session, "error")
            raise
            
        finally:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
                
            session = self.session_manager.finish(session, "success" if all_jobs else "error")
            
        return all_jobs
    
    @retry_with_backoff()
    def _navigate_to_page(self, driver: webdriver.Chrome):
        """Navigate to careers page with retry logic"""
        try:
            driver.get(self.base_url)
            
            # Wait for page to load and simulate human behavior
            time.sleep(self.behavior_sim.sleep_interval())
            
            # Wait for page to be ready
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
        except TimeoutException as e:
            self.failure_tracker.record_error(
                CrawlErrorType.TIMEOUT, f"Page load timeout: {e}", 
                "anduril", "anduril_careers"
            )
            raise
    
    def _scrape_current_page(self, driver: webdriver.Chrome) -> List[JobPosting]:
        """Scrape jobs from the current page"""
        jobs = []
        
        try:
            # Wait for job listings to load (adjust selector based on actual site)
            wait = WebDriverWait(driver, 10)
            
            job_elements = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".job-listing, .career-opportunity"))
            )
            
            logger.info(f"Found {len(job_elements)} job postings on current page")
            
            for i, job_element in enumerate(job_elements):
                try:
                    # Simulate human scroll behavior
                    if i > 0:
                        time.sleep(self.behavior_sim.sleep_interval())
                        
                    job = self._extract_job_info(driver, job_element)
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
                        "anduril", "anduril_careers", f"job_{i}"
                    )
                    logger.warning(f"Failed to extract job {i}: {e}")
                    continue
                    
        except TimeoutException:
            self.failure_tracker.record_error(
                CrawlErrorType.TIMEOUT, "Timeout waiting for job listings to load",
                "anduril", "anduril_careers"  
            )
            logger.warning("Timeout waiting for job listings to load")
        except Exception as e:
            error_type = classify_exception(e)
            self.failure_tracker.record_error(
                error_type, f"Error scraping page: {e}",
                "anduril", "anduril_careers"
            )
            raise
            
        return jobs
    
    def _extract_job_info(self, driver: webdriver.Chrome, job_element) -> Optional[JobPosting]:
        """Extract job information from a job listing element"""
        try:
            # These selectors would need to be customized based on Anduril's actual HTML structure
            
            # Extract basic info
            title_elem = job_element.find_element(By.CSS_SELECTOR, ".job-title, h3, h4")
            title = title_elem.text.strip()
            
            location_elem = job_element.find_element(By.CSS_SELECTOR, ".job-location, .location")
            location = location_elem.text.strip()
            
            # Get job URL - might be a link or need to construct
            try:
                link_elem = job_element.find_element(By.CSS_SELECTOR, "a")
                job_url = link_elem.get_attribute('href')
                if not job_url.startswith('http'):
                    job_url = f"https://www.anduril.com{job_url}"
            except NoSuchElementException:
                # If no direct link, construct URL based on title/id
                job_id = title.lower().replace(' ', '-').replace('/', '-')
                job_url = f"https://www.anduril.com/careers/{job_id}"
            
            # Try to extract department
            try:
                dept_elem = job_element.find_element(By.CSS_SELECTOR, ".department, .team")
                department = dept_elem.text.strip()
            except NoSuchElementException:
                department = "Engineering"  # Default assumption for defense tech
            
            # Get detailed description by clicking through if needed
            description, requirements = self._get_job_details(driver, job_url)
            
            return JobPosting(
                url=job_url,
                title=title,
                location=location,
                department=department,
                description=description,
                requirements=requirements,
                scraped_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract job info: {e}")
            return None
    
    def _get_job_details(self, driver: webdriver.Chrome, job_url: str) -> tuple[str, List[str]]:
        """Get detailed job description and requirements"""
        try:
            # Navigate to job detail page
            current_url = driver.current_url
            driver.get(job_url)
            
            # Wait for page load
            time.sleep(self.behavior_sim.sleep_interval())
            
            # Extract description
            try:
                desc_elem = driver.find_element(By.CSS_SELECTOR, ".job-description, .job-details")
                description = desc_elem.text.strip()
            except NoSuchElementException:
                description = "No description available"
            
            # Extract requirements
            requirements = []
            try:
                req_elements = driver.find_elements(By.CSS_SELECTOR, ".requirements li, .qualifications li")
                requirements = [elem.text.strip() for elem in req_elements if elem.text.strip()]
            except NoSuchElementException:
                pass
            
            # Go back to main page
            driver.get(current_url)
            time.sleep(self.behavior_sim.sleep_interval())
            
            return description, requirements
            
        except Exception as e:
            logger.warning(f"Failed to get job details for {job_url}: {e}")
            return "Description not available", []

    def get_job_count(self) -> int:
        """Get total number of available jobs (quick check)"""
        try:
            session = self.session_manager.start()
            driver = self._setup_driver(session.profile)
            
            driver.get(self.base_url)
            time.sleep(2)
            
            job_elements = driver.find_elements(By.CSS_SELECTOR, ".job-listing, .career-opportunity")
            count = len(job_elements)
            
            driver.quit()
            self.session_manager.finish(session, "success")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to get job count: {e}")
            return 0

# Factory function to create scraper with default config
def create_anduril_scraper(config: Dict[str, Any]) -> AndurilScraper:
    """Create Anduril scraper with configuration"""
    proxy_list = config.get('proxies', [])
    proxy_pool = ProxyPool(proxy_list) if proxy_list else None
    
    rate_limit = config.get('rate_limit_ppm', 10)
    
    return AndurilScraper(proxy_pool=proxy_pool, rate_limit_ppm=rate_limit)