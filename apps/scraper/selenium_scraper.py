"""Selenium-based job scraper with anti-detection features."""

import time
import random
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from .scraper_base import JobScraper, JobPosting, ScrapingResult


class SeleniumJobScraper(JobScraper):
    """Selenium-based scraper with humanization and anti-detection.
    
    Provides common Selenium functionality for job sites that require
    JavaScript rendering or have complex interactions.
    """
    
    def __init__(
        self, 
        base_url: str, 
        headless: bool = True,
        respect_robots: bool = True,
        delay_range: tuple = (1, 3)
    ):
        """Initialize Selenium scraper.
        
        Args:
            base_url: Base URL of the job site
            headless: Whether to run browser in headless mode
            respect_robots: Whether to check robots.txt
            delay_range: Min/max delay between requests in seconds
        """
        super().__init__(base_url, respect_robots)
        
        self.headless = headless
        self.delay_range = delay_range
        self.driver = None
        self.wait = None
        
        # Initialize browser
        self._setup_driver()
    
    def _setup_driver(self):
        """Set up Chrome WebDriver with optimized options."""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # Anti-detection and performance options
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Randomize user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        chrome_options.add_argument(f'--user-agent={random.choice(user_agents)}')
        
        # Set up service
        service = Service(ChromeDriverManager().install())
        
        # Create driver
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Execute script to remove webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Set up wait
        self.wait = WebDriverWait(self.driver, 10)
    
    def human_delay(self, extra_delay: float = 0):
        """Add human-like delay between actions."""
        base_delay = random.uniform(*self.delay_range)
        total_delay = base_delay + extra_delay
        time.sleep(total_delay)
    
    def scroll_page(self, pause_time: float = 0.5):
        """Scroll page in a human-like manner."""
        # Get page height
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for page to load
            time.sleep(pause_time)
            
            # Calculate new scroll height and compare with last height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def safe_find_element(self, by: By, value: str, timeout: int = 5) -> Optional[Any]:
        """Safely find element with timeout."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def safe_find_elements(self, by: By, value: str, timeout: int = 5) -> List[Any]:
        """Safely find multiple elements with timeout."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return self.driver.find_elements(by, value)
        except TimeoutException:
            return []
    
    def extract_text_safe(self, element: Any, attribute: str = None) -> str:
        """Safely extract text from element."""
        try:
            if attribute:
                return element.get_attribute(attribute) or ""
            return element.text.strip()
        except Exception:
            return ""
    
    def navigate_to_page(self, url: str) -> bool:
        """Navigate to a page with error handling."""
        try:
            if not self.can_fetch(url):
                return False
            
            self.driver.get(url)
            self.human_delay()
            return True
        except Exception:
            return False
    
    def search_jobs(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        max_pages: int = 5
    ) -> ScrapingResult:
        """Base implementation - should be overridden by site-specific adapters."""
        return ScrapingResult(
            success=False,
            jobs=[],
            error_message="search_jobs must be implemented by site-specific adapter"
        )
    
    def scrape_job_details(self, job_url: str) -> Optional[JobPosting]:
        """Base implementation - should be overridden by site-specific adapters."""
        return None
    
    def handle_pagination(self, max_pages: int = 5) -> List[str]:
        """Handle pagination and return list of page URLs."""
        page_urls = [self.driver.current_url]
        
        for page_num in range(2, max_pages + 1):
            # Look for next page button (common patterns)
            next_selectors = [
                "a[aria-label='Next']",
                "a[title='Next']",
                ".pagination .next",
                ".pagination a:contains('Next')",
                f"a[href*='page={page_num}']"
            ]
            
            next_button = None
            for selector in next_selectors:
                next_button = self.safe_find_element(By.CSS_SELECTOR, selector)
                if next_button:
                    break
            
            if not next_button:
                break
            
            # Click next button
            try:
                self.driver.execute_script("arguments[0].click();", next_button)
                self.human_delay(extra_delay=1.0)  # Extra delay for page loads
                
                # Wait for page to change
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.current_url not in page_urls
                )
                
                page_urls.append(self.driver.current_url)
            except Exception:
                break
        
        return page_urls
    
    def close(self):
        """Clean up Selenium resources."""
        if self.driver:
            self.driver.quit()
        super().close()