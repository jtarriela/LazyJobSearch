"""Site-specific scraping adapters for major job boards."""

import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, quote
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException

from .selenium_scraper import SeleniumJobScraper
from .scraper_base import JobPosting, ScrapingResult


class LinkedInAdapter(SeleniumJobScraper):
    """LinkedIn Jobs scraper adapter.
    
    Handles LinkedIn's job search and detail extraction.
    Note: LinkedIn has strict anti-bot measures, use carefully.
    """
    
    def __init__(self, headless: bool = True):
        super().__init__(
            base_url="https://www.linkedin.com",
            headless=headless,
            delay_range=(2, 5)  # Longer delays for LinkedIn
        )
    
    def search_jobs(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        max_pages: int = 3  # LinkedIn limits
    ) -> ScrapingResult:
        """Search LinkedIn jobs."""
        try:
            # Build search URL
            query = " ".join(keywords)
            search_url = f"https://www.linkedin.com/jobs/search/?keywords={quote(query)}"
            
            if location:
                search_url += f"&location={quote(location)}"
            
            # Navigate to search page
            if not self.navigate_to_page(search_url):
                return ScrapingResult(
                    success=False,
                    jobs=[],
                    error_message="Could not access LinkedIn jobs page"
                )
            
            jobs = []
            
            # Get job listings from current page
            job_cards = self.safe_find_elements(
                By.CSS_SELECTOR, 
                ".job-search-card, .jobs-search__results-list li"
            )
            
            for card in job_cards[:20]:  # Limit to avoid rate limiting
                job = self._extract_job_from_card(card)
                if job and self.validate_job_posting(job):
                    jobs.append(job)
                
                self.human_delay(0.5)
            
            return ScrapingResult(
                success=True,
                jobs=self.deduplicate_jobs(jobs),
                pages_scraped=1,
                jobs_found=len(jobs)
            )
            
        except Exception as e:
            return ScrapingResult(
                success=False,
                jobs=[],
                error_message=f"Error scraping LinkedIn: {str(e)}"
            )
    
    def _extract_job_from_card(self, card) -> Optional[JobPosting]:
        """Extract job info from LinkedIn job card."""
        try:
            # Get job URL
            link_element = card.find_element(By.CSS_SELECTOR, "a[href*='/jobs/view/']")
            job_url = link_element.get_attribute("href")
            
            # Extract basic info from card
            title_element = card.find_element(By.CSS_SELECTOR, ".job-search-card__title, h3")
            title = self.extract_text_safe(title_element)
            
            company_element = card.find_element(By.CSS_SELECTOR, ".job-search-card__subtitle, h4")
            company = self.extract_text_safe(company_element)
            
            location_element = card.find_element(By.CSS_SELECTOR, ".job-search-card__location")
            location = self.extract_text_safe(location_element)
            
            # Basic job posting (will be enriched by scrape_job_details)
            return JobPosting(
                url=job_url,
                title=title,
                company=company,
                location=location,
                description="",  # To be filled by detail scraping
                metadata={"source": "linkedin"}
            )
            
        except Exception:
            return None
    
    def scrape_job_details(self, job_url: str) -> Optional[JobPosting]:
        """Scrape detailed LinkedIn job posting."""
        try:
            if not self.navigate_to_page(job_url):
                return None
            
            # Wait for job details to load
            self.human_delay(2.0)
            
            # Extract detailed information
            title = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".jobs-unified-top-card__job-title h1")
            )
            
            company = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".jobs-unified-top-card__company-name")
            )
            
            location = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".jobs-unified-top-card__bullet")
            )
            
            # Job description
            description_element = self.safe_find_element(
                By.CSS_SELECTOR, 
                ".jobs-description-content__text, .jobs-box__html-content"
            )
            description = self.extract_text_safe(description_element)
            
            # Extract seniority level
            seniority = self._extract_seniority_from_description(description)
            
            return JobPosting(
                url=job_url,
                title=title,
                company=company,
                location=location,
                description=description,
                seniority=seniority,
                metadata={"source": "linkedin", "scraped_details": True}
            )
            
        except Exception:
            return None
    
    def _extract_seniority_from_description(self, description: str) -> Optional[str]:
        """Extract seniority level from job description."""
        seniority_patterns = {
            "entry": ["entry level", "junior", "associate", "intern", "graduate"],
            "mid": ["mid level", "intermediate", "experienced", "3-5 years"],
            "senior": ["senior", "lead", "principal", "5+ years", "7+ years"],
            "executive": ["director", "vp", "executive", "head of", "chief"]
        }
        
        description_lower = description.lower()
        
        for level, patterns in seniority_patterns.items():
            if any(pattern in description_lower for pattern in patterns):
                return level
        
        return None


class GreenhouseAdapter(SeleniumJobScraper):
    """Greenhouse job board scraper adapter.
    
    Many companies use Greenhouse for their job postings.
    This adapter can work with multiple company Greenhouse instances.
    """
    
    def __init__(self, company_slug: str, headless: bool = True):
        """Initialize Greenhouse adapter.
        
        Args:
            company_slug: Company identifier for Greenhouse (e.g., 'stripe', 'airbnb')
        """
        self.company_slug = company_slug
        base_url = f"https://boards.greenhouse.io/{company_slug}"
        
        super().__init__(
            base_url=base_url,
            headless=headless,
            delay_range=(1, 2)
        )
    
    def search_jobs(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        max_pages: int = 5
    ) -> ScrapingResult:
        """Search Greenhouse job board."""
        try:
            # Navigate to company jobs page
            jobs_url = f"{self.base_url}/jobs"
            
            if not self.navigate_to_page(jobs_url):
                return ScrapingResult(
                    success=False,
                    jobs=[],
                    error_message=f"Could not access {self.company_slug} jobs page"
                )
            
            # Wait for jobs to load
            self.human_delay(2.0)
            
            jobs = []
            
            # Get all job postings
            job_links = self.safe_find_elements(
                By.CSS_SELECTOR, 
                ".opening, .job-post, a[href*='/jobs/']"
            )
            
            for link in job_links:
                job_url = link.get_attribute("href")
                if job_url and "/jobs/" in job_url:
                    # Extract basic info or scrape details
                    job = self.scrape_job_details(job_url)
                    if job and self.validate_job_posting(job):
                        jobs.append(job)
                
                self.human_delay(0.5)
            
            # Filter by keywords if provided
            if keywords:
                jobs = self.filter_jobs_by_keywords(jobs, keywords)
            
            return ScrapingResult(
                success=True,
                jobs=self.deduplicate_jobs(jobs),
                pages_scraped=1,
                jobs_found=len(jobs)
            )
            
        except Exception as e:
            return ScrapingResult(
                success=False,
                jobs=[],
                error_message=f"Error scraping Greenhouse: {str(e)}"
            )
    
    def scrape_job_details(self, job_url: str) -> Optional[JobPosting]:
        """Scrape detailed Greenhouse job posting."""
        try:
            if not self.navigate_to_page(job_url):
                return None
            
            # Wait for content to load
            self.human_delay(1.0)
            
            # Extract job information
            title = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".app-title, h1, .job-post-title")
            )
            
            # Company name from URL or page
            company = self.company_slug.replace("-", " ").title()
            
            # Location
            location = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".location, .job-post-location")
            )
            
            # Job description
            content_selectors = [
                ".job-post-content",
                ".content",
                "#job_description", 
                ".description"
            ]
            
            description = ""
            for selector in content_selectors:
                desc_element = self.safe_find_element(By.CSS_SELECTOR, selector)
                if desc_element:
                    description = self.extract_text_safe(desc_element)
                    break
            
            # Extract application URL
            apply_button = self.safe_find_element(
                By.CSS_SELECTOR, 
                "a[href*='apply'], .apply-button, #apply_button"
            )
            application_url = None
            if apply_button:
                application_url = apply_button.get_attribute("href")
            
            return JobPosting(
                url=job_url,
                title=title,
                company=company,
                location=location,
                description=description,
                application_url=application_url,
                metadata={
                    "source": "greenhouse",
                    "company_slug": self.company_slug,
                    "scraped_details": True
                }
            )
            
        except Exception:
            return None


class LeverAdapter(SeleniumJobScraper):
    """Lever job board scraper adapter.
    
    Similar to Greenhouse, many companies use Lever for job postings.
    """
    
    def __init__(self, company_slug: str, headless: bool = True):
        """Initialize Lever adapter.
        
        Args:
            company_slug: Company identifier for Lever
        """
        self.company_slug = company_slug
        base_url = f"https://jobs.lever.co/{company_slug}"
        
        super().__init__(
            base_url=base_url,
            headless=headless,
            delay_range=(1, 2)
        )
    
    def search_jobs(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        max_pages: int = 5
    ) -> ScrapingResult:
        """Search Lever job board."""
        try:
            if not self.navigate_to_page(self.base_url):
                return ScrapingResult(
                    success=False,
                    jobs=[],
                    error_message=f"Could not access {self.company_slug} Lever page"
                )
            
            # Wait for jobs to load
            self.human_delay(2.0)
            
            jobs = []
            
            # Get job postings
            job_links = self.safe_find_elements(
                By.CSS_SELECTOR, 
                ".posting, .posting-name a, a[href*='/jobs/']"
            )
            
            for link in job_links:
                job_url = link.get_attribute("href")
                if job_url:
                    job = self.scrape_job_details(job_url)
                    if job and self.validate_job_posting(job):
                        jobs.append(job)
                
                self.human_delay(0.5)
            
            # Filter by keywords
            if keywords:
                jobs = self.filter_jobs_by_keywords(jobs, keywords)
            
            return ScrapingResult(
                success=True,
                jobs=self.deduplicate_jobs(jobs),
                pages_scraped=1,
                jobs_found=len(jobs)
            )
            
        except Exception as e:
            return ScrapingResult(
                success=False,
                jobs=[],
                error_message=f"Error scraping Lever: {str(e)}"
            )
    
    def scrape_job_details(self, job_url: str) -> Optional[JobPosting]:
        """Scrape detailed Lever job posting."""
        try:
            if not self.navigate_to_page(job_url):
                return None
            
            self.human_delay(1.0)
            
            # Extract information
            title = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".posting-headline h2, h1")
            )
            
            company = self.company_slug.replace("-", " ").title()
            
            location = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".posting-categories .location")
            )
            
            # Description
            description = self.extract_text_safe(
                self.safe_find_element(By.CSS_SELECTOR, ".posting-content, .content")
            )
            
            # Application URL
            apply_button = self.safe_find_element(
                By.CSS_SELECTOR, 
                ".postings-btn, .posting-apply .template-btn-submit"
            )
            application_url = apply_button.get_attribute("href") if apply_button else job_url
            
            return JobPosting(
                url=job_url,
                title=title,
                company=company,
                location=location,
                description=description,
                application_url=application_url,
                metadata={
                    "source": "lever",
                    "company_slug": self.company_slug
                }
            )
            
        except Exception:
            return None