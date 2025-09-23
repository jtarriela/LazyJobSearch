"""Base job scraper interface and common functionality."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser
import requests
from urllib.parse import urljoin, urlparse


@dataclass
class JobPosting:
    """Structured job posting data."""
    url: str
    title: str
    company: str
    location: Optional[str] = None
    seniority: Optional[str] = None
    description: str = ""
    requirements: str = ""
    salary_range: Optional[str] = None
    posted_date: Optional[str] = None
    application_url: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    success: bool
    jobs: List[JobPosting]
    error_message: Optional[str] = None
    pages_scraped: int = 0
    jobs_found: int = 0
    skipped_jobs: int = 0


class JobScraper(ABC):
    """Abstract base class for job scrapers.
    
    Provides common functionality for politeness checks, rate limiting,
    and result processing while allowing site-specific implementations.
    """
    
    def __init__(self, base_url: str, respect_robots: bool = True):
        """Initialize scraper.
        
        Args:
            base_url: Base URL of the job site
            respect_robots: Whether to check robots.txt
        """
        self.base_url = base_url
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LazyJobSearch/1.0 (Job matching research tool)'
        })
        
        self._robots_parser = None
        if respect_robots:
            self._load_robots_txt()
    
    def _load_robots_txt(self):
        """Load and parse robots.txt for the site."""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            self._robots_parser = RobotFileParser()
            self._robots_parser.set_url(robots_url)
            self._robots_parser.read()
        except Exception:
            # If robots.txt can't be loaded, err on the side of caution
            self._robots_parser = None
    
    def can_fetch(self, url: str) -> bool:
        """Check if we're allowed to fetch this URL according to robots.txt."""
        if not self.respect_robots or not self._robots_parser:
            return True
        
        return self._robots_parser.can_fetch(
            self.session.headers.get('User-Agent', '*'), 
            url
        )
    
    @abstractmethod
    def search_jobs(
        self, 
        keywords: List[str], 
        location: Optional[str] = None,
        max_pages: int = 5
    ) -> ScrapingResult:
        """Search for jobs matching the given criteria.
        
        Args:
            keywords: List of search keywords/skills
            location: Optional location filter
            max_pages: Maximum number of pages to scrape
            
        Returns:
            ScrapingResult with found job postings
        """
        pass
    
    @abstractmethod
    def scrape_job_details(self, job_url: str) -> Optional[JobPosting]:
        """Scrape detailed information for a specific job posting.
        
        Args:
            job_url: URL of the job posting
            
        Returns:
            JobPosting with detailed information or None if failed
        """
        pass
    
    def get_site_name(self) -> str:
        """Get a human-readable name for this job site."""
        parsed = urlparse(self.base_url)
        return parsed.netloc.replace('www.', '')
    
    def validate_job_posting(self, job: JobPosting) -> bool:
        """Validate that a job posting has required fields."""
        required_fields = ['url', 'title', 'company']
        
        for field in required_fields:
            if not getattr(job, field, None):
                return False
        
        # Description should have some content
        if len(job.description.strip()) < 50:
            return False
        
        return True
    
    def deduplicate_jobs(self, jobs: List[JobPosting]) -> List[JobPosting]:
        """Remove duplicate job postings based on URL."""
        seen_urls = set()
        unique_jobs = []
        
        for job in jobs:
            if job.url not in seen_urls:
                seen_urls.add(job.url)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def filter_jobs_by_keywords(
        self, 
        jobs: List[JobPosting], 
        required_keywords: List[str],
        min_matches: int = 1
    ) -> List[JobPosting]:
        """Filter jobs that mention required keywords."""
        filtered_jobs = []
        
        for job in jobs:
            # Combine title and description for keyword matching
            searchable_text = f"{job.title} {job.description}".lower()
            
            matches = sum(1 for keyword in required_keywords 
                         if keyword.lower() in searchable_text)
            
            if matches >= min_matches:
                job.metadata['keyword_matches'] = matches
                filtered_jobs.append(job)
        
        return filtered_jobs
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()