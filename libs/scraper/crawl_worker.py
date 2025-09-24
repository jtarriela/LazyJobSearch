"""
Crawl Worker Service

This module provides the main crawl worker that orchestrates the scraping process.
It handles:
- Company lookup and careers URL discovery (if missing)
- Dispatching to appropriate scraper adapters
- Job ingestion pipeline from adapter outputs to database
- Error handling and retry logic
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

from libs.db.session import get_session
from libs.db import models
from libs.scraper.careers_discovery import CareersDiscoveryService
from libs.scraper.anduril_adapter import AndurilScraper, JobPosting
from libs.scraper.ats_scrapers import GreenhouseScraper, LeverScraper, WorkdayScraper
from libs.scraper.failure_metrics import get_failure_tracker, crawl_session, company_crawl, classify_exception

logger = logging.getLogger(__name__)

class CrawlWorker:
    """Main crawler worker that orchestrates the scraping process"""
    
    def __init__(self):
        self.discovery_service = CareersDiscoveryService()
        self.failure_tracker = get_failure_tracker()
        self._scrapers = {
            'anduril': self._create_anduril_scraper,
            'greenhouse': self._create_greenhouse_scraper,
            'lever': self._create_lever_scraper,
            'workday': self._create_workday_scraper
        }
    
    def _create_anduril_scraper(self) -> AndurilScraper:
        """Create an Anduril scraper instance"""
        return AndurilScraper()
    
    def _create_greenhouse_scraper(self) -> GreenhouseScraper:
        """Create a Greenhouse scraper instance"""
        return GreenhouseScraper()
    
    def _create_lever_scraper(self) -> LeverScraper:
        """Create a Lever scraper instance"""
        return LeverScraper()
    
    def _create_workday_scraper(self) -> WorkdayScraper:
        """Create a Workday scraper instance"""
        return WorkdayScraper()
    
    def crawl_company(self, company_name: str) -> Dict[str, Any]:
        """Crawl jobs for a specific company with enhanced error tracking
        
        Args:
            company_name: Name of the company to crawl
            
        Returns:
            Dictionary with crawl results and statistics
        """
        logger.info(f"Starting crawl for company: {company_name}")
        
        with get_session() as session:
            # Look up the company
            company = session.query(models.Company).filter(
                models.Company.name.ilike(f"%{company_name}%")
            ).first()
            
            if not company:
                error_msg = f"Company '{company_name}' not found in database"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Determine scraper type first
            careers_url = company.careers_url
            if not careers_url:
                logger.info(f"No careers URL for {company.name}, attempting discovery...")
                discovered_url = self._discover_careers_url(company)
                
                if discovered_url:
                    careers_url = discovered_url
                    company.careers_url = discovered_url
                    session.commit()
                    logger.info(f"Updated careers URL for {company.name}: {discovered_url}")
                else:
                    logger.warning(f"Could not discover careers URL for {company.name}")
                    return {"error": f"Could not find careers page for {company.name}"}
            
            scraper_type = self._determine_scraper_type(careers_url)
            if scraper_type not in self._scrapers:
                error_msg = f"No scraper available for {scraper_type}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Track company crawl with metrics
            with company_crawl(company.name, scraper_type) as company_tracker:
                try:
                    # Run the scraper
                    scraper = self._scrapers[scraper_type]()
                    
                    # ATS scrapers need company domain, Anduril scraper doesn't
                    if scraper_type in ['greenhouse', 'lever', 'workday']:
                        # Extract company domain from website or use company name
                        company_domain = self._extract_company_domain(company, careers_url)
                        job_postings = scraper.search(company_domain)
                    else:
                        # Anduril and other specific scrapers
                        job_postings = scraper.search()
                    
                    logger.info(f"Scraped {len(job_postings)} jobs from {company.name}")
                    
                    # Ingest jobs into database
                    ingested_count, duplicate_count = self._ingest_jobs(session, company, job_postings)
                    
                    # Record duplicate detection metrics
                    company_tracker.record_duplicate_jobs(duplicate_count, company.name)
                    
                    # Update tracker with final counts
                    company_tracker.end_company_crawl(
                        company.name, scraper_type, True, len(job_postings), ingested_count
                    )
                    
                    return {
                        "company": company.name,
                        "careers_url": careers_url,
                        "scraper_type": scraper_type,
                        "jobs_found": len(job_postings),
                        "jobs_ingested": ingested_count,
                        "jobs_duplicate": duplicate_count,
                        "status": "success"
                    }
                    
                except Exception as e:
                    error_type = classify_exception(e, {"company": company.name, "scraper": scraper_type})
                    company_tracker.record_error(error_type, str(e), company.name, scraper_type)
                    logger.error(f"Error scraping {company.name}: {e}")
                    raise
    
    def crawl_all_companies(self) -> List[Dict[str, Any]]:
        """Crawl all companies in the database with structured metrics tracking
        
        Returns:
            List of crawl results for each company
        """
        results = []
        
        with crawl_session() as session_tracker:
            with get_session() as session:
                companies = session.query(models.Company).all()
                
                for company in companies:
                    try:
                        result = self.crawl_company(company.name)
                        results.append(result)
                    except Exception as e:
                        error_type = classify_exception(e, {"company": company.name})
                        session_tracker.record_error(
                            error_type, str(e), company.name, None
                        )
                        logger.error(f"Error crawling {company.name}: {e}")
                        results.append({
                            "company": company.name,
                            "error": str(e),
                            "status": "failed"
                        })
        
        return results
    
    def _discover_careers_url(self, company: models.Company) -> Optional[str]:
        """Discover careers URL for a company
        
        Args:
            company: Company model instance
            
        Returns:
            Discovered careers URL or None
        """
        if not company.website:
            logger.warning(f"No website URL for {company.name}")
            return None
        
        try:
            return self.discovery_service.discover_careers_url(company.website)
        except Exception as e:
            logger.error(f"Error discovering careers URL for {company.name}: {e}")
            return None
    
    def _determine_scraper_type(self, careers_url: str) -> str:
        """Determine which scraper to use based on the careers URL
        
        Args:
            careers_url: The careers page URL
            
        Returns:
            Scraper type identifier
        """
        careers_url_lower = careers_url.lower()
        
        # Company-specific scrapers
        if 'anduril.com' in careers_url_lower:
            return 'anduril'
        
        # ATS platform detection
        if 'greenhouse.io' in careers_url_lower or 'boards.greenhouse.io' in careers_url_lower:
            return 'greenhouse'
        elif 'lever.co' in careers_url_lower or 'jobs.lever.co' in careers_url_lower:
            return 'lever'
        elif 'myworkdayjobs.com' in careers_url_lower or 'workday' in careers_url_lower:
            return 'workday'
        
        # Try to detect ATS by common patterns in URL structure
        if '/greenhouse/' in careers_url_lower or '/boards/' in careers_url_lower:
            return 'greenhouse'
        elif '/lever/' in careers_url_lower:
            return 'lever'
        elif '/workday/' in careers_url_lower or '/jobs/' in careers_url_lower:
            return 'workday'
        
        # Default to greenhouse as it's quite common
        logger.info(f"Could not determine ATS type for {careers_url}, defaulting to greenhouse")
        return 'greenhouse'
    
    def _extract_company_domain(self, company: models.Company, careers_url: str) -> str:
        """Extract company domain for ATS scrapers
        
        Args:
            company: Company model instance
            careers_url: The careers page URL
            
        Returns:
            Company domain or identifier for ATS scraper
        """
        import re
        
        # First try to extract from careers URL
        if 'greenhouse.io' in careers_url:
            # Extract from greenhouse URL: https://boards.greenhouse.io/company-name
            match = re.search(r'greenhouse\.io/([^/?]+)', careers_url)
            if match:
                return match.group(1)
        elif 'lever.co' in careers_url:
            # Extract from lever URL: https://jobs.lever.co/company-name
            match = re.search(r'lever\.co/([^/?]+)', careers_url)
            if match:
                return match.group(1)
        elif 'workday' in careers_url:
            # Workday URLs are complex, return the full URL
            return careers_url
        
        # If not found in careers URL, try to extract from company website
        if company.website:
            # Remove protocol and www
            domain = company.website.lower()
            domain = re.sub(r'^https?://', '', domain)
            domain = re.sub(r'^www\.', '', domain)
            domain = domain.split('/')[0]  # Remove path
            
            # For ATS, usually the company name is the domain without TLD
            company_name = domain.split('.')[0]
            return company_name
        
        # Fallback to company name (cleaned)
        if company.name:
            # Clean company name for URL use
            clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', company.name.lower())
            clean_name = re.sub(r'\s+', '-', clean_name.strip())
            return clean_name
        
        # Ultimate fallback
        return "unknown-company"
    
    def _ingest_jobs(self, session, company: models.Company, job_postings: List[JobPosting]) -> tuple[int, int]:
        """Ingest job postings into the database with duplicate detection
        
        Args:
            session: Database session
            company: Company model instance
            job_postings: List of scraped job postings
            
        Returns:
            Tuple of (ingested_count, duplicate_count)
        """
        ingested_count = 0
        duplicate_count = 0
        
        for job in job_postings:
            try:
                # Check if job already exists (based on URL)
                existing_job = session.query(models.Job).filter(
                    models.Job.url == job.url
                ).first()
                
                if existing_job:
                    # Update existing job if content changed
                    fingerprint = self._generate_fingerprint(job.description)
                    if existing_job.scrape_fingerprint != fingerprint:
                        logger.info(f"Updating job: {job.title}")
                        existing_job.title = job.title
                        existing_job.location = job.location
                        existing_job.jd_fulltext = job.description
                        existing_job.scraped_at = job.scraped_at
                        existing_job.scrape_fingerprint = fingerprint
                        existing_job.seniority = self._extract_seniority(job.title, job.description)
                        existing_job.jd_skills_csv = self._extract_skills(job.description)
                        ingested_count += 1
                    else:
                        # Job exists and unchanged - count as duplicate
                        duplicate_count += 1
                        logger.debug(f"Job unchanged: {job.title}")
                else:
                    # Create new job
                    logger.info(f"Ingesting new job: {job.title}")
                    new_job = models.Job(
                        company_id=company.id,
                        url=job.url,
                        title=job.title,
                        location=job.location,
                        seniority=self._extract_seniority(job.title, job.description),
                        jd_fulltext=job.description,
                        jd_skills_csv=self._extract_skills(job.description),
                        scraped_at=job.scraped_at,
                        scrape_fingerprint=self._generate_fingerprint(job.description)
                    )
                    session.add(new_job)
                    ingested_count += 1
                
            except Exception as e:
                error_type = classify_exception(e)
                self.failure_tracker.record_error(
                    error_type, f"Error ingesting job {job.title}: {e}",
                    company.name, None, job.url
                )
                logger.error(f"Error ingesting job {job.title}: {e}")
                continue
        
        try:
            session.commit()
            logger.info(f"Successfully ingested {ingested_count} jobs, {duplicate_count} duplicates skipped")
        except Exception as e:
            error_type = classify_exception(e)
            self.failure_tracker.record_error(
                error_type, f"Error committing jobs to database: {e}",
                company.name
            )
            logger.error(f"Error committing jobs to database: {e}")
            session.rollback()
            ingested_count = 0
        
        return ingested_count, duplicate_count
    
    def _generate_fingerprint(self, content: str) -> str:
        """Generate a fingerprint for job content to detect changes
        
        Args:
            content: Job description content
            
        Returns:
            SHA256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _extract_seniority(self, title: str, description: str) -> Optional[str]:
        """Extract seniority level from job title and description
        
        Args:
            title: Job title
            description: Job description
            
        Returns:
            Detected seniority level or None
        """
        content = f"{title} {description}".lower()
        
        if any(keyword in content for keyword in ['senior', 'sr.', 'lead', 'principal', 'staff']):
            return 'senior'
        elif any(keyword in content for keyword in ['junior', 'jr.', 'entry', 'associate', 'intern']):
            return 'junior'
        elif any(keyword in content for keyword in ['mid', 'intermediate', 'regular']):
            return 'mid'
        
        return None
    
    def _extract_skills(self, description: str) -> str:
        """Extract skills from job description
        
        Args:
            description: Job description text
            
        Returns:
            Comma-separated list of detected skills
        """
        # Simple skill extraction - in a full system this would be more sophisticated
        common_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform', 'git',
            'machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch',
            'linux', 'windows', 'macos', 'bash', 'powershell', 'ci/cd', 'jenkins',
            'ansible', 'chef', 'puppet', 'microservices', 'rest api', 'graphql'
        ]
        
        found_skills = []
        description_lower = description.lower()
        
        for skill in common_skills:
            if skill in description_lower:
                found_skills.append(skill)
        
        return ','.join(found_skills)