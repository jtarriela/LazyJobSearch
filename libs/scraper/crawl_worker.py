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

logger = logging.getLogger(__name__)

class CrawlWorker:
    """Main crawler worker that orchestrates the scraping process"""
    
    def __init__(self):
        self.discovery_service = CareersDiscoveryService()
        self._scrapers = {
            'anduril': self._create_anduril_scraper
        }
    
    def _create_anduril_scraper(self) -> AndurilScraper:
        """Create an Anduril scraper instance"""
        return AndurilScraper()
    
    def crawl_company(self, company_name: str) -> Dict[str, Any]:
        """Crawl jobs for a specific company
        
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
                logger.error(f"Company '{company_name}' not found in database")
                return {"error": f"Company '{company_name}' not found"}
            
            # Check if careers_url is missing
            if not company.careers_url:
                logger.info(f"No careers URL for {company.name}, attempting discovery...")
                discovered_url = self._discover_careers_url(company)
                
                if discovered_url:
                    company.careers_url = discovered_url
                    session.commit()
                    logger.info(f"Updated careers URL for {company.name}: {discovered_url}")
                else:
                    logger.warning(f"Could not discover careers URL for {company.name}")
                    return {"error": f"Could not find careers page for {company.name}"}
            
            # Determine which scraper to use
            scraper_type = self._determine_scraper_type(company.careers_url)
            if scraper_type not in self._scrapers:
                logger.error(f"No scraper available for {scraper_type}")
                return {"error": f"No scraper available for {scraper_type}"}
            
            # Run the scraper
            scraper = self._scrapers[scraper_type]()
            try:
                job_postings = scraper.search()
                logger.info(f"Scraped {len(job_postings)} jobs from {company.name}")
                
                # Ingest jobs into database
                ingested_count = self._ingest_jobs(session, company, job_postings)
                
                return {
                    "company": company.name,
                    "careers_url": company.careers_url,
                    "scraper_type": scraper_type,
                    "jobs_found": len(job_postings),
                    "jobs_ingested": ingested_count,
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Error scraping {company.name}: {e}")
                return {"error": f"Scraping failed: {str(e)}"}
    
    def crawl_all_companies(self) -> List[Dict[str, Any]]:
        """Crawl all companies in the database
        
        Returns:
            List of crawl results for each company
        """
        results = []
        
        with get_session() as session:
            companies = session.query(models.Company).all()
            
            for company in companies:
                try:
                    result = self.crawl_company(company.name)
                    results.append(result)
                except Exception as e:
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
        
        if 'anduril.com' in careers_url_lower:
            return 'anduril'
        
        # Default to generic scraper (not implemented yet)
        return 'generic'
    
    def _ingest_jobs(self, session, company: models.Company, job_postings: List[JobPosting]) -> int:
        """Ingest job postings into the database
        
        Args:
            session: Database session
            company: Company model instance
            job_postings: List of scraped job postings
            
        Returns:
            Number of jobs successfully ingested
        """
        ingested_count = 0
        
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
                        ingested_count += 1
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
                logger.error(f"Error ingesting job {job.title}: {e}")
                continue
        
        try:
            session.commit()
            logger.info(f"Successfully ingested {ingested_count} jobs")
        except Exception as e:
            logger.error(f"Error committing jobs to database: {e}")
            session.rollback()
            ingested_count = 0
        
        return ingested_count
    
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