"""Job ingestion service - orchestrates job crawling, chunking, and storage pipeline.

This service implements the job ingestion workflow:
1. Job crawling and content extraction
2. Content deduplication using fingerprints
3. Job description parsing and skills extraction
4. Text chunking for embeddings
5. Embedding generation and vector storage
6. Database persistence with artifact storage

Based on requirements from the gap analysis for job crawling and storage.
"""
from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import uuid
import json

from libs.observability import get_logger, timer, counter, PerformanceMetrics
from libs.db.models import Company, Job, JobChunk
from libs.embed.versioning import EmbeddingVersionManager

logger = get_logger(__name__)

@dataclass
class CrawledJob:
    """Raw job data from crawler"""
    url: str
    title: str
    company_name: str
    description: str = ""
    location: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedJob:
    """Job after processing and enhancement"""
    url: str
    title: str
    company_id: str
    fulltext: str
    skills_csv: str
    fingerprint: str
    location: Optional[str] = None
    seniority: Optional[str] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IngestionStats:
    """Statistics from job ingestion process"""
    jobs_crawled: int
    jobs_deduplicated: int
    jobs_processed: int
    jobs_persisted: int
    chunks_created: int
    processing_time_ms: float
    errors: List[str] = field(default_factory=list)


class JobIngestionService:
    """Orchestrates job crawling, processing, and storage pipeline"""
    
    def __init__(self, db_session, crawler_service=None, embedding_service=None, 
                 chunker_service=None, embedding_version_manager: Optional[EmbeddingVersionManager] = None):
        self.db_session = db_session
        self.crawler_service = crawler_service
        self.embedding_service = embedding_service  
        self.chunker_service = chunker_service
        self.embedding_version_manager = embedding_version_manager
        
        # Deduplication cache
        self._fingerprint_cache: Set[str] = set()
    
    async def ingest_company_jobs(self, company_id: str, crawl_limit: int = 100) -> IngestionStats:
        """
        Ingest jobs for a specific company through the complete pipeline.
        
        Args:
            company_id: Company to crawl jobs for
            crawl_limit: Maximum number of jobs to crawl
            
        Returns:
            IngestionStats with processing results
        """
        start_time = asyncio.get_event_loop().time()
        stats = IngestionStats(
            jobs_crawled=0, jobs_deduplicated=0, jobs_processed=0,
            jobs_persisted=0, chunks_created=0, processing_time_ms=0, errors=[]
        )
        
        try:
            logger.info("Starting job ingestion", company_id=company_id, limit=crawl_limit)
            
            with timer("job_ingestion.total"):
                # Load fingerprint cache for deduplication
                await self._load_fingerprint_cache(company_id)
                
                # Stage 1: Crawl jobs from company
                crawled_jobs = await self._crawl_company_jobs(company_id, crawl_limit)
                stats.jobs_crawled = len(crawled_jobs)
                counter("job_ingestion.jobs_crawled", len(crawled_jobs))
                
                if not crawled_jobs:
                    logger.info("No jobs found to crawl", company_id=company_id)
                    return stats
                
                # Stage 2: Deduplication and processing
                processed_jobs = await self._process_and_deduplicate_jobs(crawled_jobs, company_id)
                stats.jobs_deduplicated = stats.jobs_crawled - len(processed_jobs)
                stats.jobs_processed = len(processed_jobs)
                counter("job_ingestion.jobs_processed", len(processed_jobs))
                
                if not processed_jobs:
                    logger.info("All jobs were duplicates", company_id=company_id)
                    return stats
                
                # Stage 3: Chunking and embeddings
                await self._generate_chunks_and_embeddings(processed_jobs)
                stats.chunks_created = sum(len(job.chunks or []) for job in processed_jobs)
                counter("job_ingestion.chunks_created", stats.chunks_created)
                
                # Stage 4: Database persistence
                persisted_count = await self._persist_jobs(processed_jobs)
                stats.jobs_persisted = persisted_count
                counter("job_ingestion.jobs_persisted", persisted_count)
                
                end_time = asyncio.get_event_loop().time()
                stats.processing_time_ms = (end_time - start_time) * 1000
                
                # Log performance metrics
                counter(PerformanceMetrics.PAGES_PER_MINUTE, stats.jobs_crawled / (stats.processing_time_ms / 60000))
                
                logger.info("Job ingestion completed",
                           company_id=company_id,
                           stats=stats)
                
                return stats
                
        except Exception as e:
            counter("job_ingestion.pipeline_failure")
            stats.errors.append(str(e))
            logger.error("Job ingestion failed", company_id=company_id, error=str(e))
            raise
    
    async def ingest_single_job(self, job_url: str, company_id: str) -> Optional[str]:
        """
        Ingest a single job posting.
        
        Args:
            job_url: URL of job posting to ingest
            company_id: Company the job belongs to
            
        Returns:
            Job ID if successful, None if failed or duplicate
        """
        try:
            logger.info("Starting single job ingestion", job_url=job_url)
            
            # Crawl single job
            if not self.crawler_service:
                raise ValueError("Crawler service not available")
            
            crawled_job = await self.crawler_service.crawl_job_url(job_url)
            if not crawled_job:
                logger.warning("Failed to crawl job", job_url=job_url)
                return None
            
            # Process and check for duplicates
            processed_jobs = await self._process_and_deduplicate_jobs([crawled_job], company_id)
            if not processed_jobs:
                logger.info("Job is duplicate", job_url=job_url)
                return None
            
            processed_job = processed_jobs[0]
            
            # Generate chunks and embeddings
            await self._generate_chunks_and_embeddings([processed_job])
            
            # Persist job
            persisted_count = await self._persist_jobs([processed_job])
            if persisted_count > 0:
                logger.info("Single job ingestion completed", job_url=job_url)
                # Return the job_id (would need to be tracked during persistence)
                return processed_job.fingerprint  # Temporary return value
            
            return None
            
        except Exception as e:
            logger.error("Single job ingestion failed", job_url=job_url, error=str(e))
            return None
    
    async def _crawl_company_jobs(self, company_id: str, limit: int) -> List[CrawledJob]:
        """Crawl jobs from company careers page"""
        try:
            with timer("job_ingestion.crawling"):
                if not self.crawler_service:
                    logger.warning("No crawler service available")
                    return []
                
                # Get company information for crawling
                company = await self._get_company(company_id)
                if not company or not company.careers_url:
                    logger.warning("Company has no careers URL", company_id=company_id)
                    return []
                
                # Crawl jobs from company careers page
                crawled_jobs = await self.crawler_service.crawl_company_jobs(
                    company.careers_url, 
                    limit=limit,
                    company_name=company.name
                )
                
                logger.debug("Job crawling completed", 
                           company_id=company_id,
                           jobs_found=len(crawled_jobs))
                
                return crawled_jobs
                
        except Exception as e:
            counter("job_ingestion.crawling_failure")
            logger.error("Job crawling failed", company_id=company_id, error=str(e))
            return []
    
    def _generate_job_fingerprint(self, job: CrawledJob) -> str:
        """Generate fingerprint for job deduplication"""
        # Combine title and description content for fingerprint
        content = f"{job.title}|{job.description[:500]}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def _get_company(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        query = await self.db_session.execute(
            "SELECT * FROM companies WHERE id = %s", (company_id,)
        )
        row = query.fetchone()
        return Company(**dict(row)) if row else None


def create_job_ingestion_service(db_session, crawler_service=None, embedding_service=None, 
                                chunker_service=None, embedding_version_manager=None) -> JobIngestionService:
    """Factory function to create job ingestion service"""
    return JobIngestionService(
        db_session=db_session,
        crawler_service=crawler_service,
        embedding_service=embedding_service,
        chunker_service=chunker_service,
        embedding_version_manager=embedding_version_manager
    )