"""
Structured Failure Metrics for Job Crawling

This module provides specialized metrics collection for tracking crawling failures,
errors, and performance issues with proper context and categorization.
"""
from __future__ import annotations
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from libs.observability.metrics import get_logger, counter, histogram, gauge

logger = get_logger(__name__)


class CrawlErrorType(Enum):
    """Types of crawling errors for structured tracking"""
    # Network and connectivity
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    CONNECTION_REFUSED = "connection_refused"
    DNS_ERROR = "dns_error"
    
    # Rate limiting and blocking
    RATE_LIMITED = "rate_limited"
    BLOCKED_BY_SITE = "blocked_by_site"
    CAPTCHA_DETECTED = "captcha_detected"
    IP_BANNED = "ip_banned"
    
    # Content and parsing
    PARSING_ERROR = "parsing_error"
    NO_JOBS_FOUND = "no_jobs_found"
    INVALID_PAGE_STRUCTURE = "invalid_page_structure"
    PAGINATION_ERROR = "pagination_error"
    
    # Database and storage
    DATABASE_ERROR = "database_error"
    DUPLICATE_CONSTRAINT = "duplicate_constraint"
    STORAGE_ERROR = "storage_error"
    
    # Configuration and setup
    MISSING_CONFIG = "missing_config"
    INVALID_URL = "invalid_url"
    SCRAPER_NOT_FOUND = "scraper_not_found"
    
    # Unknown/Other
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class CrawlMetrics:
    """Container for crawl session metrics"""
    # Basic counters
    companies_attempted: int = 0
    companies_successful: int = 0
    companies_failed: int = 0
    
    # Job metrics
    jobs_found: int = 0
    jobs_ingested: int = 0
    jobs_duplicate: int = 0
    jobs_failed: int = 0
    
    # Timing
    total_duration: float = 0.0
    avg_company_duration: float = 0.0
    
    # Errors by type
    error_counts: Dict[CrawlErrorType, int] = None
    
    # Pagination metrics
    pages_scraped: int = 0
    pages_failed: int = 0
    
    # Rate limiting metrics
    rate_limit_hits: int = 0
    retry_attempts: int = 0
    
    def __post_init__(self):
        if self.error_counts is None:
            self.error_counts = {error_type: 0 for error_type in CrawlErrorType}


class CrawlFailureTracker:
    """Tracks and reports structured failure metrics for crawling operations"""
    
    def __init__(self):
        self.current_metrics = CrawlMetrics()
        self._start_time: Optional[float] = None
        self._company_start_time: Optional[float] = None
    
    def start_crawl_session(self):
        """Start a new crawl session"""
        self.current_metrics = CrawlMetrics()
        self._start_time = time.time()
        logger.info("Started new crawl session")
    
    def end_crawl_session(self):
        """End crawl session and emit summary metrics"""
        if self._start_time:
            self.current_metrics.total_duration = time.time() - self._start_time
            
            if self.current_metrics.companies_attempted > 0:
                self.current_metrics.avg_company_duration = (
                    self.current_metrics.total_duration / self.current_metrics.companies_attempted
                )
        
        self._emit_session_metrics()
        logger.info(f"Crawl session completed: {self._format_session_summary()}")
    
    def start_company_crawl(self, company_name: str, portal_type: str):
        """Start crawling a specific company"""
        self.current_metrics.companies_attempted += 1
        self._company_start_time = time.time()
        
        logger.info("Starting company crawl", 
                   company=company_name, 
                   portal_type=portal_type,
                   attempt_number=self.current_metrics.companies_attempted)
    
    def end_company_crawl(self, company_name: str, portal_type: str, success: bool, 
                         jobs_found: int = 0, jobs_ingested: int = 0):
        """End company crawl and record metrics"""
        duration = 0.0
        if self._company_start_time:
            duration = time.time() - self._company_start_time
            self._company_start_time = None
        
        if success:
            self.current_metrics.companies_successful += 1
        else:
            self.current_metrics.companies_failed += 1
        
        self.current_metrics.jobs_found += jobs_found
        self.current_metrics.jobs_ingested += jobs_ingested
        
        # Emit company-level metrics
        tags = {"company": company_name, "portal_type": portal_type}
        counter("crawler.company.attempts", 1, tags)
        counter(f"crawler.company.{'success' if success else 'failure'}", 1, tags)
        histogram("crawler.company.duration", duration, tags)
        histogram("crawler.company.jobs_found", jobs_found, tags)
        histogram("crawler.company.jobs_ingested", jobs_ingested, tags)
        
        logger.info("Completed company crawl",
                   company=company_name,
                   portal_type=portal_type,
                   success=success,
                   duration=f"{duration:.2f}s",
                   jobs_found=jobs_found,
                   jobs_ingested=jobs_ingested)
    
    def record_error(self, error_type: CrawlErrorType, error_message: str,
                    company_name: Optional[str] = None, portal_type: Optional[str] = None,
                    job_id: Optional[str] = None, page_number: Optional[int] = None):
        """Record a structured error with full context"""
        
        self.current_metrics.error_counts[error_type] += 1
        
        # Build context tags
        tags = {"error_type": error_type.value}
        if company_name:
            tags["company"] = company_name
        if portal_type:
            tags["portal_type"] = portal_type
        if job_id:
            tags["job_id"] = job_id
        if page_number is not None:
            tags["page_number"] = str(page_number)
        
        # Emit metrics
        counter("crawler.errors.total", 1, tags)
        counter(f"crawler.errors.{error_type.value}", 1, tags)
        
        # Log with full context
        logger.error("Crawl error occurred",
                    error_type=error_type.value,
                    error_message=error_message,
                    company=company_name,
                    portal_type=portal_type,
                    job_id=job_id,
                    page_number=page_number)
    
    def record_retry_attempt(self, retry_count: int, error_type: CrawlErrorType,
                           company_name: Optional[str] = None):
        """Record retry attempts with context"""
        self.current_metrics.retry_attempts += 1
        
        tags = {
            "error_type": error_type.value,
            "retry_count": str(retry_count)
        }
        if company_name:
            tags["company"] = company_name
        
        counter("crawler.retries.total", 1, tags)
        counter(f"crawler.retries.{error_type.value}", 1, tags)
        
        logger.warning("Retry attempt",
                      retry_count=retry_count,
                      error_type=error_type.value,
                      company=company_name)
    
    def record_rate_limit_hit(self, company_name: Optional[str] = None,
                             portal_type: Optional[str] = None):
        """Record rate limiting events"""
        self.current_metrics.rate_limit_hits += 1
        
        tags = {}
        if company_name:
            tags["company"] = company_name
        if portal_type:
            tags["portal_type"] = portal_type
        
        counter("crawler.rate_limits.total", 1, tags)
        
        logger.warning("Rate limit hit",
                      company=company_name,
                      portal_type=portal_type)
    
    def record_pagination_metrics(self, pages_scraped: int, pages_failed: int,
                                 company_name: Optional[str] = None):
        """Record pagination-specific metrics"""
        self.current_metrics.pages_scraped += pages_scraped
        self.current_metrics.pages_failed += pages_failed
        
        tags = {}
        if company_name:
            tags["company"] = company_name
        
        histogram("crawler.pagination.pages_scraped", pages_scraped, tags)
        if pages_failed > 0:
            counter("crawler.pagination.pages_failed", pages_failed, tags)
        
        logger.info("Pagination metrics",
                   company=company_name,
                   pages_scraped=pages_scraped,
                   pages_failed=pages_failed)
    
    def record_duplicate_jobs(self, duplicate_count: int, company_name: Optional[str] = None):
        """Record duplicate job detection metrics"""
        self.current_metrics.jobs_duplicate += duplicate_count
        
        tags = {}
        if company_name:
            tags["company"] = company_name
        
        counter("crawler.jobs.duplicates", duplicate_count, tags)
        
        if duplicate_count > 0:
            logger.info("Duplicate jobs detected",
                       company=company_name,
                       duplicate_count=duplicate_count)
    
    def _emit_session_metrics(self):
        """Emit session-level summary metrics"""
        m = self.current_metrics
        
        # Overall session metrics
        gauge("crawler.session.companies_attempted", m.companies_attempted)
        gauge("crawler.session.companies_successful", m.companies_successful)
        gauge("crawler.session.companies_failed", m.companies_failed)
        gauge("crawler.session.jobs_found", m.jobs_found)
        gauge("crawler.session.jobs_ingested", m.jobs_ingested)
        gauge("crawler.session.jobs_duplicate", m.jobs_duplicate)
        gauge("crawler.session.total_duration", m.total_duration)
        
        if m.companies_attempted > 0:
            success_rate = m.companies_successful / m.companies_attempted
            gauge("crawler.session.success_rate", success_rate)
        
        if m.jobs_found > 0:
            ingestion_rate = m.jobs_ingested / m.jobs_found
            gauge("crawler.session.ingestion_rate", ingestion_rate)
        
        # Error breakdown
        for error_type, count in m.error_counts.items():
            if count > 0:
                gauge(f"crawler.session.errors.{error_type.value}", count)
        
        # Pagination metrics
        gauge("crawler.session.pages_scraped", m.pages_scraped)
        gauge("crawler.session.pages_failed", m.pages_failed)
        gauge("crawler.session.rate_limit_hits", m.rate_limit_hits)
        gauge("crawler.session.retry_attempts", m.retry_attempts)
    
    def _format_session_summary(self) -> str:
        """Format a human-readable session summary"""
        m = self.current_metrics
        
        success_rate = 0.0
        if m.companies_attempted > 0:
            success_rate = (m.companies_successful / m.companies_attempted) * 100
        
        total_errors = sum(m.error_counts.values())
        
        return (f"Companies: {m.companies_successful}/{m.companies_attempted} "
                f"({success_rate:.1f}% success), Jobs: {m.jobs_ingested} ingested, "
                f"Errors: {total_errors}, Duration: {m.total_duration:.1f}s")


def classify_exception(exception: Exception, context: Dict[str, Any] = None) -> CrawlErrorType:
    """Classify an exception into a structured error type
    
    Args:
        exception: The exception to classify
        context: Additional context for classification
        
    Returns:
        Appropriate CrawlErrorType
    """
    if context is None:
        context = {}
    
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()
    
    # Network and connectivity errors
    if any(term in exception_str for term in ['connection', 'network', 'resolve', 'unreachable']):
        if 'refused' in exception_str:
            return CrawlErrorType.CONNECTION_REFUSED
        elif 'timeout' in exception_str:
            return CrawlErrorType.TIMEOUT
        elif any(dns_term in exception_str for dns_term in ['resolve', 'dns', 'name']):
            return CrawlErrorType.DNS_ERROR
        else:
            return CrawlErrorType.NETWORK_ERROR
    
    # Rate limiting and blocking
    if any(term in exception_str for term in ['rate limit', '429', 'too many requests']):
        return CrawlErrorType.RATE_LIMITED
    elif any(term in exception_str for term in ['blocked', 'forbidden', '403']):
        return CrawlErrorType.BLOCKED_BY_SITE
    elif 'captcha' in exception_str:
        return CrawlErrorType.CAPTCHA_DETECTED
    
    # Database errors
    if any(term in exception_type for term in ['integrity', 'database', 'sql']):
        if 'unique' in exception_str or 'duplicate' in exception_str:
            return CrawlErrorType.DUPLICATE_CONSTRAINT
        else:
            return CrawlErrorType.DATABASE_ERROR
    
    # Parsing errors
    if any(term in exception_type for term in ['parse', 'json', 'xml', 'html']):
        return CrawlErrorType.PARSING_ERROR
    
    # Timeout errors
    if 'timeout' in exception_str or 'timeout' in exception_type:
        return CrawlErrorType.TIMEOUT
    
    # Default to unknown
    return CrawlErrorType.UNKNOWN_ERROR


# Global tracker instance
_global_tracker = None

def get_failure_tracker() -> CrawlFailureTracker:
    """Get the global failure tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CrawlFailureTracker()
    return _global_tracker


@contextmanager
def crawl_session():
    """Context manager for tracking a complete crawl session"""
    tracker = get_failure_tracker()
    tracker.start_crawl_session()
    try:
        yield tracker
    finally:
        tracker.end_crawl_session()


@contextmanager  
def company_crawl(company_name: str, portal_type: str):
    """Context manager for tracking a single company crawl"""
    tracker = get_failure_tracker()
    tracker.start_company_crawl(company_name, portal_type)
    
    success = False
    jobs_found = 0
    jobs_ingested = 0
    
    try:
        yield tracker
        success = True
    except Exception as e:
        error_type = classify_exception(e, {"company": company_name, "portal": portal_type})
        tracker.record_error(error_type, str(e), company_name, portal_type)
        raise
    finally:
        tracker.end_company_crawl(company_name, portal_type, success, jobs_found, jobs_ingested)