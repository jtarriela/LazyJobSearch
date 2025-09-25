"""
Enhanced Database Models with Deduplication Support
libs/db/models.py
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
import hashlib
import json
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, DateTime, 
    ForeignKey, UniqueConstraint, Index, JSON, Enum, 
    CheckConstraint, event, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func as sql_func
import uuid

Base = declarative_base()


class CrawlStatus(PyEnum):
    """Status of crawl sessions"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class FailureType(PyEnum):
    """Types of scraper failures"""
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    RATE_LIMIT = "rate_limit"
    CAPTCHA = "captcha"
    AUTH_REQUIRED = "auth_required"
    UNKNOWN = "unknown"


class JobSource(PyEnum):
    """Source of job listings"""
    INDEED_API = "indeed_api"
    ADZUNA_API = "adzuna_api"
    REMOTEOK_API = "remoteok_api"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    COMPANY_SITE = "company_site"
    MANUAL = "manual"


class SeniorityLevel(PyEnum):
    """Job seniority levels"""
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


def generate_uuid():
    """Generate a new UUID"""
    return str(uuid.uuid4())


def generate_content_fingerprint(title: str, description: str, 
                                location: str, salary: Optional[str] = None) -> str:
    """Generate fingerprint for content change detection"""
    content = f"{title}{description}{location}{salary or ''}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


class Company(Base):
    """Company model with enhanced tracking"""
    __tablename__ = 'companies'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, unique=True)
    website = Column(String(500))
    careers_url = Column(String(500))
    
    # Scraper configuration
    scraper_config = Column(JSONB)  # {type, selectors, auth, etc}
    detected_ats = Column(String(50))  # greenhouse, lever, workday, etc
    
    # Crawl scheduling
    last_successful_crawl = Column(DateTime)
    crawl_frequency_hours = Column(Integer, default=24)
    is_crawl_enabled = Column(Boolean, default=True)
    crawl_priority = Column(Integer, default=5)  # 1-10, higher = more important
    
    # Statistics
    total_jobs_scraped = Column(Integer, default=0)
    avg_jobs_per_crawl = Column(Float)
    success_rate = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = relationship("Job", back_populates="company", cascade="all, delete-orphan")
    crawl_sessions = relationship("CrawlSession", back_populates="company")
    scraper_failures = relationship("ScraperFailure", back_populates="company")
    
    # Indexes
    __table_args__ = (
        Index('idx_companies_next_crawl', 'last_successful_crawl', 'crawl_frequency_hours'),
        Index('idx_companies_enabled_priority', 'is_crawl_enabled', 'crawl_priority'),
    )
    
    def should_crawl(self) -> bool:
        """Check if company should be crawled"""
        if not self.is_crawl_enabled:
            return False
        
        if not self.last_successful_crawl:
            return True
        
        hours_since_crawl = (datetime.utcnow() - self.last_successful_crawl).total_seconds() / 3600
        return hours_since_crawl >= self.crawl_frequency_hours


class Job(Base):
    """Job model with deduplication and change tracking"""
    __tablename__ = 'jobs'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    company_id = Column(UUID, ForeignKey('companies.id'), nullable=False)
    
    # Core fields
    url = Column(String(1000), nullable=False, unique=True)  # Primary deduplication
    title = Column(String(500), nullable=False)
    location = Column(String(500))
    department = Column(String(255))
    
    # Job details
    jd_fulltext = Column(Text)
    jd_tsv = Column(TSVECTOR)  # Full-text search vector
    requirements = Column(Text)
    benefits = Column(Text)
    
    # Structured data
    seniority = Column(Enum(SeniorityLevel))
    job_type = Column(String(50))  # full-time, part-time, contract, etc
    salary_range = Column(String(100))
    remote = Column(Boolean, default=False)
    visa_sponsorship = Column(Boolean)
    
    # Tracking
    source = Column(Enum(JobSource), default=JobSource.COMPANY_SITE)
    external_id = Column(String(255))  # ID from external API
    content_fingerprint = Column(String(32), index=True)  # For change detection
    
    # Dates
    posted_date = Column(DateTime)
    expires_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Status
    is_active = Column(Boolean, default=True)
    change_count = Column(Integer, default=0)  # How many times content changed
    
    # Metadata
    skills_extracted = Column(JSONB)  # ["python", "kubernetes", ...]
    metadata = Column(JSONB)  # Additional unstructured data
    
    # File storage
    jd_file_url = Column(String(500))  # S3/GCS URL for full JD
    
    # Relationships
    company = relationship("Company", back_populates="jobs")
    job_chunks = relationship("JobChunk", back_populates="job", cascade="all, delete-orphan")
    matches = relationship("Match", back_populates="job")
    applications = relationship("Application", back_populates="job")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_jobs_company_active', 'company_id', 'is_active'),
        Index('idx_jobs_fingerprint', 'content_fingerprint'),
        Index('idx_jobs_posted_date', 'posted_date'),
        Index('idx_jobs_source', 'source'),
        Index('idx_jobs_seniority', 'seniority'),
        Index('idx_jobs_tsv_gin', 'jd_tsv', postgresql_using='gin'),
        CheckConstraint('change_count >= 0', name='check_change_count_positive'),
    )
    
    @validates('url')
    def validate_url(self, key, url):
        """Validate and normalize URL"""
        if not url:
            raise ValueError("URL is required")
        
        # Remove trailing slashes and fragments
        url = url.rstrip('/').split('#')[0]
        
        # Ensure it starts with http
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url
    
    def generate_fingerprint(self) -> str:
        """Generate content fingerprint"""
        return generate_content_fingerprint(
            self.title, 
            self.jd_fulltext or '',
            self.location or '',
            self.salary_range
        )
    
    def update_content(self, new_data: Dict[str, Any]):
        """Update job content and track changes"""
        old_fingerprint = self.content_fingerprint
        
        # Update fields
        for key, value in new_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Check if content changed
        new_fingerprint = self.generate_fingerprint()
        if old_fingerprint and old_fingerprint != new_fingerprint:
            self.change_count += 1
            self.content_fingerprint = new_fingerprint
            self.last_updated = datetime.utcnow()


class CrawlSession(Base):
    """Track individual crawl sessions"""
    __tablename__ = 'crawl_sessions'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    company_id = Column(UUID, ForeignKey('companies.id'))
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Status
    status = Column(Enum(CrawlStatus), default=CrawlStatus.RUNNING)
    
    # Metrics
    jobs_found = Column(Integer, default=0)
    jobs_created = Column(Integer, default=0)
    jobs_updated = Column(Integer, default=0)
    duplicates_skipped = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    
    # Source breakdown
    source_breakdown = Column(JSONB)  # {"api": 10, "scrape": 5}
    
    # Error tracking
    error_log = Column(Text)
    failure_reason = Column(String(500))
    
    # Additional metrics
    metrics = Column(JSONB)  # {pages_crawled, api_calls, proxy_rotations, etc}
    
    # Relationships
    company = relationship("Company", back_populates="crawl_sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_crawl_sessions_company_time', 'company_id', 'started_at'),
        Index('idx_crawl_sessions_status', 'status'),
    )
    
    def complete(self, status: CrawlStatus = CrawlStatus.COMPLETED):
        """Mark session as complete"""
        self.completed_at = datetime.utcnow()
        self.status = status
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class ScraperFailure(Base):
    """Track scraper failures for debugging"""
    __tablename__ = 'scraper_failures'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    company_id = Column(UUID, ForeignKey('companies.id'))
    
    # Failure details
    url = Column(Text, nullable=False)
    failure_type = Column(Enum(FailureType))
    error_message = Column(Text)
    stack_trace = Column(Text)
    
    # Debug artifacts
    screenshot_path = Column(Text)
    html_path = Column(Text)
    request_headers = Column(JSONB)
    response_headers = Column(JSONB)
    
    # Tracking
    occurred_at = Column(DateTime, default=datetime.utcnow)
    retry_count = Column(Integer, default=0)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    
    # Relationships
    company = relationship("Company", back_populates="scraper_failures")
    
    # Indexes
    __table_args__ = (
        Index('idx_scraper_failures_unresolved', 'company_id', 'occurred_at', 
              postgresql_where='resolved = false'),
        Index('idx_scraper_failures_type', 'failure_type'),
    )


class APIRateLimit(Base):
    """Track API rate limits"""
    __tablename__ = 'api_rate_limits'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    api_name = Column(String(50), nullable=False)
    endpoint = Column(String(255))
    
    # Rate limit tracking
    requests_made = Column(Integer, default=0)
    requests_limit = Column(Integer)
    window_start = Column(DateTime, default=datetime.utcnow)
    window_minutes = Column(Integer, default=60)
    
    # Last request info
    last_request = Column(DateTime)
    last_response_code = Column(Integer)
    
    # Quota info (for APIs with monthly limits)
    monthly_quota = Column(Integer)
    monthly_used = Column(Integer, default=0)
    quota_reset_date = Column(DateTime)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('api_name', 'endpoint', name='uq_api_endpoint'),
        Index('idx_api_rate_limits_name', 'api_name'),
    )
    
    def can_make_request(self) -> bool:
        """Check if we can make another request"""
        if not self.requests_limit:
            return True
        
        # Check if window has expired
        if self.window_start:
            window_expired = (datetime.utcnow() - self.window_start).total_seconds() / 60 > self.window_minutes
            if window_expired:
                self.requests_made = 0
                self.window_start = datetime.utcnow()
                return True
        
        return self.requests_made < self.requests_limit
    
    def record_request(self):
        """Record that a request was made"""
        self.requests_made += 1
        self.last_request = datetime.utcnow()
        
        if self.monthly_quota:
            self.monthly_used += 1


# Additional models for resume and matching (kept simple for now)
class Resume(Base):
    """Resume model"""
    __tablename__ = 'resumes'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    user_id = Column(UUID)  # Future user system
    
    # Content
    fulltext = Column(Text)
    sections_json = Column(JSONB)
    skills_csv = Column(Text)
    
    # Experience
    yoe_raw = Column(Float)
    yoe_adjusted = Column(Float)
    edu_level = Column(String(50))
    
    # Files
    file_url = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    matches = relationship("Match", back_populates="resume")


class Match(Base):
    """Job-Resume match scores"""
    __tablename__ = 'matches'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    job_id = Column(UUID, ForeignKey('jobs.id'), nullable=False)
    resume_id = Column(UUID, ForeignKey('resumes.id'), nullable=False)
    
    # Scores
    vector_score = Column(Float)
    llm_score = Column(Integer)
    final_score = Column(Float)
    
    # Analysis
    action = Column(String(50))  # apply, review, skip
    reasoning = Column(Text)
    strengths = Column(JSONB)
    weaknesses = Column(JSONB)
    
    # Model info
    llm_model = Column(String(50))
    prompt_hash = Column(String(32))
    
    # Timestamps
    scored_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="matches")
    resume = relationship("Resume", back_populates="matches")
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('job_id', 'resume_id', name='uq_job_resume'),
        Index('idx_matches_scores', 'final_score', 'llm_score'),
    )


class Application(Base):
    """Track job applications"""
    __tablename__ = 'applications'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    job_id = Column(UUID, ForeignKey('jobs.id'), nullable=False)
    resume_id = Column(UUID, ForeignKey('resumes.id'))
    
    # Status
    status = Column(String(50))  # draft, submitted, rejected, interview, offer
    portal = Column(String(50))  # greenhouse, lever, email, etc
    
    # Tracking
    submitted_at = Column(DateTime)
    receipt_url = Column(String(500))
    confirmation_number = Column(String(100))
    
    # Response
    response_received = Column(Boolean, default=False)
    response_date = Column(DateTime)
    response_status = Column(String(50))
    
    # Metadata
    cover_letter = Column(Text)
    answers = Column(JSONB)  # Form field answers
    
    # Relationships
    job = relationship("Job", back_populates="applications")


# Event listeners for automatic updates
@event.listens_for(Job, 'before_insert')
def job_before_insert(mapper, connection, target):
    """Generate fingerprint before inserting job"""
    if not target.content_fingerprint:
        target.content_fingerprint = target.generate_fingerprint()


@event.listens_for(Job, 'before_update')
def job_before_update(mapper, connection, target):
    """Update fingerprint and track changes"""
    new_fingerprint = target.generate_fingerprint()
    if target.content_fingerprint != new_fingerprint:
        target.change_count = (target.change_count or 0) + 1
        target.content_fingerprint = new_fingerprint
        target.last_updated = datetime.utcnow()


# Create additional chunks tables for embeddings
class JobChunk(Base):
    """Job description chunks for embedding"""
    __tablename__ = 'job_chunks'
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    job_id = Column(UUID, ForeignKey('jobs.id'), nullable=False)
    
    chunk_text = Column(Text)
    chunk_index = Column(Integer)
    embedding = Column(JSON)  # Store as JSON array
    token_count = Column(Integer)
    
    # Relationships
    job = relationship("Job", back_populates="job_chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_job_chunks_job', 'job_id', 'chunk_index'),
    )