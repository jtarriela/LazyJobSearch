"""SQLAlchemy models for LazyJobSearch with pgvector support.

This module contains the complete database schema including:
- Vector embeddings for semantic search
- Full-text search support  
- Resume and job description processing
- Application tracking and review system
"""
from __future__ import annotations
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, JSON,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()

def uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

def created_at():
    return Column(DateTime, nullable=False, default=datetime.utcnow)

class Company(Base):
    """Company information and scraping configuration."""
    __tablename__ = "companies"
    
    id = uuid_pk()
    name = Column(String(255), nullable=False, unique=True)
    website = Column(String(500))
    careers_url = Column(String(500))
    crawler_profile_json = Column(JSON)  # Selenium adapter config
    created_at = created_at()
    
    # Relationships
    jobs = relationship("Job", back_populates="company")
class Job(Base):
    """Job postings scraped from company career sites."""
    __tablename__ = "jobs"
    
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(Text)
    location = Column(Text)
    seniority = Column(String(100))
    jd_fulltext = Column(Text)
    jd_tsv = Column(TSVECTOR)  # Full-text search vector
    jd_file_url = Column(Text)  # S3/MinIO compressed storage
    jd_skills_csv = Column(Text)  # Extracted skills
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    scrape_fingerprint = Column(String(64))  # For change detection
    
    # Relationships
    company = relationship("Company", back_populates="jobs")
    chunks = relationship("JobChunk", back_populates="job")
    matches = relationship("Match", back_populates="job")
    
    __table_args__ = (
        UniqueConstraint('url', name='uq_job_url'),
        Index('ix_jobs_company_scraped', 'company_id', 'scraped_at'),
    )
    jd_file_url = Column(Text)
    jd_skills_csv = Column(Text)
    scraped_at = Column(DateTime)
    scrape_fingerprint = Column(Text)

class JobChunk(Base):
    """Semantic chunks of job descriptions with embeddings."""
    __tablename__ = "job_chunks"
    
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension
    token_count = Column(Integer)
    chunk_index = Column(Integer)  # Order within the job description
    
    # Relationships
    job = relationship("Job", back_populates="chunks")
    
    __table_args__ = (
        Index('ix_job_chunks_embedding', 'embedding', postgresql_using='ivfflat'),
    )

class Resume(Base):
    """User resume with parsed sections and metadata."""
    __tablename__ = "resumes"
    
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True))  # Future: link to user accounts
    fulltext = Column(Text)
    sections_json = Column(JSON)  # Structured sections (experience, education, etc.)
    skills_csv = Column(Text)  # Extracted skills
    yoe_raw = Column(Float)  # Years of experience calculated
    yoe_adjusted = Column(Float)  # With education bonus
    edu_level = Column(String(50))  # Bachelor's, Master's, PhD, etc.
    file_url = Column(Text)  # Original file location
    created_at = created_at()
    is_active = Column(Boolean, default=True)  # For A/B testing multiple resumes
    
    # Relationships
    chunks = relationship("ResumeChunk", back_populates="resume")
    matches = relationship("Match", back_populates="resume")

class ResumeChunk(Base):
    """Semantic chunks of resume content with embeddings."""
    __tablename__ = "resume_chunks"
    
    id = uuid_pk()
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension
    token_count = Column(Integer)
    section_type = Column(String(50))  # experience, education, skills, etc.
    chunk_index = Column(Integer)  # Order within the section
    
    # Relationships
    resume = relationship("Resume", back_populates="chunks")
    
    __table_args__ = (
        Index('ix_resume_chunks_embedding', 'embedding', postgresql_using='ivfflat'),
    )

class Match(Base):
    """Job-resume matching results with LLM scoring."""
    __tablename__ = "matches"
    
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), nullable=False)
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'), nullable=False)
    vector_score = Column(Float)  # Cosine similarity from vector search
    fts_rank = Column(Float)  # Full-text search ranking
    llm_score = Column(Integer)  # 0-100 LLM assessment
    action = Column(String(20))  # apply, skip, maybe
    reasoning = Column(Text)  # LLM explanation
    skill_gaps = Column(JSON)  # Missing required skills
    llm_model = Column(String(50))  # Which model was used
    prompt_hash = Column(String(64))  # For prompt change tracking
    scored_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="matches")
    resume = relationship("Resume", back_populates="matches")
    
    __table_args__ = (
        UniqueConstraint('job_id', 'resume_id', name='uq_match_job_resume'),
        Index('ix_matches_score_date', 'llm_score', 'scored_at'),
    )

class Review(Base):
    __tablename__ = "reviews"
    id = uuid_pk()
    resume_id = Column(UUID(as_uuid=True), nullable=False)
    job_id = Column(UUID(as_uuid=True), nullable=False)
    llm_score = Column(Integer)
    strengths_md = Column(Text)
    weaknesses_md = Column(Text)
    improvement_brief = Column(Text)
    redact_note = Column(Text)
    created_at = Column(DateTime)

class ApplicationProfile(Base):
    __tablename__ = "application_profiles"
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), nullable=False)
    profile_name = Column(Text)
    answers_json = Column(Text)  # store as JSON text for now
    files_map_json = Column(Text)
    default_profile = Column(Boolean)
    updated_at = Column(DateTime)

class Application(Base):
    __tablename__ = "applications"
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True), nullable=False)
    job_id = Column(UUID(as_uuid=True), nullable=False)
    resume_id = Column(UUID(as_uuid=True), nullable=False)
    application_profile_id = Column(UUID(as_uuid=True))
    status = Column(Text)
    portal = Column(Text)
    submitted_at = Column(DateTime)
    receipt_url = Column(Text)
    error_text = Column(Text)

class ApplicationEvent(Base):
    __tablename__ = "application_events"
    id = uuid_pk()
    application_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(Text)
    payload_json = Column(Text)
    occurred_at = Column(DateTime)

class ApplicationArtifact(Base):
    __tablename__ = "application_artifacts"
    id = uuid_pk()
    application_id = Column(UUID(as_uuid=True), nullable=False)
    kind = Column(Text)
    file_url = Column(Text)

class Portal(Base):
    __tablename__ = "portals"
    id = uuid_pk()
    name = Column(Text)
    portal_type = Column(Text)

class PortalTemplate(Base):
    __tablename__ = "portal_templates"
    id = uuid_pk()
    portal_id = Column(UUID(as_uuid=True), nullable=False)
    template_json = Column(Text)

class CompanyPortalConfig(Base):
    __tablename__ = "company_portal_configs"
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), nullable=False)
    portal_id = Column(UUID(as_uuid=True), nullable=False)
    config_json = Column(Text)

class PortalFieldDictionary(Base):
    __tablename__ = "portal_field_dictionary"
    id = uuid_pk()
    canonical_name = Column(Text)
    field_type = Column(Text)
    description = Column(Text)

# Remove duplicate Company class - keeping the one with relationships at the top
