"""SQLAlchemy model stubs matching documented schema.

NOTE: This is an initial placeholder; types and relationships are partial. The
schema validation script compares table + column names against markdown docs.

Future: Split into modules, add indexes, constraints, relationships.
"""
from __future__ import annotations
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, JSON, LargeBinary, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR, JSONB, ARRAY
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()

def uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

# User model for user management
class User(Base):
    __tablename__ = "users"
    id = uuid_pk()
    email = Column(Text, nullable=False, unique=True)
    full_name = Column(Text)
    preferences_json = Column(JSONB)  # For user preferences like locations, keywords, etc.
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

# Missing Company model that's referenced by migrations
class Company(Base):
    __tablename__ = "companies"
    id = uuid_pk()
    name = Column(Text)
    website = Column(Text)
    careers_url = Column(Text)
    crawler_profile_json = Column(Text)

class Job(Base):
    __tablename__ = "jobs"
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    url = Column(Text, nullable=False, unique=True, index=True)  # Added index for frequent queries
    title = Column(Text)
    location = Column(Text)
    seniority = Column(Text)
    jd_fulltext = Column(Text)
    jd_tsv = Column(TSVECTOR)  # Automatically managed by trigger
    jd_file_url = Column(Text)
    jd_skills_csv = Column(Text)
    scraped_at = Column(DateTime, index=True)  # Added index for frequent queries by date
    scrape_fingerprint = Column(Text)

class JobChunk(Base):
    __tablename__ = "job_chunks"
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    chunk_text = Column(Text)
    embedding = Column(Vector)  # pgvector column
    token_count = Column(Integer)
    embedding_version = Column(Text)
    embedding_model = Column(Text)
    needs_reembedding = Column(Boolean, default=False, nullable=False)

class Resume(Base):
    __tablename__ = "resumes"
    id = uuid_pk()
    fulltext = Column(Text)
    sections_json = Column(JSONB)  # Changed to JSONB for better performance
    skills_csv = Column(Text)
    yoe_raw = Column(Float)
    yoe_adjusted = Column(Float)
    edu_level = Column(Text)
    file_url = Column(Text)
    created_at = Column(DateTime)
    # Added columns to match documentation
    version = Column(Integer, default=1)
    parent_resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'))
    metadata_tags = Column(ARRAY(Text))  # Added for user-provided tags
    description = Column(Text)  # Added for user summary
    active = Column(Boolean, default=True)  # Added for active status
    source_review_id = Column(UUID(as_uuid=True), ForeignKey('reviews.id'))  # Added for review tracking
    iteration_index = Column(Integer)  # Added to mirror review iteration
    # Added for deduplication - content hash for exact matches
    content_hash = Column(Text, unique=True, index=True)  # SHA256 hash of normalized content

class ResumeChunk(Base):
    __tablename__ = "resume_chunks"
    id = uuid_pk()
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id', ondelete='CASCADE'), nullable=False)
    chunk_text = Column(Text)
    embedding = Column(Vector)  # pgvector column
    token_count = Column(Integer)
    embedding_version = Column(Text)
    embedding_model = Column(Text)
    needs_reembedding = Column(Boolean, default=False, nullable=False)

class Match(Base):
    __tablename__ = "matches"
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), nullable=False)
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'), nullable=False)
    vector_score = Column(Float)
    llm_score = Column(Integer)
    action = Column(Text)
    reasoning = Column(Text)
    llm_model = Column(Text)
    prompt_hash = Column(Text)
    scored_at = Column(DateTime)

class MatchOutcome(Base):
    __tablename__ = "match_outcomes"
    id = uuid_pk()
    match_id = Column(UUID(as_uuid=True), ForeignKey('matches.id'), nullable=False)
    got_response = Column(Boolean)
    response_time_hours = Column(Integer)
    got_interview = Column(Boolean)
    got_offer = Column(Boolean)
    user_satisfaction = Column(Integer)
    captured_at = Column(DateTime)

class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'completed', 'superseded', 'cancelled')", name='ck_review_status'),
        CheckConstraint("satisfaction IN ('NEEDS_MORE', 'SATISFIED') OR satisfaction IS NULL", name='ck_review_satisfaction'),
    )
    id = uuid_pk()
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), nullable=False)
    # Added to match documentation
    iteration = Column(Integer, default=1)  # Added iteration number
    parent_review_id = Column(UUID(as_uuid=True), ForeignKey('reviews.id'))
    llm_score = Column(Integer)
    strengths_md = Column(Text)
    weaknesses_md = Column(Text)
    suggestions_md = Column(Text)  # Keeping from existing model
    # Added columns to match documentation
    improvement_plan_json = Column(JSONB)  # Added structured directives
    proposed_new_resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'))  # Added AI-generated candidate
    accepted_new_resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'))  # Added adopted version
    satisfaction = Column(Text)  # Added satisfaction tracking
    # Keep existing fields
    iteration_count = Column(Integer, default=0)
    status = Column(Text, default='pending')
    improvement_brief = Column(Text)  # Kept from existing model
    redact_note = Column(Text)  # Kept from existing model  
    created_at = Column(DateTime)

class ApplicationProfile(Base):
    __tablename__ = "application_profiles"
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True))  # FK to users when available
    name = Column(Text, nullable=False)
    answers_json = Column(JSONB)  # JSONB for better performance
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'))
    is_default = Column(Boolean, default=False)
    profile_name = Column(Text)  # Kept from existing model
    files_map_json = Column(Text)  # Kept from existing model
    default_profile = Column(Boolean)  # Kept from existing model
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint('job_id', 'application_profile_id', name='uq_application_job_profile'),
        CheckConstraint("status IN ('pending', 'submitted', 'rejected', 'accepted', 'withdrawn')", name='ck_application_status'),
    )
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'), nullable=False)
    application_profile_id = Column(UUID(as_uuid=True), ForeignKey('application_profiles.id'), nullable=False)
    portal_id = Column(UUID(as_uuid=True), ForeignKey('portals.id'))
    status = Column(Text, default='pending')
    external_id = Column(Text)  # Portal's application ID
    applied_at = Column(DateTime)
    last_status_check = Column(DateTime)
    metadata_json = Column(JSONB)
    # Kept from existing model
    user_id = Column(UUID(as_uuid=True))
    resume_id = Column(UUID(as_uuid=True))
    portal = Column(Text)
    submitted_at = Column(DateTime)
    receipt_url = Column(Text)
    error_text = Column(Text)

class ApplicationEvent(Base):
    __tablename__ = "application_events"
    id = uuid_pk()
    application_id = Column(UUID(as_uuid=True), ForeignKey('applications.id'), nullable=False)
    at = Column(DateTime)  # Column name from migration
    kind = Column(Text, nullable=False)  # Column name from migration
    detail_json = Column(JSON)  # Column name from migration
    # Kept from existing model
    event_type = Column(Text)
    payload_json = Column(Text)
    occurred_at = Column(DateTime)

class ApplicationArtifact(Base):
    __tablename__ = "application_artifacts"
    id = uuid_pk()
    application_id = Column(UUID(as_uuid=True), ForeignKey('applications.id'), nullable=False)
    type = Column(Text, nullable=False)  # Column name from migration
    url = Column(Text, nullable=False)  # Column name from migration
    sha256 = Column(Text)  # Column name from migration
    created_at = Column(DateTime)  # Column name from migration
    # Kept from existing model
    kind = Column(Text)
    file_url = Column(Text)

class Portal(Base):
    __tablename__ = "portals"
    id = uuid_pk()
    name = Column(Text)
    kind = Column(Text)  # Added to match documentation (Greenhouse | Lever | Workday | ...)
    created_at = Column(DateTime)  # Added to match documentation
    portal_type = Column(Text)  # Keep existing field

class PortalTemplate(Base):
    __tablename__ = "portal_templates"
    id = uuid_pk()
    portal_id = Column(UUID(as_uuid=True), ForeignKey('portals.id'), nullable=False)
    template_name = Column(Text)  # Added to match documentation
    template_json = Column(JSONB)  # Changed to JSONB for better performance
    version = Column(Integer)  # Added to match documentation
    created_at = Column(DateTime)  # Added to match documentation

class CompanyPortalConfig(Base):
    __tablename__ = "company_portal_configs"
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    portal_id = Column(UUID(as_uuid=True), ForeignKey('portals.id'), nullable=False)
    config_json = Column(JSONB)  # Changed to JSONB for better performance
    active = Column(Boolean, default=True)  # Added to match documentation
    created_at = Column(DateTime)  # Added to match documentation

class PortalFieldDictionary(Base):
    __tablename__ = "portal_field_dictionary"
    id = uuid_pk()
    field_key = Column(Text, nullable=False, unique=True)  # Added to match documentation
    label = Column(Text)  # Added to match documentation
    data_type = Column(Text)  # Added to match documentation
    key = Column(Text, nullable=False, unique=True)  # Keep existing field
    description = Column(Text)
    required = Column(Boolean, default=False)
    pii_level = Column(Integer, default=0)
    created_at = Column(DateTime)

# Models from migration 0003 that were missing

class EmbeddingVersion(Base):
    __tablename__ = "embedding_versions"
    version_id = Column(Text, primary_key=True)
    model_name = Column(Text, nullable=False)
    dimensions = Column(Integer, nullable=False)
    created_at = Column(DateTime)
    deprecated_at = Column(DateTime)
    compatible_with = Column(ARRAY(Text))

class MatchingFeatureWeight(Base):
    __tablename__ = "matching_feature_weights"
    id = uuid_pk()
    created_at = Column(DateTime)
    model_version = Column(Text)
    weights_json = Column(JSONB, nullable=False)

class ScrapeSession(Base):
    __tablename__ = "scrape_sessions"
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'))
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    profile_json = Column(JSONB)
    proxy_identifier = Column(Text)
    outcome = Column(Text)
    metrics_json = Column(JSONB)

class UserCredential(Base):
    __tablename__ = "user_credentials"
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True))
    portal_family = Column(Text, nullable=False)
    username = Column(Text, nullable=False)
    password_ciphertext = Column(LargeBinary)
    totp_secret_ciphertext = Column(LargeBinary)
    updated_at = Column(DateTime)

class Session(Base):
    __tablename__ = "sessions"
    id = uuid_pk()
    user_id = Column(UUID(as_uuid=True))
    portal_family = Column(Text, nullable=False)
    cookie_jar = Column(JSON, nullable=False)
    expires_at = Column(DateTime)
    created_at = Column(DateTime)
