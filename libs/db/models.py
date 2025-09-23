"""SQLAlchemy model stubs matching documented schema.

NOTE: This is an initial placeholder; types and relationships are partial. The
schema validation script compares table + column names against markdown docs.

Future: Split into modules, add indexes, constraints, relationships.
"""
from __future__ import annotations
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, ForeignKey, DateTime, JSON
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()

def uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

class Job(Base):
    __tablename__ = "jobs"
    id = uuid_pk()
    company_id = Column(UUID(as_uuid=True), nullable=False)
    url = Column(Text, nullable=False, unique=True)
    title = Column(Text)
    location = Column(Text)
    seniority = Column(Text)
    jd_fulltext = Column(Text)
    jd_tsv = Column(TSVECTOR)
    jd_file_url = Column(Text)
    jd_skills_csv = Column(Text)
    scraped_at = Column(DateTime)
    scrape_fingerprint = Column(Text)

class JobChunk(Base):
    __tablename__ = "job_chunks"
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_text = Column(Text)
    # embedding vector omitted placeholder
    token_count = Column(Integer)

class Resume(Base):
    __tablename__ = "resumes"
    id = uuid_pk()
    fulltext = Column(Text)
    sections_json = Column(Text)
    skills_csv = Column(Text)
    yoe_raw = Column(Float)
    yoe_adjusted = Column(Float)
    edu_level = Column(Text)
    file_url = Column(Text)
    created_at = Column(DateTime)

class ResumeChunk(Base):
    __tablename__ = "resume_chunks"
    id = uuid_pk()
    resume_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_text = Column(Text)
    token_count = Column(Integer)

class Match(Base):
    __tablename__ = "matches"
    id = uuid_pk()
    job_id = Column(UUID(as_uuid=True), nullable=False)
    resume_id = Column(UUID(as_uuid=True), nullable=False)
    vector_score = Column(Float)
    llm_score = Column(Integer)
    action = Column(Text)
    reasoning = Column(Text)
    llm_model = Column(Text)
    prompt_hash = Column(Text)
    scored_at = Column(DateTime)

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
