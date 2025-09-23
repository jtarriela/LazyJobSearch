"""Initial schema with pgvector support

Revision ID: 0001_initial_core
Revises: 
Create Date: 2025-01-23

Comprehensive schema for LazyJobSearch with vector embeddings and full-text search.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001_initial_core'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    
    # Companies table
    op.create_table('companies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('website', sa.String(500)),
        sa.Column('careers_url', sa.String(500)),
        sa.Column('crawler_profile_json', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('name', name='uq_company_name')
    )
    
    # Jobs table
    op.create_table('jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('company_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('url', sa.Text, nullable=False),
        sa.Column('title', sa.Text),
        sa.Column('location', sa.Text),
        sa.Column('seniority', sa.String(100)),
        sa.Column('jd_fulltext', sa.Text),
        sa.Column('jd_tsv', postgresql.TSVECTOR),
        sa.Column('jd_file_url', sa.Text),
        sa.Column('jd_skills_csv', sa.Text),
        sa.Column('scraped_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('scrape_fingerprint', sa.String(64)),
        sa.ForeignKeyConstraint(['company_id'], ['companies.id']),
        sa.UniqueConstraint('url', name='uq_job_url')
    )
    
    # Job chunks with vector embeddings
    op.create_table('job_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_text', sa.Text, nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float), nullable=True),  # Will be vector(1536) in real usage
        sa.Column('token_count', sa.Integer),
        sa.Column('chunk_index', sa.Integer),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'])
    )
    
    # Resumes table
    op.create_table('resumes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True)),
        sa.Column('fulltext', sa.Text),
        sa.Column('sections_json', sa.JSON),
        sa.Column('skills_csv', sa.Text),
        sa.Column('yoe_raw', sa.Float),
        sa.Column('yoe_adjusted', sa.Float),
        sa.Column('edu_level', sa.String(50)),
        sa.Column('file_url', sa.Text),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean, default=True)
    )
    
    # Resume chunks with vector embeddings
    op.create_table('resume_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_text', sa.Text, nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float), nullable=True),  # Will be vector(1536) in real usage
        sa.Column('token_count', sa.Integer),
        sa.Column('section_type', sa.String(50)),
        sa.Column('chunk_index', sa.Integer),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'])
    )
    
    # Matches table for job-resume scoring
    op.create_table('matches',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vector_score', sa.Float),
        sa.Column('fts_rank', sa.Float),
        sa.Column('llm_score', sa.Integer),
        sa.Column('action', sa.String(20)),
        sa.Column('reasoning', sa.Text),
        sa.Column('skill_gaps', sa.JSON),
        sa.Column('llm_model', sa.String(50)),
        sa.Column('prompt_hash', sa.String(64)),
        sa.Column('scored_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id']),
        sa.UniqueConstraint('job_id', 'resume_id', name='uq_match_job_resume')
    )
    
    # Create indexes for performance
    op.create_index('ix_jobs_company_scraped', 'jobs', ['company_id', 'scraped_at'])
    op.create_index('ix_matches_score_date', 'matches', ['llm_score', 'scored_at'])
    op.create_index('ix_jobs_fulltext', 'jobs', ['jd_tsv'], postgresql_using='gin')


def downgrade():
    op.drop_table('matches')
    op.drop_table('resume_chunks')
    op.drop_table('resumes')
    op.drop_table('job_chunks')
    op.drop_table('jobs')
    op.drop_table('companies')
