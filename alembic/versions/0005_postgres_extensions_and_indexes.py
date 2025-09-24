"""Add PostgreSQL extensions, FTS triggers, and vector indexes

Revision ID: 0005_postgres_extensions_and_indexes
Revises: 0004_core_missing_tables
Create Date: 2024-01-15

This migration implements the database requirements from ADR 0001 and ARCHITECTURE.md:
- PostgreSQL vector and pg_trgm extensions
- Full-text search indexing on jobs.jd_tsv with automatic triggers
- IVFFLAT vector indexes for similarity search
- Performance indexes for common queries

Critical for matching pipeline performance and vector search functionality.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0005_postgres_extensions_and_indexes'
down_revision = '0004_core_missing_tables'
branch_labels = None
depends_on = None

def upgrade():
    # Enable required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    
    # Convert embedding columns from Text to Vector type (requires extension to be enabled first)
    op.execute('ALTER TABLE job_chunks ALTER COLUMN embedding TYPE vector USING embedding::vector')
    op.execute('ALTER TABLE resume_chunks ALTER COLUMN embedding TYPE vector USING embedding::vector')
    
    # Create FTS trigger function for jobs table
    op.execute("""
        CREATE OR REPLACE FUNCTION jobs_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.jd_tsv := to_tsvector('english', coalesce(NEW.jd_fulltext, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger to automatically update jd_tsv on insert/update
    op.execute("""
        CREATE TRIGGER trg_jobs_tsv 
        BEFORE INSERT OR UPDATE ON jobs
        FOR EACH ROW EXECUTE FUNCTION jobs_tsv_update();
    """)
    
    # Create FTS GIN index on jobs.jd_tsv
    op.create_index('jobs_jd_tsv_gin', 'jobs', ['jd_tsv'], postgresql_using='gin')
    
    # Create vector similarity indexes (IVFFLAT) for job and resume chunks
    # Note: These indexes require data to be present for optimal performance
    # The lists parameter should be roughly sqrt(total_rows) but we start with 100
    op.execute("""
        CREATE INDEX job_chunks_embed_ivfflat 
        ON job_chunks USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
    """)
    
    op.execute("""
        CREATE INDEX resume_chunks_embed_ivfflat 
        ON resume_chunks USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
    """)
    
    # Create additional performance indexes based on common query patterns
    
    # Skills-based search on jobs
    op.create_index('ix_jobs_skills_gin', 'jobs', ['jd_skills_csv'], postgresql_using='gin', postgresql_ops={'jd_skills_csv': 'gin_trgm_ops'})
    
    # Company lookups
    op.create_index('ix_companies_name_trgm', 'companies', ['name'], postgresql_using='gin', postgresql_ops={'name': 'gin_trgm_ops'})
    
    # Resume skills search
    op.create_index('ix_resumes_skills_gin', 'resumes', ['skills_csv'], postgresql_using='gin', postgresql_ops={'skills_csv': 'gin_trgm_ops'})
    
    # Matching query optimizations
    op.create_index('ix_matches_vector_score', 'matches', ['vector_score'])
    op.create_index('ix_matches_llm_score', 'matches', ['llm_score'])
    
    # Application tracking indexes
    op.create_index('ix_applications_applied_at', 'applications', ['applied_at'])
    
    # Embedding reprocessing indexes (for version migrations)
    op.create_index('ix_job_chunks_needs_reembed', 'job_chunks', ['needs_reembedding', 'embedding_version'])
    op.create_index('ix_resume_chunks_needs_reembed', 'resume_chunks', ['needs_reembedding', 'embedding_version'])


def downgrade():
    # Drop indexes in reverse order
    op.drop_index('ix_resume_chunks_needs_reembed', table_name='resume_chunks')
    op.drop_index('ix_job_chunks_needs_reembed', table_name='job_chunks')
    op.drop_index('ix_applications_applied_at', table_name='applications')
    op.drop_index('ix_matches_llm_score', table_name='matches')
    op.drop_index('ix_matches_vector_score', table_name='matches')
    op.drop_index('ix_resumes_skills_gin', table_name='resumes')
    op.drop_index('ix_companies_name_trgm', table_name='companies')
    op.drop_index('ix_jobs_skills_gin', table_name='jobs')
    
    # Drop vector indexes
    op.execute('DROP INDEX IF EXISTS resume_chunks_embed_ivfflat')
    op.execute('DROP INDEX IF EXISTS job_chunks_embed_ivfflat')
    
    # Drop FTS index and trigger
    op.drop_index('jobs_jd_tsv_gin', table_name='jobs')
    op.execute('DROP TRIGGER IF EXISTS trg_jobs_tsv ON jobs')
    op.execute('DROP FUNCTION IF EXISTS jobs_tsv_update()')
    
    # Convert vector columns back to text (for compatibility)
    op.execute('ALTER TABLE resume_chunks ALTER COLUMN embedding TYPE text')
    op.execute('ALTER TABLE job_chunks ALTER COLUMN embedding TYPE text')
    
    # Note: We don't drop extensions as they might be used by other databases
    # Extensions should be managed at the database level, not in migrations