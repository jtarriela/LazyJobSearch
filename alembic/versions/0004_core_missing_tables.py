"""Add missing core tables (jobs, job_chunks, resume_chunks, matches, reviews, applications)

Revision ID: 0004_core_missing_tables
Revises: 0003_embedding_feedback_antibot
Create Date: 2024-01-15

This migration adds the remaining core tables that were missing from the initial migrations.
Based on the comprehensive gap analysis, these tables are critical for MVP functionality.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0004_core_missing_tables'
down_revision = '0003_embedding_feedback_antibot'
branch_labels = None
depends_on = None

def upgrade():
    # Create jobs table (referenced by other migrations but never created)
    op.create_table(
        'jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('company_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.Text()),
        sa.Column('location', sa.Text()),
        sa.Column('seniority', sa.Text()),
        sa.Column('jd_fulltext', sa.Text()),
        sa.Column('jd_tsv', postgresql.TSVECTOR()),
        sa.Column('jd_file_url', sa.Text()),
        sa.Column('jd_skills_csv', sa.Text()),
        sa.Column('scraped_at', sa.DateTime(timezone=True)),
        sa.Column('scrape_fingerprint', sa.Text()),
        sa.ForeignKeyConstraint(['company_id'], ['companies.id'], name='fk_jobs_company'),
    )
    op.create_unique_index('jobs_url_uidx', 'jobs', ['url'])
    op.create_index('ix_jobs_company', 'jobs', ['company_id'])
    op.create_index('ix_jobs_scraped_at', 'jobs', ['scraped_at'])

    # Create job_chunks table 
    op.create_table(
        'job_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_text', sa.Text()),
        sa.Column('embedding', sa.Text()),  # Vector type will be added in extensions migration
        sa.Column('token_count', sa.Integer()),
        sa.Column('embedding_version', sa.Text()),
        sa.Column('embedding_model', sa.Text()),
        sa.Column('needs_reembedding', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name='fk_job_chunks_job', ondelete='CASCADE'),
    )
    op.create_index('ix_job_chunks_job', 'job_chunks', ['job_id'])

    # Create resume_chunks table (referenced in migration 0003 but never created)
    op.create_table(
        'resume_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_text', sa.Text()),
        sa.Column('embedding', sa.Text()),  # Vector type will be added in extensions migration
        sa.Column('token_count', sa.Integer()),
        sa.Column('embedding_version', sa.Text()),
        sa.Column('embedding_model', sa.Text()),
        sa.Column('needs_reembedding', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], name='fk_resume_chunks_resume', ondelete='CASCADE'),
    )
    op.create_index('ix_resume_chunks_resume', 'resume_chunks', ['resume_id'])

    # Create matches table (referenced by match_outcomes FK but never created)
    op.create_table(
        'matches',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vector_score', sa.Float()),
        sa.Column('llm_score', sa.Integer()),
        sa.Column('action', sa.Text()),
        sa.Column('reasoning', sa.Text()),
        sa.Column('llm_model', sa.Text()),
        sa.Column('prompt_hash', sa.Text()),
        sa.Column('scored_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name='fk_matches_job'),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], name='fk_matches_resume'),
    )
    op.create_index('ix_matches_job_resume', 'matches', ['job_id', 'resume_id'])
    op.create_index('ix_matches_resume', 'matches', ['resume_id'])
    op.create_index('ix_matches_scored_at', 'matches', ['scored_at'])

    # Create reviews table (model exists but no migration)
    op.create_table(
        'reviews',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('llm_score', sa.Integer()),
        sa.Column('strengths_md', sa.Text()),
        sa.Column('weaknesses_md', sa.Text()),
        sa.Column('suggestions_md', sa.Text()),
        sa.Column('iteration_count', sa.Integer(), server_default=sa.text('0')),
        sa.Column('parent_review_id', postgresql.UUID(as_uuid=True)),
        sa.Column('status', sa.Text(), server_default=sa.text("'pending'")),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], name='fk_reviews_resume'),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name='fk_reviews_job'),
        sa.ForeignKeyConstraint(['parent_review_id'], ['reviews.id'], name='fk_reviews_parent'),
    )
    op.create_index('ix_reviews_resume', 'reviews', ['resume_id'])
    op.create_index('ix_reviews_job', 'reviews', ['job_id'])

    # Create application_profiles table (model exists but no migration)
    op.create_table(
        'application_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True)),  # FK to users when available
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('answers_json', postgresql.JSONB()),
        sa.Column('resume_id', postgresql.UUID(as_uuid=True)),
        sa.Column('is_default', sa.Boolean(), server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['resume_id'], ['resumes.id'], name='fk_application_profiles_resume'),
    )
    op.create_index('ix_application_profiles_user', 'application_profiles', ['user_id'])

    # Create applications table (referenced by application_events/artifacts)
    op.create_table(
        'applications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_profile_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('portal_id', postgresql.UUID(as_uuid=True)),
        sa.Column('status', sa.Text(), server_default=sa.text("'pending'")),
        sa.Column('external_id', sa.Text()),  # Portal's application ID
        sa.Column('applied_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_status_check', sa.DateTime(timezone=True)),
        sa.Column('metadata_json', postgresql.JSONB()),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name='fk_applications_job'),
        sa.ForeignKeyConstraint(['application_profile_id'], ['application_profiles.id'], name='fk_applications_profile'),
        sa.ForeignKeyConstraint(['portal_id'], ['portals.id'], name='fk_applications_portal'),
    )
    op.create_index('ix_applications_job', 'applications', ['job_id'])
    op.create_index('ix_applications_profile', 'applications', ['application_profile_id'])
    op.create_index('ix_applications_status', 'applications', ['status'])

    # Fix foreign keys in existing tables that reference these new tables
    with op.batch_alter_table('application_events', schema=None) as batch_op:
        batch_op.create_foreign_key('fk_application_events_application', ['application_id'], ['applications.id'])

    with op.batch_alter_table('application_artifacts', schema=None) as batch_op:
        batch_op.create_foreign_key('fk_application_artifacts_application', ['application_id'], ['applications.id'])


def downgrade():
    # Drop foreign key constraints first
    with op.batch_alter_table('application_artifacts', schema=None) as batch_op:
        batch_op.drop_constraint('fk_application_artifacts_application', type_='foreignkey')

    with op.batch_alter_table('application_events', schema=None) as batch_op:
        batch_op.drop_constraint('fk_application_events_application', type_='foreignkey')

    # Drop tables in reverse order
    op.drop_index('ix_applications_status', table_name='applications')
    op.drop_index('ix_applications_profile', table_name='applications')
    op.drop_index('ix_applications_job', table_name='applications')
    op.drop_table('applications')
    
    op.drop_index('ix_application_profiles_user', table_name='application_profiles')
    op.drop_table('application_profiles')
    
    op.drop_index('ix_reviews_job', table_name='reviews')
    op.drop_index('ix_reviews_resume', table_name='reviews')
    op.drop_table('reviews')
    
    op.drop_index('ix_matches_scored_at', table_name='matches')
    op.drop_index('ix_matches_resume', table_name='matches')
    op.drop_index('ix_matches_job_resume', table_name='matches')
    op.drop_table('matches')
    
    op.drop_index('ix_resume_chunks_resume', table_name='resume_chunks')
    op.drop_table('resume_chunks')
    
    op.drop_index('ix_job_chunks_job', table_name='job_chunks')
    op.drop_table('job_chunks')
    
    op.drop_index('ix_jobs_scraped_at', table_name='jobs')
    op.drop_index('ix_jobs_company', table_name='jobs')
    op.drop_index('jobs_url_uidx', table_name='jobs')
    op.drop_table('jobs')