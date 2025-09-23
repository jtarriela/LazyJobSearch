"""embedding versioning, feedback loop, anti-bot session tables

Revision ID: 0003_embedding_feedback_antibot
Revises: 0002_portal_and_apply_extension
Create Date: 2025-09-23

Implements ADRs:
- 0006 Embedding Versioning & Migration
- 0007 Adaptive Matching Feedback Loop
- 0008 Production-Grade Anti-Bot Posture
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0003_embedding_feedback_antibot'
down_revision = '0002_portal_and_apply_extension'
branch_labels = None
depends_on = None

def upgrade():
    # Add embedding version columns if tables exist
    with op.batch_alter_table('job_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('embedding_version', sa.Text()))
        batch_op.add_column(sa.Column('embedding_model', sa.Text()))
        batch_op.add_column(sa.Column('needs_reembedding', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    with op.batch_alter_table('resume_chunks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('embedding_version', sa.Text()))
        batch_op.add_column(sa.Column('embedding_model', sa.Text()))
        batch_op.add_column(sa.Column('needs_reembedding', sa.Boolean(), server_default=sa.text('false'), nullable=False))

    # embedding_versions registry
    op.create_table(
        'embedding_versions',
        sa.Column('version_id', sa.Text(), primary_key=True),
        sa.Column('model_name', sa.Text(), nullable=False),
        sa.Column('dimensions', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('deprecated_at', sa.DateTime(timezone=True)),
        sa.Column('compatible_with', postgresql.ARRAY(sa.Text())),
    )

    # Feedback: match_outcomes
    op.create_table(
        'match_outcomes',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('match_id', sa.UUID(), nullable=False),
        sa.Column('got_response', sa.Boolean()),
        sa.Column('response_time_hours', sa.Integer()),
        sa.Column('got_interview', sa.Boolean()),
        sa.Column('got_offer', sa.Boolean()),
        sa.Column('user_satisfaction', sa.Integer()),
        sa.Column('captured_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id'], name='fk_match_outcomes_match'),
    )
    op.create_index('ix_match_outcomes_match', 'match_outcomes', ['match_id'])

    # Feedback: matching_feature_weights
    op.create_table(
        'matching_feature_weights',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('model_version', sa.Text()),
        sa.Column('weights_json', postgresql.JSONB(), nullable=False),
    )

    # Anti-bot: scrape_sessions
    op.create_table(
        'scrape_sessions',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('company_id', sa.UUID()),  # Optional FK to companies
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('finished_at', sa.DateTime(timezone=True)),
        sa.Column('profile_json', postgresql.JSONB()),
        sa.Column('proxy_identifier', sa.Text()),
        sa.Column('outcome', sa.Text()),
        sa.Column('metrics_json', postgresql.JSONB()),
    )
    op.create_index('ix_scrape_sessions_company', 'scrape_sessions', ['company_id'])

    # Helpful partial indexes for re-embedding backlog
    op.execute("CREATE INDEX IF NOT EXISTS idx_reembed_job_chunks ON job_chunks(embedding_version) WHERE needs_reembedding = true")
    op.execute("CREATE INDEX IF NOT EXISTS idx_reembed_resume_chunks ON resume_chunks(embedding_version) WHERE needs_reembedding = true")


def downgrade():
    # Drop indexes and tables in reverse
    op.execute('DROP INDEX IF EXISTS idx_reembed_resume_chunks')
    op.execute('DROP INDEX IF EXISTS idx_reembed_job_chunks')
    op.drop_index('ix_scrape_sessions_company', table_name='scrape_sessions')
    op.drop_table('scrape_sessions')
    op.drop_table('matching_feature_weights')
    op.drop_index('ix_match_outcomes_match', table_name='match_outcomes')
    op.drop_table('match_outcomes')
    op.drop_table('embedding_versions')

    with op.batch_alter_table('resume_chunks', schema=None) as batch_op:
        batch_op.drop_column('needs_reembedding')
        batch_op.drop_column('embedding_model')
        batch_op.drop_column('embedding_version')
    with op.batch_alter_table('job_chunks', schema=None) as batch_op:
        batch_op.drop_column('needs_reembedding')
        batch_op.drop_column('embedding_model')
        batch_op.drop_column('embedding_version')
