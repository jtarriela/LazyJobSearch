"""initial core tables (subset) placeholder

Revision ID: 0001_initial_core
Revises: 
Create Date: 2025-09-23

NOTE: This is a stub. Replace with actual SQLAlchemy model-generated migrations later.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial_core'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Example core tables (minimal skeleton) - adjust or remove once real models exist
    op.create_table(
        'companies',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('name', sa.Text()),
        sa.Column('website', sa.Text()),
        sa.Column('careers_url', sa.Text()),
        sa.Column('crawler_profile_json', sa.Text()),
    )
    op.create_table(
        'resumes',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('fulltext', sa.Text()),
        sa.Column('sections_json', sa.Text()),
        sa.Column('skills_csv', sa.Text()),
        sa.Column('yoe_raw', sa.Float()),
        sa.Column('yoe_adjusted', sa.Float()),
        sa.Column('edu_level', sa.Text()),
        sa.Column('file_url', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True)),
    )


def downgrade():
    op.drop_table('resumes')
    op.drop_table('companies')
