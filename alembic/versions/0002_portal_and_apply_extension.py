"""portal & apply extension tables stub

Revision ID: 0002_portal_and_apply_extension
Revises: 0001_initial_core
Create Date: 2025-09-23

NOTE: This is a stub for new auto-apply related tables.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_portal_and_apply_extension'
down_revision = '0001_initial_core'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'portals',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('family', sa.Text(), nullable=False),
        sa.Column('notes', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_table(
        'portal_templates',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('portal_id', sa.UUID(), sa.ForeignKey('portals.id')),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('dsl_json', sa.JSON(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_portal_templates_portal_active', 'portal_templates', ['portal_id', 'is_active'])

    op.create_table(
        'company_portal_configs',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('company_id', sa.UUID(), sa.ForeignKey('companies.id')),
        sa.Column('portal_id', sa.UUID(), sa.ForeignKey('portals.id')),
        sa.Column('login_url', sa.Text()),
        sa.Column('apply_base_url', sa.Text()),
        sa.Column('quirks_json', sa.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table(
        'portal_field_dictionary',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('key', sa.Text(), nullable=False, unique=True),
        sa.Column('description', sa.Text()),
        sa.Column('required', sa.Boolean(), server_default=sa.text('false')),
        sa.Column('pii_level', sa.Integer(), server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table(
        'user_credentials',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('user_id', sa.UUID()),  # FK to users (add later when users table exists)
        sa.Column('portal_family', sa.Text(), nullable=False),
        sa.Column('username', sa.Text(), nullable=False),
        sa.Column('password_ciphertext', sa.LargeBinary(), nullable=False),
        sa.Column('totp_secret_ciphertext', sa.LargeBinary()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_user_credentials_user_family', 'user_credentials', ['user_id', 'portal_family'])

    op.create_table(
        'sessions',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('user_id', sa.UUID()),
        sa.Column('portal_family', sa.Text(), nullable=False),
        sa.Column('cookie_jar', sa.JSON(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_sessions_user_family', 'sessions', ['user_id', 'portal_family'])

    op.create_table(
        'application_events',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('application_id', sa.UUID()),  # FK to applications (add when applications table defined)
        sa.Column('at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('detail_json', sa.JSON()),
    )
    op.create_index('ix_application_events_app_at', 'application_events', ['application_id', 'at'])

    op.create_table(
        'application_artifacts',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('application_id', sa.UUID()),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('sha256', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_application_artifacts_app', 'application_artifacts', ['application_id'])


def downgrade():
    op.drop_index('ix_application_artifacts_app', table_name='application_artifacts')
    op.drop_table('application_artifacts')
    op.drop_index('ix_application_events_app_at', table_name='application_events')
    op.drop_table('application_events')
    op.drop_index('ix_sessions_user_family', table_name='sessions')
    op.drop_table('sessions')
    op.drop_index('ix_user_credentials_user_family', table_name='user_credentials')
    op.drop_table('user_credentials')
    op.drop_table('portal_field_dictionary')
    op.drop_table('company_portal_configs')
    op.drop_index('ix_portal_templates_portal_active', table_name='portal_templates')
    op.drop_table('portal_templates')
    op.drop_table('portals')
