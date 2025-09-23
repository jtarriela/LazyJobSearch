"""Alembic environment configuration stub.

Populate target_metadata with SQLAlchemy models metadata once models are added.
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Placeholder for metadata import
# from app.db.models import Base  # noqa
# target_metadata = Base.metadata

target_metadata = None

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url", os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/lazyjobsearch"))
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    configuration = config.get_section(config.config_ini_section)
    if "sqlalchemy.url" not in configuration:
        configuration["sqlalchemy.url"] = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/lazyjobsearch")
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
