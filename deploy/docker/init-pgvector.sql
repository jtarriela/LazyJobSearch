-- Initialize pgvector extension for LazyJobSearch
-- This script runs when the PostgreSQL container starts

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For better full-text search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- For composite indexes

-- Create database if it doesn't exist (though it should from env vars)
-- SELECT 'CREATE DATABASE lazyjobsearch' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lazyjobsearch');

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE lazyjobsearch TO ljs_user;