-- Database initialization for LazyJobSearch
-- This script sets up the initial database with required extensions

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create application schema (optional - using public for simplicity)
-- CREATE SCHEMA IF NOT EXISTS ljs;

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON DATABASE lazyjobsearch TO ljs_user;
GRANT ALL ON SCHEMA public TO ljs_user;

-- Set default search path
ALTER USER ljs_user SET search_path = public;

-- Note: Tables will be created by Alembic migrations
-- This script just sets up the basic database structure and extensions