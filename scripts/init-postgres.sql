-- =============================================================================
-- DESPIECE-BOT - PostgreSQL Initialization Script
-- =============================================================================

-- Create extensions for full-text search and UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Create custom types for enums
CREATE TYPE document_status AS ENUM ('uploaded', 'processing', 'processed', 'failed');
CREATE TYPE user_role AS ENUM ('admin', 'engineer', 'viewer');
CREATE TYPE calculation_status AS ENUM ('pending', 'validated', 'failed', 'reviewed');

-- Performance optimizations
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET random_page_cost = 1.1;

-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create database specific configurations
ALTER DATABASE despiece_bot SET timezone TO 'UTC';
ALTER DATABASE despiece_bot SET log_statement TO 'all';
ALTER DATABASE despiece_bot SET log_min_duration_statement TO 1000;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE despiece_bot TO despiece_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO despiece_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO despiece_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO despiece_user;
