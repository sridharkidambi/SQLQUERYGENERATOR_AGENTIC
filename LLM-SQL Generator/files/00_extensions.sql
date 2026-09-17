-- Runs before 01_schema.sql (docker-entrypoint-initdb.d executes scripts in
-- filename order). Creates the pgvector extension in the default `public`
-- schema, where ingestion_pipeline.py's schema_embeddings table also lives.
CREATE EXTENSION IF NOT EXISTS vector;
