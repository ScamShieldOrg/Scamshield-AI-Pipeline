-- Apply once to existing volumes; fresh volumes use db/init/001_ai_service.sql.
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true;
CREATE INDEX IF NOT EXISTS rag_chunks_bge_m3_cosine_idx
    ON rag_chunks USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
    WHERE is_active = true AND embedding_model = 'BAAI/bge-m3' AND embedding IS NOT NULL
      AND vector_dims(embedding) = 1024;
