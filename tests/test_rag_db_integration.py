"""Opt-in live PostgreSQL check for citation persistence across Core retries."""

import os
import uuid

import psycopg
import pytest
from dotenv import load_dotenv

from pipeline.persistence import save_scan


@pytest.mark.skipif(os.getenv("RAG_DB_INTEGRATION") != "1", reason="requires live PostgreSQL")
def test_rag_links_survive_retry_without_duplicates():
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    core_check_id = uuid.uuid4().int % (2**62)
    with psycopg.connect(database_url) as conn:
        chunk_id = conn.execute(
            """SELECT c.id FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id
               WHERE c.is_active AND d.is_active AND d.review_status = 'VERIFIED' LIMIT 1"""
        ).fetchone()[0]
    raw = {"final_label": "LOW-RISK", "model_name": "ForensicCouncil-v2.0",
           "tier1_telemetry": {"label": "HAM", "scam_prob": 0.1, "latency_ms": 1,
                               "threshold_low": 0.35, "threshold_high": 0.85, "temperature": 1.4},
           "tier2_telemetry": {"status": "EXECUTED", "latency_ms": 2, "cost_usd": 0,
                               "rag": {"status": "USED", "sources": [{"chunk_id": chunk_id,
                                        "similarity": 0.8}]}}}
    try:
        for _ in range(2):
            save_scan(content="Kiểm thử RAG", input_type="SMS", core_check_id=core_check_id,
                      raw_result=raw, verdict="LOW_RISK", confidence=0.1,
                      explanation="Kiểm thử", guidance="Kiểm thử")
            with psycopg.connect(database_url) as conn:
                counts = conn.execute(
                    """SELECT count(*) FROM inference_retrieved_chunks r
                       JOIN ai_inferences i ON i.id = r.inference_id
                       JOIN ai_jobs j ON j.id = i.ai_job_id
                       WHERE j.core_check_id = %s""", (core_check_id,)
                ).fetchone()[0]
                assert counts == 1
    finally:
        with psycopg.connect(database_url) as conn:
            job = conn.execute("SELECT id FROM ai_jobs WHERE core_check_id = %s", (core_check_id,)).fetchone()
            if job:
                conn.execute("""DELETE FROM inference_retrieved_chunks
                                WHERE inference_id IN (SELECT id FROM ai_inferences WHERE ai_job_id = %s)""", (job[0],))
                conn.execute("DELETE FROM ai_inferences WHERE ai_job_id = %s", (job[0],))
                conn.execute("DELETE FROM ai_jobs WHERE id = %s", (job[0],))
