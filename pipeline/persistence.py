"""Persistence for AI jobs and stage-level inference results."""

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg
from psycopg.types.json import Json
from .url_signals import council_evidence


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _masked(text: str) -> str:
    """Mask common contact and account identifiers before storing message text."""
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)", "[PHONE_OR_ACCOUNT]", text)
    return text


def _result(label: Optional[str]) -> str:
    if label in {"SCAM", "SCAM_RISK", "SCAM-RISK", "DANGEROUS"}:
        return "SCAM_RISK"
    if label in {"HAM", "LOW_RISK", "LOW-RISK", "SAFE"}:
        return "LOW_RISK"
    return "UNKNOWN"


def check_database() -> bool:
    """Return whether PostgreSQL accepts a short-lived connection."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai",
    )
    try:
        with psycopg.connect(database_url, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


def _model_id(cursor, name: str, kind: str, version: str, provider: str) -> int:
    cursor.execute(
        """INSERT INTO ai_models (model_name, model_type, provider, version, is_active, created_at)
           VALUES (%s, %s, %s, %s, true, %s)
           ON CONFLICT (model_name, version) DO UPDATE SET is_active = true
           RETURNING id""",
        (name, kind, provider, version, _now()),
    )
    return cursor.fetchone()[0]


def save_scan(
    *,
    content: str,
    input_type: str,
    core_check_id: Optional[int],
    raw_result: Dict[str, Any],
    verdict: str,
    confidence: float,
    explanation: str,
    guidance: str,
) -> None:
    """Atomically upsert a completed job and its executed inference stages."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai",
    )
    now = _now()
    tier1 = raw_result.get("tier1_telemetry") or {}
    tier2 = raw_result.get("tier2_telemetry") or {}
    t1_label = tier1.get("label")
    final_label = raw_result.get("final_label")
    llm_executed = tier2.get("status") == "EXECUTED"
    url_only = input_type == "URL"
    stage = "LLM" if llm_executed else "URL_REPUTATION" if url_only else "VISOBERT"
    stage_model_name = raw_result.get("model_name") if llm_executed else "ScamShield-URL-Reputation" if url_only else "ScamShield-ViSoBERT"
    stage_version = str(tier1.get("model_version", "1"))

    with psycopg.connect(database_url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            job_values = (
                str(uuid.uuid4()), core_check_id, input_type,
                hashlib.sha256(content.encode("utf-8")).hexdigest(), _masked(content),
                "COMPLETED", verdict, stage, None, confidence, explanation, guidance,
                now, now, now,
            )
            cursor.execute(
                """INSERT INTO ai_jobs
                   (id, core_check_id, input_type, input_hash, masked_input, status, verdict,
                    decided_by_stage, category_code, confidence, explanation, safety_guidance,
                    created_at, started_at, completed_at)
                   VALUES (%s, COALESCE(%s, -nextval('ai_service_generated_core_check_id_seq')), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (core_check_id) DO UPDATE SET
                    input_type = EXCLUDED.input_type, input_hash = EXCLUDED.input_hash,
                    masked_input = EXCLUDED.masked_input, status = EXCLUDED.status,
                    verdict = EXCLUDED.verdict, decided_by_stage = EXCLUDED.decided_by_stage,
                    category_code = EXCLUDED.category_code, confidence = EXCLUDED.confidence,
                    explanation = EXCLUDED.explanation, safety_guidance = EXCLUDED.safety_guidance,
                    completed_at = EXCLUDED.completed_at
                   RETURNING id""",
                job_values,
            )
            job_id = cursor.fetchone()[0]
            cursor.execute("""DELETE FROM inference_retrieved_chunks
                              WHERE inference_id IN (SELECT id FROM ai_inferences WHERE ai_job_id = %s)""", (job_id,))
            cursor.execute("DELETE FROM ai_inferences WHERE ai_job_id = %s", (job_id,))

            if url_only:
                reputation_id = _model_id(cursor, "ScamShield-URL-Reputation", "URL_REPUTATION", "1", "PhishTank+SafeBrowsing")
                cursor.execute(
                    """INSERT INTO ai_inferences
                       (ai_job_id, stage, model_id, result, confidence, explanation, raw_response,
                        attempt_no, latency_ms, api_cost, created_at)
                       VALUES (%s, 'URL_REPUTATION', %s, %s, %s, %s, %s, 1, %s, 0, %s)""",
                    (job_id, reputation_id, _result(final_label), confidence, explanation,
                     Json(council_evidence([tier1.get("url_evidence", {})])), int(float(tier1.get("latency_ms", 0))), now),
                )
            else:
                visobert_id = _model_id(cursor, "ScamShield-ViSoBERT", "VISOBERT_CLASSIFIER", stage_version, "local")
                low = float(tier1.get("threshold_low", 0.35))
                high = float(tier1.get("threshold_high", 0.85))
                temperature = float(tier1.get("temperature", 1.0))
                cursor.execute(
                    """INSERT INTO classifier_configs
                       (model_id, low_threshold, high_threshold, temperature, configuration_version,
                        is_active, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, 1, true, %s, %s)
                       ON CONFLICT (model_id, configuration_version) DO UPDATE SET
                        low_threshold = EXCLUDED.low_threshold, high_threshold = EXCLUDED.high_threshold,
                        temperature = EXCLUDED.temperature, updated_at = EXCLUDED.updated_at
                       RETURNING id""",
                    (visobert_id, low, high, temperature, now, now),
                )
                config_id = cursor.fetchone()[0]
                cursor.execute(
                    """INSERT INTO ai_inferences
                       (ai_job_id, stage, model_id, classifier_config_id, result, confidence,
                        escalation_reason, explanation, attempt_no, latency_ms, api_cost, created_at)
                       VALUES (%s, 'VISOBERT', %s, %s, %s, %s, %s, %s, 1, %s, 0, %s)""",
                    (
                        job_id, visobert_id, config_id, _result(t1_label),
                        float(tier1.get("scam_prob", confidence)),
                        "LOW_CONFIDENCE" if llm_executed else None,
                        tier1.get("reason"), int(float(tier1.get("latency_ms", 0))), now,
                    ),
                )

            if llm_executed:
                llm_id = _model_id(cursor, str(stage_model_name or "ScamShield-LLM"), "LLM", "1.1", "cloud")
                cursor.execute(
                    """INSERT INTO ai_inferences
                       (ai_job_id, stage, model_id, result, confidence, explanation, raw_response,
                        attempt_no, latency_ms, api_cost, fallback_used, created_at)
                       VALUES (%s, 'LLM', %s, %s, %s, %s, %s, 1, %s, %s, true, %s)
                       RETURNING id""",
                    (
                        job_id, llm_id, _result(final_label), confidence, explanation,
                        Json({key: value for key, value in tier2.items() if key != "rendered_prompt"}),
                        int(float(tier2.get("latency_ms", 0))), float(tier2.get("cost_usd", 0)), now,
                    ),
                )
                inference_id = cursor.fetchone()[0]
                rag = tier2.get("rag") or {}
                if rag.get("status") == "USED":
                    for rank, source in enumerate(rag.get("sources", []), start=1):
                        cursor.execute(
                            """INSERT INTO inference_retrieved_chunks
                               (inference_id, chunk_id, retrieval_rank, similarity_score,
                                retrieval_method, used_in_prompt, retrieved_at)
                               VALUES (%s,%s,%s,%s,'BGE_M3_DENSE_COSINE',true,%s)""",
                            (inference_id, source["chunk_id"], rank, source["similarity"], now),
                        )


def save_prediction(*, message: str, core_check_id: Optional[int], result: Dict[str, Any]) -> None:
    """Persist a direct ViSoBERT prediction using the same job and stage schema."""
    decision = result["decision"]
    save_scan(
        content=message,
        input_type="SMS",
        core_check_id=core_check_id,
        raw_result={
            "final_label": decision,
            "model_name": "ScamShield-ViSoBERT",
            "tier1_telemetry": {
                "label": decision,
                "scam_prob": result["scam_probability"],
                "threshold_low": result["threshold_low"],
                "threshold_high": result["threshold_high"],
                "temperature": result["temperature"],
                "latency_ms": result["latency_ms"],
                "engine": result["engine"],
                "model_version": "1",
            },
            "tier2_telemetry": {"status": "SKIPPED"},
        },
        verdict=_result(decision),
        confidence=float(result["confidence"]),
        explanation=f"ViSoBERT classified this message as {decision} with scam probability {result['scam_probability']:.4f}.",
        guidance="",
    )
