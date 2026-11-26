"""
==============================================================================
ScamShield-VN: Tier 2 Forensic Analysis Cache Manager (Phase 5 / GĐ5)
==============================================================================
Author: ScamShield Core Engineering Team
Description:
    Manages caching of expensive Tier-2 Forensic Council and Cloud analyses.
    Follows strict security and precision invariants:
      1. Precision over recall: Exact HMAC-SHA256 text match + indicator set match.
      2. No error caching: Only successful Tier-2 executions are stored.
      3. TTL bounds: 30 days for SMS, 3 days for dynamic URLs.
      4. Single-flight lock preventing duplicate concurrent council calls.
==============================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("ScamShield.Tier2Cache")

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?84|0)(?:3|5|7|8|9)\d{8}\b")
ACCOUNT_RE = re.compile(r"\b\d{9,16}\b")


def nfc_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_cache_url(text: str) -> str:
    """Canonicalize standalone URLs for deterministic cache matching (scheme/port/trailing-slash agnostic)."""
    raw = nfc_normalize(text).strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            return raw.lower()
        scheme = (parsed.scheme or "https").lower()
        port = parsed.port
        default_port = port is None or (port == 80 and scheme == "http") or (port == 443 and scheme == "https")
        netloc = host if default_port else f"{host}:{port}"
        path = parsed.path or ""
        if path == "/" and not parsed.query:
            path = ""
        elif path.endswith("/") and len(path) > 1 and not parsed.query:
            path = path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{netloc}{path}{query}"
    except ValueError:
        return raw.lower()


def normalize_cache_text(text: str, input_type: str = "MESSAGE") -> str:
    """Standardize text for deterministic cache hashing."""
    if input_type == "URL":
        return normalize_cache_url(text)
    s = nfc_normalize(text).lower()
    return re.sub(r"\s+", " ", s).strip()


def extract_indicators(text: str, input_type: str = "MESSAGE") -> List[str]:
    """Extract structural indicators (URLs, phone numbers, account numbers)."""
    if input_type == "URL":
        canonical = normalize_cache_url(text)
        return [canonical] if canonical else []
    urls = [u.lower() for u in URL_RE.findall(text)]
    phones = PHONE_RE.findall(text)
    accounts = ACCOUNT_RE.findall(text)
    combined = sorted(set(urls + phones + accounts))
    return combined


def compute_indicator_hash(text: str, input_type: str = "MESSAGE") -> str:
    """Compute SHA-256 over sorted indicator set to prevent indicator spoofing."""
    indicators = extract_indicators(text, input_type=input_type)
    if not indicators:
        return ""
    joined = "|".join(indicators)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_exact_hash(
    text: str,
    secret_key: str = "scamshield-cache-salt-v1",
    input_type: str = "MESSAGE",
) -> str:
    """HMAC-SHA256 of normalized text to protect against dictionary rainbow tables."""
    norm = normalize_cache_text(text, input_type=input_type)
    return hmac.new(secret_key.encode("utf-8"), norm.encode("utf-8"), hashlib.sha256).hexdigest()


DEFAULT_MODEL_VERSION = "ForensicCouncil-v2.0"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_POLICY_VERSION = "v2"


def _resolve_verdict_and_label(raw_result: Dict[str, Any]) -> tuple[str, str]:
    """Preserve exact tri-state verdicts (SCAM_RISK / LOW_RISK / UNKNOWN)."""
    raw_label = str(raw_result.get("final_label") or "").strip().upper()
    if raw_label in {"SCAM-RISK", "SCAM_RISK", "DANGEROUS", "SCAM"} or raw_result.get("is_scam") is True:
        return "SCAM_RISK", "SCAM-RISK"
    if raw_label in {"UNKNOWN", "REVIEW", "SUSPICIOUS_REVIEW"} or raw_result.get("is_unknown") is True:
        return "UNKNOWN", "UNKNOWN"
    return "LOW_RISK", "LOW-RISK"


class Tier2CacheManager:
    """
    Tier-2 Analysis Cache with dual-mode storage (PostgreSQL when live, Memory fallback for dev/offline)
    and in-process single-flight deduplication for concurrent identical requests.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        default_sms_ttl_days: int = 30,
        default_url_ttl_days: int = 3,
        default_unknown_ttl_hours: int = 24,
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai",
        )
        self.default_sms_ttl_days = default_sms_ttl_days
        self.default_url_ttl_days = default_url_ttl_days
        self.default_unknown_ttl_hours = default_unknown_ttl_hours
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._inflight: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cache_key(self, input_type: str, exact_hash: str, model_version: str, prompt_version: str, policy_version: str) -> str:
        return f"{input_type}:{exact_hash}:{model_version}:{prompt_version}:{policy_version}"

    def build_key(
        self,
        input_type: str,
        content: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        policy_version: str = DEFAULT_POLICY_VERSION,
    ) -> str:
        exact_h = compute_exact_hash(content, input_type=input_type)
        return self._cache_key(input_type, exact_h, model_version, prompt_version, policy_version)

    def acquire_inflight(
        self,
        input_type: str,
        content: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        policy_version: str = DEFAULT_POLICY_VERSION,
    ) -> tuple[str, bool, threading.Event]:
        """Single-flight coordinator: returns (key, is_leader, event)."""
        key = self.build_key(input_type, content, model_version, prompt_version, policy_version)
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return key, False, existing
            event = threading.Event()
            self._inflight[key] = event
            return key, True, event

    def release_inflight(self, key: str) -> None:
        """Signal all waiting followers and clear the in-flight entry."""
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def lookup(
        self,
        input_type: str,
        content: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        policy_version: str = DEFAULT_POLICY_VERSION,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an authentic Tier-2 analysis exists in cache.
        Returns cached result dictionary if valid and unexpired; None otherwise.
        """
        import time
        started = time.perf_counter()
        exact_h = compute_exact_hash(content, input_type=input_type)
        indicator_h = compute_indicator_hash(content, input_type=input_type)
        key = self._cache_key(input_type, exact_h, model_version, prompt_version, policy_version)
        now = self._now()

        # 1. Check memory cache first
        with self._lock:
            cached = self._memory_cache.get(key)
            if cached:
                if cached["expires_at"] > now:
                    # Invariant: Indicator hash must match
                    if (cached.get("indicator_hash") or "").strip() == indicator_h:
                        cached["hit_count"] += 1
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        logger.info(f"Tier-2 Cache Hit (Memory): exact_hash={exact_h[:12]}, hits={cached['hit_count']}")
                        return self._format_cache_hit(cached, lookup_latency_ms=elapsed_ms)
                else:
                    del self._memory_cache[key]

        # 2. Check Postgres DB if available
        try:
            import psycopg
            with psycopg.connect(self.database_url, connect_timeout=1) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, verdict, risk_level, confidence, routing_path, ai_snapshot, indicator_hash, hit_count, created_at
                        FROM ai_analysis_cache
                        WHERE input_type = %s AND exact_hash = %s
                          AND model_version = %s AND prompt_version = %s AND policy_version = %s
                          AND expires_at > %s
                        LIMIT 1
                        """,
                        (input_type, exact_h, model_version, prompt_version, policy_version, now),
                    )
                    row = cur.fetchone()
                    if row:
                        cid, verdict, risk_level, confidence, routing_path, ai_snapshot, stored_ind_h, hit_count, created_at = row
                        if (stored_ind_h or "").strip() == indicator_h:
                            cur.execute("UPDATE ai_analysis_cache SET hit_count = hit_count + 1 WHERE id = %s", (cid,))
                            conn.commit()
                            elapsed_ms = (time.perf_counter() - started) * 1000.0
                            logger.info(f"Tier-2 Cache Hit (Postgres): id={cid}, hits={hit_count + 1}")
                            item = {
                                "verdict": verdict,
                                "risk_level": risk_level,
                                "confidence": float(confidence),
                                "routing_path": routing_path,
                                "snapshot": ai_snapshot or {},
                                "model_version": model_version,
                                "hit_count": hit_count + 1,
                                "created_at": created_at,
                            }
                            return self._format_cache_hit(item, lookup_latency_ms=elapsed_ms)
        except Exception as exc:
            logger.warning("Tier-2 Cache Postgres lookup failed (falling back to memory): %s", exc)

        return None

    def store(
        self,
        input_type: str,
        content: str,
        raw_result: Dict[str, Any],
        model_version: str = DEFAULT_MODEL_VERSION,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        policy_version: str = DEFAULT_POLICY_VERSION,
    ) -> bool:
        """
        Store a successful Tier-2 analysis (including tri-state SCAM_RISK, LOW_RISK, UNKNOWN).
        Strict Invariant: NEVER store failures, degraded fallbacks, or Tier 1 direct exits.
        """
        routing_path = raw_result.get("routing_path", "")
        if routing_path not in ("TIER_2_FORENSIC_COUNCIL", "TIER_2_CLOUD_FALLBACK"):
            logger.debug(f"Skipping cache store for non-tier2 routing path: {routing_path}")
            return False

        exact_h = compute_exact_hash(content, input_type=input_type)
        indicator_h = compute_indicator_hash(content, input_type=input_type)
        key = self._cache_key(input_type, exact_h, model_version, prompt_version, policy_version)
        now = self._now()

        verdict, _ = _resolve_verdict_and_label(raw_result)
        if verdict == "UNKNOWN":
            expires_at = now + timedelta(hours=self.default_unknown_ttl_hours)
        else:
            ttl_days = self.default_sms_ttl_days if input_type != "URL" else self.default_url_ttl_days
            expires_at = now + timedelta(days=ttl_days)

        default_risk = "HIGH" if verdict == "SCAM_RISK" else ("UNKNOWN" if verdict == "UNKNOWN" else "LOW")
        risk_level = raw_result.get("risk_level") or default_risk
        confidence = float(raw_result.get("final_confidence", 0.90))

        snapshot = {
            "scam_type": raw_result.get("scam_type"),
            "explanation": raw_result.get("explanation"),
            "scan_id": raw_result.get("scan_id"),
            "model_name": raw_result.get("model_name"),
            "council_deliberation": raw_result.get("council_deliberation"),
            "tier1_telemetry": raw_result.get("tier1_telemetry"),
            "tier2_telemetry": raw_result.get("tier2_telemetry"),
        }

        # 1. Save to memory cache
        with self._lock:
            self._memory_cache[key] = {
                "input_type": input_type,
                "exact_hash": exact_h,
                "indicator_hash": indicator_h,
                "verdict": verdict,
                "risk_level": risk_level,
                "confidence": confidence,
                "routing_path": routing_path,
                "snapshot": snapshot,
                "model_version": model_version,
                "prompt_version": prompt_version,
                "policy_version": policy_version,
                "hit_count": 0,
                "created_at": now,
                "expires_at": expires_at,
            }

        # 2. Save to Postgres DB if available
        try:
            import psycopg
            from psycopg.types.json import Json
            with psycopg.connect(self.database_url, connect_timeout=1) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ai_analysis_cache (
                            input_type, exact_hash, indicator_hash, verdict, risk_level,
                            confidence, routing_path, ai_snapshot, model_version, prompt_version,
                            policy_version, created_at, expires_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (input_type, exact_hash, model_version, prompt_version, policy_version)
                        DO UPDATE SET
                            indicator_hash = EXCLUDED.indicator_hash,
                            verdict = EXCLUDED.verdict,
                            risk_level = EXCLUDED.risk_level,
                            confidence = EXCLUDED.confidence,
                            routing_path = EXCLUDED.routing_path,
                            ai_snapshot = EXCLUDED.ai_snapshot,
                            expires_at = EXCLUDED.expires_at
                        """,
                        (
                            input_type, exact_h, indicator_h, verdict, risk_level,
                            confidence, routing_path, Json(snapshot), model_version,
                            prompt_version, policy_version, now, expires_at
                        ),
                    )
                    conn.commit()
                    logger.info(f"Tier-2 Cache stored in Postgres: exact_hash={exact_h[:12]}, verdict={verdict}")
        except Exception as exc:
            logger.warning("Tier-2 Cache Postgres store failed (retained in memory): %s", exc)

        return True

    def _format_cache_hit(self, item: Dict[str, Any], lookup_latency_ms: float = 0.0) -> Dict[str, Any]:
        snapshot = item.get("snapshot") or {}
        verdict = item["verdict"]
        final_label = (
            "SCAM-RISK" if verdict == "SCAM_RISK"
            else "UNKNOWN" if verdict == "UNKNOWN"
            else "LOW-RISK"
        )
        return {
            "is_scam": (verdict == "SCAM_RISK"),
            "is_unknown": (verdict == "UNKNOWN"),
            "final_label": final_label,
            "final_confidence": float(item["confidence"]),
            "risk_level": item["risk_level"],
            "scam_type": snapshot.get("scam_type") or "Tái sử dụng phân tích chuyên sâu",
            "explanation": snapshot.get("explanation") or "Kết quả tái sử dụng từ lần phân tích Tier-2 trước.",
            "model_name": snapshot.get("model_name") or item.get("model_version") or DEFAULT_MODEL_VERSION,
            "routing_path": "CACHE_HIT_TIER_2",
            "routing_reason": "Tái sử dụng kết quả phân tích chuyên sâu trước đó (không gọi lại Cloud LLM).",
            "decided_by": "CACHE",
            "tier_used": "Tier 2 (Cached Forensic Analysis)",
            "escalated": True,
            "total_cost_usd": 0.0,
            "total_latency_ms": round(lookup_latency_ms, 2),
            "cached_at": item["created_at"].isoformat(),
            "source_scan_id": snapshot.get("scan_id"),
            "council_deliberation": snapshot.get("council_deliberation"),
            "tier1_telemetry": snapshot.get("tier1_telemetry"),
            "tier2_telemetry": snapshot.get("tier2_telemetry"),
            "hit_count": item["hit_count"],
        }
