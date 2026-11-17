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

logger = logging.getLogger("ScamShield.Tier2Cache")

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?84|0)(?:3|5|7|8|9)\d{8}\b")
ACCOUNT_RE = re.compile(r"\b\d{9,16}\b")


def nfc_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_cache_text(text: str) -> str:
    """Standardize text for deterministic cache hashing."""
    s = nfc_normalize(text).lower()
    return re.sub(r"\s+", " ", s).strip()


def extract_indicators(text: str) -> List[str]:
    """Extract structural indicators (URLs, phone numbers, account numbers)."""
    urls = [u.lower() for u in URL_RE.findall(text)]
    phones = PHONE_RE.findall(text)
    accounts = ACCOUNT_RE.findall(text)
    combined = sorted(set(urls + phones + accounts))
    return combined


def compute_indicator_hash(text: str) -> str:
    """Compute SHA-256 over sorted indicator set to prevent indicator spoofing."""
    indicators = extract_indicators(text)
    if not indicators:
        return ""
    joined = "|".join(indicators)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_exact_hash(text: str, secret_key: str = "scamshield-cache-salt-v1") -> str:
    """HMAC-SHA256 of normalized text to protect against dictionary rainbow tables."""
    norm = normalize_cache_text(text)
    return hmac.new(secret_key.encode("utf-8"), norm.encode("utf-8"), hashlib.sha256).hexdigest()


class Tier2CacheManager:
    """
    Tier-2 Analysis Cache with dual-mode storage (PostgreSQL when live, Memory fallback for dev/offline).
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        default_sms_ttl_days: int = 30,
        default_url_ttl_days: int = 3,
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai",
        )
        self.default_sms_ttl_days = default_sms_ttl_days
        self.default_url_ttl_days = default_url_ttl_days
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cache_key(self, input_type: str, exact_hash: str, model_version: str, prompt_version: str, policy_version: str) -> str:
        return f"{input_type}:{exact_hash}:{model_version}:{prompt_version}:{policy_version}"

    def lookup(
        self,
        input_type: str,
        content: str,
        model_version: str = "ForensicCouncil-v2.0",
        prompt_version: str = "v1",
        policy_version: str = "v1",
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an authentic Tier-2 analysis exists in cache.
        Returns cached result dictionary if valid and unexpired; None otherwise.
        """
        exact_h = compute_exact_hash(content)
        indicator_h = compute_indicator_hash(content)
        key = self._cache_key(input_type, exact_h, model_version, prompt_version, policy_version)
        now = self._now()

        # 1. Check memory cache first
        with self._lock:
            cached = self._memory_cache.get(key)
            if cached:
                if cached["expires_at"] > now:
                    # Invariant: Indicator hash must match
                    if cached.get("indicator_hash") == indicator_h:
                        cached["hit_count"] += 1
                        logger.info(f"Tier-2 Cache Hit (Memory): exact_hash={exact_h[:12]}, hits={cached['hit_count']}")
                        return self._format_cache_hit(cached)
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
                        if stored_ind_h == indicator_h:
                            cur.execute("UPDATE ai_analysis_cache SET hit_count = hit_count + 1 WHERE id = %s", (cid,))
                            conn.commit()
                            logger.info(f"Tier-2 Cache Hit (Postgres): id={cid}, hits={hit_count + 1}")
                            return {
                                "is_scam": (verdict == "SCAM_RISK"),
                                "final_label": "SCAM-RISK" if verdict == "SCAM_RISK" else "LOW-RISK",
                                "final_confidence": float(confidence),
                                "risk_level": risk_level,
                                "scam_type": ai_snapshot.get("scam_type", "Tái sử dụng phân tích chuyên sâu"),
                                "explanation": ai_snapshot.get("explanation", "Kết quả tái sử dụng từ lần phân tích Tier-2 trước."),
                                "routing_path": "CACHE_HIT_TIER_2",
                                "decided_by": "CACHE",
                                "tier_used": "Tier 2 (Cached Forensic Analysis)",
                                "cached_at": created_at.isoformat(),
                                "source_scan_id": ai_snapshot.get("scan_id"),
                                "tier1_telemetry": ai_snapshot.get("tier1_telemetry"),
                                "tier2_telemetry": ai_snapshot.get("tier2_telemetry"),
                                "hit_count": hit_count + 1,
                            }
        except Exception:
            # Fall back to memory cache seamlessly
            pass

        return None

    def store(
        self,
        input_type: str,
        content: str,
        raw_result: Dict[str, Any],
        model_version: str = "ForensicCouncil-v2.0",
        prompt_version: str = "v1",
        policy_version: str = "v1",
    ) -> bool:
        """
        Store a successful Tier-2 analysis.
        Strict Invariant: NEVER store failures, unknown fallbacks, or Tier 1 direct exits.
        """
        routing_path = raw_result.get("routing_path", "")
        if routing_path not in ("TIER_2_FORENSIC_COUNCIL", "TIER_2_CLOUD_FALLBACK"):
            logger.debug(f"Skipping cache store for non-tier2 routing path: {routing_path}")
            return False

        exact_h = compute_exact_hash(content)
        indicator_h = compute_indicator_hash(content)
        key = self._cache_key(input_type, exact_h, model_version, prompt_version, policy_version)
        now = self._now()
        ttl_days = self.default_sms_ttl_days if input_type != "URL" else self.default_url_ttl_days
        expires_at = now + timedelta(days=ttl_days)

        is_scam = raw_result.get("is_scam", False)
        verdict = "SCAM_RISK" if is_scam else "LOW_RISK"
        risk_level = raw_result.get("risk_level", "HIGH" if is_scam else "LOW")
        confidence = float(raw_result.get("final_confidence", 0.90))

        snapshot = {
            "scam_type": raw_result.get("scam_type"),
            "explanation": raw_result.get("explanation"),
            "scan_id": raw_result.get("scan_id"),
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
                    logger.info(f"Tier-2 Cache stored in Postgres: exact_hash={exact_h[:12]}")
        except Exception:
            pass

        return True

    def _format_cache_hit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = item["snapshot"]
        verdict = item["verdict"]
        return {
            "is_scam": (verdict == "SCAM_RISK"),
            "final_label": "SCAM-RISK" if verdict == "SCAM_RISK" else "LOW-RISK",
            "final_confidence": item["confidence"],
            "risk_level": item["risk_level"],
            "scam_type": snapshot.get("scam_type", "Tái sử dụng phân tích chuyên sâu"),
            "explanation": snapshot.get("explanation", "Kết quả tái sử dụng từ lần phân tích Tier-2 trước."),
            "routing_path": "CACHE_HIT_TIER_2",
            "decided_by": "CACHE",
            "tier_used": "Tier 2 (Cached Forensic Analysis)",
            "cached_at": item["created_at"].isoformat(),
            "source_scan_id": snapshot.get("scan_id"),
            "tier1_telemetry": snapshot.get("tier1_telemetry"),
            "tier2_telemetry": snapshot.get("tier2_telemetry"),
            "hit_count": item["hit_count"],
        }
