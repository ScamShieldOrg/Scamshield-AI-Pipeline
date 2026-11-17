"""
Unit tests for Tier-2 Forensic Analysis Cache Manager (Phase 5 / GĐ5)
"""

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from pipeline.tier2_cache import (
    Tier2CacheManager,
    compute_exact_hash,
    compute_indicator_hash,
    extract_indicators,
)


def test_indicator_extraction():
    text = "Tai khoan VCB 0123456789 bi khoa. Truy cap https://vcb-fake.xyz hoac lien he 0912345678."
    indicators = extract_indicators(text)
    assert "https://vcb-fake.xyz" in indicators
    assert "0912345678" in indicators
    assert "0123456789" in indicators


def test_indicator_hash_difference_prevents_cache_poisoning():
    """If scammer reuses exact same wording but changes malicious URL, indicator hash MUST differ."""
    text1 = "Nhan thuong 500k tai https://gift-a.xyz"
    text2 = "Nhan thuong 500k tai https://gift-b.xyz"

    h1 = compute_indicator_hash(text1)
    h2 = compute_indicator_hash(text2)
    assert h1 != h2, "Indicator hash must differ when URLs differ!"


def test_tier2_cache_store_and_lookup():
    cache = Tier2CacheManager()

    content = "Viettel: Quy khach duoc tang 50k vao tai khoan."
    council_result = {
        "routing_path": "TIER_2_FORENSIC_COUNCIL",
        "is_scam": False,
        "final_confidence": 0.92,
        "risk_level": "LOW",
        "scam_type": "Chương trình khuyến mãi chính danh",
        "explanation": "Hội đồng 3 chuyên gia xác nhận không có yếu tố chiếm đoạt.",
        "scan_id": "SC-TEST-001",
        "tier1_telemetry": {"label": "HAM", "confidence": 0.82},
        "tier2_telemetry": {"status": "EXECUTED", "consensus_score": 0.08},
    }

    # Store
    stored = cache.store("MESSAGE", content, council_result)
    assert stored is True

    # Lookup hit
    hit = cache.lookup("MESSAGE", content)
    assert hit is not None
    assert hit["is_scam"] is False
    assert hit["decided_by"] == "CACHE"
    assert hit["hit_count"] == 1

    # Second lookup increments hit_count
    hit2 = cache.lookup("MESSAGE", content)
    assert hit2 is not None
    assert hit2["hit_count"] == 2


def test_tier2_cache_rejects_tier1_fastpath():
    """Cache must ONLY store Tier-2 analyses, never Tier-1 fastpath or failures."""
    cache = Tier2CacheManager()

    content = "Thong bao giao dich."
    fastpath_result = {
        "routing_path": "TIER_1_DIRECT_FASTPATH",
        "is_scam": False,
        "final_confidence": 0.99,
    }

    stored = cache.store("MESSAGE", content, fastpath_result)
    assert stored is False, "Tier 1 fastpath must NOT be cached!"

    hit = cache.lookup("MESSAGE", content)
    assert hit is None


def test_tier2_cache_rejects_degradation_failure():
    """Cache must NEVER store graceful degradation or offline errors."""
    cache = Tier2CacheManager()

    content = "Nghi van lua dao."
    failure_result = {
        "routing_path": "TIER_1_GRACEFUL_DEGRADATION",
        "is_scam": False,
        "final_confidence": 0.85,
    }

    stored = cache.store("MESSAGE", content, failure_result)
    assert stored is False, "Degradation failures must NOT be cached!"
