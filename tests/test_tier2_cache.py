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


def test_url_cache_normalizes_scheme_port_and_trailing_slash():
    cache = Tier2CacheManager()
    council_result = {
        "routing_path": "TIER_2_FORENSIC_COUNCIL",
        "is_scam": True,
        "final_confidence": 0.94,
        "risk_level": "HIGH",
        "scam_type": "Phishing URL",
        "explanation": "Domain giả mạo ngân hàng.",
        "scan_id": "SC-URL-001",
    }

    assert cache.store("URL", "https://CPS.ONL:443/ymajtwftyg/", council_result) is True

    hit_no_scheme = cache.lookup("URL", "cps.onl/ymajtwftyg")
    assert hit_no_scheme is not None
    assert hit_no_scheme["decided_by"] == "CACHE"
    assert hit_no_scheme["is_scam"] is True

    hit_http = cache.lookup("URL", "http://cps.onl:80/ymajtwftyg#frag")
    assert hit_http is not None
    assert hit_http["hit_count"] == 2


def test_tier2_cache_preserves_unknown_verdict_with_24h_ttl():
    """Successful Tier-2 Council UNKNOWN verdict must be cached as UNKNOWN (not LOW_RISK) with 24h TTL."""
    cache = Tier2CacheManager(default_unknown_ttl_hours=24)
    content = 'Chao Nam Moi 2026 ! Tang Lop Hoc 2 Ngay, Hoc Qua Zoom "DONG TIEN HANH PHUC" Vao nhom Zalo https://zalo.me/g/wsqcpu908'
    council_unknown = {
        "routing_path": "TIER_2_FORENSIC_COUNCIL",
        "final_label": "UNKNOWN",
        "is_scam": False,
        "is_unknown": True,
        "final_confidence": 0.55,
        "risk_level": "UNKNOWN",
        "scam_type": "Chưa đủ cơ sở kết luận",
        "explanation": "Hội đồng chuyên gia nhận thấy chưa đủ bằng chứng kết luận lừa đảo.",
        "model_name": "ForensicCouncil-v1.1 (FullPanel)",
        "scan_id": "SC-UNK-001",
        "council_deliberation": {"summary": "Mixed signals"},
        "tier1_telemetry": {"label": "HAM", "confidence": 0.62},
        "tier2_telemetry": {
            "status": "EXECUTED",
            "consensus_score": 0.55,
            "specialist_votes": {
                "linguistic": {"display_name": "Linguistic", "vote": "UNKNOWN", "scam_prob": 0.55, "analysis": "Ambiguous"}
            },
        },
    }

    assert cache.store("MESSAGE", content, council_unknown) is True
    hit = cache.lookup("MESSAGE", content)
    assert hit is not None
    assert hit["final_label"] == "UNKNOWN"
    assert hit["is_scam"] is False
    assert hit["is_unknown"] is True
    assert hit["escalated"] is True
    assert hit["decided_by"] == "CACHE"
    assert hit["routing_path"] == "CACHE_HIT_TIER_2"
    assert hit["total_cost_usd"] == 0.0
    assert hit["model_name"] == "ForensicCouncil-v1.1 (FullPanel)"
    assert hit["council_deliberation"] == {"summary": "Mixed signals"}

    # Verify 24h TTL on memory entry
    key = cache.build_key("MESSAGE", content)
    entry = cache._memory_cache[key]
    ttl_hours = (entry["expires_at"] - entry["created_at"]).total_seconds() / 3600.0
    assert 23.9 <= ttl_hours <= 24.1


def test_single_flight_deduplicates_concurrent_identical_requests():
    import threading
    import time

    cache = Tier2CacheManager()
    content = "Tang khoa hoc Zoom mien phi https://zalo.me/g/test123"
    key1, is_leader1, ev1 = cache.acquire_inflight("MESSAGE", content)
    key2, is_leader2, ev2 = cache.acquire_inflight("MESSAGE", content)

    assert key1 == key2
    assert is_leader1 is True
    assert is_leader2 is False
    assert ev1 is ev2

    released = []

    def follower():
        ev2.wait(timeout=2.0)
        released.append(cache.lookup("MESSAGE", content))

    t = threading.Thread(target=follower)
    t.start()
    time.sleep(0.05)
    cache.store(
        "MESSAGE",
        content,
        {
            "routing_path": "TIER_2_FORENSIC_COUNCIL",
            "final_label": "UNKNOWN",
            "is_scam": False,
            "is_unknown": True,
            "final_confidence": 0.5,
            "risk_level": "UNKNOWN",
            "explanation": "Cached after leader finished.",
        },
    )
    cache.release_inflight(key1)
    t.join(timeout=2.0)

    assert len(released) == 1
    assert released[0] is not None
    assert released[0]["final_label"] == "UNKNOWN"
    assert released[0]["routing_path"] == "CACHE_HIT_TIER_2"
