"""
Integration Tests for 2-Tier Cascaded AI Router with 3-Agent Forensic Council & API Endpoints
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from fastapi.testclient import TestClient
from api.app import app
from pipeline.two_tier_router import TwoTierRouter


@pytest.fixture
def client():
    return TestClient(app)


def test_tier1_fastpath_direct_exit_ham():
    """Verify that highly confident HAM messages exit via Tier 1 without calling Tier 2."""
    router = TwoTierRouter(use_council=True)
    # Mock Tier 1 to return very low scam prob (HAM)
    with patch.object(router.tier1_classifier, "predict") as mock_t1:
        mock_t1.return_value = {
            "label": "HAM",
            "class_id": 0,
            "confidence": 0.98,
            "calibrated_scam_prob": 0.02,
            "latency_ms": 25.0,
            "matched_cues": [],
            "temperature": 1.4015,
            "threshold_low": 0.35,
            "threshold_high": 0.85,
            "engine": "onnx",
        }
        res = router.classify("Thong bao: Quy khach da thanh toan tien dien thanh cong 500.000d.")
        assert res["routing_path"] == "TIER_1_DIRECT_FASTPATH"
        assert res["escalated"] is False
        assert res["final_label"] in ["HAM", "LOW-RISK"]
        assert res["total_cost_usd"] == 0.0


def test_tier1_fastpath_direct_exit_scam():
    """Verify that highly confident SCAM messages exit via Tier 1 without calling Tier 2."""
    router = TwoTierRouter(use_council=True)
    with patch.object(router.tier1_classifier, "predict") as mock_t1:
        mock_t1.return_value = {
            "label": "SCAM",
            "class_id": 1,
            "confidence": 0.99,
            "calibrated_scam_prob": 0.99,
            "latency_ms": 28.0,
            "matched_cues": ["SuspiciousDomain[xyz]"],
            "temperature": 1.4015,
            "threshold_low": 0.35,
            "threshold_high": 0.85,
            "engine": "onnx",
        }
        res = router.classify("Tai khoan cua ban bi khoa, bam vao https://vcb-scam.xyz de mo khoa.")
        assert res["routing_path"] == "TIER_1_DIRECT_FASTPATH"
        assert res["escalated"] is False
        assert res["final_label"] in ["SCAM", "SCAM-RISK"]
        assert res["total_cost_usd"] == 0.0


def test_tier1_review_escalation_to_3agent_council():
    """Verify that ambiguous messages in REVIEW band escalate to Tier 2 3-Agent Council."""
    router = TwoTierRouter(use_council=True)
    with patch.object(router.tier1_classifier, "predict") as mock_t1:
        mock_t1.return_value = {
            "label": "REVIEW",
            "class_id": 0,
            "confidence": 0.55,
            "calibrated_scam_prob": 0.55,
            "latency_ms": 30.0,
            "matched_cues": [],
            "temperature": 1.4015,
            "threshold_low": 0.35,
            "threshold_high": 0.85,
            "engine": "onnx",
        }
        # Mock Tier 2 3-Agent Council
        mock_council_res = {
            "message": "Ngan hang thong bao cap nhat sinh trac hoc theo QĐ 2345.",
            "final_label": "SCAM-RISK",
            "consensus_scam_prob": 0.82,
            "risk_level": "HIGH",
            "total_latency_ms": 1200.0,
            "early_exit_triggered": False,
            "defender_intervened": False,
            "total_api_calls": 3,
            "specialist_votes": {
                "cyber_financial_analyst": {"role_id": "ROLE_CYBER_BANK", "vote": "SCAM-RISK", "scam_prob": 0.90, "confidence": 0.90, "analysis": "Fake link", "display_name": "Cyber-Bank Analyst", "model_used": "gpt-6.1-sol"},
                "psycholinguistic_slang_profiler": {"role_id": "ROLE_PSYCHO_TEENCODE", "vote": "SCAM-RISK", "scam_prob": 0.80, "confidence": 0.80, "analysis": "Urgency cues", "display_name": "Psycholinguistic Profiler", "model_used": "gpt-6.1-sol"},
                "public_defender_advocate": {"role_id": "ROLE_DEFENDER", "vote": "LOW-RISK", "scam_prob": 0.30, "confidence": 0.70, "analysis": "Mentions QD 2345", "display_name": "Public Defender", "model_used": "gpt-6.1-sol"},
            },
            "unified_rationale": "Council unanimously detected phishing intent.",
            "active_council_size": 3,
            "estimated_council_cost_usd": 0.00045,
        }
        with patch.object(router.council, "evaluate_sync", return_value=mock_council_res):
            res = router.classify("Ngan hang thong bao cap nhat sinh trac hoc theo QĐ 2345.")
            assert res["routing_path"] == "TIER_2_FORENSIC_COUNCIL"
            assert res["escalated"] is True
            assert res["final_label"] == "SCAM-RISK"
            assert res["tier2_telemetry"]["architecture"] == "3-Agent Multi-LLM Heterogeneous Council"
            assert res["tier2_telemetry"]["status"] == "EXECUTED"


def test_api_scan_endpoint_schema_contract(client):
    """Verify /api/v1/scan adheres to the standard schema expected by Spring Boot backend."""
    payload = {"content": "Vietcombank: Giao dich thanh toan 50.000 VND tai Circle K.", "check_type": "SMS"}
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert "scan_id" in data
    assert "tier_executed" in data
    assert "verdict" in data
    assert "radar_metrics" in data
    assert "telemetry" in data
    assert data["verdict"]["risk_level"] in ["SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"]


def test_scan_requires_content_and_does_not_document_message(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ScanRequest"]
    assert "content" in schema["required"]
    assert "message" not in schema["properties"]
    assert client.post("/api/v1/scan", json={"message": "Tin nhan cu", "check_type": "SMS"}).status_code == 422


@pytest.mark.parametrize("check_type", ["TEXT", "PHONE", "BANK_ACCOUNT"])
def test_scan_rejects_retired_check_types(client, check_type):
    response = client.post("/api/v1/scan", json={"content": "0123456789", "check_type": check_type})
    assert response.status_code == 422


def test_scan_requires_explicit_check_type(client):
    assert client.post("/api/v1/scan", json={"content": "Tin nhan"}).status_code == 422


def test_api_url_without_match_is_unknown_and_skips_visobert(client):
    router = TwoTierRouter(use_council=False)
    router.url_signals.analyze = MagicMock(return_value={
        "known_threat_match": False, "normalized_value": "https://translate.google.com/",
        "hostname": "translate.google.com",
        "signals": [{"source": "phishtank", "status": "NO_MATCH"},
                    {"source": "safe_browsing", "status": "NO_MATCH"}],
    })
    with patch("api.app.router_instance", router), patch.object(router.tier1_classifier, "predict") as predict, patch("api.app.save_scan") as save:
        response = client.post("/api/v1/scan", json={"content": "https://translate.google.com/", "check_type": "URL"})
    assert response.status_code == 200
    assert response.json()["data"]["verdict"]["risk_level"] == "UNKNOWN"
    assert response.json()["data"]["verdict"]["risk_score"] is None
    predict.assert_not_called()
    assert save.call_args.kwargs["input_type"] == "URL"
