from unittest.mock import Mock

import pytest
import api.app as api
from api.schemas import ScanRequest
from pipeline.verdict_contract import normalize_verdict


@pytest.mark.parametrize("label,expected", [
    ("SCAM", "SCAM_RISK"), ("SCAM-RISK", "SCAM_RISK"),
    ("SCAM_RISK", "SCAM_RISK"), ("HAM", "LOW_RISK"),
    ("LOW-RISK", "LOW_RISK"), ("LOW_RISK", "LOW_RISK"),
    ("UNKNOWN", "UNKNOWN"), ("REVIEW", "UNKNOWN"),
    ("SUSPICIOUS_REVIEW", "UNKNOWN"), ("SUSPICIOUS", "UNKNOWN"),
    ("", "UNKNOWN"), ("invalid", "UNKNOWN"),
])
def test_public_verdict_and_persistence_agree(monkeypatch, label, expected):
    router = Mock()
    router.classify.return_value = {
        "final_label": label, "is_scam": True, "final_confidence": 0.7,
        "explanation": "Chưa đủ bằng chứng hoặc kết quả phân tích tiếng Việt.",
        "escalated": True,
        "tier2_telemetry": {"specialist_votes": {"reviewer": {"vote": label}}},
    }
    monkeypatch.setattr(api, "router_instance", router)
    saved = Mock()
    monkeypatch.setattr(api, "save_scan", saved)
    response = api.scan_message(ScanRequest(content="Bạn xem giúp thông báo này nhé", check_type="SMS"))
    assert normalize_verdict(label) == expected
    assert response.data.verdict.risk_level == expected
    assert response.data.verdict.summary.strip()
    assert response.data.council_breakdown.agents[0].verdict == expected
    assert saved.call_args.kwargs["verdict"] == expected
    if expected == "UNKNOWN":
        assert response.data.verdict.risk_score is None


def test_blank_explanation_gets_a_reason(monkeypatch):
    router = Mock()
    router.classify.return_value = {"final_label": "REVIEW", "explanation": ""}
    monkeypatch.setattr(api, "router_instance", router)
    assert api.scan_message(ScanRequest(content="Thông báo", check_type="SMS")).data.verdict.summary.strip()
