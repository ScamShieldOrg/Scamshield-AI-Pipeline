import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import pytest
except ImportError:
    pytest = None
from fastapi.testclient import TestClient
from api.app import app, startup_event

# Run startup hook
startup_event()
client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ScamShield AI Pipeline Engine"
    assert data["tier1_ready"] is True


def test_scan_tier1_fastpath_scam():
    payload = {
        "content": "Vietcombank: Tai khoan cua ban tam khoa do bat thuong. Dang nhap https://vcb-xacthuc.xyz de mo khoa.",
        "check_type": "SMS"
    }
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert data["verdict"]["risk_level"] == "SCAM_RISK"
    assert data["tier_executed"] == 1
    assert len(data["actionable_recommendations"]) > 0


def test_scan_tier1_fastpath_ham():
    payload = {
        "content": "Ma OTP xac thuc giao dich ShopeePay cua ban la 839201. Ma co hieu luc trong 3 phut. Khong chia se cho bat ky ai.",
        "check_type": "SMS"
    }
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert data["verdict"]["risk_level"] == "LOW_RISK"
    assert data["tier_executed"] == 1


def test_visobert_direct_endpoint():
    payload = {
        "message": "Tai khoan cua quy khach bi khoa. Truy cap https://vcb-verify.xyz de xac thuc ngay."
    }
    response = client.post("/api/v1/visobert/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["SCAM", "REVIEW"]
    assert "scam_probability" in data
    assert "latency_ms" in data




def test_scan_with_sms_check_type_contract():
    """Verify GĐ1: Core Spring Boot sends check_type='SMS' and API successfully accepts it."""
    payload = {
        "content": "[QC] Mua data 5G nhan ngay voucher giam 50k tai Shopee.",
        "check_type": "SMS"
    }
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    v = res["data"]["verdict"]
    assert "analysis_status" in v
    assert "warning_level" in v
    assert "model_score" in v
    assert "decided_by" in v
    assert v["analysis_status"] == "OK"
    assert v["decided_by"] in ["TIER1", "COUNCIL", "CLOUD"]
    if v["risk_score"] < 70.0:
        assert v["risk_score"] != 88.5

if __name__ == "__main__":
    test_health_check()
    test_scan_tier1_fastpath_scam()
    test_scan_tier1_fastpath_ham()
    test_visobert_direct_endpoint()
    print("All tests passed successfully!")
