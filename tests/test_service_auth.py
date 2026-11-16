from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.app as api


def test_openapi_describes_bearer_authentication():
    schema = api.app.openapi()
    assert schema["components"]["securitySchemes"]["ServiceBearer"]["type"] == "http"
    assert schema["components"]["securitySchemes"]["ServiceBearer"]["scheme"] == "bearer"
    for path in api.PROTECTED_PATHS:
        operation = schema["paths"][path]["post"]
        assert operation["security"] == [{"ServiceBearer": []}]
        assert "401" in operation["responses"]
    assert "security" not in schema["paths"]["/health"]["get"]


@pytest.mark.parametrize("path,payload", [
    ("/api/v1/scan", {"content": "private content", "check_type": "SMS"}),
    ("/api/v1/scan/raw", {"content": "private content", "check_type": "SMS"}),
    ("/api/v1/visobert/predict", {"message": "private content"}),
])
@pytest.mark.parametrize("authorization", [None, "Basic test-service-token", "Bearer wrong-token", "Bearer "])
def test_rejects_credentials_before_inference(monkeypatch, caplog, path, payload, authorization):
    router = MagicMock()
    monkeypatch.setattr(api, "router_instance", router)
    headers = {"Authorization": authorization} if authorization is not None else {}
    response = TestClient(api.app).post(path, json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": {"code": "UNAUTHORIZED"}}
    assert "private content" not in response.text
    assert router.method_calls == []
    assert "private content" not in caplog.text
    assert "test-service-token" not in caplog.text
    assert "httpStatus=401 errorCode=UNAUTHORIZED" in caplog.text


def test_correct_token_passes_authentication(monkeypatch):
    # Validation runs after authentication, so malformed bodies reach each route.
    monkeypatch.setattr(api, "router_instance", MagicMock())
    client = TestClient(api.app)
    for path in ("/api/v1/scan", "/api/v1/scan/raw", "/api/v1/visobert/predict"):
        response = client.post(path, json={}, headers={"Authorization": "Bearer test-service-token"})
        assert response.status_code == 422


def test_health_remains_public(monkeypatch):
    monkeypatch.setattr(api, "router_instance", None)
    monkeypatch.setattr(api, "check_database", lambda: False)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200


def test_missing_service_token_prevents_startup(monkeypatch):
    monkeypatch.delenv("AI_SERVICE_TOKEN")
    with pytest.raises(RuntimeError, match="AI_SERVICE_TOKEN is required"):
        api.startup_event()
