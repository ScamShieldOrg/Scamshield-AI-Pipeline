"""Exercise the scan HTTP route and real routing/prompt construction offline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.app as api
from pipeline.two_tier_router import TwoTierRouter


@pytest.fixture
def scan_stack(monkeypatch):
    # Keep the real router, URL extraction/adapters, Council and HTTP serializer.
    # Replace model loading, external reputation services and cloud responses.
    with patch("pipeline.two_tier_router.LocalPhoBERTClassifier") as classifier, \
         patch("pipeline.two_tier_router.ModelRegistry") as registry:
        retriever = MagicMock()
        retriever.retrieve.return_value = ([{
            "chunk_id": 7, "title": "Reference", "source_url": "https://source.example/",
            "text": "Verified reference evidence",
        }], {"status": "USED", "sources": [], "cited_chunk_ids": []})
        router = TwoTierRouter(rag_retriever=retriever)
    classifier.return_value.predict.return_value = {
        "label": "HAM", "class_id": 0, "confidence": 0.636,
        "calibrated_scam_prob": 0.364, "latency_ms": 1,
    }
    for key in ("openai_api_key", "anthropic_api_key", "groq_api_key"):
        setattr(router.council, key, "")
    router.council.gemini_api_key = "test-key"
    provider = AsyncMock(return_value={
        "vote": "SCAM-RISK", "scam_prob": 0.9, "confidence": 0.9,
        "risk_level": "HIGH", "analysis": "Detected scam cues",
    })
    monkeypatch.setattr(router.council, "_call_gemini_api", provider)
    monkeypatch.setattr(router.url_signals.phishtank, "lookup", MagicMock(return_value={
        "source": "phishtank", "status": "NO_MATCH",
    }))
    monkeypatch.setattr(router.url_signals.safe_browsing, "lookup", MagicMock(return_value={
        "source": "safe_browsing", "status": "NO_MATCH",
    }))
    registry.return_value.generate_content.return_value = {
        "success": True, "text": '{"is_scam": true, "confidence": 0.9, "cited_chunk_ids": [7, 999]}',
        "tier": "lite", "model_used": "test-model", "latency_ms": 2,
    }
    monkeypatch.setattr(api, "router_instance", router)
    return TestClient(api.app), router, provider


@pytest.mark.parametrize("with_url", [False, True])
@pytest.mark.parametrize("route", ["fastpath", "council", "registry", "offline"])
def test_sms_scan_routes(scan_stack, with_url, route):
    client, router, provider = scan_stack
    content = "Xac minh 0912345678"
    if with_url:
        content += " https://bad.example/login?token=private"
    if route == "fastpath":
        router.tier1_classifier.predict.return_value.update(confidence=0.98)
    elif route in {"registry", "offline"}:
        router.council.evaluate_sync = MagicMock(side_effect=RuntimeError("Council unavailable"))
        if route == "offline":
            router.tier2_registry.generate_content.return_value = {"success": False, "error": "offline"}

    response = client.post("/api/v1/scan", json={"content": content, "check_type": "SMS"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    data = body["data"]
    assert data["tier_executed"] == (1 if route == "fastpath" else 2)
    assert data["verdict"]["risk_level"] == {
        "fastpath": "LOW_RISK", "council": "SCAM_RISK", "registry": "SCAM_RISK", "offline": "UNKNOWN",
    }[route]
    assert router.url_signals.phishtank.lookup.call_count == int(with_url)
    if route == "fastpath":
        router.rag_retriever.retrieve.assert_not_called()
        provider.assert_not_called()
        return
    normalized = router.rag_retriever.retrieve.call_args.args[0]
    assert "0912345678" not in normalized
    assert "[PHONE_REDACTED]" in normalized
    if route == "council":
        assert provider.await_count == len(router.council.specialists)
        prompts = [call.args[1] for call in provider.call_args_list]
        router.tier2_registry.generate_content.assert_not_called()
    else:
        prompts = [router.tier2_registry.generate_content.call_args.kwargs["prompt"]]
        if route == "registry":
            assert data["telemetry"]["tier2"]["rag"]["cited_chunk_ids"] == [7]
    for prompt in prompts:
        assert normalized in prompt
        assert "[chunk_id:7]" in prompt
        assert "0912345678" not in prompt
        assert "token=private" not in prompt
        assert ('"hostname": "bad.example"' in prompt) == with_url
        if not with_url:
            assert "Bằng chứng URL: []" in prompt


@pytest.mark.parametrize("matched", [False, True])
def test_direct_url_scan_skips_text_models(scan_stack, matched):
    client, router, provider = scan_stack
    router.url_signals.phishtank.lookup.return_value = {
        "source": "phishtank", "status": "MATCH" if matched else "NO_MATCH",
    }
    response = client.post("/api/v1/scan", json={"content": "https://bad.example/login", "check_type": "URL"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["tier_executed"] == 1
    assert data["verdict"]["risk_level"] == ("SCAM_RISK" if matched else "UNKNOWN")
    router.tier1_classifier.predict.assert_not_called()
    router.rag_retriever.retrieve.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize("citations,expected", [
    ([7, "7", 999, True, 7.9, None, "bogus"], [7]),
    ([], []), (None, []), ("7", []),
])
def test_council_citations_are_filtered_and_exposed(scan_stack, citations, expected):
    client, router, provider = scan_stack
    provider.return_value["cited_chunk_ids"] = citations
    response = client.post("/api/v1/scan", json={
        "content": "Công an thông báo phạt nguội", "check_type": "SMS", "force_tier2": True,
    })
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["telemetry"]["tier2"]["rag"]["cited_chunk_ids"] == expected
    for vote in data["telemetry"]["tier2"]["specialist_votes"].values():
        assert vote["cited_chunk_ids"] == expected
    for agent in data["council_breakdown"]["agents"]:
        assert agent["cited_chunk_ids"] == expected
    assert all('"cited_chunk_ids": []' in call.args[1] for call in provider.call_args_list)


def test_council_unions_different_agent_citations(scan_stack):
    client, router, provider = scan_stack
    chunks, meta = router.rag_retriever.retrieve.return_value
    chunks.append({**chunks[0], "chunk_id": 8})
    reply = provider.return_value
    provider.side_effect = [{**reply, "cited_chunk_ids": [7]},
                            {**reply, "cited_chunk_ids": [8, 999]},
                            {**reply, "cited_chunk_ids": []}]
    response = client.post("/api/v1/scan", json={"content": "Phạt nguội", "check_type": "SMS"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["telemetry"]["tier2"]["rag"]["cited_chunk_ids"] == [7, 8]
    assert sorted(agent["cited_chunk_ids"] for agent in data["council_breakdown"]["agents"]) == [[], [7], [8]]


@pytest.mark.parametrize("local_fallback", [False, True])
def test_council_without_evidence_or_cloud_never_invents_citations(scan_stack, local_fallback):
    client, router, provider = scan_stack
    provider.return_value["cited_chunk_ids"] = [7]
    if local_fallback:
        provider.side_effect = RuntimeError("Provider unavailable")
    else:
        router.rag_retriever = None
    response = client.post("/api/v1/scan", json={"content": "Phạt nguội", "check_type": "SMS"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["telemetry"]["tier2"]["rag"]["cited_chunk_ids"] == []
    assert all(agent["cited_chunk_ids"] == [] for agent in data["council_breakdown"]["agents"])


@pytest.mark.parametrize("content,check_type", [
    ("Tin nhan", "SMS"), ("Xem https://bad.example/", "SMS"), ("https://bad.example/", "URL"),
])
def test_scan_inference_error_returns_500(scan_stack, content, check_type):
    client, router, _ = scan_stack
    router.tier1_classifier.predict.side_effect = RuntimeError("Inference failed")
    router.url_signals.phishtank.lookup.side_effect = RuntimeError("Inference failed")
    with patch.object(api, "save_scan") as save:
        response = client.post("/api/v1/scan", json={"content": content, "check_type": check_type})
    assert response.status_code == 500
    assert response.json() == {"status": "error", "data": None, "error": "Model inference failed: Inference failed"}
    save.assert_not_called()
    error_schema = client.get("/openapi.json").json()["paths"]["/api/v1/scan"]["post"]["responses"]["500"]
    assert error_schema["content"]["application/json"]["schema"]["$ref"].endswith("/StandardScanResponse")


def test_scan_uninitialized_returns_503(monkeypatch):
    monkeypatch.setattr(api, "router_instance", None)
    monkeypatch.setattr(api, "startup_event", lambda: None)
    response = TestClient(api.app).post("/api/v1/scan", json={"content": "Tin nhan", "check_type": "SMS"})
    assert response.status_code == 503


def test_scan_invalid_request_returns_422(scan_stack):
    client, router, _ = scan_stack
    response = client.post("/api/v1/scan", json={"content": " ", "check_type": "SMS"})
    assert response.status_code == 422
    router.tier1_classifier.predict.assert_not_called()
