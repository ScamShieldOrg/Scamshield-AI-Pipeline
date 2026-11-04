"""Behavioral checks for URL-only routing and the two threat-list adapters."""

import csv
import gzip
import io
import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

import api.app as app_module
from pipeline.two_tier_router import TwoTierRouter
from pipeline.safe_browsing_proto import SearchUrlsResponse
from pipeline.url_signals import PhishTankFeed, SafeBrowsingLookup, UrlSignals, extract_urls, normalize_url
from scripts.update_phishtank import main as download_phishtank_feed


def test_normalize_url_rejects_spoofed_or_non_public_hosts():
    assert normalize_url("https://translate.google.com/")[1] == "translate.google.com"
    assert normalize_url("https://translate.google.com.evil.example/")[1] == "translate.google.com.evil.example"
    for value in ["https://google.com@evil.example/", "http://127.0.0.1/", "http://localhost/", "https://google.com\\@evil.example/"]:
        with pytest.raises(ValueError):
            normalize_url(value)


def test_phishtank_feed_exact_url_even_when_snapshot_is_old(tmp_path):
    path = tmp_path / "feed.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phish_id", "url", "phish_detail_url", "submission_time", "verified", "verification_time", "online", "target"])
        writer.writeheader()
        writer.writerow({"phish_id": "42", "url": "https://bad.example/login", "verified": "yes", "online": "yes", "target": "Example Bank"})
        writer.writerow({"phish_id": "43", "url": "https://old.example/", "verified": "no", "online": "yes"})
    old_time = time.time() - 30 * 24 * 60 * 60
    os.utime(path, (old_time, old_time))
    feed = PhishTankFeed(path)
    assert feed.lookup("https://bad.example/login")["status"] == "MATCH"
    assert feed.lookup("https://bad.example/other")["status"] == "NO_MATCH"
    assert feed.lookup("https://old.example/")["status"] == "NO_MATCH"


def test_phishtank_downloader_downloads_only_when_missing(tmp_path, monkeypatch):
    path = tmp_path / "feed.csv.gz"
    monkeypatch.setenv("PHISHTANK_FEED_PATH", str(path))
    monkeypatch.delenv("PHISHTANK_APP_KEY", raising=False)
    payload = io.BytesIO()
    with gzip.open(payload, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phish_id", "url", "verified", "online"])
        writer.writeheader()
        writer.writerow({"phish_id": "42", "url": "https://bad.example/login", "verified": "yes", "online": "yes"})
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [payload.getvalue()]
    with patch("scripts.update_phishtank.requests.get", return_value=response) as get:
        assert download_phishtank_feed() == 0
        assert path.is_file()
        assert download_phishtank_feed() == 0
        get.assert_called_once()
        assert get.call_args.args[0] == "https://data.phishtank.com/data/online-valid.csv.gz"


def test_api_starts_one_background_download_only_when_feed_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PHISHTANK_AUTO_DOWNLOAD", "true")
    with patch.object(app_module, "feed_path", return_value=tmp_path / "missing.csv.gz"), \
         patch.object(app_module, "TwoTierRouter"), \
         patch.object(app_module.threading, "Thread") as thread, \
         patch.object(app_module, "_phishtank_bootstrap_started", False), \
         patch.object(app_module, "router_instance", None):
        app_module.startup_event()
        app_module.startup_event()
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()


def test_api_logs_configured_sources_without_exposing_key(tmp_path, monkeypatch, caplog):
    path = tmp_path / "feed.csv.gz"
    path.write_bytes(b"existing snapshot")
    router = MagicMock()
    router.url_signals.safe_browsing.api_key = "secret-test-key"
    monkeypatch.setenv("PHISHTANK_AUTO_DOWNLOAD", "true")
    with caplog.at_level(logging.INFO, logger="ScamShield.API"), \
         patch.object(app_module, "feed_path", return_value=path), \
         patch.object(app_module, "TwoTierRouter", return_value=router), \
         patch.object(app_module.threading, "Thread") as thread, \
         patch.object(app_module, "router_instance", None):
        app_module.startup_event()
        thread.assert_not_called()
    assert "Safe Browsing configured" in caplog.text
    assert "PhishTank feed file found" in caplog.text
    assert "secret-test-key" not in caplog.text


def test_api_logs_background_download_failure(caplog):
    with patch.object(app_module, "download_phishtank_feed", return_value=1):
        app_module._download_missing_phishtank_feed()
    assert "PhishTank download failed" in caplog.text


def test_safe_browsing_v5_match_and_negative_cache():
    service = SafeBrowsingLookup(api_key="test-key")
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = {"threats": [{"url": "https://bad.example/", "threatTypes": ["SOCIAL_ENGINEERING"]}], "cacheDuration": "60s"}
    with patch("pipeline.url_signals.requests.get", return_value=response) as get:
        match = service.lookup("https://bad.example/")
        assert match["status"] == "MATCH"
        assert match["lookup_status"] == "SUCCESS"
        assert match["known_threat_match"] is True
        assert match["threat_types"] == ["SOCIAL_ENGINEERING"]
        assert service.lookup("https://bad.example/")["cached"] is True
        assert get.call_count == 1
    response.json.return_value = {"threats": [], "cacheDuration": "10s"}
    with patch("pipeline.url_signals.requests.get", return_value=response) as get:
        no_match = service.lookup("https://unknown.example/")
        assert no_match["status"] == "NO_MATCH"
        assert no_match["lookup_status"] == "SUCCESS"
        assert no_match["known_threat_match"] is False
        assert no_match["http_status"] == 200
        assert service.lookup("https://unknown.example/")["cached"] is True
        assert get.call_count == 1


def test_safe_browsing_v5_protobuf_match_and_negative_cache():
    service = SafeBrowsingLookup(api_key="test-key")
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/x-protobuf"}
    message = SearchUrlsResponse()
    message.cache_duration.seconds = 60
    threat = message.threats.add()
    threat.url = "https://bad.example/"
    threat.threat_types.append(2)
    response.content = message.SerializeToString()
    with patch("pipeline.url_signals.requests.get", return_value=response) as get:
        result = service.lookup("https://bad.example/")
        assert result["status"] == "MATCH"
        assert result["threat_types"] == ["SOCIAL_ENGINEERING"]
        assert service.lookup("https://bad.example/")["cached"] is True
        assert get.call_count == 1

    negative = SearchUrlsResponse()
    negative.cache_duration.seconds = 10
    response.content = negative.SerializeToString()
    with patch("pipeline.url_signals.requests.get", return_value=response) as get:
        result = service.lookup("https://translate.google.com/")
        assert result["status"] == "NO_MATCH"
        assert result["lookup_status"] == "SUCCESS"
        assert result["known_threat_match"] is False
        assert result["http_status"] == 200
        assert result["cache_duration_seconds"] == 10
        assert service.lookup("https://translate.google.com/")["cached"] is True
        assert get.call_count == 1


def test_safe_browsing_invalid_protobuf_is_not_a_negative_match():
    service = SafeBrowsingLookup(api_key="test-key")
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/x-protobuf"}
    response.content = b"not a protobuf response"
    with patch("pipeline.url_signals.requests.get", return_value=response):
        result = service.lookup("https://unknown.example/")
    assert result["status"] == "ERROR"
    assert result["lookup_status"] == "FAILED"
    assert result["known_threat_match"] is None
    assert result["reason"] == "invalid_response"


def test_safe_browsing_timeout_is_logged_without_url_path(caplog):
    service = SafeBrowsingLookup(api_key="test-key")
    with patch("pipeline.url_signals.requests.get", side_effect=requests.Timeout):
        result = service.lookup("https://unknown.example/private-token?value=secret")
    assert result["status"] == "ERROR"
    assert result["lookup_status"] == "FAILED"
    assert result["known_threat_match"] is None
    assert result["http_status"] is None
    assert result["source"] == "safe_browsing"
    assert result["reason"] == "timeout"
    assert "reason=timeout" in caplog.text
    assert "private-token" not in caplog.text
    assert "value=secret" not in caplog.text


def test_safe_browsing_http_error_logs_status_without_key(caplog):
    service = SafeBrowsingLookup(api_key="secret-api-key")
    response = MagicMock()
    response.status_code = 403
    response.raise_for_status.side_effect = requests.HTTPError(
        "request failed with key=secret-api-key", response=response
    )
    with patch("pipeline.url_signals.requests.get", return_value=response):
        result = service.lookup("https://unknown.example/path?token=private")
    assert result["reason"] == "http_error"
    assert result["lookup_status"] == "FAILED"
    assert result["known_threat_match"] is None
    assert result["http_status"] == 403
    assert "reason=http_error status=403" in caplog.text
    assert "secret-api-key" not in caplog.text
    assert "token=private" not in caplog.text


def test_safe_browsing_without_key_does_not_claim_no_match():
    result = SafeBrowsingLookup(api_key="").lookup("https://translate.google.com/")
    assert result["status"] == "UNAVAILABLE"
    assert result["lookup_status"] == "NOT_RUN"
    assert result["known_threat_match"] is None
    assert result["http_status"] is None


def test_standalone_url_never_uses_visobert_and_no_match_is_unknown():
    router = TwoTierRouter(use_council=False)
    router.url_signals = UrlSignals(
        phishtank=MagicMock(lookup=MagicMock(return_value={"source": "phishtank", "status": "NO_MATCH"})),
        safe_browsing=MagicMock(lookup=MagicMock(return_value={"source": "safe_browsing", "status": "NO_MATCH"})),
    )
    with patch.object(router.tier1_classifier, "predict") as predict:
        result = router.classify("https://translate.google.com/", check_type="URL")
    predict.assert_not_called()
    assert result["final_label"] == "UNKNOWN"
    assert result["tier1_telemetry"]["url_evidence"]["normalized_value"] == "https://translate.google.com/"


def test_standalone_url_known_threat_is_direct_warning():
    router = TwoTierRouter(use_council=False)
    router.url_signals = UrlSignals(
        phishtank=MagicMock(lookup=MagicMock(return_value={"source": "phishtank", "status": "MATCH", "phish_id": "42"})),
        safe_browsing=MagicMock(lookup=MagicMock(return_value={"source": "safe_browsing", "status": "NO_MATCH"})),
    )
    with patch.object(router.tier1_classifier, "predict") as predict:
        result = router.classify("https://bad.example/login", check_type="URL")
    predict.assert_not_called()
    assert result["final_label"] == "SCAM-RISK"
    assert result["routing_path"] == "TIER_1_URL_REPUTATION"


def test_sms_ham_with_threat_match_cannot_exit_safe():
    router = TwoTierRouter(use_council=False)
    router.tier2_available = False
    router.url_signals.analyze = MagicMock(return_value={"known_threat_match": True, "hostname": "bad.example", "signals": [{"source": "phishtank", "status": "MATCH"}]})
    with patch.object(router.tier1_classifier, "predict", return_value={
        "label": "HAM", "class_id": 0, "confidence": 0.98, "calibrated_scam_prob": 0.02,
        "latency_ms": 1.0, "threshold_low": 0.35, "threshold_high": 0.85,
    }):
        result = router.classify("Truy cap https://bad.example/login", check_type="SMS")
    assert result["final_label"] == "UNKNOWN"
    assert result["tier2_telemetry"]["fallback_trigger"] == "URL_THREAT_CONFLICT"


def test_extract_urls_trims_sentence_punctuation():
    assert extract_urls("Xem https://bad.example/login. Hoac https://good.example/!") == ["https://bad.example/login", "https://good.example/"]
