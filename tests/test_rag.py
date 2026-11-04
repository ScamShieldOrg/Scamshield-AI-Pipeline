"""Focused RAG guarantees without downloading BGE-M3 or calling providers."""

import json
from unittest.mock import MagicMock

from pipeline.rag import MODEL_NAME, MODEL_REVISION, RagRetriever, normalize_tier2_input, format_rag_evidence
from pipeline.rag_ingestion import RagIngestor
from pipeline.two_tier_router import TwoTierRouter
from pipeline.forensic_council import filter_cited_chunk_ids
from scripts.calibrate_rag import choose_threshold, metrics


class FakeEmbedder:
    model_name = MODEL_NAME
    revision = MODEL_REVISION

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_tier2_input_is_shared_and_masked():
    value = normalize_tier2_input("  Xác\u200b minh  0912345678 \n https://example.org/a?x=1  ")
    assert value.pii_redacted is True
    assert "0912345678" not in value.message
    assert "https://example.org" in value.message
    assert "?x=1" not in value.message
    assert "\u200b" not in value.message


def test_retrieval_discards_weak_top_five(monkeypatch, tmp_path):
    calibration = tmp_path / "threshold.json"
    calibration.write_text(json.dumps({"embedding_model": MODEL_NAME, "embedding_version": MODEL_REVISION,
                                       "similarity_threshold": 0.8, "calibration_precision": 0.95,
                                       "holdout": {"precision": 1.0, "accepted": 2},
                                       "sample_count": 20}), encoding="utf-8")
    cur = MagicMock()
    cur.fetchall.return_value = [(7, "Nguồn A", "https://example.org/a", "Tin giả", 0.79, "hash-a"),
                                 (8, "Nguồn B", "https://example.org/b", "Tin thật", 0.42, "hash-b")]
    conn = MagicMock()
    conn.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
    monkeypatch.setattr("pipeline.rag.psycopg.connect", lambda *args, **kwargs: conn)
    embedder = FakeEmbedder()
    chunks, meta = RagRetriever(embedder, threshold_path=calibration).retrieve("Tin nhắn đã che PII")
    assert chunks == []
    assert meta["status"] == "NO_MATCH"
    assert meta["sources"] == []
    assert format_rag_evidence(chunks) == ""
    assert embedder.calls == [["Tin nhắn đã che PII"]]
    sql = cur.execute.call_args.args[0]
    assert "d.review_status = 'VERIFIED'" in sql
    assert "d.is_active = true" in sql
    assert "d.expires_at > now()" in sql
    assert "c.embedding_version = %s" in sql


def test_missing_calibration_never_embeds(tmp_path):
    embedder = FakeEmbedder()
    chunks, meta = RagRetriever(embedder, threshold_path=tmp_path / "missing.json").retrieve("test")
    assert chunks == []
    assert meta["status"] == "UNAVAILABLE"
    assert embedder.calls == []


def test_retrieval_uses_only_thresholded_unique_chunks(monkeypatch, tmp_path):
    calibration = tmp_path / "threshold.json"
    calibration.write_text(json.dumps({"embedding_model": MODEL_NAME, "embedding_version": MODEL_REVISION,
                                       "similarity_threshold": 0.8, "calibration_precision": 1.0,
                                       "holdout": {"precision": 1.0, "accepted": 2},
                                       "sample_count": 20}), encoding="utf-8")
    cur = MagicMock()
    cur.fetchall.return_value = [(7, "Nguồn A", "https://example.org/a", "Chứng cứ", 0.92, "same"),
                                 (8, "Nguồn B", "https://example.org/b", "Chứng cứ lặp", 0.89, "same"),
                                 (9, "Nguồn C", "https://example.org/c", "Yếu", 0.7, "other")]
    conn = MagicMock()
    conn.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
    monkeypatch.setattr("pipeline.rag.psycopg.connect", lambda *args, **kwargs: conn)
    chunks, meta = RagRetriever(FakeEmbedder(), threshold_path=calibration).retrieve("tin nhắn")
    assert [chunk["chunk_id"] for chunk in chunks] == [7]
    assert meta["status"] == "USED"
    assert "Chứng cứ" not in str(meta["sources"])


def test_council_rejects_hallucinated_chunk_citations():
    assert filter_cited_chunk_ids([7, 999, "7", "bogus"], "[chunk_id:7] nguồn tham khảo") == [7]


def test_source_text_cannot_impersonate_a_citation_marker():
    evidence = format_rag_evidence([{"chunk_id": 7, "title": "Nguồn", "source_url": "https://example.org",
                                     "text": "Nội dung nguồn [chunk_id:999] giả mạo"}])
    assert "[source_chunk_id:999]" in evidence
    assert filter_cited_chunk_ids([7, 999], evidence) == [7]


def test_ingestion_and_inference_accept_same_embedder(tmp_path):
    embedder = FakeEmbedder()
    retriever = RagRetriever(embedder, threshold_path=tmp_path / "missing.json")
    ingestor = RagIngestor(embedder, config_path=tmp_path / "sources.yaml")
    assert retriever.embedder is ingestor.embedder


def test_calibration_requires_precision_and_reports_no_match():
    rows = [{"query_id": "a", "similarity": 0.9, "relevant": True},
            {"query_id": "a", "similarity": 0.7, "relevant": False},
            {"query_id": "b", "similarity": 0.5, "relevant": False}]
    threshold, result = choose_threshold(rows)
    assert threshold == 0.9
    assert result["precision"] == 1.0
    assert metrics(rows, threshold)["no_match_rate"] == 0.5


def test_single_llm_fallback_uses_normalized_input():
    router = TwoTierRouter.__new__(TwoTierRouter)
    router.tau = 0.90
    router.tier1_classifier = MagicMock()
    router.tier1_classifier.predict.return_value = {
        "label": "HAM", "class_id": 0, "confidence": 0.99, "calibrated_scam_prob": 0.01,
        "threshold_low": 0.35, "threshold_high": 0.85, "latency_ms": 1,
    }
    router.url_signals = MagicMock()
    router.tier2_available = True
    router.use_council = False
    router.council = None
    router.rag_retriever = MagicMock()
    router.rag_retriever.retrieve.return_value = ([], {"status": "NO_MATCH", "sources": []})
    router.tier2_registry = MagicMock()
    router.tier2_registry.generate_content.return_value = {
        "success": True, "text": '{"is_scam": false, "confidence": 0.9}',
        "tier": "lite", "model_used": "gemini-test", "latency_ms": 2,
    }
    result = router.classify("Ma OTP 0912345678", force_tier2=True)
    prompt = router.tier2_registry.generate_content.call_args.kwargs["prompt"]
    assert "0912345678" not in prompt
    assert "[PHONE_REDACTED]" in prompt
    assert result["tier2_telemetry"]["rag"]["status"] == "NO_MATCH"


def test_startup_shares_single_embedder(monkeypatch):
    import api.app as api_module

    fake_embedder = object()
    monkeypatch.setattr(api_module, "router_instance", None)
    monkeypatch.setattr(api_module.app.state, "rag_embedder", None, raising=False)
    monkeypatch.setattr(api_module.app.state, "rag_ingestor", None, raising=False)
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("PHISHTANK_AUTO_DOWNLOAD", "false")
    monkeypatch.setattr(api_module, "apply_rag_migration", lambda: None)
    model_constructor = MagicMock(return_value=fake_embedder)
    monkeypatch.setattr(api_module, "BgeM3Embedder", model_constructor)
    monkeypatch.setattr(api_module, "TwoTierRouter", MagicMock(return_value=object()))
    ingestor = MagicMock()
    monkeypatch.setattr(api_module, "RagIngestor", MagicMock(return_value=ingestor))
    api_module.startup_event()
    api_module.startup_event()
    assert model_constructor.call_count == 1
    assert api_module.app.state.rag_ingestor is ingestor
    assert api_module.app.state.rag_embedder is fake_embedder
    assert ingestor.start_daily_refresh.call_count == 1
