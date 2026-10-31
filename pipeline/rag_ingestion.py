"""Allowlisted source refresh in the API process, sharing its BGE-M3 instance."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import requests
import yaml

from .rag import BgeM3Embedder, ROOT, normalize_tier2_input

LOG = logging.getLogger(__name__)
SOURCES_PATH = ROOT / "config" / "rag_sources.yaml"
CASES_DIR = ROOT / "data" / "rag" / "approved_cases"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _chunks(text: str, words_per_chunk: int = 220) -> list[str]:
    words = text.split()
    return [" ".join(words[start:start + words_per_chunk]) for start in range(0, len(words), words_per_chunk)]


class RagIngestor:
    def __init__(self, embedder: BgeM3Embedder, database_url: str | None = None,
                 config_path: Path = SOURCES_PATH, cases_dir: Path = CASES_DIR):
        self.embedder = embedder
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai")
        self.config_path = config_path
        self.cases_dir = cases_dir
        self._refresh_lock = threading.Lock()

    def _config(self) -> dict:
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _allowed(url: str, allowlist: set[str]) -> bool:
        parts = urlsplit(url)
        return parts.scheme == "https" and parts.hostname in allowlist and not parts.username and not parts.password

    def _fetch(self, url: str, allowlist: set[str]) -> str:
        if not self._allowed(url, allowlist):
            raise ValueError("source outside HTTPS allowlist")
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=10, allow_redirects=False,
                                headers={"User-Agent": "ScamShield-RAG/1.0"})
        response.raise_for_status()
        if response.is_redirect or "text/html" not in response.headers.get("content-type", ""):
            raise ValueError("redirect or unsupported source content")
        if len(response.content) > 2_000_000:
            raise ValueError("source content exceeds size limit")
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()
        articles = soup.find_all("article")
        main = (soup.select_one(".tinymce-content") or soup.find("main") or
                (max(articles, key=lambda node: len(node.get_text(" ", strip=True))) if articles else None) or
                soup.body)
        if main is None:
            raise ValueError("source has no readable body")
        text = " ".join(main.stripped_strings)
        if len(text) < 100:
            raise ValueError("source text too short")
        return text[:100_000]

    def _ingest_document(self, *, source_key: str, title: str, text: str,
                         source_type: str, source_url: str | None, case_label: str | None = None):
        text = normalize_tier2_input(text).message
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logical_key = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:24]
        document_key = f"{logical_key}:{content_hash[:24]}:{self.embedder.revision[:12]}"
        chunks = _chunks(text)
        if not chunks:
            return
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM rag_documents WHERE document_key = %s AND is_active = true", (document_key,))
                if cur.fetchone():
                    if source_type == "OFFICIAL_WEB":
                        cur.execute("UPDATE rag_documents SET expires_at = %s, updated_at = %s WHERE document_key = %s",
                                    (_now() + timedelta(days=7), _now(), document_key))
                    return
        vectors = self.embedder.encode(chunks)
        now = _now()
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE rag_documents SET is_active = false, updated_at = %s WHERE document_key LIKE %s AND is_active = true",
                            (now, logical_key + ":%"))
                cur.execute(
                    """INSERT INTO rag_documents
                       (document_key, title, source_type, source_name, source_url, source_version,
                        language_code, case_label, review_status, is_active, content_hash,
                        effective_at, expires_at, ingested_at, reviewed_at, created_at, updated_at)
                       VALUES (%s,%s,%s,%s,%s,%s,'vi',%s,'VERIFIED',true,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (document_key) DO UPDATE SET is_active = true,
                           expires_at = EXCLUDED.expires_at, updated_at = EXCLUDED.updated_at
                       RETURNING id""",
                    (document_key, title, source_type, urlsplit(source_url).hostname if source_url else "approved_case",
                     source_url, content_hash[:24], case_label, content_hash, now,
                     now + timedelta(days=7) if source_type == "OFFICIAL_WEB" else None,
                     now, now, now, now),
                )
                document_id = cur.fetchone()[0]
                for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    literal = "[" + ",".join(str(float(value)) for value in vector) + "]"
                    cur.execute(
                        """INSERT INTO rag_chunks
                           (document_id, chunk_no, chunk_text, chunk_hash, embedding,
                            embedding_model, embedding_version, token_count, metadata, is_active, created_at)
                           VALUES (%s,%s,%s,%s,%s::vector,%s,%s,%s,'{}'::json,true,%s)
                           ON CONFLICT (document_id, chunk_no) DO UPDATE SET is_active = true""",
                        (document_id, index, chunk, hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                         literal, self.embedder.model_name, self.embedder.revision, len(chunk.split()), now),
                    )

    def refresh(self) -> dict[str, int]:
        if not self._refresh_lock.acquire(blocking=False):
            return {"skipped": 1}
        counts = {"ingested": 0, "failed": 0}
        try:
            config = self._config()
            allowlist = set(config.get("allowed_hosts", []))
            for source in config.get("seeds", []):
                try:
                    url = source["url"]
                    content = self._fetch(url, allowlist)
                    self._ingest_document(source_key=url, title=source["title"], text=content,
                                          source_type="OFFICIAL_WEB", source_url=url)
                    counts["ingested"] += 1
                except Exception as exc:
                    LOG.warning("RAG source refresh failed: %s (%s)", source.get("url"), type(exc).__name__)
                    counts["failed"] += 1
            for path in sorted(self.cases_dir.glob("*.json")) if self.cases_dir.is_dir() else []:
                try:
                    case = json.loads(path.read_text(encoding="utf-8"))
                    if case.get("review_status") != "VERIFIED" or case.get("case_label") not in {"SCAM", "HAM"}:
                        raise ValueError("case lacks approved label")
                    self._ingest_document(source_key="case:" + path.name, title=case["title"],
                                          text=case["text"], source_type="APPROVED_CASE",
                                          source_url=case.get("source_url"), case_label=case["case_label"])
                    counts["ingested"] += 1
                except Exception as exc:
                    LOG.warning("RAG approved case rejected: %s (%s)", path.name, type(exc).__name__)
                    counts["failed"] += 1
            return counts
        finally:
            self._refresh_lock.release()

    def start_daily_refresh(self) -> threading.Thread:
        def run():
            while True:
                try:
                    self.refresh()
                except Exception as exc:
                    LOG.warning("RAG refresh unavailable: %s", type(exc).__name__)
                threading.Event().wait(24 * 60 * 60)

        thread = threading.Thread(target=run, name="rag-ingestion", daemon=True)
        thread.start()
        return thread
