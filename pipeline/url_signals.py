"""URL validation and known-threat signals for Tier 1.

No user-supplied URL is fetched. PhishTank is read from a local snapshot;
Safe Browsing is queried only when a dedicated API key is configured.
"""

import csv
import gzip
import ipaddress
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from .safe_browsing_proto import decode_search_urls_response


ROOT = Path(__file__).resolve().parents[1]
URL_IN_TEXT = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v5/urls:search"
logger = logging.getLogger("ScamShield.URLSignals")


def normalize_url(value: str) -> tuple[str, str]:
    """Return a conservative lookup URL and hostname, or raise ValueError."""
    if not value or len(value) > 8192 or "\\" in value or any(ord(ch) < 32 or ch.isspace() for ch in value):
        raise ValueError("URL trống, quá dài hoặc chứa khoảng trắng/ký tự điều khiển")
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            raise ValueError("Chỉ hỗ trợ URL HTTP(S) có hostname")
        if parts.username is not None or parts.password is not None:
            raise ValueError("URL chứa thông tin đăng nhập trong authority")
        host = (parts.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port  # Accessing this also rejects malformed/out-of-range ports.
        if not host or len(host) > 253:
            raise ValueError("Hostname không hợp lệ")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or any(not DOMAIN_LABEL.fullmatch(label) for label in host.split(".")):
                raise ValueError("Hostname không hợp lệ") from None
            if all(label.isdigit() for label in host.split(".")):
                raise ValueError("Địa chỉ IP không hợp lệ")
        else:
            if not address.is_global:
                raise ValueError("Địa chỉ IP không công khai")
        if port is not None and port in {80, 443} and parts.scheme.lower() == ("http" if port == 80 else "https"):
            port = None
        authority = f"[{host}]" if ":" in host else host
        if port is not None:
            authority += f":{port}"
        normalized = urlunsplit((parts.scheme.lower(), authority, parts.path or "/", parts.query, ""))
        return normalized, host
    except (UnicodeError, ValueError) as exc:
        raise ValueError(str(exc) or "URL không hợp lệ") from exc


def extract_urls(message: str) -> list[str]:
    """Extract distinct HTTP(S) URLs, leaving validation to normalize_url."""
    result = []
    for match in URL_IN_TEXT.finditer(message):
        url = match.group(0).rstrip(".,;!?)]}")
        if url not in result:
            result.append(url)
    return result


class PhishTankFeed:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("PHISHTANK_FEED_PATH", ROOT / "data" / "phishtank_online_valid.csv.gz"))
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self._stamp = None
        self._records = {}
        self._lock = threading.Lock()

    def lookup(self, normalized_url: str) -> dict:
        host = urlsplit(normalized_url).hostname or "unknown"
        try:
            stamp = self.path.stat().st_mtime_ns
        except OSError:
            logger.warning("PhishTank lookup unavailable: feed file missing at %s", self.path)
            return {"source": "phishtank", "status": "UNAVAILABLE", "reason": "feed_missing"}
        with self._lock:
            if stamp != self._stamp:
                try:
                    records = {}
                    with gzip.open(self.path, "rt", encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        required = {"url", "phish_id", "verified", "online"}
                        if not required.issubset(reader.fieldnames or []):
                            raise ValueError("PhishTank feed thiếu cột bắt buộc")
                        for row in reader:
                            if (row.get("verified") or "").lower() != "yes" or (row.get("online") or "").lower() != "yes":
                                continue
                            try:
                                key, _ = normalize_url(row["url"])
                            except ValueError:
                                continue
                            records[key] = {
                                "phish_id": row["phish_id"],
                                "verified_at": row.get("verification_time"),
                                "target": row.get("target"),
                                "detail_url": row.get("phish_detail_url"),
                            }
                    if not records:
                        raise ValueError("PhishTank feed không có bản ghi hợp lệ")
                    self._records, self._stamp = records, stamp
                except (OSError, ValueError, csv.Error, UnicodeError) as exc:
                    logger.warning("PhishTank lookup failed: invalid feed at %s (%s)", self.path, type(exc).__name__)
                    return {"source": "phishtank", "status": "ERROR", "reason": "feed_invalid"}
            record = self._records.get(normalized_url)
        logger.info("PhishTank lookup host=%s status=%s", host, "MATCH" if record else "NO_MATCH")
        return {
            "source": "phishtank",
            "status": "MATCH" if record else "NO_MATCH",
            "feed_updated_at": datetime.fromtimestamp(stamp / 1e9, timezone.utc).isoformat(),
            **(record or {}),
        }


class SafeBrowsingLookup:
    def __init__(self, api_key: str | None = None, timeout: float = 2.0):
        self.api_key = api_key if api_key is not None else os.getenv("SAFE_BROWSING_API_KEY", "")
        self.timeout = timeout
        self._cache = {}
        self._lock = threading.Lock()

    @staticmethod
    def _failure(reason: str, http_status: int | None = None) -> dict:
        return {"source": "safe_browsing", "status": "ERROR", "lookup_status": "FAILED",
                "known_threat_match": None, "http_status": http_status, "reason": reason}

    def lookup(self, normalized_url: str) -> dict:
        if not self.api_key or self.api_key.lower().startswith("your_"):
            return {"source": "safe_browsing", "status": "UNAVAILABLE", "lookup_status": "NOT_RUN",
                    "known_threat_match": None, "http_status": None, "reason": "api_key_missing"}
        host = urlsplit(normalized_url).hostname or "unknown"
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(normalized_url)
            if cached and cached[0] > now:
                logger.info("Safe Browsing lookup host=%s status=%s cache=hit", host, cached[1]["status"])
                return {**cached[1], "cached": True}
        logger.info("Safe Browsing lookup host=%s cache=miss; calling Google", host)
        response = None
        try:
            response = requests.get(
                SAFE_BROWSING_ENDPOINT,
                params={"key": self.api_key, "urls": normalized_url},
                timeout=self.timeout,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type == "application/json":
                try:
                    payload = response.json()
                except ValueError:
                    logger.warning("Safe Browsing lookup failed host=%s reason=invalid_json", host)
                    return self._failure("invalid_json", response.status_code)
                if not isinstance(payload, dict):
                    raise ValueError("Safe Browsing response không hợp lệ")
                threats = payload.get("threats", [])
                if not isinstance(threats, list):
                    raise ValueError("Safe Browsing response không hợp lệ")
                threat_types = sorted({
                    kind for item in threats if isinstance(item, dict)
                    for kind in item.get("threatTypes", []) if isinstance(kind, str)
                })
                duration = payload.get("cacheDuration", "")
                if not isinstance(duration, str) or not re.fullmatch(r"\d+(?:\.\d+)?s", duration):
                    raise ValueError("Safe Browsing thiếu cacheDuration hợp lệ")
                seconds = float(duration[:-1])
                has_threat = bool(threats)
            else:
                threat_types, seconds = decode_search_urls_response(response.content)
                has_threat = bool(threat_types)
            # The service may return a matching host/path expression instead of the input URL.
            result = {
                "source": "safe_browsing",
                "status": "MATCH" if has_threat else "NO_MATCH",
                "lookup_status": "SUCCESS",
                "known_threat_match": has_threat,
                "http_status": response.status_code,
                "threat_types": threat_types,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            result["cache_duration_seconds"] = seconds
            result["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
            if seconds > 0:
                with self._lock:
                    self._cache[normalized_url] = (time.monotonic() + seconds, result)
            logger.info("Safe Browsing lookup host=%s status=%s threat_types=%s", host, result["status"], ",".join(threat_types) or "none")
            return result
        except requests.HTTPError as exc:
            error_response = exc.response if exc.response is not None else response
            http_status = getattr(error_response, "status_code", None)
            http_status = http_status if isinstance(http_status, int) else None
            logger.warning("Safe Browsing lookup failed host=%s reason=http_error status=%s", host, http_status)
            return self._failure("http_error", http_status)
        except requests.Timeout:
            logger.warning("Safe Browsing lookup failed host=%s reason=timeout timeout_seconds=%s", host, self.timeout)
            return self._failure("timeout")
        except requests.ConnectionError as exc:
            logger.warning("Safe Browsing lookup failed host=%s reason=connection_error error_type=%s", host, type(exc).__name__)
            return self._failure("connection_error")
        except requests.RequestException as exc:
            logger.warning("Safe Browsing lookup failed host=%s reason=request_error error_type=%s", host, type(exc).__name__)
            return self._failure("request_error")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Safe Browsing lookup failed host=%s reason=invalid_response error_type=%s", host, type(exc).__name__)
            return self._failure("invalid_response", response.status_code if response is not None else None)


class UrlSignals:
    def __init__(self, phishtank: PhishTankFeed | None = None, safe_browsing: SafeBrowsingLookup | None = None):
        self.phishtank = phishtank or PhishTankFeed()
        self.safe_browsing = safe_browsing or SafeBrowsingLookup()

    def analyze(self, value: str) -> dict:
        try:
            normalized, host = normalize_url(value)
        except ValueError as exc:
            return {"input_type": "URL", "normalized_value": None, "hostname": None,
                    "signals": [{"source": "local_parser", "status": "INVALID", "reason": str(exc)}],
                    "evidence_sufficient": False, "known_threat_match": False}
        signals = [self.phishtank.lookup(normalized), self.safe_browsing.lookup(normalized)]
        matched = any(item["status"] == "MATCH" for item in signals)
        return {"input_type": "URL", "normalized_value": normalized, "hostname": host,
                "signals": signals, "known_threat_match": matched, "evidence_sufficient": matched}


def council_evidence(items: list[dict]) -> list[dict]:
    """Keep provenance and match details without duplicating raw URL/query data."""
    return [
        {
            "hostname": item.get("hostname"),
            "signals": [
                {key: signal[key] for key in ("source", "status", "lookup_status", "known_threat_match", "reason", "phish_id", "target", "threat_types", "checked_at", "verified_at") if key in signal}
                for signal in item.get("signals", [])
            ],
        }
        for item in items
    ]
