"""
Deterministic feature extraction helper for ScamShield-VN.
Extracts IOCs (URLs, phone numbers, OTP patterns, brand identities, urgency keywords)
for explainability and radar metric calculation.
"""

import re
import urllib.parse
from typing import Dict, Any, List, Optional

OFFICIAL_BANK_DOMAINS = [
    "vietcombank.com.vn",
    "bidv.com.vn",
    "vietinbank.vn",
    "mbbank.com.vn",
    "techcombank.com.vn",
    "tpbank.vn",
    "vpbank.com.vn",
    "acb.com.vn",
    "agribank.com.vn",
    "sacombank.com.vn",
    "shb.com.vn",
    "hdbank.com.vn",
    "ocb.com.vn",
    "msb.com.vn",
    "vib.com.vn",
    "scb.com.vn",
    "vcb.com.vn",
    "vneid.gov.vn",
    "dichvucong.gov.vn",
    "bocongan.gov.vn",
    "chinhphu.vn",
    "mobifone.vn",
    "viettel.vn",
    "vietteltelecom.vn",
    "vinaphone.com.vn",
    "vnpt.vn",
    "vnpt.com.vn",
    "fpt.vn",
    "fpt.com.vn",
]

REGISTERED_BRANDNAMES = [
    "VCB",
    "VIETCOMBANK",
    "BIDV",
    "VIETINBANK",
    "MBBANK",
    "TECHCOMBANK",
    "TPBANK",
    "VPBANK",
    "ACB",
    "AGRIBANK",
    "SACOMBANK",
    "HDBANK",
    "MSB",
    "VIB",
    "VIETTEL",
    "VINAPHONE",
    "MOBIFONE",
    "VNEID",
    "DICHVUCONG",
    "BOCONGAN",
]

URGENCY_PATTERNS = [
    r"trong\s+24h?",
    r"khoa\s+vinh\s+vien",
    r"tam\s+khoa",
    r"khan\s+cap",
    r"ngay\s+lap\s+tuc",
    r"tru\s+no",
    r"phong\s+toa",
    r"cong\s+an",
    r"vien\s+kiem\s+sat",
    r"lenh\s+bat",
    r"chuyen\s+an",
    r"sinh\s+trac\s+hoc",
    r"qd\s*2345",
]


def extract_deterministic_features(content: str, sender: Optional[str] = None) -> Dict[str, Any]:
    """Trích xuất bảng kiểm đặc trưng định lượng tại biên thiết bị."""
    url_pattern = (
        r"https?://[^\s<>\"']+"
        r"|www\.[^\s<>\"']+"
        r"|(?<![\w@.])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|net|org|info|biz|vn|io|co|me|to|cc|tv|xyz|top|vip|shop|online|site|club|app|link|live|store|asia|tk|ml|ga|cf|gq|icu|cyou|bond|sbs|click|fun|pro|win|bet|best|life|world|today|news|space|website|tech|dev|ly|gl|gd|in|us|uk|cn|ru)\b(?:/[^\s<>\"']*)?"
    )
    raw_matches = re.findall(url_pattern, content or "")

    urls: List[str] = []
    ignored_suffixes = {"e.g", "i.e", "tp.hcm", "tp.hn"}
    for raw in raw_matches:
        cleaned = raw.strip().rstrip(".,;:!?)'\"")
        if cleaned and cleaned.lower() not in ignored_suffixes:
            if not re.match(r"^\d+\.\d+$", cleaned) and "@" not in cleaned:
                urls.append(cleaned)
    urls = list(dict.fromkeys(urls))

    is_whitelist = True
    domain_reputation = "LEGITIMATE"

    if urls:
        is_whitelist = False
        domain_reputation = "MALICIOUS_SIMILARITY"
        for u in urls:
            try:
                full_url = u if "://" in u else f"http://{u}"
                parsed = urllib.parse.urlparse(full_url)
                hostname = (parsed.hostname or "").lower()
                for official in OFFICIAL_BANK_DOMAINS:
                    if hostname == official or hostname.endswith("." + official):
                        is_whitelist = True
                        domain_reputation = "LEGITIMATE"
                        break
                if is_whitelist:
                    break
            except Exception:
                pass

    phone_pattern = r"(?<!\d)(?:(?:\+84|0)[35789]\d{8}|(?:1800|1900)\d{4,6})(?!\d)"
    detected_phones = list(dict.fromkeys([p.strip() for p in re.findall(phone_pattern, content or "")]))

    has_otp = bool(re.search(r"\b\d{6}\b", content or "") and re.search(r"(otp|ma\s+xac\s+thuc|code)", content or "", re.IGNORECASE))
    otp_pattern_match = "Chuỗi 6 chữ số hợp chuẩn mã xác thực" if has_otp else "Không phát hiện mã OTP"

    sender_clean = (sender or "").strip().upper()
    is_registered = sender_clean in REGISTERED_BRANDNAMES
    if not sender_clean and content:
        for b in REGISTERED_BRANDNAMES:
            if content.upper().startswith(b + ":") or content.upper().startswith(b + " "):
                sender_clean = b
                is_registered = True
                break

    matched_urgency = []
    for pat in URGENCY_PATTERNS:
        m = re.search(pat, content or "", re.IGNORECASE)
        if m:
            matched_urgency.append(m.group(0))
    urgency_score = min(100, len(matched_urgency) * 30)

    return {
        "url_analysis": {
            "detected_urls": urls,
            "is_in_official_whitelist": is_whitelist if urls else True,
            "domain_reputation": domain_reputation if urls else "LEGITIMATE",
        },
        "urgency_indicators": {
            "matched_keywords": matched_urgency,
            "urgency_score": urgency_score,
        },
        "phone_analysis": {
            "detected_phones": detected_phones,
            "is_official_hotline": any(p.startswith("1800") or p.startswith("1900") for p in detected_phones),
            "phone_type": "HOTLINE" if any(p.startswith("1800") or p.startswith("1900") for p in detected_phones) else ("PERSONAL_MOBILE" if detected_phones else "NONE"),
        },
        "otp_structure": {
            "has_otp_pattern": has_otp,
            "pattern_match_type": otp_pattern_match,
        },
        "brand_identity": {
            "claimed_brand": sender_clean if is_registered else "UNREGISTERED",
            "is_registered_brand": is_registered,
        },
        "visobert_token_attribution": [],
    }
