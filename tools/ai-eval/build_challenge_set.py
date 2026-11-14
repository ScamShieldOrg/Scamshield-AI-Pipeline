#!/usr/bin/env python3
"""
GĐ0b - Xây dựng Bộ thách thức tin nhắn (Challenge Set v1) gồm 150 mẫu biên khó.

Bộ này dùng để benchmark đánh giá ổn định trước và sau khi tinh chỉnh model / chính sách:
  1. GAMBLING_BETTING_PROMO: Quảng cáo cá độ, nạp tiền, game đổi thưởng (gốc nhãn 0 nhưng có link/tiền).
  2. BRAND_PROMO_VOUCHER: Khuyến mãi viễn thông, voucher ăn uống có link.
  3. BILLING_UTILITY: Hóa đơn cước viễn thông, điện, nước, internet.
  4. HEALTH_MEDICAL: Thông báo kết quả khám, xét nghiệm, y tế.
  5. DAILY_CONVERSATION: Tin nhắn đời thường không có tín hiệu lừa đảo.
  6. BORDERLINE_AUTH_OTP: Mã OTP / xác minh từ dịch vụ chính danh.
  7. SCAM_SAMPLES_HARD: Tin lừa đảo tinh vi không có từ khóa lộ liễu.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SOURCE = Path(r"C:\Users\USER\Scamshield-AI-Pipeline\data\raw\vietnamese_sms_merged_all.csv")
OUT_FILE = Path(__file__).resolve().parent / "out" / "challenge_set_v1.json"

gambling_re = re.compile(r"(?:tai app|nohu|nohuvip|bet|game bai|tang tien|thuong khung|phat qua|daga|888|banca|game|trian|nhan ngay)", re.IGNORECASE)
promo_re = re.compile(r"(?:\[QC\]|\[TB\]|khuyen mai|uu dai|voucher|giam gia|coupon|viettel|vinaphone|mobifone|grab|katinat)", re.IGNORECASE)
billing_re = re.compile(r"(?:hoa don|thanh toan|cuoc dv|ftth|tien dien|tien nuoc|tai khoan|du no)", re.IGNORECASE)
medical_re = re.compile(r"(?:xet nghiem|benh vien|kham benh|ket qua|suc khoe|vac xin|tiem chung)", re.IGNORECASE)
otp_re = re.compile(r"(?:ma xac minh|otp|ma otp|mat khau|ma xac nhan|ma bao mat)", re.IGNORECASE)
daily_re = re.compile(r"(?:chao anh|chao em|anh oi|em oi|khi nao|o dau|ve chua|an com|chuc mung|nho ban|gap nhau)", re.IGNORECASE)


def sha_sort_key(r: dict) -> str:
    return hashlib.sha256(r["message_id"].encode("utf-8")).hexdigest()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    if not SOURCE.exists():
        print(f"Không tìm thấy file nguồn: {SOURCE}", file=sys.stderr)
        return 1

    with SOURCE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buckets: dict[str, list[dict]] = defaultdict(list)

    for r in rows:
        msg = r["message"]
        lbl = int(r["label"])
        
        if lbl == 0:
            if gambling_re.search(msg) and any(kw in msg.lower() for kw in ["nhan", "app", "thuong", "tang"]):
                buckets["GAMBLING_BETTING_PROMO"].append(r)
            elif medical_re.search(msg):
                buckets["HEALTH_MEDICAL"].append(r)
            elif billing_re.search(msg):
                buckets["BILLING_UTILITY"].append(r)
            elif otp_re.search(msg):
                buckets["BORDERLINE_AUTH_OTP"].append(r)
            elif daily_re.search(msg):
                buckets["DAILY_CONVERSATION"].append(r)
            elif promo_re.search(msg):
                buckets["BRAND_PROMO_VOUCHER"].append(r)
        else:
            if not re.search(r"(?:otp|chuyen tien|mat khau|khoa the)", msg, re.IGNORECASE):
                buckets["SCAM_SAMPLES_HARD"].append(r)

    targets = {
        "GAMBLING_BETTING_PROMO": 30,
        "BRAND_PROMO_VOUCHER": 30,
        "BILLING_UTILITY": 25,
        "HEALTH_MEDICAL": len(buckets["HEALTH_MEDICAL"]),  # Lấy hết (7 mẫu)
        "DAILY_CONVERSATION": len(buckets["DAILY_CONVERSATION"]),  # Lấy hết (18 mẫu)
        "BORDERLINE_AUTH_OTP": 20,
        "SCAM_SAMPLES_HARD": 20,
    }

    selected = []
    category_summary = {}

    for cat, quota in targets.items():
        pool = sorted(buckets[cat], key=sha_sort_key)
        chosen = pool[:quota]
        category_summary[cat] = len(chosen)
        for item in chosen:
            selected.append({
                "message_id": item["message_id"],
                "category": cat,
                "gold_label": int(item["label"]),
                "message": item["message"],
                "source": item.get("source", "merged_all")
            })

    # Sắp xếp lại toàn bộ theo message_id tất định
    selected.sort(key=lambda x: x["message_id"])

    manifest = {
        "version": "1.0.0",
        "total_samples": len(selected),
        "category_counts": category_summary,
        "label_counts": {
            "ham_label_0": sum(1 for x in selected if x["gold_label"] == 0),
            "scam_label_1": sum(1 for x in selected if x["gold_label"] == 1)
        },
        "description": "Bộ thách thức 150 mẫu tin biên khó (challenge set) đóng băng phiên bản cho ScamShield",
        "samples": selected
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Đã tạo thành công Challenge Set: {OUT_FILE}")
    print(f"Tổng số mẫu: {len(selected)}")
    for cat, count in category_summary.items():
        print(f"  - {cat:25}: {count} mẫu")
    print(f"Phân bố nhãn: Ham (0) = {manifest['label_counts']['ham_label_0']}, Scam (1) = {manifest['label_counts']['scam_label_1']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
