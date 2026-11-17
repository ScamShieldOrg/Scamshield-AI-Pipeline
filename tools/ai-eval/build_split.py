#!/usr/bin/env python3
"""
GĐ0 - Làm sạch dữ liệu và tách holdout theo thành phần liên thông (Connected Components).

Mục tiêu:
  1. Báo cáo vệ sinh dữ liệu: mojibake, trùng lặp, xung đột nhãn trong cùng cụm.
  2. Tách holdout sao cho 0 mẫu holdout nào có bản gần-trùng (SimHash-64 Hamming <= near_dup_hamming) trong train.
  3. Tất định tuyệt đối: không dùng RNG, thứ tự chọn suy ra từ SHA-256 của thành phần liên thông.

Không sửa file nguồn trong repo pipeline; chỉ đọc file nguồn và ghi kết quả ra thư mục out/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT_DIR / "data" / "raw" / "vietnamese_sms_merged_all.csv"
OUT_DIR = Path(__file__).resolve().parent / "out"

PLACEHOLDER_RE = re.compile(r"\[(?:number|money|time|date|point|otp|email|url)\]", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^0-9a-zàáảãạăâđêôơưýỳỷỹỵ\s]")

# True mojibake regex: chuỗi byte UTF-8 của ký tự tiếng Việt bị giải mã nhầm qua Latin-1/CP1252.
# Tuyệt đối không dùng ký tự đơn như "Ã" hay "Â" vì chúng là chữ in hoa hợp lệ trong tiếng Việt ("ĐÃ", "ÂM", "ƯU ĐÃI").
MOJIBAKE_RE = re.compile(r"(?:Ä[ƒ‘]|Ã[¡à£©ª]|á[º»][\x80-\xbf]|Æ°|â€[™œ“]|Ã„)")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def cluster_key(text: str) -> str:
    """
    Khoá chuẩn hoá: bỏ placeholder, dấu câu, gộp khoảng trắng, viết thường.
    Giữ nguyên dấu tiếng Việt để phân biệt ngữ nghĩa.
    """
    s = nfc(text).lower()
    s = PLACEHOLDER_RE.sub(" ", s)
    s = NON_WORD_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_mojibake(text: str) -> bool:
    return bool(MOJIBAKE_RE.search(text))


def simhash64(text: str, ngram: int = 4) -> int:
    """SimHash 64-bit trên n-gram ký tự của chuỗi chuẩn hoá."""
    s = cluster_key(text)
    if len(s) < ngram:
        s = s or " "
    grams = [s[i : i + ngram] for i in range(max(1, len(s) - ngram + 1))]
    vector = [0] * 64
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit in range(64):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--holdout-share", type=float, default=0.20)
    parser.add_argument("--near-dup-hamming", type=int, default=3)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Không thấy file nguồn: {args.source}", file=sys.stderr)
        return 2

    rows = load_rows(args.source)
    if not rows:
        print("File nguồn rỗng.", file=sys.stderr)
        return 2

    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()

    # --- 1. Vệ sinh dữ liệu ban đầu ---
    mojibake = [r for r in rows if is_mojibake(r["message"])]
    not_nfc = [r for r in rows if nfc(r["message"]) != r["message"]]
    empty = [r for r in rows if not r["message"].strip()]
    very_short = [r for r in rows if len(cluster_key(r["message"])) < 12]

    exact_counter = Counter(r["message"].strip() for r in rows)
    exact_dupes = {k: v for k, v in exact_counter.items() if v > 1}

    for idx, row in enumerate(rows):
        row["_idx"] = idx
        row["_key"] = cluster_key(row["message"])
        row["_label"] = int(row["label"])
        row["_hash"] = simhash64(row["message"])

    # --- 2. Gom cụm liên thông (Connected Components) theo khoảng cách Hamming ---
    # Nếu hamming(simhash_A, simhash_B) <= near_dup_hamming thì A và B cùng thuộc một cụm.
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if hamming(rows[i]["_hash"], rows[j]["_hash"]) <= args.near_dup_hamming:
                union(i, j)

    comp_dict: dict[int, list[dict]] = defaultdict(list)
    for i in range(n):
        comp_dict[find(i)].append(rows[i])

    comp_list = []
    for root_id, grp in comp_dict.items():
        labels = {r["_label"] for r in grp}
        comp_id = hashlib.sha256(",".join(sorted(r["message_id"] for r in grp)).encode("utf-8")).hexdigest()
        comp_list.append({
            "comp_id": comp_id,
            "rows": grp,
            "labels": labels,
            "is_conflict": len(labels) > 1,
            "size": len(grp)
        })

    conflict_comps = [c for c in comp_list if c["is_conflict"]]
    pure_0 = [c for c in comp_list if not c["is_conflict"] and 0 in c["labels"]]
    pure_1 = [c for c in comp_list if not c["is_conflict"] and 1 in c["labels"]]

    # Sắp xếp tất định theo SHA-256 ID của cụm (không dùng RNG)
    pure_0.sort(key=lambda c: c["comp_id"])
    pure_1.sort(key=lambda c: c["comp_id"])

    # --- 3. Tách Holdout và Train ---
    # Cụm có xung đột nhãn (cả 0 và 1 có khoảng cách SimHash <= threshold) KHÔNG BAO GIỜ vào holdout.
    holdout: list[dict] = []
    train: list[dict] = []

    for c in conflict_comps:
        train.extend(c["rows"])

    # Chọn ~20% pure_0 vào holdout
    target_0 = int(round(args.holdout_share * sum(c["size"] for c in pure_0)))
    taken_0 = 0
    for c in pure_0:
        if taken_0 < target_0:
            holdout.extend(c["rows"])
            taken_0 += c["size"]
        else:
            train.extend(c["rows"])

    # Chọn ~20% pure_1 vào holdout
    target_1 = int(round(args.holdout_share * sum(c["size"] for c in pure_1)))
    taken_1 = 0
    for c in pure_1:
        if taken_1 < target_1:
            holdout.extend(c["rows"])
            taken_1 += c["size"]
        else:
            train.extend(c["rows"])

    # --- 4. Kiểm chứng: 0 mẫu holdout nào có bản gần-trùng trong train ---
    holdout_hashes = [r["_hash"] for r in holdout]
    train_hashes = [r["_hash"] for r in train]
    violations = []
    min_distance = 64

    for h_row in holdout:
        for t_row in train:
            dist = hamming(h_row["_hash"], t_row["_hash"])
            if dist < min_distance:
                min_distance = dist
            if dist <= args.near_dup_hamming:
                violations.append((h_row["message_id"], t_row["message_id"], dist))

    key_overlap = {r["_key"] for r in holdout} & {r["_key"] for r in train}
    conflict_rows = [r for c in conflict_comps for r in c["rows"]]

    # --- 5. Xuất các file dữ liệu sạch ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_columns = ["message_id", "message", "label", "source"]

    def dump_csv(filename: str, dataset: list[dict], cols: list[str]) -> None:
        with (OUT_DIR / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dataset)

    dump_csv("train.csv", train, base_columns)
    dump_csv("holdout.csv", holdout, base_columns)
    dump_csv("label_conflicts.csv", sorted(conflict_rows, key=lambda r: r["_key"]), base_columns)
    dump_csv("mojibake_rows.csv", mojibake, base_columns)

    def label_breakdown(data: list[dict]) -> dict:
        counter = Counter(r["_label"] for r in data)
        return {str(k): v for k, v in sorted(counter.items())}

    manifest = {
        "source": str(args.source),
        "source_sha256": source_digest,
        "algorithm": {
            "method": "Connected Components on SimHash-64 Graph (Atomic Component Assignment)",
            "cluster_key": "NFC + lowercase + strip placeholders + remove punctuation + collapse spaces",
            "selection_order": "SHA-256(comp_id) ascending, no RNG (strictly deterministic)",
            "holdout_share_target": args.holdout_share,
            "near_dup_metric": f"SimHash64 char-4gram, violation if hamming <= {args.near_dup_hamming}",
        },
        "counts": {
            "total": len(rows),
            "train": len(train),
            "holdout": len(holdout),
            "components_total": len(comp_list),
            "conflict_components": len(conflict_comps),
            "conflict_rows": len(conflict_rows),
            "mojibake_rows": len(mojibake),
            "not_nfc_rows": len(not_nfc),
            "empty_rows": len(empty),
            "very_short_rows": len(very_short),
            "exact_duplicate_values": len(exact_dupes),
        },
        "labels": {"train": label_breakdown(train), "holdout": label_breakdown(holdout)},
        "verification": {
            "exact_key_overlap": len(key_overlap),
            "near_dup_violations": len(violations),
            "min_hamming_holdout_vs_train": min_distance,
            "passed": len(key_overlap) == 0 and len(violations) == 0 and min_distance > args.near_dup_hamming,
        },
        "sample_violations": violations[:10],
    }

    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Nguồn       : {args.source}")
    print(f"Tổng dòng   : {len(rows)}")
    print(f"Train       : {len(train)}  {label_breakdown(train)}")
    print(f"Holdout     : {len(holdout)}  {label_breakdown(holdout)}")
    print(f"Thành phần  : {len(comp_list)} cụm liên thông (xung đột nhãn: {len(conflict_comps)})")
    print()
    print("--- Vệ sinh dữ liệu ---")
    print(f"  mojibake            : {len(mojibake)}")
    print(f"  chưa ở dạng NFC     : {len(not_nfc)}")
    print(f"  rỗng                : {len(empty)}")
    print(f"  quá ngắn (<12 ký tự): {len(very_short)}")
    print(f"  trùng tuyệt đối     : {len(exact_dupes)} giá trị")
    print(f"  cụm xung đột nhãn   : {len(conflict_comps)} cụm ({len(conflict_rows)} dòng)")
    print()
    print("--- Kiểm chứng ---")
    print(f"  trùng khoá cụm      : {len(key_overlap)}")
    print(f"  vi phạm gần-trùng   : {len(violations)}")
    print(f"  khoảng cách nhỏ nhất: {min_distance}")
    print(f"  KẾT LUẬN            : {'ĐẠT (PASSED)' if manifest['verification']['passed'] else 'KHÔNG ĐẠT (FAILED)'}")
    print(f"\nĐã ghi các tệp kết quả vào: {OUT_DIR}")

    return 0 if manifest["verification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
