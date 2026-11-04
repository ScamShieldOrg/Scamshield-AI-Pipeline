"""Calibrate BGE-M3 cosine cutoff from human-labelled Vietnamese query/passage pairs.

Input JSONL rows: query_id, query, passage, relevant (bool), source_group, split
where split is `calibration` or `holdout`. Keep source_group in only one split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.rag import BgeM3Embedder, CALIBRATION_PATH


def resolve_passages(rows: list[dict], model: str, version: str) -> None:
    """Resolve benchmark chunk references against the active, verified corpus."""
    pending = [row for row in rows if not row.get("passage")]
    if not pending:
        return
    import psycopg
    import yaml
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    source_urls = {source["key"]: source["url"] for source in
                   yaml.safe_load((ROOT / "config" / "rag_sources.yaml").read_text(encoding="utf-8"))["seeds"]}
    url = os.getenv("DATABASE_URL", "postgresql://scamshield:scamshield@localhost:5432/scamshield_ai")
    with psycopg.connect(url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            for row in pending:
                source_url = row.get("source_url") or source_urls[row["source_key"]]
                cur.execute(
                    """SELECT c.chunk_text FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id
                       WHERE d.source_url = %s AND c.chunk_no = %s AND d.is_active = true
                         AND d.review_status = 'VERIFIED' AND c.is_active = true
                         AND c.embedding_model = %s AND c.embedding_version = %s""",
                    (source_url, row["chunk_no"], model, version),
                )
                found = cur.fetchone()
                if not found:
                    raise ValueError(f"benchmark chunk unavailable: {source_url} #{row['chunk_no']}")
                row["passage"] = found[0]


def top_five(rows: list[dict], scores: list[float]) -> list[dict]:
    by_query = defaultdict(list)
    for row, score in zip(rows, scores):
        by_query[(row["split"], row["query_id"])].append({**row, "similarity": score})
    return [item for candidates in by_query.values()
            for item in sorted(candidates, key=lambda value: value["similarity"], reverse=True)[:5]]


def metrics(rows: list[dict], threshold: float) -> dict[str, float]:
    accepted = [row for row in rows if row["similarity"] >= threshold]
    true_positive = sum(bool(row["relevant"]) for row in accepted)
    positives = sum(bool(row["relevant"]) for row in rows)
    query_ids = {row["query_id"] for row in rows}
    matched_ids = {row["query_id"] for row in accepted}
    return {"precision": true_positive / len(accepted) if accepted else 0.0,
            "recall": true_positive / positives if positives else 0.0,
            "no_match_rate": 1 - len(matched_ids) / len(query_ids) if query_ids else 1.0,
            "accepted": len(accepted), "queries": len(query_ids)}


def choose_threshold(rows: list[dict]) -> tuple[float, dict[str, float]]:
    if not rows or not any(row["relevant"] for row in rows) or not any(not row["relevant"] for row in rows):
        raise ValueError("calibration requires relevant and irrelevant pairs")
    candidates = sorted({float(row["similarity"]) for row in rows})
    eligible = [(threshold, metrics(rows, threshold)) for threshold in candidates]
    eligible = [(threshold, result) for threshold, result in eligible
                if result["accepted"] and result["precision"] >= 0.95]
    if not eligible:
        raise ValueError("no threshold reaches 95% calibration precision")
    return max(eligible, key=lambda item: (item[1]["recall"], item[1]["accepted"], -item[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 20 or {row.get("split") for row in rows} != {"calibration", "holdout"}:
        raise ValueError("benchmark needs >=20 labelled pairs and both splits")
    groups = defaultdict(set)
    for row in rows:
        if not all(key in row for key in ("query_id", "query", "relevant", "source_group", "split")):
            raise ValueError("benchmark row missing required field")
        if not row.get("passage") and not ("chunk_no" in row and ("source_url" in row or "source_key" in row)):
            raise ValueError("benchmark row needs passage or source_key/source_url + chunk_no")
        if not isinstance(row["relevant"], bool):
            raise ValueError("relevant must be boolean")
        groups[row["source_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("source_group leaks across calibration and holdout")
    embedder = BgeM3Embedder()
    resolve_passages(rows, embedder.model_name, embedder.revision)
    unique_texts = list(dict.fromkeys([text for row in rows for text in (row["query"], row["passage"])]))
    vectors = dict(zip(unique_texts, embedder.encode(unique_texts)))
    scores = [sum(a * b for a, b in zip(vectors[row["query"]], vectors[row["passage"]])) for row in rows]
    ranked = top_five(rows, scores)
    calibration = [row for row in ranked if row["split"] == "calibration"]
    holdout = [row for row in ranked if row["split"] == "holdout"]
    threshold, calibration_metrics = choose_threshold(calibration)
    holdout_metrics = metrics(holdout, threshold)
    report = {"embedding_model": embedder.model_name, "embedding_version": embedder.revision,
              "similarity_threshold": round(threshold, 7),
              "calibration_precision": calibration_metrics["precision"],
              "calibration": calibration_metrics, "holdout": holdout_metrics,
              "sample_count": len(rows), "calibrated_at": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if holdout_metrics["precision"] < 0.95 or holdout_metrics["accepted"] == 0:
        raise ValueError("holdout precision below 95%; threshold not activated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
