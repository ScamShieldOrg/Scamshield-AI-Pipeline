"""
==============================================================================
ScamShield-VN: Comprehensive Two-Tier Benchmark Runner
==============================================================================
Author: ScamShield Research & Evaluation Team
Description:
    Evaluates:
      - Arm 1: Solo Tier 1 Edge (ViSoBERT Fast-Path)
      - Arm 2: Solo Tier 2 (Consolidated 3-Agent Forensic Council)
      - Arm 3: Cascaded Two-Tier System (Tier 1 + Tier 2 Escalation)
    Complies strictly with AGENTS.md:
      - Logs per-sample telemetry in real-time
      - Records exact count of API calls made, latency, cost, and routing path
      - Stores results into SQLite DB and JSONL
      - Built-in SQLite Cache for Tier-2 LLM responses (prevents re-billing)
==============================================================================
"""

import os
import sys
import csv
import json
import time
import uuid
import hashlib
import sqlite3
import argparse
import logging
from typing import Dict, List, Any, Optional

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.visobert_classifier import get_visobert_classifier
from pipeline.forensic_council import ForensicCouncil
from pipeline.two_tier_router import TwoTierRouter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TwoTierBenchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tier ScamShield Architecture")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "tier1_only", "tier2_only", "cascaded"],
        default="cascaded",
        help="Evaluation arm to run (default: cascaded)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "raw", "vietnamese_sms_merged_all.csv"),
        help="Path to CSV/JSONL dataset file",
    )
    parser.add_argument("--limit", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 samples smoke test")
    parser.add_argument("--rag", action="store_true", help="Use shared BGE-M3 RAG in the cascaded arm")
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.path.join(ROOT_DIR, "experiments", "benchmark_results.db"),
        help="SQLite database path for persisting run telemetry and cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results"),
        help="Directory to save summary reports and telemetry",
    )
    return parser.parse_args()


def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sample_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        mode TEXT NOT NULL,
        sample_id TEXT NOT NULL,
        gold_label TEXT NOT NULL,
        predicted_label TEXT NOT NULL,
        is_correct INTEGER NOT NULL,
        confidence REAL NOT NULL,
        latency_ms REAL NOT NULL,
        cost_usd REAL NOT NULL,
        api_calls_count INTEGER NOT NULL,
        routing_path TEXT NOT NULL,
        escalated INTEGER NOT NULL,
        defender_intervened INTEGER NOT NULL,
        message TEXT,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tier2_response_cache (
        msg_hash TEXT PRIMARY KEY,
        message TEXT,
        response_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def get_cached_council_response(db_path: str, message: str) -> Optional[Dict[str, Any]]:
    msg_hash = hashlib.sha256(message.strip().encode("utf-8")).hexdigest()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT response_json FROM tier2_response_cache WHERE msg_hash = ?", (msg_hash,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def set_cached_council_response(db_path: str, message: str, response: Dict[str, Any]):
    msg_hash = hashlib.sha256(message.strip().encode("utf-8")).hexdigest()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO tier2_response_cache (msg_hash, message, response_json) VALUES (?, ?, ?)",
        (msg_hash, message[:500], json.dumps(response, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def load_dataset(filepath: str, limit: int = 0) -> List[Dict[str, Any]]:
    samples = []
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                msg = item.get("message") or item.get("text") or item.get("content") or ""
                gold = item.get("label") or item.get("gold_label") or item.get("ground_truth") or ""
                gold_norm = "SCAM" if any(k in str(gold).upper() for k in ["SCAM", "SPAM", "1"]) else "HAM"
                sid = item.get("id") or item.get("sample_id") or f"SAMPLE-{idx:04d}"
                if msg.strip():
                    samples.append({"sample_id": str(sid), "message": msg.strip(), "gold_label": gold_norm})
                if limit > 0 and len(samples) >= limit:
                    break
    else:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                msg = row.get("message") or row.get("\ufeffmessage") or ""
                gold = row.get("label") or ""
                gold_norm = "SCAM" if any(k in str(gold).upper() for k in ["SCAM", "SPAM", "1"]) else "HAM"
                sid = row.get("message_id") or row.get("id") or f"SMS-{idx:04d}"
                if msg.strip():
                    samples.append({"sample_id": str(sid), "message": msg.strip(), "gold_label": gold_norm})
                if limit > 0 and len(samples) >= limit:
                    break
    return samples


def run_benchmark():
    args = parse_args()
    limit = 5 if args.dry_run else args.limit
    init_db(args.db_path)
    os.makedirs(args.output_dir, exist_ok=True)

    samples = load_dataset(args.dataset, limit=limit)
    total_samples = len(samples)
    dataset_name = os.path.basename(args.dataset)
    run_id = f"RUN-{datetime_tag()}-{args.mode.upper()}{'-RAG' if args.rag else '-BASE'}"
    logger.info(f"Loaded {total_samples} samples from {dataset_name}. Starting mode: {args.mode} (run_id: {run_id})")

    # Initialize components according to mode
    tier1_classifier = None
    council = None
    router = None

    if args.mode in ("all", "tier1_only"):
        tier1_classifier = get_visobert_classifier()
    if args.mode in ("all", "tier2_only"):
        council = ForensicCouncil()
    if args.mode in ("all", "cascaded"):
        rag_retriever = None
        if args.rag:
            from pipeline.rag import BgeM3Embedder, RagRetriever
            rag_retriever = RagRetriever(BgeM3Embedder())
        router = TwoTierRouter(use_council=True, rag_retriever=rag_retriever)
        # Hook council evaluate_sync with SQLite response cache
        if router.council:
            original_eval = router.council.evaluate_sync

            def cached_eval(message: str, active_roles=None, **kwargs):
                cached = None if args.rag else get_cached_council_response(args.db_path, message)
                if cached:
                    return cached
                res = original_eval(message, active_roles=active_roles, **kwargs)
                if not args.rag:
                    set_cached_council_response(args.db_path, message, res)
                return res

            router.council.evaluate_sync = cached_eval

    modes_to_execute = [args.mode] if args.mode != "all" else ["tier1_only", "tier2_only", "cascaded"]

    telemetry_records = []

    for active_mode in modes_to_execute:
        logger.info(f"--- Executing Arm: {active_mode} ({total_samples} samples) ---")
        correct_count = 0
        tp = fp = tn = fn = 0
        total_latency = 0.0
        total_cost = 0.0
        total_calls = 0
        bench_start = time.perf_counter()

        for idx, s in enumerate(samples, 1):
            sid = s["sample_id"]
            msg = s["message"]
            gold = s["gold_label"]

            t0 = time.perf_counter()

            if active_mode == "tier1_only":
                res = tier1_classifier.predict(msg)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                pred = res["model_label"]  # SCAM or HAM
                confidence = res["confidence"]
                routing_path = "SOLO_TIER1_EDGE"
                escalated = 0
                defender_intervened = 0
                api_calls = 0
                cost = 0.0

            elif active_mode == "tier2_only":
                cached_res = get_cached_council_response(args.db_path, msg)
                if cached_res:
                    res = cached_res
                else:
                    res = council.evaluate_sync(msg)
                    set_cached_council_response(args.db_path, msg, res)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                final_lbl = res["final_label"]
                pred = "SCAM" if final_lbl == "SCAM-RISK" else ("HAM" if final_lbl == "LOW-RISK" else "UNKNOWN")
                confidence = res["consensus_scam_prob"]
                routing_path = "SOLO_TIER2_COUNCIL"
                escalated = 1
                defender_intervened = 1 if res.get("defender_intervened") else 0
                api_calls = res.get("total_api_calls", 3)
                cost = res.get("estimated_council_cost_usd", 0.00045)

            elif active_mode == "cascaded":
                res = router.classify(msg)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                final_lbl = res["final_label"]
                pred = "SCAM" if final_lbl in ("SCAM-RISK", "SCAM") else ("HAM" if final_lbl in ("LOW-RISK", "HAM") else "UNKNOWN")
                confidence = float(res.get("final_confidence", 0.5))
                routing_path = res.get("routing_path", "UNKNOWN")
                escalated = 1 if res.get("escalated") else 0
                t2_tel = res.get("tier2_telemetry", {})
                defender_intervened = 1 if t2_tel.get("defender_intervened") else 0
                api_calls = t2_tel.get("total_api_calls", 3 if escalated else 0)
                cost = res.get("total_cost_usd", 0.0)

            is_correct = 1 if pred == gold else 0
            if gold == "SCAM" and pred == "SCAM":
                tp += 1
            elif gold == "HAM" and pred == "SCAM":
                fp += 1
            elif gold == "HAM" and pred == "HAM":
                tn += 1
            elif gold == "SCAM" and pred == "HAM":
                fn += 1
            if is_correct:
                correct_count += 1
            total_latency += elapsed_ms
            total_cost += cost
            total_calls += api_calls

            rec = {
                "run_id": run_id,
                "mode": active_mode,
                "sample_id": sid,
                "gold_label": gold,
                "predicted_label": pred,
                "is_correct": is_correct,
                "confidence": round(confidence, 4),
                "latency_ms": round(elapsed_ms, 2),
                "cost_usd": round(cost, 6),
                "api_calls_count": api_calls,
                "routing_path": routing_path,
                "escalated": escalated,
                "defender_intervened": defender_intervened,
                "message": msg[:100] + ("..." if len(msg) > 100 else ""),
                "details": json.dumps(res, ensure_ascii=False),
            }
            telemetry_records.append(rec)

            # Log periodic status or significant events (escalations/errors)
            if idx % 50 == 0 or idx == total_samples or escalated:
                elapsed_total = time.perf_counter() - bench_start
                avg_time = elapsed_total / idx
                eta_s = avg_time * (total_samples - idx)
                logger.info(
                    f"[{idx}/{total_samples} - {idx/total_samples*100:.1f}%] Mode={active_mode} ID={sid} "
                    f"Gold={gold} Pred={pred} Lat={elapsed_ms:.1f}ms Calls={api_calls} Path={routing_path} "
                    f"| Acc={correct_count/idx*100:.1f}% | ETA={eta_s/60:.1f}m"
                )

        # Print summary for this mode
        acc = (correct_count / total_samples) * 100.0 if total_samples else 0.0
        avg_lat = total_latency / total_samples if total_samples else 0.0
        false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
        scam_recall = tp / (tp + fn) if tp + fn else 0.0
        logger.info(
            f"=== ARM SUMMARY [{active_mode}] ===\n"
            f"Total Samples: {total_samples} | Accuracy: {acc:.2f}%\n"
            f"FPR: {false_positive_rate:.3f} | SCAM recall: {scam_recall:.3f}\n"
            f"Total Cost: ${total_cost:.5f} | Total Calls: {total_calls} | Avg Latency: {avg_lat:.1f}ms\n"
        )

    # Persist records to SQLite
    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    cur.executemany("""
    INSERT INTO sample_telemetry (
        run_id, mode, sample_id, gold_label, predicted_label, is_correct,
        confidence, latency_ms, cost_usd, api_calls_count, routing_path,
        escalated, defender_intervened, message, details
    ) VALUES (
        :run_id, :mode, :sample_id, :gold_label, :predicted_label, :is_correct,
        :confidence, :latency_ms, :cost_usd, :api_calls_count, :routing_path,
        :escalated, :defender_intervened, :message, :details
    )
    """, telemetry_records)
    conn.commit()
    conn.close()

    # Save to JSONL telemetry
    jsonl_path = os.path.join(args.output_dir, f"telemetry_{run_id}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in telemetry_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(telemetry_records)} telemetry rows to {args.db_path} and {jsonl_path}")
    return run_id


def datetime_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    run_benchmark()
