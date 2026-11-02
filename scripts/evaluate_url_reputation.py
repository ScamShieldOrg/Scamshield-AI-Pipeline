"""Measure URL-only Tier-1 coverage and false alerts on an independent CSV.

Input columns: url,label where label is 1 for a confirmed threat and 0 for a
confirmed benign URL. UNKNOWN is counted separately, never as a true negative.
Freeze labels independently of the live PhishTank/Google feeds before evaluation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.url_signals import UrlSignals  # noqa: E402


def evaluate(path: Path) -> dict:
    signals = UrlSignals()
    counts = {"benign": 0, "threat": 0, "false_alerts": 0, "true_alerts": 0,
              "unknown_benign": 0, "unknown_threat": 0}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"url", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain url,label columns")
        for row in reader:
            if row["label"] not in {"0", "1"}:
                raise ValueError("Labels must be 0 or 1")
            label = "threat" if row["label"] == "1" else "benign"
            counts[label] += 1
            result = signals.analyze(row["url"])
            if result["known_threat_match"]:
                counts["true_alerts" if label == "threat" else "false_alerts"] += 1
            else:
                counts[f"unknown_{label}"] += 1
    counts["false_alert_rate"] = counts["false_alerts"] / counts["benign"] if counts["benign"] else None
    counts["known_threat_recall"] = counts["true_alerts"] / counts["threat"] if counts["threat"] else None
    counts["unknown_rate"] = (counts["unknown_benign"] + counts["unknown_threat"]) / (counts["benign"] + counts["threat"]) if counts["benign"] + counts["threat"] else None
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input_csv), ensure_ascii=False, indent=2))
