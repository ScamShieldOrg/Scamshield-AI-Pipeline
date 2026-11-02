"""Download the verified, online PhishTank CSV feed once if it is missing.

The public feed works without an app key but is rate limited. Use --force only
when an explicit refresh is needed; a failed download preserves the old file.
"""

import csv
import gzip
import argparse
import os
import sys
import tempfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
def feed_path() -> Path:
    path = Path(os.getenv("PHISHTANK_FEED_PATH", ROOT / "data" / "phishtank_online_valid.csv.gz"))
    return path if path.is_absolute() else ROOT / path


def main(force: bool = False) -> int:
    destination = feed_path()
    if destination.is_file() and not force:
        print(f"PhishTank feed already exists: {destination}")
        return 0
    key = os.getenv("PHISHTANK_APP_KEY", "")
    has_key = bool(key and not key.lower().startswith("your_"))
    url = (f"https://data.phishtank.com/data/{key}/online-valid.csv.gz"
           if has_key else "https://data.phishtank.com/data/online-valid.csv.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(url, headers={"User-Agent": "phishtank/scamshield-ai-pipeline"}, timeout=30, stream=True) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="phishtank-", suffix=".csv.gz", delete=False) as target:
                temporary = Path(target.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        target.write(chunk)
        with gzip.open(temporary, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"phish_id", "url", "verified", "online"}.issubset(reader.fieldnames or []):
                raise ValueError("Downloaded feed has an unexpected schema")
            if not any(row.get("verified", "").lower() == "yes" and row.get("online", "").lower() == "yes" for row in reader):
                raise ValueError("Downloaded feed has no verified online records")
        temporary.replace(destination)
        print(f"PhishTank feed downloaded: {destination}")
        return 0
    except (OSError, requests.RequestException, ValueError, csv.Error) as exc:
        print(f"PhishTank feed update failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing feed with a fresh download")
    parser.add_argument("--anonymous", action="store_true", help="Legacy option; a missing feed already uses the public URL without a key")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
