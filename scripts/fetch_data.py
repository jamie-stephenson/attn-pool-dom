"""Fetch CAA contrastive datasets into cache/caa/<behaviour>/.

Pulls the `generate`, A/B `test`, and open-ended `test` JSON for a behaviour from
the CAA repo. Refusal data (AdvBench/Alpaca) streams via `datasets`, so it isn't
fetched here.

Run: uv run python scripts/fetch_data.py --behaviour sycophancy
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/nrimsky/CAA/{branch}/datasets/{split}/{behaviour}/{fname}"
FILES = {
    "generate": "generate_dataset.json",
    "test": "test_dataset_ab.json",
    "test_open": "test_dataset_open_ended.json",
}
SPLIT = {"generate": "generate", "test": "test", "test_open": "test"}


def _download(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            if r.status == 200:
                return r.read()
    except Exception:
        return None
    return None


def fetch(behaviour: str, out_dir: Path) -> None:
    out = out_dir / behaviour
    out.mkdir(parents=True, exist_ok=True)
    for key, fname in FILES.items():
        data = None
        for branch in ("main", "master"):
            url = RAW.format(branch=branch, split=SPLIT[key], behaviour=behaviour, fname=fname)
            data = _download(url)
            if data:
                break
        if not data:
            print(f"  WARN: could not fetch {fname}")
            continue
        (out / fname).write_bytes(data)
        print(f"  wrote {out / fname} ({len(data)} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", default="sycophancy")
    ap.add_argument("--out", default="cache/caa")
    args = ap.parse_args()
    print(f"fetching CAA '{args.behaviour}' -> {args.out}/")
    fetch(args.behaviour, Path(args.out))


if __name__ == "__main__":
    main()
