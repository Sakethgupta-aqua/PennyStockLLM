"""
One command that keeps every data cache current. Safe to run from cron as often as you like.

Each dataset checks its own age first, so a run costs three SQLite reads when everything is fresh.
A lock file prevents a slow run overlapping the next scheduled one.

  python scripts/refresh_data.py --status            # what is cached and what is stale
  python scripts/refresh_data.py                     # refresh only what is stale
  python scripts/refresh_data.py --force             # refresh everything
  python scripts/refresh_data.py --only peers        # one dataset
  python scripts/refresh_data.py --print-cron        # a crontab line you can install

Exit codes: 0 = fine, 1 = at least one dataset failed, 2 = another run holds the lock.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.data.maintenance import refresh_all, single_flight, status  # noqa: E402

CRON = ("# PennyStockLLM: keep data caches warm. Runs daily at 07:30; each job self-checks staleness.\n"
        "30 7 * * *  cd {root} && {python} scripts/refresh_data.py >> {root}/refresh.log 2>&1\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh PennyStockLLM data caches.")
    ap.add_argument("--status", action="store_true", help="report freshness and exit")
    ap.add_argument("--force", action="store_true", help="refresh even if fresh")
    ap.add_argument("--only", help="comma-separated: universe,peers,ownership,metrics")
    ap.add_argument("--print-cron", action="store_true", help="print a crontab line and exit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    if args.print_cron:
        print(CRON.format(root=ROOT, python=sys.executable), end="")
        print("\n# install with:  crontab -e   (then paste the line above)")
        return 0

    if args.status:
        rows = status()
        print(json.dumps(rows, indent=2) if args.json else "")
        if not args.json:
            print(f"{'DATASET':<12}{'CADENCE':<10}{'STALE':<8}DETAIL")
            for r in rows:
                print(f"{r['dataset']:<12}{r['cadence']:<10}{str(r['stale']):<8}{r['detail']}")
        return 0

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    try:
        with single_flight():
            results = refresh_all(only=only, force=args.force)
    except RuntimeError as e:
        print(f"SKIPPED: {e}")
        return 2

    print(json.dumps(results, indent=2) if args.json else "")
    if not args.json:
        for r in results:
            secs = f" ({r['seconds']}s)" if "seconds" in r else ""
            print(f"  {r['dataset']:<12} {r['action']}{secs} - {r['detail']}")
    return 1 if any(r["action"] == "FAILED" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
