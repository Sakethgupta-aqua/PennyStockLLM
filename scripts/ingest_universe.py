"""
Refresh the NSE symbol universe in market.db (the table the name->ticker resolver reads).

The real work lives in pennystock/data/universe.py so the pipeline can call it too - this is
just the manual/cron entry point. demo.py refreshes automatically when the data is older than
`universe_max_age_days`, so you normally do not need to run this by hand.

  python scripts/ingest_universe.py            # refresh only if stale
  python scripts/ingest_universe.py --force    # refresh regardless
  python scripts/ingest_universe.py --status   # just report freshness
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                                  # noqa: E402
from pennystock.data.universe import ensure_fresh, is_stale, last_updated  # noqa: E402


def counts() -> tuple[int, int]:
    with sqlite3.connect(str(settings.market_db)) as con:
        total = con.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
        active = con.execute("SELECT COUNT(*) FROM universe WHERE is_active=1").fetchone()[0]
    return total, active


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh the NSE symbol universe.")
    ap.add_argument("--force", action="store_true", help="refresh even if the data is fresh")
    ap.add_argument("--status", action="store_true", help="report freshness and exit")
    args = ap.parse_args()

    if args.status:
        total, active = counts()
        print(f"last updated : {last_updated()}\nstale        : {is_stale()} "
              f"(threshold {settings.universe_max_age_days}d)\nsymbols      : {active} active / {total} total")
        return 0

    print(ensure_fresh(force=args.force))
    total, active = counts()
    print(f"{active} active / {total} total symbols in {settings.market_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
