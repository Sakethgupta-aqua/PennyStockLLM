"""
Fetch, cache and inspect quarterly shareholding - and record promoter PLEDGE by hand.

Shareholding is filed quarterly, so this is a periodic job, not a per-query fetch. Cached rows
live in market.db and are served instantly to the agent.

  # cache shareholding for specific stocks
  python scripts/ingest_ownership.py CIPLA SUZLON RPOWER

  # cache the whole active universe (slow - be polite to the source)
  python scripts/ingest_ownership.py --from-universe --limit 50

  # PLEDGE override / backfill (NSE serves only the LATEST quarter; history builds up as you re-run)
  python scripts/ingest_ownership.py --set-pledge SUZLON "Jun 2026" 12.5
  python scripts/ingest_ownership.py --import-pledge-csv pledges.csv     # symbol,quarter,pledge_pct

  # see what the agent will see
  python scripts/ingest_ownership.py --show CIPLA
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                                    # noqa: E402
from pennystock.data.ownership import summarize_ownership                 # noqa: E402
from pennystock.data.ownership_providers import OwnershipStore, get_ownership  # noqa: E402


def universe_symbols(limit: int | None) -> list[str]:
    with sqlite3.connect(str(settings.market_db)) as con:
        rows = con.execute("SELECT symbol FROM universe WHERE is_active=1 ORDER BY symbol").fetchall()
    syms = [r[0] for r in rows]
    return syms[:limit] if limit else syms


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache quarterly shareholding; record promoter pledge.")
    ap.add_argument("symbols", nargs="*", help="NSE symbols to fetch")
    ap.add_argument("--from-universe", action="store_true", help="fetch every active symbol in market.db")
    ap.add_argument("--limit", type=int, help="cap --from-universe")
    ap.add_argument("--refresh", action="store_true", help="refetch even if the cache is fresh")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests (be polite)")
    ap.add_argument("--set-pledge", nargs=3, metavar=("SYMBOL", "QUARTER", "PCT"),
                    help='record a pledge %%, e.g. --set-pledge SUZLON "Jun 2026" 12.5')
    ap.add_argument("--import-pledge-csv", metavar="FILE",
                    help="bulk pledge import; CSV columns: symbol,quarter,pledge_pct")
    ap.add_argument("--show", metavar="SYMBOL", help="print the summary the agent would receive")
    args = ap.parse_args()

    store = OwnershipStore()

    if args.set_pledge:
        sym, quarter, pct = args.set_pledge
        store.set_pledge(sym, quarter, float(pct), source="manual")
        print(f"Recorded pledge {pct}% for {sym.upper()} ({quarter}).")
        return 0

    if args.import_pledge_csv:
        n = 0
        with open(args.import_pledge_csv, newline="") as f:
            for row in csv.DictReader(f):
                store.set_pledge(row["symbol"], row["quarter"], float(row["pledge_pct"]), source="manual")
                n += 1
        print(f"Imported {n} pledge row(s) from {args.import_pledge_csv}.")
        return 0

    if args.show:
        print(json.dumps(summarize_ownership(store.load(args.show),
                                             settings.ownership_lookback_quarters),
                         indent=2, default=str))
        return 0

    symbols = universe_symbols(args.limit) if args.from_universe else [s.upper() for s in args.symbols]
    if not symbols:
        ap.print_help()
        return 1

    ok = miss = 0
    for i, sym in enumerate(symbols, 1):
        own = get_ownership(sym, refresh=args.refresh)
        if own.quarters:
            ok += 1
            latest = own.quarters[-1]
            print(f"  [{i}/{len(symbols)}] {sym:<12} {len(own.quarters):>2}q  "
                  f"latest {latest.quarter}: promoter {latest.promoter_pct}%  "
                  f"pledge {latest.pledge_pct if latest.pledge_pct is not None else 'UNKNOWN'}")
        else:
            miss += 1
            print(f"  [{i}/{len(symbols)}] {sym:<12} no data")
        if i < len(symbols):
            time.sleep(args.delay)

    print(f"\nCached {ok} symbol(s); {miss} without data. Stored in {settings.market_db}")
    if ok:
        print("Pledge comes from NSE's statutory filing. Use --set-pledge only to override or to\n"
              "backfill quarters NSE no longer serves (it returns the latest quarter only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
