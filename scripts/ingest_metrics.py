"""
Populate the SCREENING metrics cache in market.db - one compact row per stock.

Each row costs several network calls and a few seconds, so this is an OVERNIGHT job. Once cached,
scripts/screen.py ranks the whole universe in milliseconds and re-scoring never needs a re-fetch.

  python scripts/ingest_metrics.py --index totalmarket,microcap250   # ~1000 real companies
  python scripts/ingest_metrics.py --from-universe --limit 500
  python scripts/ingest_metrics.py TATASTEEL JSWSTEEL SAIL
  python scripts/ingest_metrics.py --from-universe --resume          # skip what is already cached
  python scripts/ingest_metrics.py --coverage                        # what is cached, and how stale

Interrupt-safe: rows are written in batches, so stopping the job keeps everything fetched so far.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

from pennystock.config import settings                                        # noqa: E402
from pennystock.data.metrics import (cached_symbols, fetch_metrics,           # noqa: E402
                                     load_metrics, save_metrics)
from pennystock.data.peers import INDEX_LISTS, index_symbols                  # noqa: E402

BATCH = 20


def universe_symbols(limit: int | None) -> list[str]:
    with sqlite3.connect(str(settings.market_db)) as con:
        rows = con.execute("SELECT symbol FROM universe WHERE is_active=1 ORDER BY symbol").fetchall()
    syms = [r[0] for r in rows]
    return syms[:limit] if limit else syms


def coverage() -> str:
    rows = load_metrics()
    if not rows:
        return "stock_metrics is empty - run this script first."
    ages = [(datetime.now() - datetime.fromisoformat(r.fetched_at)).days
            for r in rows if r.fetched_at]
    complete = sum(1 for r in rows if r.altman_zem is not None and r.turnover_cr is not None)
    return (f"{len(rows)} symbols cached | {complete} with full forensics + liquidity\n"
            f"age: newest {min(ages)}d, oldest {max(ages)}d" if ages else f"{len(rows)} cached")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache per-stock screening metrics.")
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--index", help="NSE index list(s), comma separated. Choices: " + ", ".join(INDEX_LISTS))
    ap.add_argument("--from-universe", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true", help="skip symbols already in the cache")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between symbols")
    ap.add_argument("--coverage", action="store_true", help="report what is cached and exit")
    args = ap.parse_args()

    if args.coverage:
        print(coverage()); return 0

    if args.index:
        symbols = index_symbols(args.index.split(","))
    elif args.from_universe:
        symbols = universe_symbols(None)
    else:
        symbols = [s.upper() for s in args.symbols]
    if args.resume:
        already = cached_symbols()
        symbols = [s for s in symbols if s not in already]
    if args.limit:
        symbols = symbols[: args.limit]
    if not symbols:
        ap.print_help(); return 1

    print(f"Fetching screening metrics for {len(symbols)} symbols "
          f"(~{len(symbols) * (args.delay + 3) / 60:.0f} min)...\n")
    batch, saved, missed, t0 = [], 0, 0, time.time()
    try:
        for i, sym in enumerate(symbols, 1):
            row = fetch_metrics(sym)
            if row is None:
                missed += 1
                print(f"  [{i}/{len(symbols)}] {sym:<14} -- no data")
            else:
                batch.append(row)
                print(f"  [{i}/{len(symbols)}] {sym:<14} mcap {str(row.market_cap_cr):>10}cr  "
                      f"turnover {str(row.turnover_cr):>7}cr  ROCE {str(row.roce_pct):>6}  "
                      f"Z-EM {str(row.altman_zem):>6}  flags {row.red_flag_count}")
            if len(batch) >= BATCH:                     # partial progress survives an interrupt
                saved += save_metrics(batch); batch = []
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what was fetched")
    saved += save_metrics(batch)
    print(f"\nSaved {saved} rows ({missed} had no usable data) in {(time.time()-t0)/60:.1f} min.")
    print("Next: python scripts/screen.py --top 25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
