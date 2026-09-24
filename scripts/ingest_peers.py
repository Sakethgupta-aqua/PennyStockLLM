"""
Populate the peer-metrics cache in market.db (sector/industry + valuation ratios per symbol).

Peer comparison needs one network call per company, which is far too slow inside an agent run -
so we cache offline and serve instantly. Re-run monthly, or after a big market move.

  python scripts/ingest_peers.py --index totalmarket,microcap250   # ~1000 real companies by size
  python scripts/ingest_peers.py --index nifty500                  # the Nifty 500
  python scripts/ingest_peers.py --seed-liquid            # ~60 large, liquid names across sectors
  python scripts/ingest_peers.py TATASTEEL JSWSTEEL SAIL  # specific symbols
  python scripts/ingest_peers.py --from-universe --limit 300
  python scripts/ingest_peers.py --coverage               # what is cached
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                       # noqa: E402
from pennystock.data.peers import (INDEX_LISTS, SEED_LIQUID, coverage,  # noqa: E402
                                   fetch_one, index_symbols, save_peers)


def universe_symbols(limit: int | None) -> list[str]:
    with sqlite3.connect(str(settings.market_db)) as con:
        rows = con.execute("SELECT symbol FROM universe WHERE is_active=1 ORDER BY symbol").fetchall()
    syms = [r[0] for r in rows]
    return syms[:limit] if limit else syms


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache peer metrics for sector comparison.")
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--seed-liquid", action="store_true", help="cache a broad liquid starter set")
    ap.add_argument("--index", help="NSE index list(s), comma separated. Choices: "
                                    + ", ".join(INDEX_LISTS))
    ap.add_argument("--from-universe", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    ap.add_argument("--coverage", action="store_true", help="report what is cached and exit")
    args = ap.parse_args()

    if args.coverage:
        print(coverage()); return 0

    if args.index:
        symbols = index_symbols(args.index.split(","))
        if args.limit:
            symbols = symbols[:args.limit]
    elif args.seed_liquid:
        symbols = SEED_LIQUID
    elif args.from_universe:
        symbols = universe_symbols(args.limit)
    else:
        symbols = [s.upper() for s in args.symbols]
    if not symbols:
        ap.print_help(); return 1

    batch, missed = [], 0
    for i, sym in enumerate(symbols, 1):
        row = fetch_one(sym)
        if row:
            batch.append(row)
            print(f"  [{i}/{len(symbols)}] {sym:<12} {str(row.industry)[:30]:<32} P/E {row.pe}")
        else:
            missed += 1
            print(f"  [{i}/{len(symbols)}] {sym:<12} no data")
        if len(batch) >= 25:                                  # flush periodically - crash-safe
            save_peers(batch); batch = []
        time.sleep(args.delay)
    if batch:
        save_peers(batch)
    print(f"\nCached {len(symbols) - missed} symbol(s); {missed} without data. {coverage()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
