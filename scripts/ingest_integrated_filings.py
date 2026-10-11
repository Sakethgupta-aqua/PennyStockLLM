"""Warm the NSE integrated-filing cache (audited figures + the half-yearly related-party disclosure) for every
company in the documents library, and report coverage.

  python scripts/ingest_integrated_filings.py --status
  python scripts/ingest_integrated_filings.py              # every company; parsed files are never re-fetched
  python scripts/ingest_integrated_filings.py --symbol DIXON
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pennystock.config import settings                                      # noqa: E402
from pennystock.data.integrated_filings import IntegratedFilings            # noqa: E402


def companies() -> list[str]:
    with sqlite3.connect(str(settings.filings_db)) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM filing_docs ORDER BY symbol")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol", nargs="*")
    args = ap.parse_args()
    store = IntegratedFilings(offline=args.status)            # a status report reads the cache only
    symbols = [s.upper() for s in args.symbol] if args.symbol else companies()
    years = rpts = 0
    for n, sym in enumerate(symbols, 1):
        fy, rpt = store.filed_year(sym), store.rpt_disclosure(sym)
        years += fy is not None
        rpts += rpt is not None
        if not args.status:
            print(f"[{n}/{len(symbols)}] {sym:<12} filed year: {fy.fiscal_year if fy else '-':<5} "
                  f"related parties: {f'{rpt.half}, {len(rpt.rows)} rows' if rpt else '-'}", flush=True)
            time.sleep(settings.refresh_delay_s)
    print(f"\n{len(symbols)} companies: filed year {years} ({years / max(len(symbols), 1):.0%}), "
          f"related-party disclosure {rpts} ({rpts / max(len(symbols), 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
