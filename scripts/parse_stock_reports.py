"""
Trendlyne Stock Reports -> snapshots: Trendlyne's scores, checklist and analyst consensus, read from
every report on file and kept month by month beside our own analysis.

  python scripts/parse_stock_reports.py                  # read every report this parser has not read
  python scripts/parse_stock_reports.py --status         # what is read; which fields came back blank
  python scripts/parse_stock_reports.py --symbol AUBANK  # one company (and show its latest snapshot)
  python scripts/parse_stock_reports.py --reparse        # read everything again
  python scripts/parse_stock_reports.py --no-ledger      # read, but do not record calls in the ledger

Each snapshot also becomes a ledger call (source "trendlyne"): its D/V/M lean, dated to the first
session after the report's data timestamp, graded like any broker call by track_outcomes.py.

Trendlyne's scores are a vendor's OPINION: shown attributed, never as this system's view.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.data import trendlyne_scores as ts   # noqa: E402


def print_status() -> None:
    st = ts.status()
    print("\nTRENDLYNE STOCK REPORTS")
    print(f"  reports on file      {st['reports']:>5}")
    print(f"  read into snapshots  {st['parsed']:>5}   ({st['companies']} companies; "
          f"{st['older_parser']} by an older parser)")
    if st["unreadable"]:
        print("  blank fields (not printed in that report, or not readable):")
        for f, n in sorted(st["unreadable"].items(), key=lambda kv: -kv[1]):
            print(f"    {f:<20} {n:>4}")


def print_latest(symbol: str) -> None:
    got = ts.latest(symbol)
    if not got:
        print(f"  no snapshot for {symbol.upper()}")
        return
    key, s = got
    direction, conviction = ts.lean(s)
    print(f"\n  {symbol.upper()}  as of {s.as_of}  price {s.price}   [{key}]")
    print(f"  D {s.durability} ({s.durability_label})  V {s.valuation} ({s.valuation_label})  "
          f"M {s.momentum} ({s.momentum_label})  -> {s.classification}; lean {direction} ({conviction})")
    print(f"  checklist {s.checklist_pass_pct}% ({s.checklist_passed} passed / {s.checklist_failed} failed) "
          f"{s.checklist_groups}")
    print(f"  consensus {s.consensus} of {s.analysts} analysts, target {s.target_price} ({s.upside_pct}%), "
          f"P/E zone {s.pe_zone} {s.pe_zone_pct}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol")
    ap.add_argument("--reparse", action="store_true")
    ap.add_argument("--no-ledger", action="store_true", help="do not record Trendlyne's view as ledger calls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.status:
        print_status()
        return 0
    tally = ts.parse_all(reparse=args.reparse, symbol=args.symbol)
    print(f"read {tally['parsed']}, already current {tally['current']}, failed {tally['failed']}, "
          f"file missing {tally['missing']}")
    if not args.no_ledger:
        added = ts.record_predictions()
        print(f"ledger: {added} new Trendlyne call(s)"
              + (" - grade them with scripts/track_outcomes.py" if added else ""))
    if args.symbol:
        print_latest(args.symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
