"""
Download every company's key corporate filings from NSE into Corporate_Filings/.

  python scripts/fetch_filings.py --status                  # what the library holds
  python scripts/fetch_filings.py --dry-run --symbol AUBANK # what would be fetched, nothing written
  python scripts/fetch_filings.py                           # every company in the broker folder
  python scripts/fetch_filings.py --symbol BHEL --limit 20

Kept: results, investor presentations, earnings-call transcripts, credit ratings and material events
from the last two years, plus the newest two annual reports (pennystock/data/filings.py says how
each is recognised). Companies = every symbol with a Broker_Reports folder or a flat broker report.

Idempotent: a filing already in the documents catalog with its file on disk is skipped, so an
interrupted run resumes where it stopped. NSE is someone else's server: one download a second.
No Trendlyne quota is used - these are the exchange's own public documents.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

from pennystock.config import settings                   # noqa: E402
from pennystock.data import documents, filings            # noqa: E402
from pennystock.data.broker_corpus import company_symbols  # noqa: E402
from pennystock.data.nse_client import nse_api, nse_session  # noqa: E402


def status() -> int:
    held = documents.list_documents(kind="filing")
    by_cat = Counter(d.category for d in held)
    print(f"\nCORPORATE FILINGS ({settings.filings_dir})")
    print(f"  files {len(held):>6}   companies {len({d.symbol for d in held}):>4}   "
          f"size {sum(d.bytes for d in held) / 1e9:.2f} GB")
    for cat in filings.CATEGORIES:
        print(f"  {cat:<24}{by_cat.get(cat, 0):>6}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched; write nothing")
    ap.add_argument("--symbol", action="append", help="only this company (repeatable)")
    ap.add_argument("--limit", type=int, help="stop after N downloads this run")
    ap.add_argument("--days", type=int, default=settings.filings_days)
    args = ap.parse_args()
    if args.status:
        return status()

    symbols = [s.upper() for s in args.symbol] if args.symbol else company_symbols(ROOT / "Broker_Reports")
    tally: Counter = Counter()
    downloads = 0
    print(f"{len(symbols)} companies, filings from the last {args.days} days\n")
    for n, sym in enumerate(symbols, 1):
        if args.limit and downloads >= args.limit:
            break
        session = None
        for attempt in (1, 2):                  # NSE times out now and then; one patient retry
            try:
                session = nse_session(sym)
                break
            except Exception as e:
                print(f"[{n}/{len(symbols)}] {sym}: NSE session failed, attempt {attempt} ({type(e).__name__})")
                time.sleep(10)
        if session is None:
            tally["no_session"] += 1
            continue
        found = filings.discover(sym, lambda path: nse_api(session, path), days=args.days)
        cats = Counter(f.category for f in found)
        print(f"[{n}/{len(symbols)}] {sym:<12} {len(found):>3} to keep  "
              + "  ".join(f"{c}={k}" for c, k in sorted(cats.items())))
        if args.dry_run:
            for f in found[:60]:
                print(f"      {f.date}  {f.category:<22} {filings.filing_path(f).name[:90]}")
            continue
        get = filings.nse_getter(session)
        for f in found:
            if args.limit and downloads >= args.limit:
                break
            result = filings.download(f, get)
            tally[result] += 1
            if result != "have":
                downloads += 1
                time.sleep(settings.filings_delay_s)
        filings.write_manifest(sym)

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())) if tally else "")
    if not args.dry_run:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
