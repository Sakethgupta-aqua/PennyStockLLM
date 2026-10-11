"""Read every company's contingent-liability note into verified numbers: the local model lays the table out,
code verifies every figure against the page. Resumable - a report already read at this version is skipped.

  python scripts/extract_contingent_liabilities.py --status
  python scripts/extract_contingent_liabilities.py                  # every company with an annual report
  python scripts/extract_contingent_liabilities.py --symbol DIXON --force
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pennystock.config import settings                                      # noqa: E402
from pennystock.data import contingent_liabilities as cl                     # noqa: E402


def companies() -> list[str]:
    with sqlite3.connect(str(settings.filings_db)) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM filing_docs WHERE doc_type = "
                                          "'annual_report' ORDER BY symbol")]


def lenders() -> set[str]:
    from pennystock.data.metrics import load_metrics
    from pennystock.data.sectors import profile_for
    return {m.symbol for m in load_metrics() if profile_for(m.sector, m.industry).name == "financials"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    symbols = [s.upper() for s in args.symbol] if args.symbol else companies()
    results = {}
    for n, sym in enumerate(symbols, 1):
        if args.status:
            results[sym] = cl.stored(sym) or {"status": "not_read"}
            continue
        t0 = time.time()
        results[sym] = r = cl.extract(sym, force=args.force)
        kept, dropped = len(r.get("rows") or []), len(r.get("dropped") or [])
        print(f"[{n}/{len(symbols)}] {sym:<12} {r['status']:<9} page {r.get('page', '-')} "
              f"unit {r.get('unit', '-')} rows {kept} dropped {dropped} ({time.time() - t0:.0f}s)",
              flush=True)
    by = Counter(r["status"] for r in results.values())
    banks = lenders()
    others = [s for s in results if s not in banks]
    verified = sum(results[s]["status"] == "verified" for s in others)
    print(f"\n{len(results)} reports: {dict(by)}")
    share = verified / max(len(others), 1)
    print(f"verified coverage of non-lenders: {verified}/{len(others)} ({share:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
