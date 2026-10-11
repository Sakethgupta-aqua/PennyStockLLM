"""
The documents catalog: what the company library holds, and whether every file is still intact.

  python scripts/catalog_documents.py                # summary: files, companies and size per kind
  python scripts/catalog_documents.py --backfill     # register Broker_Reports + Stock_Reports already on disk
  python scripts/catalog_documents.py --verify       # re-hash every catalogued file
  python scripts/catalog_documents.py --symbol BHEL  # everything held for one company

Three libraries, one table (market.db `documents`): Broker_Reports/ (sell-side research),
Stock_Reports/ (Trendlyne snapshots), Corporate_Filings/ (NSE). The backfill only catalogues files
where they already are - nothing is moved or copied.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.data import documents, stock_reports              # noqa: E402
from pennystock.data.broker_corpus import ReportRecord, discover  # noqa: E402


def broker_document(rec: ReportRecord) -> dict | None:
    """documents.register(...) arguments for one downloaded broker report; None if no PDF or symbol."""
    if not (rec.pdf and rec.symbol and rec.report_date):
        return None
    title = ": ".join(x for x in (rec.broker, rec.headline) if x) or None
    return {"doc_key": f"broker_report:trendlyne:{rec.report_id}" if rec.report_id else f"broker_report:sha:{rec.doc_id}",
            "symbol": rec.symbol, "kind": "broker_report", "doc_date": rec.report_date, "source": "trendlyne",
            "path": rec.pdf, "title": title,
            "source_url": f"https://trendlyne.com/get-document/report/pdf/{rec.report_id}/" if rec.report_id else None}


def backfill() -> None:
    added = skipped = 0
    for rec in discover(ROOT / "Broker_Reports"):
        kw = broker_document(rec)
        if kw is None:
            skipped += 1
            continue
        added += documents.register(**kw)
    print(f"broker reports: {added} newly catalogued ({skipped} listed but not downloaded, or unlisted)")
    print(f"stock reports:  {stock_reports.register_existing()} newly catalogued")


def summary(symbol: str | None) -> None:
    if symbol:
        for d in documents.list_documents(symbol):
            print(f"  {d.doc_date}  {d.kind:<14}{(d.category or ''):<22}{d.local_path}")
        return
    print("\nCOMPANY DOCUMENTS LIBRARY")
    for kind, c in sorted(documents.counts().items()):
        print(f"  {kind:<14} {c['files']:>6} files  {c['companies']:>4} companies  {c['bytes'] / 1e9:>6.2f} GB")
    cats = Counter(d.category for d in documents.list_documents(kind="filing"))
    if cats:
        print("  filings by type: " + "  ".join(f"{k}={v}" for k, v in sorted(cats.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--symbol")
    args = ap.parse_args()
    if args.backfill:
        backfill()
    if args.verify:
        bad = documents.verify()
        print(f"verify: {len(bad['missing'])} missing, {len(bad['changed'])} changed")
        for k in bad["missing"][:20] + bad["changed"][:20]:
            print(f"  ! {k}")
    summary(args.symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
