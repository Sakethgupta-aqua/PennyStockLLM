"""
File Trendlyne Stock Report downloads into Stock_Reports/ and the documents catalog.

  python scripts/file_stock_reports.py                  # scan ~/Downloads
  python scripts/file_stock_reports.py --from DIR       # scan another folder
  python scripts/file_stock_reports.py --register       # only catalogue what is already in Stock_Reports/

Picks up <SYM>-StockReport-<YYYYMMDD>-<HHMM>[__TL<id>].pdf downloads (Trendlyne's own names) and the
legacy tlsr_batch_*.bin bundles, keeps only real PDFs, and moves each filed original into
Stock_Reports/_filed_downloads/ so the Downloads folder stays clean - nothing is deleted. A report is
never overwritten: a new month's snapshot is a new dated file. Rewrites Stock_Reports/index.csv.

Monthly refresh in the browser: open each company's report at
trendlyne.com/equity/stock-report/get-pdf/<id>/ (one of 70 daily opens each), then run this script.
With an exported cookie, scripts/fetch_stock_reports.py does the fetching itself.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings              # noqa: E402
from pennystock.data import documents, stock_reports, trendlyne_scores  # noqa: E402


def file_downloads(src: Path) -> tuple[int, int]:
    """(filed, rejected) for every Stock Report download and bundle in `src`."""
    done = Path(settings.stock_reports_dir) / "_filed_downloads"
    done.mkdir(parents=True, exist_ok=True)
    filed = rejected = 0
    for f in sorted(src.glob("*-StockReport-*.pdf")):
        name = stock_reports.parse_name(f.name)
        if not name:
            continue
        if stock_reports.file_snapshot(f.read_bytes(), name.symbol, name.stamp, name.tl_id):
            filed += 1
            shutil.move(str(f), done / f.name)
        else:
            rejected += 1
            print(f"  not a report, left in place: {f.name}")
    for f in sorted(src.glob("tlsr_batch_*.bin")):
        fallback = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y%m%d-%H%M")
        for sym, tl_id, pdf, server in stock_reports.unpack_bundle(f.read_bytes()):
            name = stock_reports.parse_name(server)
            if stock_reports.file_snapshot(pdf, sym, name.stamp if name else fallback, tl_id):
                filed += 1
            else:
                rejected += 1
                print(f"  bundle {f.name}: {sym} is not a report")
        shutil.move(str(f), done / f.name)
    return filed, rejected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--register", action="store_true", help="only catalogue the PDFs already filed")
    args = ap.parse_args()

    # the existing index carries each file's real symbol and Trendlyne id - read it BEFORE rewriting it
    added = stock_reports.register_existing()
    filed = rejected = 0
    if not args.register:
        filed, rejected = file_downloads(args.src)
    index = stock_reports.write_index()
    held = documents.list_documents(kind="stock_report")
    print(f"registered {added} existing, filed {filed} new, rejected {rejected}")
    print(f"Stock Reports held: {len(held)} snapshots of {len({d.symbol for d in held})} companies "
          f"({sum(d.bytes for d in held) / 1e6:.0f} MB) - index: {index.relative_to(ROOT)}")
    tl = trendlyne_scores.parse_all()
    calls = trendlyne_scores.record_predictions()
    print(f"read {tl['parsed']} new report(s) into Trendlyne snapshots ({tl['failed']} unreadable); "
          f"{calls} new Trendlyne call(s) in the ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
