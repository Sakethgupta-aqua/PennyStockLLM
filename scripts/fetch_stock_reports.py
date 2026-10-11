"""
Fetch this month's Trendlyne Stock Report for every company, over an exported browser cookie.

  python scripts/fetch_stock_reports.py --dry-run                      # who still needs one this month
  python scripts/fetch_stock_reports.py --cookies cookies.txt          # at most 70 opens
  python scripts/fetch_stock_reports.py --cookies cookies.txt --symbol AUBANK

Each report costs one of the account's 70 daily stock-report opens (1,000 a month). The run stops at
Trendlyne's daily-runout page - the cap exists to stop bulk downloaders, so it is never worked
around - and skips every company that already has a snapshot this calendar month. Reports are filed
straight into Stock_Reports/ and the documents catalog. Without a cookie, fetch in the browser and
run scripts/file_stock_reports.py (see its docstring).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                 # noqa: E402
from pennystock.data import stock_reports, trendlyne_scores  # noqa: E402
from pennystock.data.trendlyne_session import UA, load_cookies  # noqa: E402
from pennystock.data.broker_corpus import company_symbols  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cookies", type=Path, help="Netscape cookies.txt from a logged-in browser")
    ap.add_argument("--cookie-header", help='raw header: "sessionid=...; csrftoken=..."')
    ap.add_argument("--symbol", action="append", help="only this company (repeatable)")
    ap.add_argument("--limit", type=int, default=settings.stock_report_daily_opens)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.today()
    symbols = [s.upper() for s in args.symbol] if args.symbol else company_symbols(ROOT / "Broker_Reports")
    need = [s for s in symbols if not stock_reports.has_snapshot_in_month(s, today)]
    print(f"{len(symbols)} companies; {len(need)} without a {today:%B %Y} snapshot")
    if args.dry_run:
        print("  " + " ".join(need))
        return 0
    if not (args.cookies or args.cookie_header):
        print("No cookie given: Trendlyne serves these reports only to a logged-in session.\n"
              "Export cookies.txt (see scripts/fetch_midcap_reports.py --help-cookies) or fetch in the "
              "browser and run scripts/file_stock_reports.py.")
        return 2

    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Referer": "https://trendlyne.com/"})
    load_cookies(session, args.cookies, args.cookie_header)
    tally = stock_reports.fetch(need, session, today, limit=args.limit)
    stock_reports.write_index()
    print("  ".join(f"{k}={v}" for k, v in tally.items()))
    tl = trendlyne_scores.parse_all()
    calls = trendlyne_scores.record_predictions()
    print(f"read {tl['parsed']} new report(s) into Trendlyne snapshots ({tl['failed']} unreadable); "
          f"{calls} new Trendlyne call(s) in the ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
