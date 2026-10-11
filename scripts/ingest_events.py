"""
Event signals: insider trades, promoter pledges, credit-rating actions and severe announcements - typed,
stored with the time the market learned them, and graded in the ledger.

  python scripts/ingest_events.py                  # everything since settings.events_since (resumes)
  python scripts/ingest_events.py --status         # what is stored; which feed months came back empty
  python scripts/ingest_events.py --symbol IRB     # one company's announcements only
  python scripts/ingest_events.py --since 2026-09-01
  python scripts/ingest_events.py --retype         # re-derive stored events under the current rules
  python scripts/ingest_events.py --no-ledger      # store events, record no ledger calls

NSE only - about one request a second, no Trendlyne quota. A full first run takes 1.5-2 hours; after
that only new months and companies not fetched in the last week are asked for.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                     # noqa: E402
from pennystock.data import event_signals as es             # noqa: E402
from pennystock.data.nse_client import NseFeed              # noqa: E402

logger = logging.getLogger("ingest_events")


def print_status() -> None:
    st = es.status()
    print("\nEVENT SIGNALS")
    print(f"  events stored     {st['events']:>7}   ({st['companies']} companies, {st['first']} -> {st['last']})")
    for t, n in sorted(st["by_type"].items(), key=lambda kv: -kv[1]):
        print(f"    {t:<24} {n:>6}   [{es.TYPES[t][0]}]")
    print(f"  feed months done  {st['months_done']}")
    print(f"  announcements     {st['announcement_companies']} companies fetched; last fetch {st['last_fetch']}")
    if st["empty_months"]:
        print(f"  ! months that came back EMPTY (check the feed): {', '.join(st['empty_months'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol", help="fetch one company's announcements only")
    ap.add_argument("--since", default=settings.events_since, help="YYYY-MM-DD (default: settings.events_since)")
    ap.add_argument("--retype", action="store_true")
    ap.add_argument("--no-ledger", action="store_true", help="do not record ledger calls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("pennystock.data.nse_client").setLevel(logging.ERROR)
    if args.status:
        print_status()
        return 0
    if args.retype:
        print(f"re-typed {es.retype()} stored event(s) under rules v{es.RULES_VERSION}")
        return 0

    from pennystock.data.metrics import load_metrics
    nse, since = NseFeed(), date.fromisoformat(args.since)
    # ledger calls only for companies the screen can pick: tiny illiquid names would add noise to the
    # track records the analyst is shown
    universe = {m.symbol for m in load_metrics()}
    if args.symbol:
        got = es.ingest_announcements(args.symbol, nse.api, since.isoformat())
        print(f"{args.symbol.upper()}: {'no answer from NSE' if got is None else f'{got} new event(s)'}")
        universe = {args.symbol.upper()}
    else:
        print(f"ingesting since {since}: market-wide feeds by month, then {len(universe)} companies' announcements")
        print(es.ingest(since, date.today(), nse.api, nse.get, universe))
    if not args.no_ledger:
        added = es.record_predictions(symbols=universe)
        print(f"ledger: {added} new event call(s)" + (" - graded by scripts/track_outcomes.py" if added else ""))
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
