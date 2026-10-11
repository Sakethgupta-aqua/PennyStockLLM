"""
Turn every known broker report into a ledger prediction - downloaded or not.

A broker call is four facts: who, when, which way, and a target. The listings carry all four for
every report - including the 296 Midcap-100 notes whose PDFs still sit behind Trendlyne's login -
so a call is recorded from the listing alone. That is what Layer 2 (the consensus panel) reads, and
what the nightly grader scores, so a broker's track record starts accumulating today.

  python scripts/backfill_broker_predictions.py --dry-run   # show what would be written
  python scripts/backfill_broker_predictions.py             # write them
  python scripts/backfill_broker_predictions.py --dir ./more_pdfs

Safe to re-run: a call already in the ledger - same company, same broker, same report date - is
skipped, never rewritten, whichever listing recorded it first. The PDF's own numbers (estimates,
sector KPIs) are not copied here; scripts/ingest_broker_reports.py keeps them in broker_metrics.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

import pandas as pd                                                          # noqa: E402

from pennystock.config import settings                                       # noqa: E402
from pennystock.data.broker_corpus import ReportRecord, broker_key, discover  # noqa: E402
from pennystock.data.ledger import (Prediction, conviction_from_rating,      # noqa: E402
                                    direction_from_rating, load_predictions,
                                    make_prediction_id, rating_score, save_predictions)
from pennystock.data.outcomes import (_sessions, actionable_date,             # noqa: E402
                                      fetch_closes, history_period)
from pennystock.data.sectors import profile_for                              # noqa: E402


def natural_key(p: Prediction) -> tuple[str, str, str]:
    """(symbol, broker, report date): the same call reached through two listings is one call."""
    return (p.symbol.upper(), broker_key(p.payload.get("broker")),
            str(p.payload.get("report_date") or p.as_of_date))


def to_prediction(rec: ReportRecord, closes: pd.Series | None, lag: int,
                  sector: str | None = None, industry: str | None = None
                  ) -> tuple[Prediction | None, str]:
    """One report -> (Prediction, "ok") or (None, why not).

    The reason is returned rather than swallowed: lumping "a results note", "a company that would
    not resolve" and "no date" under one silent skip is how a ledger quietly loses a third of its
    input and nobody notices.
    """
    rating = (rec.rating or "").strip()
    if rating_score(rating) is None:
        return None, "not a call"
    if not rec.symbol:
        return None, "symbol unresolved"
    if not rec.report_date:
        return None, "no report date"

    as_of = actionable_date(rec.report_date, closes, lag)
    # The listing's price at the recommendation is the point-in-time price: recorded on the day,
    # so no later dividend or split adjustment can have rewritten it. Strictly better than the
    # feed, whose history is adjusted backwards - which is why the source is recorded either way.
    entry, entry_asof, source = rec.price, rec.report_date, "listing"
    if entry is None and closes is not None and not closes.empty:
        idx = _sessions(closes)
        upto = closes[idx <= pd.Timestamp(as_of)]
        if not upto.empty:
            entry, entry_asof, source = round(float(upto.iloc[-1]), 2), \
                str(idx[len(upto) - 1].date()), "feed_adjusted"
    if entry is None:
        source = "none"

    payload = {"broker": rec.broker, "company": rec.company, "rating": rating,
               "target_price": rec.target, "report_date": rec.report_date,
               "report_id": rec.report_id, "change_vs_prev": rec.change_vs_prev,
               "headline": rec.headline}
    return Prediction(
        symbol=rec.symbol, source="broker", as_of_date=as_of,
        direction=direction_from_rating(rating), conviction=conviction_from_rating(rating),
        horizon="3m",                      # broker targets are 12-month; 3m is our longest
        entry_price=entry, entry_price_asof=entry_asof if entry is not None else None,
        sector=sector, rulebook=profile_for(sector, industry).name, payload=payload,
        provenance={"report_id": rec.report_id, "file": rec.pdf.name if rec.pdf else None,
                    "backfilled_at": datetime.now().isoformat(timespec="seconds"),
                    "entry_price_source": source},
        prediction_id=make_prediction_id(as_of, rec.symbol, {
            "source": "broker", "broker": rec.broker, "report_id": rec.report_id or "",
            "report_date": rec.report_date})), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Broker_Reports")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lag", type=int, default=settings.broker_action_lag_days,
                    help="trading days after the report date before you could act")
    args = ap.parse_args()

    from pennystock.data.metrics import load_metrics
    records = discover(args.dir)
    if not records:
        print(f"no broker reports found under {Path(args.dir).resolve()}")
        return 1
    seen = {natural_key(p) for p in load_predictions(source="broker", include_superseded=True)}
    sectors = {m.symbol: (m.sector, m.industry) for m in load_metrics()}

    print(f"{len(records)} report(s), action lag +{args.lag} trading day(s)\n")
    hdr = f"  {'SYMBOL':<12}{'BROKER':<24}{'RATING':<12}{'REPORT':<12}{'ACTIONABLE':<12}{'ENTRY':>10}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))

    new: list[Prediction] = []
    why_not: Counter = Counter()
    for rec in records:
        # cheap checks first, so a newsletter or a known call never costs a network fetch
        if rating_score(rec.rating) is None or not rec.symbol or not rec.report_date:
            why_not[to_prediction(rec, None, args.lag)[1]] += 1
            continue
        if (rec.symbol, broker_key(rec.broker), rec.report_date) in seen:
            why_not["already in the ledger"] += 1
            continue
        closes = fetch_closes(rec.symbol, history_period(rec.report_date))
        p, why = to_prediction(rec, closes, args.lag, *sectors.get(rec.symbol, (None, None)))
        if p is None:
            why_not[why] += 1
            continue
        seen.add(natural_key(p))
        new.append(p)
        print(f"  {p.symbol:<12}{str(rec.broker)[:22]:<24}{rec.rating[:10]:<12}{rec.report_date:<12}"
              f"{p.as_of_date:<12}{f'{p.entry_price:,.2f}' if p.entry_price else '-':>10}")

    print(f"\n{len(new)} new call(s)")
    for why, n in why_not.most_common():
        print(f"  skipped {n}: {why}")
    if args.dry_run:
        print("dry run - nothing written")
        return 0
    if new:
        save_predictions(new)
        print(f"wrote {len(new)} prediction(s). Next: python scripts/track_outcomes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
