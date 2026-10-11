"""
Grade every prediction whose horizon has arrived.

The nightly half of the loop: the ledger remembers what was said, this finds out what
happened. For each prediction old enough to judge, it fetches the stock and the
benchmark, lines them up on the days both actually traded, and records the ABNORMAL
return - the stock minus the market, which is the only number worth grading.

  python scripts/track_outcomes.py --dry-run      # show what would be graded
  python scripts/track_outcomes.py                # grade and save
  python scripts/track_outcomes.py --horizons 1,5 # only the short ones
  python scripts/track_outcomes.py --status       # what is in the ledger right now

Safe to run repeatedly: anything already conclusively graded is skipped, and anything
still `open` is retried. Suitable for cron.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

from pennystock.config import settings                                       # noqa: E402
from pennystock.data.ledger import counts, load_predictions, pending         # noqa: E402
from pennystock.data.ledger import load_outcomes                             # noqa: E402
from pennystock.data.outcomes import scorecard, track                        # noqa: E402


def _graded_at(horizon_days: int):
    return load_outcomes(horizon_days=horizon_days)


def show_status() -> int:
    c = counts()
    print("\nLEDGER\n" + "-" * 62)
    print(f"  predictions stored        {c['predictions']:>6}")
    for src, n in sorted(c["by_source"].items()):
        print(f"      from {src:<18} {n:>6}")
    print(f"  outcomes graded           {c['graded_outcomes']:>6}")
    print(f"  outcomes still open       {c['open_outcomes']:>6}")
    # "0 due" and "0 left to do" are NOT the same thing. A prediction made last week has
    # no +60d outcome because the horizon is still ahead, not because it is finished --
    # and a status line that conflates the two says the work is done when it is not.
    total = len(load_predictions())
    print(f"\n  {'HORIZON':<22}{'DUE NOW':>9}{'GRADED':>8}{'NOT YET DUE':>13}")
    for h in settings.ledger_horizons:
        due = len(pending(h))
        done = len([o for o in _graded_at(h) if o.label != "open"])
        print(f"      +{h:>2} trading days    {due:>9}{done:>8}{max(total - due - done, 0):>13}")
    if not c["predictions"]:
        print("\n  Empty. Run scripts/backfill_broker_predictions.py, or demo.py.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="compute but do not save")
    ap.add_argument("--horizons", help="comma separated trading-day horizons")
    ap.add_argument("--limit", type=int, help="grade at most N per horizon")
    ap.add_argument("--status", action="store_true", help="show ledger state and exit")
    ap.add_argument("--scorecard", action="store_true", help="hit rate after grading")
    ap.add_argument("--today", help="pretend today is this ISO date (for testing)")
    args = ap.parse_args()

    if args.status:
        return show_status()

    horizons = (tuple(int(h) for h in args.horizons.split(","))
                if args.horizons else tuple(settings.ledger_horizons))
    today = date.fromisoformat(args.today) if args.today else date.today()

    graded = track(horizons=horizons, today=today, limit=args.limit, dry_run=args.dry_run)
    if not graded:
        print("\nNothing due for grading yet.")
        return show_status()

    preds = {p.prediction_id: p for p in load_predictions()}
    print(f"\n{'SYMBOL':<13}{'SRC':<8}{'DIR':<9}{'+d':>4}{'STOCK%':>9}{'MKT%':>8}"
          f"{'ABNORM%':>9}  LABEL")
    print("-" * 74)
    for o in sorted(graded, key=lambda x: (x.prediction_id, x.horizon_days)):
        p = preds.get(o.prediction_id)
        f = lambda v: "-" if v is None else f"{v:+.2f}"          # noqa: E731
        print(f"{(p.symbol if p else '?'):<13}{(p.source if p else '?'):<8}"
              f"{(p.direction if p else '?'):<9}{o.horizon_days:>4}"
              f"{f(o.stock_return_pct):>9}{f(o.benchmark_return_pct):>8}"
              f"{f(o.abnormal_return_pct):>9}  {o.label}")

    tally = Counter(o.label for o in graded)
    print(f"\n{len(graded)} outcome(s): {dict(tally)}")
    if args.dry_run:
        print("dry run - nothing saved")

    if args.scorecard:
        for src in ("broker", "agent"):
            s = scorecard(source=src, horizon_days=max(horizons))
            if s["graded"]:
                print(f"\n{src.upper()} @ +{s['horizon_days']}d: hit rate "
                      f"{s['hit_rate']} over {s['decided_sample']} decided call(s)")
                print(f"  {s['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
