"""Layer 2: the consensus panel - what the brokers collectively think, how much they disagree,
and which way they are moving.

One report is an opinion; several brokers over time are a dataset. The panel is built the way the
research literature says it has to be, because the naive version - "average target 1,250, 3 Buys"
- is the version that misleads:

  ONE VOTE PER BROKER, ON ITS LATEST CALL. A broker's newer note replaces its own older one, and a
  call older than `broker_max_age_days` is not current (I/B/E/S drops stale estimates; Trendlyne's
  listing still carries ATGL's 2022 HOLD). Otherwise the most prolific broker outvotes the rest.

  DISPERSION IS A FIRST-CLASS OUTPUT. When targets agree, the consensus upside has carried
  information; when they are widely spread it has been unreliable and even contrarian, largely
  because some targets are stale after bad news (Palley, Steffen & Zhang, Management Science 2025).
  Dispersion in forecasts also predicts lower returns (Diether, Malloy & Scherbina 2002).

  CHANGES BEAT LEVELS. The quarterly change in the consensus recommendation predicts returns
  robustly where its level does not (Jegadeesh, Kim, Krische & Lee 2004); sell calls drift further
  than buys (Womack 1996); earnings-estimate revisions predict returns (Chan, Jegadeesh &
  Lakonishok 1996). So revisions are measured per broker against ITS OWN previous call.

  CONTEXT FOR ANY ONE BROKER. A Buy from a broker that rates nearly everything Buy says less than
  one from a broker that rarely does (Barber, Lehavy, McNichols & Trueman 2006).

Ratings use the I/B/E/S scale: 1 = Buy ... 5 = Sell. Broker calls come from the ledger (immutable,
graded nightly); estimates come from broker_metrics.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from pennystock.config import settings
from pennystock.data.broker_corpus import broker_key
from pennystock.data.broker_metrics import Observation


@dataclass(frozen=True)
class Call:
    """One broker's call on one stock."""
    broker: str
    report_date: str           # ISO
    rating: str
    score: int | None          # I/B/E/S 1..5; None = not a rating, never votes
    target: float | None


def _stance(score: int) -> str:
    return "bullish" if score <= 2 else "neutral" if score == 3 else "bearish"


def _latest_per_broker(calls: Iterable[Call], as_of: str) -> dict[str, Call]:
    latest: dict[str, Call] = {}
    for c in calls:
        if c.score is None or c.report_date > as_of:
            continue
        k = broker_key(c.broker)
        if k not in latest or c.report_date > latest[k].report_date:
            latest[k] = c
    return latest


def current_calls(calls: Iterable[Call], as_of: date, max_age_days: int) -> tuple[list[Call], int]:
    """(each broker's latest call inside the window, newest first; how many brokers were stale)."""
    cutoff = (as_of - timedelta(days=max_age_days)).isoformat()
    latest = _latest_per_broker(calls, as_of.isoformat())
    current = sorted((c for c in latest.values() if c.report_date >= cutoff),
                     key=lambda c: c.report_date, reverse=True)
    return current, len(latest) - len(current)


def _previous(calls: list[Call], c: Call) -> Call | None:
    """The same broker's call immediately before this one."""
    earlier = [x for x in calls if x.score is not None and broker_key(x.broker) == broker_key(c.broker)
               and x.report_date < c.report_date]
    return max(earlier, key=lambda x: x.report_date) if earlier else None


def _mean(scores: list[int]) -> float | None:
    return round(sum(scores) / len(scores), 2) if scores else None


def summarize_calls(calls: list[Call], as_of: date, price: float | None = None,
                    max_age_days: int | None = None, revision_days: int | None = None,
                    high_dispersion: float | None = None) -> dict:
    """The panel for one stock, from its broker calls. Pure: no I/O, so every rule is testable."""
    max_age = max_age_days or settings.broker_max_age_days
    window = revision_days or settings.broker_revision_days
    wide = high_dispersion if high_dispersion is not None else settings.broker_high_dispersion
    current, stale = current_calls(calls, as_of, max_age)
    since = (as_of - timedelta(days=window)).isoformat()
    flags: list[str] = []

    entries, raised, cut, same, ups, downs = [], 0, 0, 0, 0, 0
    for c in current:
        prev = _previous(calls, c)
        entries.append({"broker": c.broker, "date": c.report_date, "rating": c.rating,
                        "target": c.target, "age_days": (as_of - date.fromisoformat(c.report_date)).days,
                        "previous": None if prev is None else
                        {"date": prev.report_date, "rating": prev.rating, "target": prev.target}})
        if prev is None or c.report_date < since:
            continue
        if c.target and prev.target:
            if c.target > prev.target * 1.001:
                raised += 1
            elif c.target < prev.target * 0.999:
                cut += 1
            else:
                same += 1
        if c.score < prev.score:
            ups += 1
        elif c.score > prev.score:
            downs += 1
            flags.append(f"{c.broker} downgraded {prev.rating} -> {c.rating} on {c.report_date}"
                         + (f" (target {prev.target:,.0f} -> {c.target:,.0f})"
                            if c.target and prev.target else ""))
    revised = raised + cut + same

    scores = [c.score for c in current]
    targets = sorted(c.target for c in current if c.target)
    target = None
    if targets:
        med = float(statistics.median(targets))
        target = {"median": med, "low": float(targets[0]), "high": float(targets[-1]),
                  "dispersion_pct": round((targets[-1] - targets[0]) / med * 100, 1)
                  if len(targets) > 1 else None,
                  "implied_upside_pct": round((med / price - 1) * 100, 2) if price else None}

    then, _ = current_calls(calls, as_of - timedelta(days=window), max_age)
    drift = None
    if current and then:
        now_m, then_m = _mean(scores), _mean([c.score for c in then])
        change = round(now_m - then_m, 2)
        drift = {"mean_score_then": then_m, "mean_score_now": now_m, "change": change,
                 "direction": "unchanged" if abs(change) < 0.01 else
                 "more bearish" if change > 0 else "more bullish"}

    if not current:
        flags.insert(0, f"No broker call within the last {max_age} days - there is no current "
                        f"consensus ({stale} older call(s) ignored as stale).")
    elif len(current) == 1:
        flags.insert(0, "Only one broker covers this stock - an opinion, not a consensus.")
    if stale and current:
        flags.append(f"{stale} older call(s) ignored as stale (> {max_age} days).")
    if target and target["dispersion_pct"] is not None and target["dispersion_pct"] >= wide * 100:
        flags.append(f"Targets disagree widely (range = {target['dispersion_pct']:.0f}% of the "
                     "median). When targets are this dispersed the consensus upside has "
                     "historically been unreliable, even contrarian - often because some targets "
                     "are stale.")
    for c in current:
        if c.score >= 4:
            flags.append(f"{c.broker} rates it {c.rating}" +
                         (f" (target {c.target:,.0f})" if c.target else "") +
                         ": sell-side ratings skew to Buy, so a sell call is rare - and has "
                         "historically carried more information than a buy.")

    return {"brokers": len(current), "stale_excluded": stale, "calls": entries,
            "rating_mix": {s: sum(1 for x in scores if _stance(x) == s)
                           for s in ("bullish", "neutral", "bearish")},
            "mean_score": _mean(scores), "target": target,
            "revisions": {"window_days": window, "target_raised": raised, "target_cut": cut,
                          "upgrades": ups, "downgrades": downs,
                          "breadth": round((raised - cut) / revised, 2) if revised else None},
            "drift": drift, "flags": flags}


def eps_revisions(observations: Iterable[Observation], as_of: date,
                  window_days: int | None = None) -> dict | None:
    """Earnings-estimate revisions: each broker's newest report inside the window against its own
    previous report, on the nearest fiscal year both forecast. None until some broker has two."""
    since = (as_of - timedelta(days=window_days or settings.broker_revision_days)).isoformat()
    docs: dict[str, dict[str, dict[str, Observation]]] = defaultdict(lambda: defaultdict(dict))
    for o in observations:
        if o.metric == "eps" and o.period_kind == "estimate" and o.report_date <= as_of.isoformat():
            docs[broker_key(o.broker)][o.report_date][o.period] = o
    raised = cut = same = 0
    details = []
    for reports in docs.values():
        dates = sorted(reports)
        if len(dates) < 2 or dates[-1] < since:
            continue
        new, old = reports[dates[-1]], reports[dates[-2]]
        common = sorted(set(new) & set(old), key=lambda p: (new[p].period_end, p))
        if not common:
            continue
        p = common[0]
        a, b = old[p].value, new[p].value
        if b > a * 1.005:
            raised += 1
        elif b < a * 0.995:
            cut += 1
        else:
            same += 1
        details.append(f"{new[p].broker}: {p} EPS {a:g} -> {b:g} ({dates[-2]} -> {dates[-1]})")
    total = raised + cut + same
    if not total:
        return None
    return {"raised": raised, "cut": cut, "unchanged": same,
            "breadth": round((raised - cut) / total, 2), "details": details}


def bullish_share_by_broker(calls: Iterable[Call], min_calls: int = 10) -> dict[str, tuple[float, int]]:
    """{broker: (share of its calls that are bullish, number of calls)} for brokers with enough calls
    to say anything - below `min_calls` a share is anecdote."""
    tally: dict[str, list[int]] = defaultdict(list)
    for c in calls:
        if c.score is not None:
            tally[broker_key(c.broker)].append(c.score)
    return {b: (round(sum(1 for s in sc if s <= 2) / len(sc), 2), len(sc))
            for b, sc in tally.items() if len(sc) >= min_calls}


def calls_from_predictions(preds) -> list[Call]:
    """The ledger's broker predictions as calls. The REPORT date is what a broker's view is dated
    by (the ledger's as_of is the next tradeable session)."""
    from pennystock.data.ledger import rating_score
    out = []
    for p in preds:
        rating = p.payload.get("rating")
        day = str(p.payload.get("report_date") or p.as_of_date)[:10]
        target = p.payload.get("target_price")
        out.append(Call(broker=p.payload.get("broker") or "unknown broker", report_date=day,
                        rating=str(rating or ""), score=rating_score(rating),
                        target=float(target) if target else None))
    return out


def broker_consensus(symbol: str, price: float | None = None, as_of: date | None = None) -> dict:
    """The full panel for one stock: calls from the ledger, EPS revisions from broker_metrics."""
    from pennystock.data.broker_metrics import load_observations
    from pennystock.data.ledger import load_predictions
    as_of = as_of or date.today()
    preds = load_predictions(source="broker")
    out = summarize_calls(calls_from_predictions([p for p in preds if p.symbol == symbol.upper()]),
                          as_of, price)
    shares = bullish_share_by_broker(calls_from_predictions(preds))
    for c in out["calls"]:
        if share := shares.get(broker_key(c["broker"])):
            c["broker_bullish_share"], c["broker_calls_on_record"] = share
    out["eps_revisions"] = eps_revisions(load_observations(symbol, ["eps"]), as_of)
    return out
