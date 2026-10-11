"""Layer 2: what the brokers collectively think, how much they disagree, and which way they move.

Cases are taken from the real Midcap-100 listing (Sep 2026). The rules they pin come from the
research literature: one vote per broker on its latest call (I/B/E/S), stale calls excluded,
dispersion reported because a dispersed consensus has historically pointed the wrong way (Palley,
Steffen & Zhang 2025), and CHANGES - revisions, drift - because they predict better than levels
(Jegadeesh, Kim, Krische & Lee 2004).
"""
from datetime import date

import pytest

from pennystock.data.broker_metrics import Observation
from pennystock.data.consensus import (Call, bullish_share_by_broker, eps_revisions,
                                       summarize_calls)

AS_OF = date(2026, 10, 2)


def call(broker, day, rating, target=None):
    from pennystock.data.ledger import rating_score
    return Call(broker=broker, report_date=day, rating=rating, score=rating_score(rating),
                target=target)


def summary(calls, as_of=AS_OF, **kw):
    return summarize_calls(calls, as_of=as_of, max_age_days=180, revision_days=90,
                           high_dispersion=0.25, **kw)


# --- who is in the consensus -----------------------------------------------------------------
def test_each_broker_counts_once_with_its_latest_call():
    s = summary([call("Motilal Oswal", "2026-04-21", "Buy", 1300),
                 call("Motilal Oswal", "2026-07-16", "Buy", 1350),
                 call("ICICI Securities", "2026-08-01", "Hold", 1200)])
    assert s["brokers"] == 2
    assert [(c["broker"], c["target"]) for c in s["calls"]] == \
        [("ICICI Securities", 1200), ("Motilal Oswal", 1350)]          # newest first


def test_calls_older_than_the_window_are_excluded_and_counted():
    """ATGL's newest calls are from 2022 and Jan 2025. A two-year-old HOLD read as today's view is
    exactly the stale-target trap; saying 'no current consensus' is the honest answer."""
    s = summary([call("ICICI Securities", "2022-02-06", "Hold", 1775),
                 call("Ventura", "2025-01-04", "Buy", 1106)])
    assert (s["brokers"], s["stale_excluded"]) == (0, 2)
    assert any("No broker call" in f for f in s["flags"])


def test_a_report_type_never_votes():
    s = summary([call("BP Wealth", "2026-08-18", "Results Update"),
                 call("Motilal Oswal", "2026-08-14", "Buy", 198)])
    assert s["brokers"] == 1


# --- what they say ---------------------------------------------------------------------------
def test_rating_mix_and_mean_score_on_the_ibes_scale():
    s = summary([call("A", "2026-09-01", "Buy", 100), call("B", "2026-09-02", "Accumulate", 100),
                 call("C", "2026-09-03", "Sell", 100)])
    assert s["rating_mix"] == {"bullish": 2, "neutral": 0, "bearish": 1}
    assert s["mean_score"] == 2.67                                    # (1 + 2 + 5) / 3


def test_target_median_spread_and_upside():
    s = summary([call("ICICI", "2026-09-04", "Buy", 1250), call("Axis", "2026-07-27", "Buy", 1225),
                 call("IDBI", "2026-08-10", "Buy", 1300)], price=1100.0)
    assert s["target"] == {"median": 1250.0, "low": 1225.0, "high": 1300.0,
                           "dispersion_pct": 6.0, "implied_upside_pct": 13.64}


def test_widely_dispersed_targets_are_flagged():
    """PAGEIND, Aug 2026: ICICI SELL at 31,500 against Motilal BUY at 45,000. A median of those is
    not a forecast; the honest output is the disagreement itself."""
    s = summary([call("ICICI Securities", "2026-08-14", "Sell", 31500),
                 call("Motilal Oswal", "2026-08-13", "Buy", 45000),
                 call("Geojit", "2026-06-26", "Hold", 44951)])
    assert s["target"]["dispersion_pct"] == pytest.approx(30.0, abs=0.1)
    assert any("disagree widely" in f for f in s["flags"])


def test_a_single_broker_is_not_a_consensus():
    s = summary([call("IDBI Capital", "2026-08-14", "Buy", 610)])
    assert any("Only one broker" in f for f in s["flags"])


def test_a_sell_call_is_highlighted():
    """Sell-side ratings skew heavily to Buy, so a Sell is rare - and sells have historically
    carried more information than buys (Womack 1996)."""
    s = summary([call("ICICI Securities", "2026-08-09", "Sell", 2200),
                 call("Motilal Oswal", "2026-08-08", "Neutral", 2500)])
    assert any("ICICI Securities" in f and "Sell" in f for f in s["flags"])


# --- which way they are moving ----------------------------------------------------------------
def test_revisions_compare_each_broker_with_its_own_previous_call():
    """APARINDS: Prabhudas Lilladhar went Hold @13,309 (29 May) -> Accumulate @14,720 (25 Jul)."""
    s = summary([call("Prabhudas Lilladhar", "2026-03-16", "Accumulate", 9629),
                 call("Prabhudas Lilladhar", "2026-05-29", "Hold", 13309),
                 call("Prabhudas Lilladhar", "2026-07-25", "Accumulate", 14720)],
                as_of=date(2026, 8, 30))
    r = s["revisions"]
    assert (r["target_raised"], r["target_cut"], r["upgrades"], r["downgrades"], r["breadth"]) == \
        (1, 0, 1, 0, 1.0)


def test_a_downgrade_is_named():
    """BDL: Motilal Oswal cut from Buy @1,800 to Neutral @1,150 on 28 May 2026."""
    s = summary([call("Motilal Oswal", "2026-01-31", "Buy", 1800),
                 call("Motilal Oswal", "2026-05-28", "Neutral", 1150)], as_of=date(2026, 6, 15))
    assert (s["revisions"]["downgrades"], s["revisions"]["target_cut"]) == (1, 1)
    assert any("Motilal Oswal downgraded" in f for f in s["flags"])


def test_drift_compares_the_consensus_now_with_a_quarter_ago():
    """The quarterly CHANGE in consensus is the robust predictor, not its level."""
    s = summary([call("A", "2026-03-01", "Buy", 100), call("A", "2026-08-01", "Hold", 100),
                 call("B", "2026-04-01", "Buy", 100)], as_of=date(2026, 9, 1))
    assert s["drift"] == {"mean_score_then": 1.0, "mean_score_now": 2.0, "change": 1.0,
                          "direction": "more bearish"}


# --- earnings estimates from the PDFs (Layer 1 data, Layer 2 signal) ---------------------------
def eps(broker, day, period, value):
    return Observation(doc_id=f"{broker}-{day}", symbol="X", broker=broker, report_date=day,
                       metric="eps", period=period, period_kind="estimate",
                       period_end="2027-03-31", value=value, label="eps", page=1)


def test_eps_revisions_compare_consecutive_reports_from_one_broker():
    got = eps_revisions([eps("X", "2026-05-01", "FY27", 50.0), eps("X", "2026-08-01", "FY27", 52.0),
                         eps("Y", "2026-04-01", "FY27", 50.0), eps("Y", "2026-08-05", "FY27", 48.0)],
                        as_of=date(2026, 9, 1), window_days=90)
    assert (got["raised"], got["cut"], got["unchanged"], got["breadth"]) == (1, 1, 0, 0.0)


def test_eps_revisions_need_two_reports_from_the_same_broker():
    assert eps_revisions([eps("X", "2026-08-01", "FY27", 52.0), eps("Y", "2026-08-05", "FY27", 48.0)],
                         as_of=date(2026, 9, 1), window_days=90) is None


# --- context for reading any one broker -------------------------------------------------------
def test_a_brokers_buy_habit_is_reported_once_it_has_enough_calls():
    """A Buy from a broker that says Buy 90% of the time carries less information than one from a
    broker that rarely does (Barber, Lehavy, McNichols & Trueman 2006)."""
    calls = ([call("Habitual", f"2026-0{1 + i % 9}-01", "Buy") for i in range(9)]
             + [call("Habitual", "2026-09-15", "Hold")]
             + [call("Rare", "2026-09-01", "Buy") for _ in range(3)])
    assert bullish_share_by_broker(calls, min_calls=10) == {"habitual": (0.9, 10)}
