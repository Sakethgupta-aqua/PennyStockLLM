"""Broker calls into the ledger - Layer 2's raw material.

A call is who, when, which way, and a target. The Midcap-100 listing carries all four for 297
reports, 296 of which have no PDF yet - so a call must be recordable from the listing alone.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backfill_broker_predictions import actionable_date, natural_key, to_prediction  # noqa: E402

from pennystock.data.broker_corpus import ReportRecord                               # noqa: E402
from pennystock.data.ledger import Prediction                                        # noqa: E402


def closes(start="2026-09-01", end="2026-09-30"):
    days = pd.bdate_range(start, end)
    return pd.Series([100.0 + i for i in range(len(days))], index=days)


def record(**kw):
    base = dict(report_id="100339", symbol="360ONE", company="360 One Wam", broker="Motilal Oswal",
                report_date="2026-09-18", rating="Buy", target=1300.0)
    return ReportRecord(**(base | kw))


def test_a_listed_call_without_its_pdf_becomes_a_prediction():
    """Report 18 Sep (a Friday): the first session anyone outside Motilal's client list could act
    on is Monday 21 Sep, and the entry is that day's close."""
    pred, why = to_prediction(record(), closes(), lag=1)
    assert why == "ok"
    assert (pred.symbol, pred.direction, pred.as_of_date, pred.entry_price) == \
        ("360ONE", "bullish", "2026-09-21", 114.0)
    assert (pred.payload["target_price"], pred.payload["report_date"]) == (1300.0, "2026-09-18")
    assert pred.provenance["entry_price_source"] == "feed_adjusted"


def test_the_listed_price_at_the_call_is_the_entry_price():
    pred, _ = to_prediction(record(price=1072.0), closes(), lag=1)
    assert (pred.entry_price, pred.provenance["entry_price_source"]) == (1072.0, "listing")


def test_a_report_type_is_not_a_call():
    """'Results Update' is a kind of note, not an opinion; recording it would invent a call."""
    assert to_prediction(record(rating="Results Update"), closes(), lag=1) == (None, "not a call")


def test_a_report_older_than_the_price_history_keeps_its_own_date():
    """ATGL's Feb-2022 ICICI call with a 2024-onward price series used to be re-dated to the first
    session the feed returned - two years after the call was made."""
    assert actionable_date("2022-02-06", closes("2024-01-01", "2024-03-01"), lag=1) == "2022-02-06"


def test_the_same_call_from_two_listings_has_one_identity():
    existing = Prediction(symbol="AUBANK", source="broker", as_of_date="2026-09-07",
                          direction="bullish",
                          payload={"broker": "ICICI Securities Limited", "report_date": "2026-09-04"})
    new, _ = to_prediction(record(symbol="AUBANK", broker="ICICI Securities Ltd.",
                                  report_date="2026-09-04"), closes(), lag=1)
    assert natural_key(existing) == natural_key(new)
