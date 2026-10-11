"""The prediction ledger: storage only.

Every behaviour pinned here exists because re-fetching the past gives a different
answer. Yahoo applies dividends backwards, so ITC's close two years ago reads 426.19
on the day and 379.91 today - a 10.9% gap. A ledger that recomputes its own entry
prices reports gains that never happened, so these tests guard the one property that
makes the whole thing trustworthy: a prediction, once written, never changes.
"""
import sqlite3
from datetime import date

import pytest

from pennystock.data import ledger as lg
from pennystock.data.ledger import (Outcome, Prediction, load_outcomes, load_predictions,
                                    make_prediction_id, pending, save_outcomes,
                                    save_predictions)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Each test gets its own market.db so the ledger tests never touch the real one."""
    monkeypatch.setattr(lg.settings, "market_db", tmp_path / "test.db")
    yield


def pred(symbol="HINDALCO", **kw) -> Prediction:
    base = dict(source="agent", as_of_date="2026-09-27", direction="bullish",
                entry_price=640.5, conviction=4, horizon="1m")
    return Prediction(symbol=symbol, **(base | kw))


# --- the id ---------------------------------------------------------------------------
def test_id_starts_with_the_date_so_sorting_by_id_sorts_by_time():
    ids = [make_prediction_id("2026-01-15", "A", {}), make_prediction_id("2026-09-27", "B", {})]
    assert sorted(ids) == ids


def test_id_carries_the_symbol_so_a_human_can_read_it():
    assert "HINDALCO" in make_prediction_id("2026-09-27", "HINDALCO", {})


def test_two_calls_on_the_same_stock_the_same_day_get_different_ids():
    """A broker and the agent can both call HINDALCO on one day; so can two brokers."""
    a = make_prediction_id("2026-09-27", "HINDALCO", {"source": "agent"})
    b = make_prediction_id("2026-09-27", "HINDALCO", {"source": "broker", "broker": "ICICI"})
    assert a != b


# --- writing and reading --------------------------------------------------------------
def test_a_saved_prediction_comes_back_unchanged():
    save_predictions([pred()])
    got = load_predictions()[0]
    assert (got.symbol, got.direction, got.entry_price) == ("HINDALCO", "bullish", 640.5)
    assert got.prediction_id and got.created_at


def test_the_payload_survives_the_round_trip():
    """Source-specific detail lives in JSON; it has to come back as a dict, not a string."""
    p = pred(source="broker", payload={"broker": "ICICI Securities", "rating": "HOLD",
                                       "target_price": 4000.0, "prev_target_price": 2670.0,
                                       "estimates": {"FY28E": {"eps": 329.1}}})
    save_predictions([p])
    got = load_predictions()[0].payload
    assert got["rating"] == "HOLD"
    assert got["estimates"]["FY28E"]["eps"] == 329.1


def test_provenance_records_where_the_call_came_from():
    save_predictions([pred(provenance={"trace_id": "20260927T101500-ab12cd",
                                       "analyst_model": "qwen3.5:35b"})])
    assert load_predictions()[0].provenance["analyst_model"] == "qwen3.5:35b"


def test_predictions_can_be_filtered_by_source():
    save_predictions([pred("A", source="agent"), pred("B", source="broker"),
                      pred("C", source="broker")])
    assert len(load_predictions(source="broker")) == 2
    assert [p.symbol for p in load_predictions(source="agent")] == ["A"]


def test_saving_nothing_is_a_no_op():
    assert save_predictions([]) == 0


# --- immutability: the property the whole ledger rests on -----------------------------
def test_the_same_prediction_cannot_be_written_twice():
    """Silently overwriting what you predicted would make every score meaningless."""
    p = pred()
    save_predictions([p])
    with pytest.raises(sqlite3.IntegrityError):
        save_predictions([p])


def test_a_later_price_feed_cannot_rewrite_a_stored_entry_price():
    """The ITC case: re-fetching 2023 gives 379.91 where the screen said 426.19. If that
    could overwrite the stored entry price, the ledger would report a gain nobody made."""
    p = pred(symbol="ITC", entry_price=426.19)
    save_predictions([p])
    with pytest.raises(sqlite3.IntegrityError):
        save_predictions([pred(symbol="ITC", entry_price=379.91,
                               prediction_id=p.prediction_id)])
    assert load_predictions()[0].entry_price == 426.19


# --- outcomes: derived, and deliberately NOT immutable --------------------------------
def out(pid, horizon=20, **kw) -> Outcome:
    base = dict(outcome_date="2026-10-28", stock_return_pct=-1.71,
                benchmark_symbol="^CRSLDX", benchmark_return_pct=5.99,
                abnormal_return_pct=-7.71, label="miss")
    return Outcome(prediction_id=pid, horizon_days=horizon, **(base | kw))


def test_an_outcome_round_trips():
    save_predictions([pred()])
    pid = load_predictions()[0].prediction_id
    save_outcomes([out(pid)])
    got = load_outcomes(pid)[0]
    assert (got.abnormal_return_pct, got.label) == (-7.71, "miss")


def test_outcomes_CAN_be_recomputed_unlike_predictions():
    """If the abnormal-return method improves, outcomes are rebuilt while the
    predictions they grade stay untouched. That is why they are separate tables."""
    save_predictions([pred()])
    pid = load_predictions()[0].prediction_id
    save_outcomes([out(pid, label="open", abnormal_return_pct=None)])
    save_outcomes([out(pid, label="miss", abnormal_return_pct=-7.71)])
    rows = load_outcomes(pid)
    assert len(rows) == 1                      # replaced, not duplicated
    assert rows[0].label == "miss"


def test_one_prediction_holds_several_horizons_side_by_side():
    save_predictions([pred()])
    pid = load_predictions()[0].prediction_id
    save_outcomes([out(pid, horizon=h) for h in (1, 5, 20, 60)])
    assert sorted(o.horizon_days for o in load_outcomes(pid)) == [1, 5, 20, 60]


# --- finding what still needs grading -------------------------------------------------
def test_pending_returns_calls_old_enough_to_grade():
    """20 trading days is about 34 calendar days, so a call from 60 days ago is due."""
    save_predictions([pred("OLD", as_of_date="2026-07-01"),
                      pred("NEW", as_of_date="2026-09-26")])
    due = pending(horizon_days=20, today=date(2026, 9, 27))
    assert [p.symbol for p in due] == ["OLD"]


def test_pending_skips_what_is_already_graded():
    save_predictions([pred("OLD", as_of_date="2026-07-01")])
    pid = load_predictions()[0].prediction_id
    assert len(pending(20, date(2026, 9, 27))) == 1
    save_outcomes([out(pid, horizon=20)])
    assert pending(20, date(2026, 9, 27)) == []


def test_a_still_open_outcome_is_returned_again_for_regrading():
    """'open' means the horizon had not arrived yet - it must be retried, not skipped."""
    save_predictions([pred("OLD", as_of_date="2026-07-01")])
    pid = load_predictions()[0].prediction_id
    save_outcomes([out(pid, horizon=20, label="open", abnormal_return_pct=None)])
    assert [p.symbol for p in pending(20, date(2026, 9, 27))] == ["OLD"]


def test_pending_on_an_empty_ledger_is_empty_not_an_error():
    assert pending(20, date(2026, 9, 27)) == []


# --- normalising what different sources call the same thing ----------------------------
def test_every_bullish_word_the_brokers_use_becomes_bullish():
    """Your 12 downloaded reports alone use BUY, ADD, ACCUMULATE and IPO SUBSCRIBE.
    Grading needs one word, not twenty."""
    for r in ("BUY", "Buy", "ADD", "ACCUMULATE", "OUTPERFORM", "OVERWEIGHT",
              "STRONG BUY", "IPO SUBSCRIBE", "SUBSCRIBE"):
        assert lg.direction_from_rating(r) == "bullish", r


def test_every_bearish_word_becomes_bearish():
    for r in ("SELL", "REDUCE", "UNDERPERFORM", "UNDERWEIGHT", "STRONG SELL"):
        assert lg.direction_from_rating(r) == "bearish", r


def test_hold_and_neutral_are_neutral_not_bullish():
    """The commonest mistake in reading sell-side research: a HOLD is not a soft buy."""
    for r in ("HOLD", "NEUTRAL", "Neutral", "MARKET PERFORM", "EQUAL WEIGHT"):
        assert lg.direction_from_rating(r) == "neutral", r


def test_an_unrecognised_or_missing_rating_is_neutral_never_a_guess():
    for r in ("DAILY NOTE", "UNDER REVIEW", "NOT RATED", "", None):
        assert lg.direction_from_rating(r) == "neutral"


def test_conviction_tracks_how_strongly_the_rating_is_worded():
    assert lg.conviction_from_rating("STRONG BUY") == 5
    assert lg.conviction_from_rating("BUY") == 4
    assert lg.conviction_from_rating("ADD") == 3
    assert lg.conviction_from_rating("HOLD") == 2
    assert lg.conviction_from_rating("SELL") == 4        # a firm call, just the other way
    assert lg.conviction_from_rating("REDUCE") == 3


def test_an_unknown_rating_gets_low_conviction():
    assert lg.conviction_from_rating("DAILY NOTE") == 1
    assert lg.conviction_from_rating(None) == 1


def test_the_agents_own_actions_map_too():
    """The analyst says buy/avoid/watch; a broker says BUY/SELL/HOLD. One vocabulary."""
    assert lg.direction_from_rating("buy") == "bullish"
    assert lg.direction_from_rating("avoid") == "bearish"
    assert lg.direction_from_rating("watch") == "neutral"


# --- non-destructive correction -------------------------------------------------------
# The third invariant of an immutable ledger. Extraction is fallible: the model can read
# "we would avoid this" as a buy. You must be able to fix that WITHOUT deleting the
# original, because "we recorded the wrong call on Tuesday" is itself a fact about the
# system, and a ledger you can quietly edit proves nothing.
def test_a_wrong_call_is_superseded_not_deleted():
    save_predictions([pred(symbol="HINDALCO", direction="bullish")])
    old = load_predictions()[0]
    lg.supersede(old.prediction_id,
                 pred(symbol="HINDALCO", direction="bearish"),
                 reason="extraction read 'we would avoid' as a buy")
    rows = load_predictions(include_superseded=True)
    assert len(rows) == 2                                   # both survive
    assert {r.direction for r in rows} == {"bullish", "bearish"}


def test_the_superseded_row_records_what_replaced_it_and_why():
    save_predictions([pred()])
    old_id = load_predictions()[0].prediction_id
    new_id = lg.supersede(old_id, pred(direction="bearish"), reason="mis-read")
    old = [r for r in load_predictions(include_superseded=True)
           if r.prediction_id == old_id][0]
    assert old.superseded_by == new_id
    assert "mis-read" in old.provenance["superseded_reason"]


def test_superseded_rows_are_hidden_by_default():
    """A scorecard must not count a call twice, so the default read shows live rows only."""
    save_predictions([pred()])
    lg.supersede(load_predictions()[0].prediction_id, pred(direction="bearish"), reason="x")
    live = load_predictions()
    assert len(live) == 1 and live[0].direction == "bearish"


def test_a_superseded_call_is_never_graded_again():
    save_predictions([pred(as_of_date="2026-07-01")])
    old_id = load_predictions()[0].prediction_id
    lg.supersede(old_id, pred(symbol="HINDALCO", as_of_date="2026-07-01",
                              direction="bearish"), reason="x")
    due = pending(20, date(2026, 9, 27))
    assert old_id not in {p.prediction_id for p in due}
    assert len(due) == 1


def test_superseding_something_that_does_not_exist_raises():
    with pytest.raises(KeyError):
        lg.supersede("no-such-id", pred(), reason="x")


# --- schema evolution -------------------------------------------------------------------
def test_every_row_records_the_schema_it_was_written_under():
    """Events are immutable, so a later schema change cannot rewrite old rows. Stamping the
    version is what lets a reader upcast them instead of misreading them."""
    save_predictions([pred()])
    assert load_predictions()[0].schema_version == lg.SCHEMA_VERSION


# --- schema migration -------------------------------------------------------------------
def test_a_ledger_written_by_an_older_version_still_opens(tmp_path, monkeypatch):
    """CREATE TABLE IF NOT EXISTS does NOT add columns to a table that already exists, so
    a new column silently breaks every database written before it. Caught in practice: the
    real market.db raised `no such column: schema_version` the moment two columns were
    added. Additive migration runs on connect so old ledgers keep working.
    """
    db = tmp_path / "old.db"
    monkeypatch.setattr(lg.settings, "market_db", db)
    con = sqlite3.connect(str(db))                       # a v0 table: no new columns
    con.execute("""CREATE TABLE predictions (
        prediction_id TEXT PRIMARY KEY, source TEXT NOT NULL, symbol TEXT NOT NULL,
        exchange TEXT, as_of_date TEXT NOT NULL, as_of_ts TEXT, direction TEXT NOT NULL,
        conviction INTEGER, horizon TEXT, entry_price REAL, entry_price_asof TEXT,
        invalidation_price REAL, sector TEXT, rulebook TEXT,
        payload TEXT DEFAULT '{}', provenance TEXT DEFAULT '{}', created_at TEXT)""")
    con.execute("INSERT INTO predictions (prediction_id, source, symbol, as_of_date, "
                "direction, created_at) VALUES ('old-1','agent','ITC','2026-01-01',"
                "'bullish','2026-01-01T00:00:00')")
    con.commit()
    con.close()

    rows = load_predictions()                            # must not raise
    assert len(rows) == 1
    assert rows[0].symbol == "ITC"
    assert rows[0].schema_version == 0                   # honestly reported as pre-versioning
    assert rows[0].superseded_by is None


def test_migration_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(lg.settings, "market_db", tmp_path / "x.db")
    save_predictions([pred()])
    for _ in range(3):                                   # connecting repeatedly must be safe
        assert len(load_predictions()) == 1


# --- the rating scale a consensus is built on ----------------------------------------------
@pytest.mark.parametrize("rating,score", [
    ("BUY", 1), ("Strong Buy", 1), ("Accumulate", 2), ("ADD", 2), ("Outperform", 2),
    ("Hold", 3), ("Neutral", 3), ("Reduce", 4), ("Underperform", 4), ("SELL", 5),
])
def test_ratings_map_onto_the_ibes_five_point_scale(rating, score):
    """I/B/E/S maps every broker's scale onto 1 (strongest buy) .. 5 (strongest sell). Indian brokers
    rate Buy / Accumulate / Hold / Reduce / Sell, which lands one-to-one."""
    assert lg.rating_score(rating) == score


@pytest.mark.parametrize("not_a_rating", ["Results Update", "Daily Note", "Strategy Note",
                                          "IPO Note", "Not Rated", "", None])
def test_a_report_type_is_not_a_rating(not_a_rating):
    """The ledger reads an unknown label as 'neutral' - right for a single call, wrong for a
    consensus, where a 'Results Update' would be counted as a broker saying HOLD."""
    assert lg.rating_score(not_a_rating) is None
