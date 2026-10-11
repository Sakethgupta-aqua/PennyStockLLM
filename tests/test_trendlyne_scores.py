"""Trendlyne Stock Reports read into snapshots: parsing, bands, storage, the ledger, agreement.

All offline. The page texts below are shaped like the cleaned page 1 of real reports (AU Small
Finance Bank; Fortis with Durability "Not Eligible") - no Trendlyne PDF is ever committed.
"""
from __future__ import annotations

from pennystock.data import trendlyne_scores as ts

AU_P1 = """AU SMALL FINANCE BANK LTD.
Trendlyne Stock Report
Microfinance Institutions
Oct 01, 2026 03:31 PM
Banking and Finance
₹987 987
(-2.3% ) 52 Week High Low Market Cap: 741.6B Avg Daily Volume: 2.7M
720.7 1144.1
NSE | Oct 01, 2026 03:31 PM
DVM SUMMARY
Mid-range Performer
D V M
These stocks have average quality, financial and technical momentum, indicating a mid range performer
Durability Score (D) Medium Financial Strength Valuation Score (V) Mid Valuation Momentum Score (M) Technically Neutral
Bad Medium Good Bad Medium Good Bad Medium Good
45 38 47
0 35 55 100 0 30 50 100 0 35 60 100
ANALYST RECOMMENDATION TRENDLYNE CHECKLIST
Consensus Recommendation Target Price 46.2%
BUY 1121 (13.59%) Passed in checklist
13 4 4 4 3 Financials 3 5
Total : 28 Ownership 3 1
Strong Buy Buy Hold Sell Strong Sell
Peer Comparison 2 1
Current Price
987
Value & Momentum 4 4
Total 12 14
800 1121.1 1285
Low Est. Avg. Est. High Est.
The consensus recommendation is based on 28 analyst recommendations."""
# FORTIS: Durability "Not Eligible", three buckets printed, three-count checklist lines
FORTIS_P1 = AU_P1.replace("Durability Score (D) Medium Financial Strength",
                          "Durability Score (D) Not Eligible") \
    .replace("45 38 47", "31 37").replace("13 4 4 4 3 Financials 3 5", "16 2 2 Financials 5 2 1") \
    .replace("Total 12 14", "Total 13 9 1").replace("46.2%", "59.1%") \
    .replace("BUY 1121 (13.59%)", "STRONG BUY 1114 (41.92%)").replace("based on 28", "based on 20")
ZONE_PAGE = ("below the current P/E 76.1. This puts it in the PE Strong Buy\nZone\n"
             "Strong Buy Zone 3.1% into P/E buy sell zone This is based on")
SENTENCES = ["X Ltd. has a durability score of 45, which indicates Medium Financial Strength.",
             "X Ltd. has a valuation score of 38, which indicates an Mid Valuation.",
             "X Ltd. has a Momentum score of 47, which indicates that it is Technically Neutral."]


# --- Task 1: reading a report ------------------------------------------------------------------

def test_page_one_yields_the_full_snapshot():
    s = ts.parse_snapshot([AU_P1, "", "", ZONE_PAGE])
    assert (s.as_of, s.price) == ("2026-10-01T15:31", 987.0)
    assert (s.durability, s.valuation, s.momentum) == (45, 38, 47)
    assert (s.durability_label, s.valuation_label, s.momentum_label) == (
        "Medium Financial Strength", "Mid Valuation", "Technically Neutral")
    assert s.classification == "Mid-range Performer"
    assert (s.checklist_pass_pct, s.checklist_passed, s.checklist_failed) == (46.2, 12, 14)
    assert s.checklist_groups == {"Financials": [3, 5], "Ownership": [3, 1], "Peer Comparison": [2, 1],
                                  "Value & Momentum": [4, 4]}
    assert (s.consensus, s.target_price, s.upside_pct, s.analysts) == ("BUY", 1121.0, 13.59, 28)
    assert (s.est_low, s.est_avg, s.est_high) == (800.0, 1121.1, 1285.0)
    assert (s.n_strong_buy, s.n_buy, s.n_hold, s.n_sell, s.n_strong_sell) == (13, 4, 4, 4, 3)
    assert (s.pe_zone, s.pe_zone_pct) == ("Strong Buy", 3.1)


def test_not_eligible_durability_and_omitted_buckets_are_left_blank():
    s = ts.parse_snapshot([FORTIS_P1])
    assert (s.durability, s.valuation, s.momentum) == (None, 31, 37)
    assert s.durability_label == "Not Eligible"
    assert (s.consensus, s.analysts) == ("STRONG BUY", 20)
    assert s.n_strong_buy is None                       # "16 2 2" - which buckets? unknown
    assert (s.checklist_passed, s.checklist_failed) == (13, 9)
    assert s.checklist_groups["Financials"] == [5, 2]


def test_score_sentences_fill_scores_the_first_page_lost():
    broken = AU_P1.replace("45 38 47\n", "")
    s = ts.parse_snapshot([broken, SENTENCES[0], "", SENTENCES[1], SENTENCES[2]])
    assert (s.durability, s.valuation, s.momentum) == (45, 38, 47)


def test_a_report_without_analyst_coverage_still_yields_its_scores():
    bare = AU_P1.split("ANALYST RECOMMENDATION")[0]
    s = ts.parse_snapshot([bare])
    assert (s.durability, s.consensus, s.analysts, s.target_price) == (45, None, None, None)


def test_a_missing_classification_line_is_blank_not_the_axis_labels():
    s = ts.parse_snapshot([AU_P1.replace("Mid-range Performer\n", "")])
    assert s.classification is None


def test_with_a_not_counted_column_two_printed_counts_are_left_out():
    # Fortis's Total line has a third (not counted) count, so its "Ownership 3 1" could be
    # 3 passed / 1 not counted; MEESHO's real bars show exactly that. Only three counts are certain.
    s = ts.parse_snapshot([FORTIS_P1])
    assert s.checklist_groups == {"Financials": [5, 2]}


def test_a_group_printed_with_one_count_is_left_out():
    # the other segment was zero and is not drawn: "Ownership 4" is 4 passed OR 4 failed
    s = ts.parse_snapshot([AU_P1.replace("Ownership 3 1", "Ownership 4")])
    assert "Ownership" not in s.checklist_groups and s.checklist_groups["Financials"] == [3, 5]


def test_a_long_momentum_label_that_pushes_its_header_up_is_still_read():
    # APARINDS: "Technically Moderately Bullish" is long, so "Momentum Score" lands on the line above
    line = ("Durability Score (D) High Financial Strength Valuation Score (V) Mid Valuation "
            "(M) Technically Moderately Bullish")
    s = ts.parse_snapshot([AU_P1.replace(AU_P1.split("\n")[13], line)])
    assert (s.durability_label, s.valuation_label, s.momentum_label) == (
        "High Financial Strength", "Mid Valuation", "Technically Moderately Bullish")


def test_one_analyst_gives_a_target_but_no_consensus_word():
    # RVNL prints "- 165 (-16.24%)": no consensus recommendation, still a target and its upside
    s = ts.parse_snapshot([AU_P1.replace("BUY 1121 (13.59%)", "- 165 (-16.24%)")])
    assert (s.consensus, s.target_price, s.upside_pct) == (None, 165.0, -16.24)


def test_one_analyst_prints_only_the_average_estimate():
    s = ts.parse_snapshot([AU_P1.replace("800 1121.1 1285\nLow Est. Avg. Est. High Est.", "390\nAvg.\nEst.")])
    assert (s.est_low, s.est_avg, s.est_high) == (None, 390.0, None)


GREEN, RED = (0.0, 0.6353, 0.3569), (0.9882, 0.3529, 0.3529)
AMBER, BLACK = (1.0, 0.7725, 0.2588), (0.0, 0.0, 0.0)


def seg(x0, x1, top, colour):
    return {"x0": x0, "x1": x1, "top": top, "bottom": top + 8.0, "non_stroking_color": colour}


def word(text, x0, top):
    return {"text": text, "x0": x0, "x1": x0 + 2.4 * len(text), "top": top, "bottom": top + 5.6}


def test_the_bar_colours_say_which_bucket_a_printed_count_belongs_to():
    # MEESHO's Ownership row prints "3 1": a green 3 then an AMBER 1 - not counted, not failed.
    # Each segment is also stroked in black, which is no bucket.
    words = [word("Ownership", 336.5, 363.2), word("3", 448.8, 363.3), word("1", 465.4, 363.3),
             word("Peer", 321.2, 380.4), word("Comparison", 333.4, 380.4), word("2", 465.6, 380.5),
             word("1", 476.8, 380.5)]
    rects = [seg(437.4, 462.6, 361.0, GREEN), seg(437.4, 462.6, 361.0, BLACK),
             seg(462.6, 470.6, 361.0, AMBER), seg(462.6, 470.6, 361.0, BLACK),
             seg(459.4, 474.2, 378.2, GREEN), seg(474.2, 481.4, 378.2, RED)]
    assert ts.checklist_bars(words, rects) == {"Ownership": [3, 0], "Peer Comparison": [2, 1]}


def test_a_bar_segment_without_its_count_leaves_the_group_out():
    words = [word("Ownership", 336.5, 363.2), word("3", 448.8, 363.3)]
    rects = [seg(437.4, 462.6, 361.0, GREEN), seg(462.6, 470.6, 361.0, RED)]
    assert ts.checklist_bars(words, rects) == {}


def test_the_analyst_count_is_read_when_the_pdf_drops_the_space():
    s = ts.parse_snapshot([AU_P1.replace("based on 28 analyst", "based on 28analyst")])
    assert s.analysts == 28


def g(text, x0, top, font="F-Bold"):
    return {"text": text, "x0": x0, "x1": x0 + 3.6, "top": top, "fontname": font}


def test_drop_shadow_copies_are_removed_and_real_double_letters_kept():
    chars = [g("4", 108.45, 251.67), g("5", 112.04, 251.67), g("4", 108.45, 250.11), g("5", 112.04, 250.11),
             g("D", 115.45, 169.24), g("D", 115.45, 168.20),
             g("l", 40.0, 300.0, "F-Reg"), g("l", 41.8, 300.0, "F-Reg")]       # "Sell": adjacent, kept
    drop = ts.overprint_duplicates(chars)
    # the shadow is the copy drawn LOWER on the page (larger top); the upper copy is the glyph
    assert drop == {("4", 108.45, 251.67), ("5", 112.04, 251.67), ("D", 115.45, 169.24)}


# --- Task 2: bands, lean, storage -----------------------------------------------------------------
import pytest  # noqa: E402

from pennystock.data import documents  # noqa: E402


def snap(d=None, v=None, m=None, **kw):
    return ts.Snapshot(durability=d, valuation=v, momentum=m, **kw)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(ts.settings, "market_db", tmp_path / "test.db")
    monkeypatch.setattr(ts.settings, "project_root", tmp_path)
    return tmp_path


@pytest.mark.parametrize("kind, score, want", [("durability", 55, "medium"), ("durability", 56, "good"),
    ("durability", 34, "bad"), ("valuation", 51, "good"), ("valuation", 29, "bad"),
    ("momentum", 59, "medium"), ("momentum", 60, "good"), ("momentum", 34, "bad"), ("momentum", None, None)])
def test_the_published_bands(kind, score, want):
    assert ts.band(kind, score) == want


@pytest.mark.parametrize("d, v, m, want", [(65, 31, 67, ("bullish", 3)), (70, 60, 65, ("bullish", 4)),
    (45, 38, 47, ("neutral", 2)), (75, 28, 68, ("neutral", 2)), (20, 25, 50, ("bearish", 3)),
    (20, 25, 30, ("bearish", 4)), (None, 60, 70, ("bullish", 3)), (None, 60, 40, ("neutral", 2))])
def test_the_lean_follows_the_bands(d, v, m, want):
    assert ts.lean(snap(d, v, m)) == want


def test_at_a_rounded_boundary_trendlynes_own_label_decides_the_band():
    # printed 30 is "Expensive" on some reports and "Mid" on others: the label was set on the
    # unrounded score, so it - not the printed integer - says which side of the mark it fell
    assert ts.lean(snap(70, 30, 65)) == ("bullish", 3)
    assert ts.lean(snap(70, 30, 65, valuation_label="Expensive Valuation")) == ("neutral", 2)
    assert ts.lean(snap(70, 30, 65, valuation_label="Some New Wording")) == ("bullish", 3)


def test_a_reparse_replaces_the_snapshot_and_history_keeps_months_apart(db):
    ts.save_snapshot("k1", "AUBANK", snap(45, 38, 47, as_of="2026-10-01T15:31"))
    ts.save_snapshot("k1", "AUBANK", snap(46, 38, 47, as_of="2026-10-01T15:31"))
    ts.save_snapshot("k2", "AUBANK", snap(50, 30, 30, as_of="2026-11-02T15:31"))
    hist = ts.history("AUBANK")
    assert [(k, s.durability) for k, s in hist] == [("k1", 46), ("k2", 50)]
    assert ts.latest("AUBANK")[0] == "k2"
    assert ts.changes(hist[0][1], hist[1][1])["durability"] == 4


def test_a_snapshot_survives_the_round_trip_whole(db):
    s = ts.parse_snapshot([AU_P1, "", "", ZONE_PAGE])
    ts.save_snapshot("k1", "aubank", s)
    assert ts.latest("AUBANK") == ("k1", s)


def _report_on_file(root, key="stock_report:trendlyne:1:20261003-2318"):
    path = root / "Stock_Reports" / "AUBANK" / "AUBANK-StockReport-20261003-2318.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF-1.7\n" + b"x" * 100)
    documents.register(doc_key=key, symbol="AUBANK", kind="stock_report", doc_date="2026-10-03",
                       source="trendlyne", path=path)
    return path


def test_each_report_is_read_once_until_the_parser_changes(db, monkeypatch):
    _report_on_file(db)
    reads = []
    monkeypatch.setattr(ts, "parse_report",
                        lambda p: reads.append(p) or snap(45, 38, 47, as_of="2026-10-01T15:31"))
    assert ts.parse_all()["parsed"] == 1
    assert ts.parse_all() == {"parsed": 0, "current": 1, "failed": 0, "missing": 0}
    monkeypatch.setattr(ts, "PARSER_VERSION", ts.PARSER_VERSION + 1)
    assert ts.parse_all()["parsed"] == 1 and len(reads) == 2


def test_a_report_that_cannot_be_read_is_counted_not_fatal(db, monkeypatch):
    _report_on_file(db)

    def broken(path):
        raise ValueError("not a Trendlyne layout")
    monkeypatch.setattr(ts, "parse_report", broken)
    assert ts.parse_all()["failed"] == 1 and ts.latest("AUBANK") is None


# --- Task 3: Trendlyne's view in the ledger -------------------------------------------------------
import pandas as pd  # noqa: E402

from pennystock.data.ledger import load_predictions  # noqa: E402


def test_a_snapshot_becomes_a_graded_prediction_dated_to_the_next_session():
    closes = pd.Series([100.0, 101.0, 102.0],
                       index=pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-03"]))
    p = ts.to_prediction("APARINDS", "k9", snap(65, 31, 67, as_of="2026-10-01T15:31", price=987.0), closes)
    assert (p.source, p.direction, p.conviction, p.as_of_date) == ("trendlyne", "bullish", 3, "2026-10-03")
    assert (p.entry_price, p.entry_price_asof) == (987.0, "2026-10-01")
    assert p.payload["durability"] == 65 and p.provenance["doc_key"] == "k9"


def test_a_fresh_snapshot_is_never_dated_on_its_own_day():
    # made after the last close in the feed: the next session has not happened yet
    closes = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-09-30", "2026-10-01"]))
    p = ts.to_prediction("AUBANK", "k1", snap(45, 38, 47, as_of="2026-10-01T15:31", price=987.0), closes)
    assert p.as_of_date == "2026-10-02"


def test_a_snapshot_without_a_view_is_not_a_prediction():
    # one readable score cannot make "two agree"; no date cannot be graded - neither is "neutral"
    assert ts.to_prediction("X", "k1", snap(None, None, 47, as_of="2026-10-01T15:31")) is None
    assert ts.to_prediction("X", "k1", snap(45, 38, 47)) is None


def test_recording_twice_adds_nothing_the_second_time(db, monkeypatch):
    ts.save_snapshot("k1", "AUBANK", snap(45, 38, 47, as_of="2026-10-01T15:31", price=987.0))
    monkeypatch.setattr(ts, "fetch_closes", lambda *a, **k: None)
    assert ts.record_predictions() == 1
    assert ts.record_predictions() == 0
    [p] = load_predictions(source="trendlyne")
    assert (p.symbol, p.payload["as_of"], p.direction) == ("AUBANK", "2026-10-01T15:31", "neutral")


# --- Task 4: beside our own score -----------------------------------------------------------------
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_strong_disagreement_is_flagged():
    rows = ts.agreement({"quality": 0.8, "valuation": 0.2, "momentum": 0.5}, snap(20, 70, 47))
    assert [(r["ours"], r["theirs"], r["verdict"]) for r in rows] == [
        ("quality", "durability", "DISAGREE"), ("valuation", "valuation", "DISAGREE"),
        ("momentum", "momentum", "agree")]


def test_near_misses_and_gaps_are_not_called_agreement():
    rows = ts.agreement({"quality": 0.9, "valuation": None, "momentum": 0.7}, snap(70, 38, 47))
    assert [r["verdict"] for r in rows] == ["agree", "n/a", "one step apart"]


def test_the_view_carries_bands_lean_and_what_changed(db):
    ts.save_snapshot("k1", "AUBANK", snap(45, 38, 47, as_of="2026-09-01T15:31"))
    ts.save_snapshot("k2", "AUBANK", snap(60, 38, 62, as_of="2026-10-01T15:31"))
    v = ts.view("AUBANK")
    assert v["bands"] == {"durability": "good", "valuation": "medium", "momentum": "good"}
    assert (v["lean"], v["lean_conviction"], v["snapshots_on_file"]) == ("bullish", 3, 2)
    assert v["changes_since_previous"]["durability"] == 15
    assert ts.view("NOPE") is None


def test_the_panel_says_when_no_report_is_on_file(db, capsys):
    from screen import print_trendlyne_panel
    print_trendlyne_panel("NOPE", {})
    out = capsys.readouterr().out
    assert "no Trendlyne Stock Report on file" in out and "neutral" not in out.lower()


def test_the_panel_attributes_the_scores_and_flags_disagreement(db, capsys):
    from screen import print_trendlyne_panel
    ts.save_snapshot("k1", "AUBANK",
                     snap(20, 70, 47, as_of="2026-10-01T15:31", classification="Falling Comet"))
    print_trendlyne_panel("AUBANK", {"quality": 0.8, "valuation": 0.2, "momentum": 0.5})
    out = capsys.readouterr().out
    assert "TRENDLYNE'S VIEW" in out and "not this analysis" in out
    assert "Falling Comet" in out and "DISAGREE" in out
