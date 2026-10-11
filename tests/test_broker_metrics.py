"""Layer 1 storage: sector KPIs read from broker reports.

The rule every test here protects: a broker KPI is a fact about WHAT A BROKER SAID, on a date, on
their basis - never a fact about the company. So nothing is overwritten across reports, conflicts
are resolved only when reading, and a forecast is never allowed to pose as a reported number.
"""
import sqlite3
from datetime import date

import pytest

from pennystock.data import broker_metrics as bm
from pennystock.data.broker_metrics import (Observation, kpi_snapshot, load_observations,
                                            save_document, screening_kpis, summarize)
from pennystock.data.broker_reports import EstimateRow


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.settings, "market_db", tmp_path / "test.db")
    yield


def row(metric, period, value, label=None, page=1):
    return EstimateRow(page=page, label=label or metric, metric=metric, period=period, value=value)


def save(doc_id, rows, symbol="AUBANK", broker="ICICI Securities", report_date="2026-09-04"):
    return save_document(doc_id=doc_id, symbol=symbol, broker=broker, report_date=report_date,
                         source_file=f"{doc_id}.pdf", rows=rows)


def obs(metric, period, value, broker="A", kind="actual", end="2026-03-31",
        report_date="2026-09-04", label=None, symbol="AUBANK"):
    return Observation(doc_id=f"{broker}-{report_date}", symbol=symbol, broker=broker,
                       report_date=report_date, metric=metric, period=period, period_kind=kind,
                       period_end=end, value=value, label=label or metric, page=1)


# --- writing ---------------------------------------------------------------------------------
def test_a_saved_document_comes_back_with_canonical_periods():
    save("d1", [row("gnpa_pct", "FY26A", 2.0, "gnpa (%)"), row("gnpa_pct", "FY27E", 1.6, "gnpa (%)")])
    got = [(o.metric, o.period, o.period_kind, o.period_end, o.value, o.label)
           for o in load_observations("AUBANK")]
    assert got == [("gnpa_pct", "FY26", "actual", "2026-03-31", 2.0, "gnpa (%)"),
                   ("gnpa_pct", "FY27", "estimate", "2027-03-31", 1.6, "gnpa (%)")]


def test_reparsing_a_document_replaces_only_that_document():
    """The numbers are a pure function of the PDF bytes, so a parser fix may rewrite ITS rows. It
    must never touch another report's - that is how a later note would rewrite an earlier one."""
    save("d1", [row("nim_pct", "FY26", 12.0)], broker="ICICI Securities")
    save("d2", [row("nim_pct", "FY26", 12.6)], broker="Motilal Oswal")
    save("d1", [row("nim_pct", "FY26", 12.1)], broker="ICICI Securities")
    assert sorted((o.doc_id, o.value) for o in load_observations("AUBANK")) == \
        [("d1", 12.1), ("d2", 12.6)]


def test_within_a_document_the_reported_actual_beats_a_forecast_of_the_same_year():
    """Physicswallah's cover table says FY26E revenue 3,900; its income statement, later in the
    same note, gives FY26 at 3,899.4 - reported, because the year was filed by then."""
    save("pw", [row("revenue_cr", "FY26E", 3900.0, page=3), row("revenue_cr", "FY26", 3899.4, page=42)],
         symbol="PWL")
    assert [(o.period, o.period_kind, o.value) for o in load_observations("PWL")] == \
        [("FY26", "actual", 3899.4)]


def test_within_a_document_the_first_table_wins():
    """AU SF Bank prints FY26 credit cost as 0.9 (DuPont, p10) and 1.0 (key ratios, p11) - two
    bases. One value per document per year, and the earlier table is the broker's headline."""
    save("d1", [row("credit_cost_pct", "FY26", 0.9, "credit cost", page=10),
                row("credit_cost_pct", "FY26", 1.0, "credit cost (incl. cgfmu)", page=11)])
    assert [(o.value, o.label) for o in load_observations("AUBANK")] == [(0.9, "credit cost")]


def test_a_row_that_says_it_is_a_percentage_beats_a_bare_label():
    """Motilal's asset-quality table prints 'GNPA 45.6' - the amount, in INR b - and two rows later
    'GNPA Ratio (%) 1.6'. IDFC First's GNPA is not 45.6%: for a percentage KPI, the row whose label
    says it is a percentage wins over a bare label, whichever comes first."""
    save("mo", [row("gnpa_pct", "FY26", 45.6, "gnpa", page=12),
                row("gnpa_pct", "FY26", 1.6, "gnpa ratio (%)", page=12)], symbol="IDFCFIRSTB")
    assert [(o.value, o.label) for o in load_observations("IDFCFIRSTB")] == [(1.6, "gnpa ratio (%)")]


@pytest.mark.parametrize("field", ["symbol", "report_date", "broker"])
def test_a_document_without_its_identity_is_refused(field):
    kwargs = dict(doc_id="d1", symbol="AUBANK", broker="ICICI", report_date="2026-09-04",
                  source_file="x.pdf", rows=[row("nim_pct", "FY26", 5.7)])
    kwargs[field] = None
    with pytest.raises(ValueError):
        save_document(**kwargs)


def test_the_value_column_refuses_text():
    """Four stock_metrics rows hold the TEXT 'Infinity' in a numeric column and crash the screen.
    These tables are STRICT, so that class of bug cannot get in."""
    save("d1", [row("nim_pct", "FY26", 5.7)])
    with sqlite3.connect(bm.settings.market_db) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE broker_metrics SET value = 'Infinity'")


# --- reading: the reduction ------------------------------------------------------------------
def test_a_brokers_newer_report_replaces_its_own_older_one():
    """One vote per broker. Counting a broker's February and August notes separately would let
    the most prolific broker outvote the rest."""
    views = summarize([obs("nim_pct", "FY26", 12.0, broker="Motilal Oswal", report_date="2026-02-13"),
                       obs("nim_pct", "FY26", 12.6, broker="Motilal Oswal", report_date="2026-08-02"),
                       obs("nim_pct", "FY26", 11.5, broker="Joindre", report_date="2026-08-10")])
    v = views[0]
    assert (v.n, v.median, v.low, v.high, v.brokers) == (2, 12.05, 11.5, 12.6, ["Joindre", "Motilal Oswal"])


def test_a_broker_name_spelled_two_ways_is_one_broker():
    views = summarize([obs("nim_pct", "FY26", 12.0, broker="ICICI Securities Limited", report_date="2026-07-26"),
                       obs("nim_pct", "FY26", 12.4, broker="ICICI Securities Ltd.", report_date="2026-09-04")])
    assert (views[0].n, views[0].median) == (1, 12.4)


@pytest.mark.parametrize("values,status", [
    ([2.0], "single"),
    ([2.0, 2.1], "agree"),             # one decimal of rounding apart: the same disclosed number
    ([2.4, 2.6], "differ"),            # two bases (standalone vs consolidated, average vs closing)
    ([280.3, 279.0], "agree"),         # large values get a 1% tolerance
])
def test_agreement_means_equal_within_rounding(values, status):
    views = summarize([obs("gnpa_pct", "FY26", v, broker=f"B{i}") for i, v in enumerate(values)])
    assert views[0].status == status


def test_a_stale_forecast_does_not_vote_against_a_reported_number():
    """A broker's February note forecast FY26; another's September note reports it. Once a year is
    reported, forecasts of it are history, not opinions about it."""
    views = summarize([obs("nim_pct", "FY26", 12.4, broker="IDBI", kind="estimate", report_date="2026-02-13"),
                       obs("nim_pct", "FY26", 12.0, broker="ICICI", kind="actual")])
    assert [(v.period_kind, v.n, v.median) for v in views] == [("actual", 1, 12.0)]


def test_the_snapshot_shows_the_latest_actual_then_the_next_forecasts():
    save("d1", [row("gnpa_pct", "FY25", 2.3), row("gnpa_pct", "FY26", 2.0),
                row("gnpa_pct", "Q1FY27", 2.1), row("gnpa_pct", "FY27E", 1.6),
                row("gnpa_pct", "FY28E", 1.7), row("eps", "FY27E", 50.2)])
    snap = kpi_snapshot("AUBANK")
    assert [(v.metric, v.period, v.period_kind, v.median) for v in snap] == \
        [("gnpa_pct", "Q1FY27", "actual", 2.1), ("gnpa_pct", "FY27", "estimate", 1.6),
         ("gnpa_pct", "FY28", "estimate", 1.7)]          # eps is an estimate, not a sector KPI


def test_screening_takes_the_latest_reported_year_median_inside_the_age_window():
    """Ranking needs the SAME kind of number for every company: a reported fiscal year (quarters
    vary in timing), from a recent note, never a forecast."""
    save("a1", [row("gnpa_pct", "FY26", 2.0), row("gnpa_pct", "Q1FY27", 2.1),
                row("gnpa_pct", "FY27E", 1.6)], broker="ICICI", report_date="2026-09-04")
    save("a2", [row("gnpa_pct", "FY26", 2.4)], broker="IDBI", report_date="2026-08-10")
    save("b1", [row("gnpa_pct", "FY25", 3.0)], symbol="OLDBANK", broker="ICICI",
         report_date="2025-12-01")
    got = screening_kpis({"gnpa_pct"}, today=date(2026, 10, 2), max_age_days=180)
    assert got == {"AUBANK": {"gnpa_pct": 2.2}}


def test_counts_show_what_was_parsed_and_embedded():
    save("d1", [row("nim_pct", "FY26", 5.7), row("car_pct", "FY26", 18.7)])
    save("d2", [], symbol="TRENT", broker="Joindre")
    bm.mark_embedded(["d1"])
    c = bm.counts()
    assert (c["documents"], c["embedded"], c["metric_rows"], c["symbols"]) == (2, 1, 2, 2)


def _obs(broker, metric, period, kind, value, label, date="2026-08-08"):
    end = {"FY26": "2026-03-31", "FY27": "2027-03-31", "FY28": "2028-03-31"}[period]
    return bm.Observation(f"{broker}-{date}", "DIXON", broker, date, metric, period, kind, end, value, label, 1)


DIXON_OBS = [  # DIXON broker_metrics rows, 2026-10-11
    _obs("Keynote Capitals Ltd", "net_profit_cr", "FY26", "actual", 1400.0, "net profit"),
    _obs("Aditya Birla Money Limited", "net_profit_cr", "FY26", "actual", 845.1, "adjusted pat", "2026-08-02"),
    _obs("Motilal Oswal", "net_profit_cr", "FY26", "actual", 850.0, "pat", "2026-08-01"),
    _obs("Keynote Capitals Ltd", "net_profit_cr", "FY27", "estimate", 800.0, "net profit"),
    _obs("Aditya Birla Money Limited", "net_profit_cr", "FY27", "estimate", 1124.6, "adjusted pat", "2026-08-02"),
    _obs("Motilal Oswal", "net_profit_cr", "FY27", "estimate", 950.0, "pat", "2026-08-01"),
    _obs("Keynote Capitals Ltd", "eps", "FY26", "actual", 236.6, "eps"),
    _obs("Motilal Oswal", "eps", "FY26", "actual", 139.7, "eps", "2026-08-01"),
]


def test_a_reported_year_shows_the_filed_figure_and_the_brokers_adjusted_one(monkeypatch):
    # the median across bases (850) was once presented as the 'reported' FY26 profit
    monkeypatch.setattr(bm, "load_observations", lambda symbol, metrics=None, since=None: list(DIXON_OBS))
    views = bm.anchored_estimates("DIXON", {("net_profit_cr", "FY26"): 1438.64})
    fy26 = next(v for v in views if v["kpi"] == "Net profit cr" and v["period"] == "FY26")
    assert fy26["filed"] == 1438.64
    assert [b["broker"] for b in fy26["brokers_on_filed_basis"]] == ["Keynote Capitals Ltd"]
    assert fy26["brokers_adjusted"]["median"] == 847.55 and "ADJUSTED" in fy26["meaning"]


def test_forecasts_are_grouped_by_each_brokers_basis(monkeypatch):
    monkeypatch.setattr(bm, "load_observations", lambda symbol, metrics=None, since=None: list(DIXON_OBS))
    fy27 = next(v for v in bm.anchored_estimates("DIXON", {("net_profit_cr", "FY26"): 1438.64})
                if v["period"] == "FY27")
    assert fy27["basis"].startswith("adjusted (2 of 3")
    assert fy27["median"] == 1037.3 and fy27["other_basis"] == [
        {"broker": "Keynote Capitals Ltd", "basis": "filed", "value": 800.0}]


def test_eps_takes_its_basis_from_the_brokers_profit(monkeypatch):
    monkeypatch.setattr(bm, "load_observations", lambda symbol, metrics=None, since=None: list(DIXON_OBS))
    eps = next(v for v in bm.anchored_estimates("DIXON", {("net_profit_cr", "FY26"): 1438.64})
               if v["kpi"] == "EPS" and v["period"] == "FY26")
    assert [b["broker"] for b in eps["brokers_on_filed_basis"]] == ["Keynote Capitals Ltd"]
    assert eps["brokers_adjusted"]["brokers"] == ["Motilal Oswal"]


def test_without_a_filed_figure_the_view_says_so(monkeypatch):
    monkeypatch.setattr(bm, "load_observations", lambda symbol, metrics=None, since=None: list(DIXON_OBS))
    fy26 = next(v for v in bm.anchored_estimates("DIXON", {}) if v["period"] == "FY26" and v["kpi"] == "Net profit cr")
    assert fy26["filed"] is None and "no filed figure" in fy26["note"]
