"""Layer 1: sector KPIs read out of broker reports, stored as what each broker SAID.

The parser has always read GNPA 2.3% off AU SF Bank's cover - and then flattened it into a sentence
for the vector store and threw the number away. This module keeps it, and keeps it honestly:

  A BROKER KPI IS NOT A FACT ABOUT THE COMPANY. It is a fact about what one broker printed, on
  one date, on its own basis. Four brokers put Muthoot's FY26 NIM - a settled historical number -
  anywhere from 11.5% to 12.7%, because NIM has no single definition (on AUM? on total assets?
  standalone or consolidated?). So:

  * nothing is ever overwritten ACROSS reports: every (document, metric, period) is its own row;
  * conflicts are resolved when READING - median, range, how many brokers - never when writing;
  * the row label is kept, because "nim on aum" vs "nims (cal)" IS the basis;
  * actuals and forecasts are told apart (`period_kind`), so an analyst's FY27 guess can never be
    ranked as if the company had reported it;
  * valid time (`report_date`) and transaction time (`parsed_at`) are both stored - the same
    point-in-time discipline as the ledger, so the screen can always say what it knew, and when.

A re-parse REPLACES one document's rows. That is not rewriting history: the numbers are a pure
function of the PDF's bytes and the parser, so a parser fix corrects extraction without ever
letting a later report overwrite an earlier one.
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

from pennystock.config import settings
from pennystock.data.broker_corpus import broker_key
from pennystock.data.broker_reports import (FORWARD_ESTIMATES, KPI_LABELS, KPI_ROWS, PARSER_VERSION, EstimateRow,
                                           period_info)

logger = logging.getLogger(__name__)

# The order KPIs are displayed in: the order the vocabulary declares them, lenders first.
KPI_ORDER = [k for k, _, _ in KPI_ROWS]

# STRICT: a numeric column refuses text. stock_metrics is not strict, and four of its rows hold
# the TEXT 'Infinity' in `pe` - which is how a single bad feed value crashes a whole screen.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS broker_documents (
    doc_id          TEXT PRIMARY KEY,
    report_id       TEXT,
    symbol          TEXT NOT NULL,
    company         TEXT,
    broker          TEXT NOT NULL,
    report_date     TEXT NOT NULL,
    rating          TEXT,
    source_file     TEXT NOT NULL,
    pages           INTEGER,
    metric_rows     INTEGER NOT NULL DEFAULT 0,
    parser_version  INTEGER NOT NULL,
    parsed_at       TEXT NOT NULL,
    embedded_at     TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS broker_metrics (
    doc_id       TEXT NOT NULL REFERENCES broker_documents(doc_id) ON DELETE CASCADE,
    metric       TEXT NOT NULL,
    period       TEXT NOT NULL,
    period_kind  TEXT NOT NULL CHECK (period_kind IN ('actual', 'estimate')),
    period_end   TEXT NOT NULL,
    value        REAL NOT NULL,
    label        TEXT NOT NULL,
    page         INTEGER,
    symbol       TEXT NOT NULL,
    broker       TEXT NOT NULL,
    report_date  TEXT NOT NULL,
    PRIMARY KEY (doc_id, metric, period)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_broker_metrics_symbol ON broker_metrics(symbol, metric);
"""

_OBS_COLS = ("doc_id", "symbol", "broker", "report_date", "metric", "period", "period_kind",
             "period_end", "value", "label", "page")


@dataclass(frozen=True)
class Observation:
    """One stored number: what one broker printed for one metric and period, in one report."""
    doc_id: str
    symbol: str
    broker: str
    report_date: str
    metric: str
    period: str
    period_kind: str
    period_end: str
    value: float
    label: str
    page: int | None


@dataclass(frozen=True)
class KpiView:
    """Every broker's latest word on one metric for one period, reduced to what a reader needs."""
    metric: str
    period: str
    period_kind: str
    period_end: str
    median: float
    low: float
    high: float
    n: int                    # distinct brokers
    brokers: list[str]
    labels: list[str]         # the bases, as printed
    latest_report: str
    status: str               # single | agree | differ


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.market_db))
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    return con


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ writing
def _says_percent(metric: str, label: str) -> bool:
    """Does the row label itself say its number is a percentage ('gnpa ratio (%)', 'nim (%)')?"""
    return metric.endswith("_pct") and ("%" in label or re.search(r"\bratio\b", label) is not None)


def _canonical(rows: Iterable[EstimateRow], report_date: str) -> list[tuple]:
    """One value per (metric, period) for this document.

    A reported ACTUAL beats a forecast of the same year (Physicswallah's cover says FY26E, its
    income statement reports FY26). For a percentage KPI, a row whose label says it is a percentage
    beats a bare one: Motilal prints 'GNPA 45.6' (the amount, INR b) above 'GNPA Ratio (%) 1.6'.
    Otherwise the first table wins, because the cover summary is the broker's headline and later
    tables are detail on other bases.
    """
    chosen: dict[tuple[str, str], tuple] = {}
    for r in rows:
        if not math.isfinite(r.value):
            continue
        period, kind, end = period_info(r.period, report_date)
        key = (r.metric, period)
        current = chosen.get(key)
        if (current is None or (kind == "actual" and current[1] == "estimate")
                or (kind == current[1] and _says_percent(r.metric, r.label)
                    and not _says_percent(r.metric, current[4]))):
            chosen[key] = (period, kind, end, float(r.value), r.label, r.page)
    return [(metric, *v) for (metric, _), v in chosen.items()]


def save_document(*, doc_id: str, symbol: str | None, broker: str | None, report_date: str | None,
                  source_file: str, rows: list[EstimateRow], report_id: str | None = None,
                  company: str | None = None, rating: str | None = None,
                  pages: int | None = None) -> int:
    """Record one parsed PDF and every KPI row it yielded, replacing any earlier parse of it.

    Refuses a document with no symbol, broker or date: a number nobody can attribute to a company,
    a broker and a day is not a broker KPI, it is noise.
    """
    if not (doc_id and symbol and broker and report_date):
        raise ValueError(f"document {doc_id!r} needs a symbol, a broker and a report date "
                         f"(got {symbol!r}, {broker!r}, {report_date!r})")
    canonical = _canonical(rows, report_date)
    with _connect() as con:
        con.execute("DELETE FROM broker_metrics WHERE doc_id = ?", (doc_id,))
        # a re-parse may change the text too, so the vector copy is marked stale (embedded_at NULL)
        con.execute(
            "INSERT INTO broker_documents (doc_id, report_id, symbol, company, broker, report_date, "
            "rating, source_file, pages, metric_rows, parser_version, parsed_at, embedded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL) ON CONFLICT(doc_id) DO UPDATE SET "
            "report_id=excluded.report_id, symbol=excluded.symbol, company=excluded.company, "
            "broker=excluded.broker, report_date=excluded.report_date, rating=excluded.rating, "
            "source_file=excluded.source_file, pages=excluded.pages, "
            "metric_rows=excluded.metric_rows, parser_version=excluded.parser_version, "
            "parsed_at=excluded.parsed_at, embedded_at=NULL",
            (doc_id, report_id, symbol, company, broker, report_date, rating, source_file, pages,
             len(canonical), PARSER_VERSION, _now()))
        con.executemany(
            "INSERT INTO broker_metrics (doc_id, metric, period, period_kind, period_end, value, "
            "label, page, symbol, broker, report_date) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(doc_id, *c, symbol, broker, report_date) for c in canonical])
    return len(canonical)


def parsed_documents() -> dict[str, tuple[int, str | None]]:
    """doc_id -> (parser_version, embedded_at). The ingest's memory of what it has already done."""
    with _connect() as con:
        return {d: (v, e) for d, v, e in con.execute(
            "SELECT doc_id, parser_version, embedded_at FROM broker_documents")}


def mark_embedded(doc_ids: list[str]) -> None:
    with _connect() as con:
        con.executemany("UPDATE broker_documents SET embedded_at = ? WHERE doc_id = ?",
                        [(_now(), d) for d in doc_ids])


def clear_embedded() -> None:
    """After the vector collection is rebuilt from scratch, nothing in it is embedded any more."""
    with _connect() as con:
        con.execute("UPDATE broker_documents SET embedded_at = NULL")


def counts() -> dict:
    with _connect() as con:
        docs, embedded, symbols = con.execute(
            "SELECT COUNT(*), COUNT(embedded_at), COUNT(DISTINCT symbol) FROM broker_documents").fetchone()
        rows = con.execute("SELECT COUNT(*) FROM broker_metrics").fetchone()[0]
        stale = con.execute("SELECT COUNT(*) FROM broker_documents WHERE parser_version < ?",
                            (PARSER_VERSION,)).fetchone()[0]
    return {"documents": docs, "embedded": embedded, "symbols": symbols, "metric_rows": rows,
            "parsed_by_older_parser": stale}


# ------------------------------------------------------------------ reading
def load_observations(symbol: str | None = None, metrics: Iterable[str] | None = None,
                      since: str | None = None) -> list[Observation]:
    sql = f"SELECT {', '.join(_OBS_COLS)} FROM broker_metrics"
    where, args = [], []
    if symbol:
        where.append("symbol = ?")
        args.append(symbol.upper())
    if metrics is not None:
        metrics = list(metrics)
        where.append(f"metric IN ({', '.join('?' * len(metrics))})")
        args.extend(metrics)
    if since:
        where.append("report_date >= ?")
        args.append(since)
    if where:
        sql += " WHERE " + " AND ".join(where)
    with _connect() as con:
        return [Observation(*r) for r in con.execute(sql + " ORDER BY symbol, metric, period_end", args)]


def _agrees(values: list[float], median: float) -> bool:
    """Equal within rounding. Brokers copying the same disclosed number print it to one decimal,
    so two honest copies differ by at most 0.1; anything more is a different basis."""
    return max(values) - min(values) <= max(0.15, abs(median) * 0.01)


def _latest_votes(observations: Iterable[Observation]) -> list[Observation]:
    """One vote per broker per (metric, period, kind): its newest report replaces its own older ones."""
    latest: dict[tuple, Observation] = {}
    for o in observations:
        k = (broker_key(o.broker), o.metric, o.period, o.period_kind)
        if k not in latest or (o.report_date, o.doc_id) > (latest[k].report_date, latest[k].doc_id):
            latest[k] = o
    return list(latest.values())


# the filed line each metric's basis is judged on: EPS follows the broker's profit (basic EPS in filings and in
# broker tables differ by share-count conventions, so it is not anchored on its own)
_ANCHOR = {"revenue_cr": "revenue_cr", "net_profit_cr": "net_profit_cr", "eps": "net_profit_cr"}


def _spread(votes: list[Observation]) -> dict:
    vals = [o.value for o in votes]
    return {"median": round(statistics.median(vals), 2), "low": min(vals), "high": max(vals),
            "brokers": sorted(o.broker for o in votes)}


def anchored_estimates(symbol: str, filed: dict[tuple[str, str], float], forecasts: int = 2,
                       tolerance_pct: float | None = None) -> list[dict]:
    """Fiscal-year estimates on ONE basis, anchored to the company's filed figures.

    Brokers print profit on different bases - DIXON FY26 (2026-10-11): Keynote Rs 1,400 Cr (the filed Rs 1,438.64
    Cr), Aditya Birla Money Rs 845.1 Cr and Motilal Oswal Rs 850 Cr (adjusted) - and their median (850) was shown
    as the 'reported' figure. Following I/B/E/S, which drops estimates made on a different basis from the
    majority: a reported year's headline is the FILED figure, brokers that differ are shown as adjusted, each
    broker's basis is inferred from that year and its forecasts are grouped by it."""
    tol = (settings.broker_filed_tolerance_pct if tolerance_pct is None else tolerance_pct) / 100
    votes = [o for o in _latest_votes(load_observations(symbol, FORWARD_ESTIMATES)) if o.period.startswith("FY")]
    basis: dict[str, dict[str, str]] = {}                  # anchor metric -> broker -> filed | adjusted
    out: list[dict] = []
    for metric in sorted(FORWARD_ESTIMATES, key=lambda m: m == "eps"):   # profit's basis is known before EPS
        mv = [o for o in votes if o.metric == metric]
        if not mv:
            continue
        actuals = [o for o in mv if o.period_kind == "actual"]
        anchor = _ANCHOR.get(metric)
        label = KPI_LABELS.get(metric, metric)
        if actuals:
            period = max(actuals, key=lambda o: o.period_end).period
            year = [o for o in actuals if o.period == period]
            f = filed.get((metric, period))
            if metric == anchor and f:
                basis[anchor] = {broker_key(o.broker): "filed" if abs(o.value - f) <= tol * abs(f) else "adjusted"
                                 for o in year}
            known = basis.get(anchor or "", {})
            on_filed = [o for o in year if known.get(broker_key(o.broker)) == "filed"]
            adjusted = [o for o in year if known.get(broker_key(o.broker)) == "adjusted"]
            if not known:
                out.append({"kpi": label, "period": period, "kind": "reported", "filed": f,
                            "brokers_median": _spread(year)["median"], "brokers": _spread(year)["brokers"],
                            "note": "no filed figure to check against - as each broker prints it"})
            else:
                view = {"kpi": label, "period": period, "kind": "reported", "filed": f,
                        "brokers_on_filed_basis": [{"broker": o.broker, "value": o.value} for o in on_filed]}
                if adjusted:
                    view["brokers_adjusted"] = _spread(adjusted)
                    view["meaning"] = (f"{len(adjusted)} of {len(year)} brokers print an ADJUSTED {period} figure "
                                       f"(median {view['brokers_adjusted']['median']:,}) - a different basis from the "
                                       "filed one; compare their forecasts with the adjusted figure")
                out.append(view)
        cutoff = max((o.period_end for o in actuals), default="")
        ahead = sorted({(o.period_end, o.period) for o in mv if o.period_kind == "estimate" and o.period_end > cutoff})
        for _, period in ahead[:forecasts]:
            est = [o for o in mv if o.period == period and o.period_kind == "estimate"]
            known = basis.get(anchor or "", {})
            groups: dict[str, list[Observation]] = {}
            for o in est:
                groups.setdefault(known.get(broker_key(o.broker), "unknown"), []).append(o)
            named = {k: v for k, v in groups.items() if k != "unknown"}
            major = max(named, key=lambda k: (len(named[k]), k == "filed")) if named else None
            main = named.get(major, []) + groups.get("unknown", [])
            view = {"kpi": label, "period": period, "kind": "forecast",
                    "basis": f"{major} ({len(named[major])} of {len(est)} brokers)" if major
                    else "as each broker prints it", **_spread(main)}
            others = [o for k, v in named.items() if k != major for o in v]
            if others:
                view["other_basis"] = [{"broker": o.broker, "basis": known[broker_key(o.broker)], "value": o.value}
                                       for o in others]
            out.append(view)
    return out


def summarize(observations: Iterable[Observation]) -> list[KpiView]:
    """Reduce raw rows to one view per (symbol-agnostic) metric and period.

    One vote per broker: a broker's newest report replaces its own older ones. And once ANY broker
    reports a year as actual, forecasts of that year are history rather than opinions about it.
    """
    groups: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for o in _latest_votes(observations):
        groups[(o.metric, o.period)].append(o)

    views = []
    for (metric, period), group in groups.items():
        actual = [o for o in group if o.period_kind == "actual"]
        group = actual or group
        # a broker that printed BOTH kinds over time keeps only its newest word
        by_broker: dict[str, Observation] = {}
        for o in sorted(group, key=lambda o: o.report_date):
            by_broker[broker_key(o.broker)] = o
        group = list(by_broker.values())
        values = [o.value for o in group]
        med = round(statistics.median(values), 4)
        views.append(KpiView(
            metric=metric, period=period, period_kind=group[0].period_kind,
            period_end=group[0].period_end, median=med, low=min(values), high=max(values),
            n=len(group), brokers=sorted({o.broker for o in group}),
            labels=sorted({o.label for o in group}),
            latest_report=max(o.report_date for o in group),
            status="single" if len(group) == 1 else "agree" if _agrees(values, med) else "differ"))
    order = {m: i for i, m in enumerate(KPI_ORDER)}
    return sorted(views, key=lambda v: (order.get(v.metric, len(order)), v.metric, v.period_end))


def kpi_snapshot(symbol: str, metrics: Iterable[str] | None = None,
                 forecasts: int = 2) -> list[KpiView]:
    """Per sector KPI: the latest REPORTED value (quarters included - a bank's Q1 GNPA is fresher
    than its full-year number), then the next fiscal-year forecasts after it."""
    order = list(metrics) if metrics is not None else KPI_ORDER
    views = summarize(load_observations(symbol, order))
    out: list[KpiView] = []
    by_metric: dict[str, list[KpiView]] = defaultdict(list)
    for v in views:
        by_metric[v.metric].append(v)
    for metric in [m for m in order if m in by_metric]:
        vs = by_metric[metric]
        actuals = [v for v in vs if v.period_kind == "actual"]
        last = max(actuals, key=lambda v: v.period_end) if actuals else None
        cutoff = last.period_end if last else ""
        ahead = sorted((v for v in vs if v.period_kind == "estimate" and v.period.startswith("FY")
                        and v.period_end > cutoff), key=lambda v: v.period_end)[:forecasts]
        out.extend(([last] if last else []) + ahead)
    return out


def screening_kpis(metrics: Iterable[str], today: date | None = None,
                   max_age_days: int | None = None) -> dict[str, dict[str, float]]:
    """{symbol: {metric: value}} for ranking: the latest REPORTED FISCAL YEAR, median across
    brokers, from notes inside the age window.

    Fiscal years only, because ranking needs the same kind of number for every company and
    quarter-ends arrive at different times; reported only, because a forecast is an opinion.
    """
    today = today or date.today()
    since = (today - timedelta(days=max_age_days or settings.broker_max_age_days)).isoformat()
    by_symbol: dict[str, list[Observation]] = defaultdict(list)
    for o in load_observations(metrics=metrics, since=since):
        by_symbol[o.symbol].append(o)
    out: dict[str, dict[str, float]] = {}
    for symbol, rows in by_symbol.items():
        best: dict[str, KpiView] = {}
        for v in summarize(rows):
            if (v.period_kind == "actual" and v.period.startswith("FY")
                    and (v.metric not in best or v.period_end > best[v.metric].period_end)):
                best[v.metric] = v
        if best:
            out[symbol] = {m: v.median for m, v in best.items()}
    return out
