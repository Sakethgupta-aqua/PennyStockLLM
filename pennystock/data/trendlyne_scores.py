"""Trendlyne Stock Reports read into structured snapshots - shown beside our own analysis, graded.

Trendlyne scores every stock 0-100 on Durability (financial strength: revenue, cash flow, debt),
Valuation (P/E, P/BV) and Momentum (30+ technical indicators, recomputed daily). Those are a
VENDOR'S OPINION: "stocks that score high tend to outperform" is Trendlyne's own claim, with no
independent evidence found. So a snapshot is attributed wherever it is shown, tracked month to
month, and graded in the ledger like any broker call - never trusted on the vendor's word.

Reading the PDF takes two tricks, both measured on the 119 reports on disk:

  1. Bold text is drawn twice - a grey copy 1.0-1.6 pt LOWER, then the black glyph over it - so
     naive extraction reads "45 38 47" as "4455 3388 4477". pdfplumber's position-tolerance dedupe
     cannot tell that shadow from real double letters ("Sell", "Technically") at any tolerance; the
     exact rule can: same glyph, same font, same x, a little lower -> the shadow.
  2. Checklist counts are labels inside coloured bar segments - green passed, red failed, amber not
     counted - and a zero segment is not drawn. So the printed "3 1" can mean 3 passed / 1 failed
     or 3 passed / 1 not counted; only the colours say which (`checklist_bars`).

Anything that cannot be read is None - never guessed.
Spec: docs/superpowers/specs/2026-10-04-stock-report-reader-design.md
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

from pennystock.config import settings
from pennystock.data import documents
from pennystock.data.ledger import Prediction, load_predictions, make_prediction_id, save_predictions
from pennystock.data.outcomes import actionable_date, fetch_closes, history_period
from pennystock.data.sectors import profile_for

logger = logging.getLogger(__name__)

PARSER_VERSION = 1
PAGES_READ = 7                  # page 1 + the score sentences (pp. 2-6) and the P/E zone (p. 4 or 5)

KINDS = ("durability", "valuation", "momentum")
NOT_ELIGIBLE = "Not Eligible"
CHECKLIST_GROUPS = ("Financials", "Ownership", "Peer Comparison", "Value & Momentum")
BUCKETS = ("n_strong_buy", "n_buy", "n_hold", "n_sell", "n_strong_sell")
# checklist bar segments, as filled in every report: (r, g, b) -> bucket
SEGMENT_COLOURS = {(0.0, 0.6353, 0.3569): "passed", (0.9882, 0.3529, 0.3529): "failed",
                   (1.0, 0.7725, 0.2588): "not_counted"}

# The gauge marks printed on every report (D 35/55, V 30/50, M 35/60) as (bad below, good from):
# good when D > 55, V > 50, M >= 60; bad when D < 35, V < 30, M < 35; medium otherwise.
BANDS = {"durability": (35, 56), "valuation": (30, 51), "momentum": (35, 60)}
# Trendlyne's own words for each band. It labels the UNROUNDED score, so at a printed boundary
# (V 30 appears as both "Expensive" and "Mid") the label, not the integer, says which side it fell.
LABEL_BANDS = {
    "durability": {"High Financial Strength": "good", "Medium Financial Strength": "medium",
                   "Low Financial Strength": "bad"},
    "valuation": {"Affordable Valuation": "good", "Mid Valuation": "medium", "Expensive Valuation": "bad"},
    "momentum": {"Technically Moderately Bullish": "good", "Technically Neutral": "medium",
                 "Technically Bearish": "bad"},
}

RE_ASOF = re.compile(r"([A-Z][a-z]{2} \d{2}, \d{4} \d{2}:\d{2} [AP]M)")
RE_PRICE = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)")
# a long Momentum label ("Technically Moderately Bullish") pushes "Momentum Score" up a line
RE_LABELS = re.compile(r"^Durability Score \(D\)\s*(.*?)\s*Valuation Score \(V\)\s*(.*?)\s*"
                       r"(?:Momentum Score\s*)?\(M\)\s*(.*?)\s*$", re.M)
RE_GAUGE = re.compile(r"^0 35 55 100\b", re.M)
RE_SCORES = re.compile(r"\d{1,3}(?: \d{1,3}){0,2}")
RE_SENTENCE = re.compile(r"has a (durability|valuation|momentum) score of (\d{1,3})\b", re.I)
RE_CLASS = re.compile(r"^DVM SUMMARY\s*\n(.+)$", re.M)
RE_REC = re.compile(r"^(STRONG BUY|BUY|HOLD|SELL|STRONG SELL|-)\s+([\d,]+(?:\.\d+)?)"
                    r"\s+\((-?[\d.]+)%\s*\)", re.M)
RE_PASS_PCT = re.compile(r"(\d+(?:\.\d+)?)%\s*\n[^\n]*Passed in checklist")
RE_ANALYSTS = re.compile(r"based on (\d+)\s*analyst", re.I)
RE_ESTIMATES = re.compile(r"^([\d,]+(?:\.\d+)?) ([\d,]+(?:\.\d+)?) ([\d,]+(?:\.\d+)?)\s*\nLow Est\.", re.M)
RE_EST_AVG_ONLY = re.compile(r"^([\d,]+(?:\.\d+)?)\s*\nAvg\.\s*Est\.", re.M)   # one analyst
RE_GROUP = re.compile(r"(Financials|Ownership|Peer Comparison|Value & Momentum)"
                      r" (\d+)(?: (\d+))?(?: (\d+))?\s*$", re.M)
RE_TOTAL = re.compile(r"^Total (\d+) (\d+)(?: (\d+))?\s*$", re.M)
RE_BUCKET_LINE = re.compile(r"^((?:\d+ ){1,5})\S.*\nTotal : (\d+)", re.M)
RE_ZONE = re.compile(r"\b(Strong Buy|Strong Sell|Buy|Sell|Neutral) Zone (\d+(?:\.\d+)?)% into P/E")


@dataclass(frozen=True)
class Snapshot:
    as_of: str | None = None                # data timestamp printed in the report, "YYYY-MM-DDTHH:MM"
    price: float | None = None              # price printed in the report (point-in-time)
    durability: int | None = None
    valuation: int | None = None
    momentum: int | None = None
    durability_label: str | None = None
    valuation_label: str | None = None
    momentum_label: str | None = None
    classification: str | None = None       # Trendlyne's name for the mix, e.g. "Falling Comet"
    checklist_pass_pct: float | None = None
    checklist_passed: int | None = None
    checklist_failed: int | None = None
    checklist_groups: dict = field(default_factory=dict)    # group -> [passed, failed]
    consensus: str | None = None            # Trendlyne's analyst consensus, e.g. "STRONG BUY"
    target_price: float | None = None
    upside_pct: float | None = None
    est_low: float | None = None
    est_avg: float | None = None
    est_high: float | None = None
    analysts: int | None = None
    n_strong_buy: int | None = None         # the five-bucket split: only when all five are printed
    n_buy: int | None = None
    n_hold: int | None = None
    n_sell: int | None = None
    n_strong_sell: int | None = None
    pe_zone: str | None = None              # where today's P/E sits in its own history
    pe_zone_pct: float | None = None


# ------------------------------------------------------------------ the PDF edge
def overprint_duplicates(chars, dy: float = 2.0) -> set[tuple]:
    """(text, x0, top) of every drop-shadow copy: same glyph, same font, same x (+-0.05 pt), drawn
    up to `dy` pt BELOW a copy already kept. Real double letters sit side by side, never stacked."""
    groups = defaultdict(list)
    for c in chars:
        groups[(c["text"], c.get("fontname"))].append(c)
    drop: set[tuple] = set()
    for same in groups.values():
        same.sort(key=lambda c: (c["top"], c["x0"]))
        kept: list[dict] = []
        for c in same:
            if any(abs(c["x0"] - k["x0"]) <= 0.05 and 0 <= c["top"] - k["top"] < dy
                   for k in reversed(kept) if c["top"] - k["top"] < dy):
                drop.add((c["text"], c["x0"], c["top"]))
            else:
                kept.append(c)
    return drop


def _without_shadows(page):
    drop = overprint_duplicates(page.chars)
    if not drop:
        return page
    return page.filter(lambda o: not (o.get("object_type") == "char"
                                      and (o["text"], o["x0"], o["top"]) in drop))


def clean_page_text(page) -> str:
    return _without_shadows(page).extract_text() or ""


def _bucket(colour) -> str | None:
    if not isinstance(colour, (tuple, list)) or len(colour) != 3:
        return None
    for rgb, name in SEGMENT_COLOURS.items():
        if all(abs(a - b) <= 0.02 for a, b in zip(colour, rgb)):
            return name
    return None


def checklist_bars(words, rects) -> dict[str, list[int]]:
    """Each group's [passed, failed], read from its row of coloured bar segments: the count
    printed inside each segment, bucketed by the segment's colour. A group whose segments cannot
    all be read (no label, or two) is left out rather than half-counted."""
    segments = [r for r in rects if _bucket(r.get("non_stroking_color"))]
    out: dict[str, list[int]] = {}
    for name in CHECKLIST_GROUPS:
        first = name.split()[0]
        for w in words:
            if w["text"] != first:
                continue
            row = [r for r in segments if r["top"] - 1 <= w["top"] <= r["bottom"] and r["x0"] >= w["x1"]]
            if not row:
                continue
            counts = {"passed": 0, "failed": 0, "not_counted": 0}
            for r in row:
                labels = [x for x in words if x["text"].isdigit() and r["top"] - 1 <= x["top"] <= r["bottom"]
                          and r["x0"] <= (x["x0"] + x["x1"]) / 2 <= r["x1"]]
                if len(labels) != 1:
                    break
                counts[_bucket(r["non_stroking_color"])] += int(labels[0]["text"])
            else:
                out[name] = [counts["passed"], counts["failed"]]
            break
    return out


# ------------------------------------------------------------------ reading the text
def _num(s: str | None) -> float | None:
    return float(s.replace(",", "")) if s else None


def _scores_from_page_one(p1: str, labels: dict[str, str | None]) -> dict[str, int]:
    """The score line sits directly above the gauge marks; a "Not Eligible" score prints no number,
    so the numbers belong, in order, to the scores that are eligible."""
    gauge = RE_GAUGE.search(p1)
    if not gauge:
        return {}
    above = p1[:gauge.start()].rstrip("\n").rsplit("\n", 1)[-1].strip()
    if not RE_SCORES.fullmatch(above):
        return {}
    numbers = [int(n) for n in above.split()]
    eligible = [k for k in KINDS if labels.get(k) != NOT_ELIGIBLE]
    return dict(zip(eligible, numbers)) if len(numbers) == len(eligible) else {}


def _checklist_groups(p1: str, total_has_not_counted: bool) -> dict[str, list[int]]:
    """Groups the text alone settles. Three counts are passed/failed/not counted. Two counts are
    passed/failed only when the report counts nothing as 'not counted' - otherwise "3 1" could be
    3 passed / 1 not counted. One count (the other segment was zero) is never settled by text."""
    out = {}
    for m in RE_GROUP.finditer(p1):
        counts = [int(x) for x in m.groups()[1:] if x]
        if len(counts) == 3 or (len(counts) == 2 and not total_has_not_counted):
            out[m.group(1)] = counts[:2]
    return out


def parse_snapshot(pages: list[str]) -> Snapshot:
    """One report's text (page 1 first) -> Snapshot. Every field is read on its own, so a layout
    change loses that field, never the whole report."""
    p1 = pages[0] if pages else ""
    joined = " ".join(" ".join(pages).split())
    f: dict = {}

    m = RE_ASOF.search(p1)
    if m:
        f["as_of"] = datetime.strptime(m.group(1), "%b %d, %Y %I:%M %p").strftime("%Y-%m-%dT%H:%M")
    m = RE_PRICE.search(p1)
    f["price"] = _num(m.group(1)) if m else None

    m = RE_LABELS.search(p1)
    labels = dict(zip(KINDS, (x or None for x in m.groups()))) if m else {}
    for k in KINDS:
        f[f"{k}_label"] = labels.get(k)
    scores = _scores_from_page_one(p1, labels)
    for k, n in RE_SENTENCE.findall(joined):          # fallback: the plain sentences on pp. 2-6
        k = k.lower()
        if k not in scores and labels.get(k) != NOT_ELIGIBLE:
            scores[k] = int(n)
    f.update(scores)

    m = RE_CLASS.search(p1)
    if m:
        line = m.group(1).strip()
        axis_or_number = line == "D V M" or re.fullmatch(r"[\d\s.]+", line)
        if not axis_or_number and not line.startswith("Durability Score"):
            f["classification"] = line

    m = RE_PASS_PCT.search(p1)
    f["checklist_pass_pct"] = float(m.group(1)) if m else None
    total = RE_TOTAL.search(p1)
    if total:
        f["checklist_passed"], f["checklist_failed"] = int(total.group(1)), int(total.group(2))
    f["checklist_groups"] = _checklist_groups(p1, bool(total and total.group(3)))

    if "ANALYST RECOMMENDATION" in p1:
        m = RE_REC.search(p1)
        if m:
            # "-" is one analyst's target with no consensus word: keep the target, not a rating
            f["consensus"] = None if m.group(1) == "-" else m.group(1)
            f["target_price"], f["upside_pct"] = _num(m.group(2)), float(m.group(3))
        m = RE_ANALYSTS.search(p1)
        f["analysts"] = int(m.group(1)) if m else None
        m = RE_ESTIMATES.search(p1)
        if m:
            f["est_low"], f["est_avg"], f["est_high"] = (_num(x) for x in m.groups())
        elif m := RE_EST_AVG_ONLY.search(p1):
            f["est_avg"] = _num(m.group(1))
        m = RE_BUCKET_LINE.search(p1)
        if m:
            split = [int(x) for x in m.group(1).split()]
            # zero buckets are omitted, so only all five printed (and adding up) is unambiguous
            if len(split) == 5 and sum(split) == int(m.group(2)):
                f.update(zip(BUCKETS, split))

    m = RE_ZONE.search(joined)
    if m:
        f["pe_zone"], f["pe_zone_pct"] = m.group(1), float(m.group(2))
    return Snapshot(**f)


def parse_report(path: str | Path) -> Snapshot:
    """A Stock Report PDF -> Snapshot: the text of the first pages, plus the checklist bars."""
    import pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages[:PAGES_READ]
        texts = [clean_page_text(p) for p in pages]
        first = _without_shadows(pages[0]) if pages else None
        bars = checklist_bars(first.extract_words(), first.rects) if first is not None else {}
    snap = parse_snapshot(texts)
    return replace(snap, checklist_groups={**snap.checklist_groups, **bars}) if bars else snap


# ------------------------------------------------------------------ bands and lean
def band(kind: str, score: int | None) -> str | None:
    """'good' / 'medium' / 'bad' by the printed gauge marks; None when there is no score."""
    if score is None:
        return None
    bad_below, good_from = BANDS[kind]
    return "bad" if score < bad_below else "good" if score >= good_from else "medium"


def band_of(snap: Snapshot, kind: str) -> str | None:
    """Trendlyne's own label when it is one we know; the printed score otherwise."""
    return LABEL_BANDS[kind].get(getattr(snap, f"{kind}_label") or "") or band(kind, getattr(snap, kind))


def lean(snap: Snapshot) -> tuple[str, int]:
    """(direction, conviction) from the published bands: bullish when no readable score is bad and
    at least two are good; bearish when at least two are bad; neutral otherwise. Conviction is 4
    when all three agree, 3 for two, 2 for neutral."""
    bands = [b for b in (band_of(snap, k) for k in KINDS) if b]
    good, bad = bands.count("good"), bands.count("bad")
    if not bad and good >= 2:
        return "bullish", 4 if good == 3 else 3
    if bad >= 2:
        return "bearish", 4 if bad == 3 else 3
    return "neutral", 2


# ------------------------------------------------------------------ storage (market.db)
_FIELDS = [f.name for f in fields(Snapshot)]
_TYPES = {"str": "TEXT", "float": "REAL", "int": "INTEGER"}


def _column_type(f) -> str:
    return "TEXT NOT NULL" if f.name == "checklist_groups" else _TYPES[str(f.type).split()[0]]


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS trendlyne_snapshots (\n"
    "    doc_key TEXT PRIMARY KEY,\n    symbol TEXT NOT NULL,\n"
    + "".join(f"    {f.name} {_column_type(f)},\n" for f in fields(Snapshot))
    + "    parser_version INTEGER NOT NULL,\n    parsed_at TEXT NOT NULL\n) STRICT;\n"
    "CREATE INDEX IF NOT EXISTS idx_trendlyne_symbol ON trendlyne_snapshots(symbol, as_of);\n")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def save_snapshot(doc_key: str, symbol: str, snap: Snapshot) -> None:
    """One row per report: a snapshot is a pure function of the PDF and the parser, so re-parsing
    the same report replaces its row."""
    row = asdict(snap)
    row["checklist_groups"] = json.dumps(row["checklist_groups"], sort_keys=True)
    cols = ["doc_key", "symbol", *_FIELDS, "parser_version", "parsed_at"]
    vals = [doc_key, symbol.upper(), *(row[f] for f in _FIELDS), PARSER_VERSION,
            datetime.now().isoformat(timespec="seconds")]
    with _connect() as con:
        con.execute(f"INSERT OR REPLACE INTO trendlyne_snapshots ({', '.join(cols)}) "
                    f"VALUES ({', '.join('?' * len(cols))})", vals)


def _snapshot(row: sqlite3.Row) -> Snapshot:
    d = {f: row[f] for f in _FIELDS}
    d["checklist_groups"] = json.loads(d["checklist_groups"] or "{}")
    return Snapshot(**d)


def history(symbol: str) -> list[tuple[str, Snapshot]]:
    """Every snapshot on file for a company, oldest first."""
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trendlyne_snapshots WHERE symbol = ? ORDER BY as_of, doc_key",
                           (symbol.upper(),)).fetchall()
    return [(r["doc_key"], _snapshot(r)) for r in rows]


def latest(symbol: str) -> tuple[str, Snapshot] | None:
    hist = history(symbol)
    return hist[-1] if hist else None


_NUMERIC_CHANGES = ("price", "durability", "valuation", "momentum", "checklist_pass_pct", "target_price",
                    "upside_pct", "analysts", "pe_zone_pct")
_TEXT_CHANGES = ("classification", "consensus", "pe_zone", "durability_label", "valuation_label",
                 "momentum_label")


def changes(prev: Snapshot, cur: Snapshot) -> dict:
    """What moved between two snapshots: numbers as cur - prev, words as "was -> now". Only fields
    readable in both and actually different."""
    out: dict = {}
    for f in _NUMERIC_CHANGES:
        a, b = getattr(prev, f), getattr(cur, f)
        if a is not None and b is not None and a != b:
            out[f] = round(b - a, 2)
    for f in _TEXT_CHANGES:
        a, b = getattr(prev, f), getattr(cur, f)
        if a and b and a != b:
            out[f] = f"{a} -> {b}"
    return out


def _parsed_versions() -> dict[str, int]:
    with _connect() as con:
        return dict(con.execute("SELECT doc_key, parser_version FROM trendlyne_snapshots"))


def parse_all(reparse: bool = False, symbol: str | None = None) -> dict:
    """Parse every catalogued Stock Report not yet read by this PARSER_VERSION (all, with reparse).
    A report that cannot be read is counted and logged; it never stops the run."""
    tally = {"parsed": 0, "current": 0, "failed": 0, "missing": 0}
    done = _parsed_versions()
    for doc in documents.list_documents(symbol, "stock_report"):
        if not reparse and done.get(doc.doc_key) == PARSER_VERSION:
            tally["current"] += 1
            continue
        path = Path(settings.project_root) / doc.local_path
        if not path.is_file():
            tally["missing"] += 1
            continue
        try:
            save_snapshot(doc.doc_key, doc.symbol, parse_report(path))
            tally["parsed"] += 1
        except Exception as e:                  # one odd PDF must not stop the other reports
            logger.warning(f"could not read {path.name}: {type(e).__name__}: {e}")
            tally["failed"] += 1
    return tally


# ------------------------------------------------------------------ beside our own score
# our pillar -> Trendlyne's score that measures the same thing
PAIRS = (("quality", "durability"), ("valuation", "valuation"), ("momentum", "momentum"))
_MATCH = {"high": "good", "mid": "medium", "low": "bad"}


def _tier(pct: float | None) -> str | None:
    if pct is None:
        return None
    return "high" if pct >= 0.67 else "low" if pct <= 0.33 else "mid"


def agreement(ours: dict[str, float | None], snap: Snapshot) -> list[dict]:
    """Our pillar percentile (0-1) beside Trendlyne's matching score. High (>= 0.67) with good, mid
    with medium, low (<= 0.33) with bad agree; high against bad, or low against good, is DISAGREE -
    the case worth reading: one of the two sees something the other does not."""
    rows = []
    for mine, theirs in PAIRS:
        tier, their_band = _tier(ours.get(mine)), band_of(snap, theirs)
        if tier is None or their_band is None:
            verdict = "n/a"
        elif _MATCH[tier] == their_band:
            verdict = "agree"
        elif {tier, their_band} in ({"high", "bad"}, {"low", "good"}):
            verdict = "DISAGREE"
        else:
            verdict = "one step apart"
        rows.append({"ours": mine, "our_score": ours.get(mine), "our_tier": tier, "theirs": theirs,
                     "their_score": getattr(snap, theirs), "their_band": their_band, "verdict": verdict})
    return rows


def view(symbol: str) -> dict | None:
    """The latest snapshot as the analyst tool hands it over: its readable fields, the bands, the
    lean the ledger grades, and what changed since the previous report. None when none is on file."""
    hist = history(symbol)
    if not hist:
        return None
    key, cur = hist[-1]
    out = {k: v for k, v in asdict(cur).items() if v not in (None, {})}
    out["bands"] = {k: band_of(cur, k) for k in KINDS}
    out["lean"], out["lean_conviction"] = lean(cur)
    out["snapshots_on_file"] = len(hist)
    if len(hist) > 1:
        prev = hist[-2][1]
        out["changes_since_previous"] = {"previous_as_of": prev.as_of, **changes(prev, cur)}
    out["report"] = key
    return out


def _all_snapshots() -> list[tuple[str, str, Snapshot]]:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trendlyne_snapshots ORDER BY symbol, as_of").fetchall()
    return [(r["doc_key"], r["symbol"], _snapshot(r)) for r in rows]


# ------------------------------------------------------------------ grading Trendlyne's view
def to_prediction(symbol: str, doc_key: str, snap: Snapshot, closes=None, lag: int = 1,
                  sector: str | None = None, industry: str | None = None) -> Prediction | None:
    """A snapshot as a ledger call: direction and conviction from the bands, dated to the first
    session after the report's data timestamp, entered at the price the report printed.

    None when the snapshot holds no view: no data timestamp (cannot be dated), or fewer than two
    readable scores ("two agree" is impossible - recording that as neutral would invent a view)."""
    if not snap.as_of or sum(1 for k in KINDS if band_of(snap, k)) < 2:
        return None
    direction, conviction = lean(snap)
    report_day = snap.as_of[:10]
    as_of = actionable_date(report_day, closes, lag)
    payload = {"durability": snap.durability, "valuation": snap.valuation, "momentum": snap.momentum,
               "classification": snap.classification, "checklist_pass_pct": snap.checklist_pass_pct,
               "consensus": snap.consensus, "pe_zone": snap.pe_zone, "as_of": snap.as_of, "doc_key": doc_key}
    return Prediction(
        symbol=symbol.upper(), source="trendlyne", as_of_date=as_of, direction=direction,
        conviction=conviction,
        horizon="1m",                       # a new snapshot replaces this view every month
        entry_price=snap.price, entry_price_asof=report_day if snap.price is not None else None,
        sector=sector, rulebook=profile_for(sector, industry).name, payload=payload,
        provenance={"doc_key": doc_key, "parser_version": PARSER_VERSION, "entry_price_source": "report",
                    "recorded_at": datetime.now().isoformat(timespec="seconds")},
        prediction_id=make_prediction_id(as_of, symbol, {"source": "trendlyne", "as_of": snap.as_of,
                                                         "doc_key": doc_key}))


def record_predictions(lag: int = 1) -> int:
    """One ledger call per (company, report data timestamp), never rewritten: running it again,
    or re-parsing a report, adds nothing. Returns how many calls were added."""
    seen = {(p.symbol, p.payload.get("as_of"))
            for p in load_predictions(source="trendlyne", include_superseded=True)}
    try:
        from pennystock.data.metrics import load_metrics
        sectors = {m.symbol: (m.sector, m.industry) for m in load_metrics()}
    except Exception as e:                      # a missing metrics cache only costs the rulebook
        logger.info(f"no sector data for the rulebook: {type(e).__name__}: {e}")
        sectors = {}
    new: list[Prediction] = []
    for doc_key, symbol, snap in _all_snapshots():
        if (symbol, snap.as_of) in seen or not snap.as_of:
            continue
        closes = fetch_closes(symbol, history_period(snap.as_of[:10]))
        p = to_prediction(symbol, doc_key, snap, closes, lag, *sectors.get(symbol, (None, None)))
        if p:
            seen.add((symbol, snap.as_of))
            new.append(p)
    return save_predictions(new)


def status() -> dict:
    """Reports catalogued vs read, and how often each field came back unreadable (NULL)."""
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trendlyne_snapshots").fetchall()
    return {"reports": len(documents.list_documents(kind="stock_report")), "parsed": len(rows),
            "companies": len({r["symbol"] for r in rows}),
            "older_parser": sum(1 for r in rows if r["parser_version"] != PARSER_VERSION),
            "unreadable": {f: n for f in _FIELDS if f != "checklist_groups"
                           if (n := sum(1 for r in rows if r[f] is None))}}


def pe_zone_meaning(zone: str | None, pct: float | None) -> str | None:
    """pe_zone_pct in words: the share of its own history the stock's P/E spent BELOW today's. Read raw, the model
    turned 'Strong Buy zone, 12%' into 'P/E in the top 12% of its history' (DIXON, 2026-10-10)."""
    if pct is None:
        return None
    verdict = "cheap" if pct <= 30 else "expensive" if pct >= 70 else "middling"
    only = "only " if pct < 50 else ""
    named = f" (Trendlyne calls this zone '{zone}')" if zone else ""
    return f"today's P/E is higher than {only}{pct:g}% of its own history - {verdict} against its past{named}"
