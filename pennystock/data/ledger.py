"""The prediction ledger: what was said, when, and at what price.

Every analysis this system runs produces a report and then forgets it. Nothing records
"on 2026-09-27 I said buy HINDALCO at 640" - so nothing can ever be checked. This module
is the memory: one append-only row per call, from the agent or from a broker, plus a
separate, recomputable table for what the price actually did afterwards.

TWO PROPERTIES MAKE IT TRUSTWORTHY, and both are enforced here rather than by convention:

  PREDICTIONS ARE IMMUTABLE. The entry price is captured once and never recomputed,
  because the past changes. Yahoo applies dividends and splits backwards, so ITC's close
  on 2023-09-25 reads 426.19 on the day and 379.91 if you fetch it today - a 10.9% gap.
  A ledger that re-derived its own entry prices would report gains nobody ever made.
  `save_predictions` therefore uses a plain INSERT: a duplicate id RAISES.

  OUTCOMES ARE DERIVED. They arrive at different times (+1d tomorrow, +60d in three
  months) and may be rebuilt wholesale if the abnormal-return method improves. They
  upsert freely, and live in their own table precisely so that rebuilding them cannot
  touch the predictions they grade.

This module does storage only - no network, no return arithmetic. That lives in
`outcomes.py`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timedelta

from pennystock.config import settings

logger = logging.getLogger(__name__)

# Stamped on every row. Events are immutable, so a later schema change can never rewrite
# old rows - the version is what lets a reader upcast them rather than misread them.
SCHEMA_VERSION = 1

SOURCES = ("agent", "broker")
DIRECTIONS = ("bullish", "bearish", "neutral")
LABELS = ("hit", "miss", "flat", "invalidated", "open", "unavailable")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      TEXT PRIMARY KEY,
    source             TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    exchange           TEXT NOT NULL DEFAULT 'NSE',
    as_of_date         TEXT NOT NULL,
    as_of_ts           TEXT,
    direction          TEXT NOT NULL,
    conviction         INTEGER,
    horizon            TEXT,
    entry_price        REAL,
    entry_price_asof   TEXT,
    invalidation_price REAL,
    sector             TEXT,
    rulebook           TEXT,
    payload            TEXT NOT NULL DEFAULT '{}',
    provenance         TEXT NOT NULL DEFAULT '{}',
    schema_version     INTEGER NOT NULL DEFAULT 1,
    superseded_by      TEXT,
    created_at         TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_pred_source ON predictions(source, as_of_date);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id              TEXT NOT NULL,
    horizon_days               INTEGER NOT NULL,
    outcome_date               TEXT,
    stock_return_pct           REAL,
    benchmark_symbol           TEXT,
    benchmark_return_pct       REAL,
    abnormal_return_pct        REAL,
    sector_benchmark_symbol    TEXT,
    sector_abnormal_return_pct REAL,
    label                      TEXT NOT NULL,
    computed_at                TEXT NOT NULL,
    PRIMARY KEY (prediction_id, horizon_days));
"""


@dataclass
class Prediction:
    """One call. Write once; never edit."""
    symbol: str
    source: str
    as_of_date: str                       # the first day this could have been acted on
    direction: str
    entry_price: float | None = None      # raw close as known that day - frozen forever
    exchange: str = "NSE"
    conviction: int = 3
    horizon: str = "1m"
    as_of_ts: str | None = None
    entry_price_asof: str | None = None
    invalidation_price: float | None = None
    sector: str | None = None
    rulebook: str | None = None
    payload: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    prediction_id: str | None = None      # generated on save when absent
    schema_version: int = SCHEMA_VERSION
    superseded_by: str | None = None      # set only by supersede(); never edited in place
    created_at: str | None = None


@dataclass
class Outcome:
    """What the price did, for one prediction at one horizon. Recomputable."""
    prediction_id: str
    horizon_days: int
    label: str
    outcome_date: str | None = None
    stock_return_pct: float | None = None
    benchmark_symbol: str | None = None
    benchmark_return_pct: float | None = None
    abnormal_return_pct: float | None = None
    sector_benchmark_symbol: str | None = None
    sector_abnormal_return_pct: float | None = None
    computed_at: str | None = None


_PRED_COLS = [f.name for f in fields(Prediction)]
_OUT_COLS = [f.name for f in fields(Outcome)]
_JSON_COLS = {"payload", "provenance"}


# Brokers and the agent say the same three things in about twenty different words. The
# 12 reports downloaded so far already use BUY, ADD, HOLD, Neutral, SELL and IPO
# SUBSCRIBE. Grading needs ONE vocabulary, so everything is normalised on the way in.
_BULLISH = {"buy", "strong buy", "add", "accumulate", "outperform", "overweight",
            "subscribe", "ipo subscribe", "long"}
_BEARISH = {"sell", "strong sell", "reduce", "underperform", "underweight", "avoid",
            "short"}
_NEUTRAL = {"hold", "neutral", "market perform", "equal weight", "watch", "sector perform"}

# How firmly it was said. A HOLD is a real but weak statement; an unrecognised label
# (a daily newsletter, "UNDER REVIEW") is barely a statement at all.
_CONVICTION = {"strong buy": 5, "strong sell": 5, "buy": 4, "sell": 4,
               "add": 3, "reduce": 3, "accumulate": 3, "outperform": 3, "underperform": 3,
               "overweight": 3, "underweight": 3, "subscribe": 3, "ipo subscribe": 3,
               "hold": 2, "neutral": 2, "watch": 2, "market perform": 2, "equal weight": 2}


def _norm(rating: str | None) -> str:
    return " ".join(str(rating or "").lower().split())


def direction_from_rating(rating: str | None) -> str:
    """Any source's rating -> `bullish` | `bearish` | `neutral`.

    An unrecognised rating is NEUTRAL, never a guess. A daily newsletter tagged
    'DAILY NOTE' is not a bullish call, and treating it as one would quietly stuff the
    ledger with calls nobody made.
    """
    r = _norm(rating)
    if r in _BULLISH:
        return "bullish"
    if r in _BEARISH:
        return "bearish"
    if r in _NEUTRAL:
        return "neutral"
    return "neutral"


def conviction_from_rating(rating: str | None) -> int:
    """1-5. Unknown ratings score 1 so they carry almost no weight in any hit rate."""
    return _CONVICTION.get(_norm(rating), 1)


# I/B/E/S's consensus scale: every broker's ratings mapped onto 1 (strongest buy) .. 5 (strongest
# sell). Indian brokers rate Buy / Accumulate-Add / Hold-Neutral / Reduce / Sell, which lands
# one-to-one; Buy is their top rating, so it scores 1 as I/B/E/S maps a contributor's top rating.
_SCORE = {"strong buy": 1, "buy": 1, "long": 1,
          "add": 2, "accumulate": 2, "outperform": 2, "overweight": 2, "subscribe": 2,
          "ipo subscribe": 2,
          "hold": 3, "neutral": 3, "market perform": 3, "equal weight": 3, "sector perform": 3,
          "watch": 3,
          "reduce": 4, "underperform": 4, "underweight": 4,
          "sell": 5, "strong sell": 5, "avoid": 5, "short": 5}


def rating_score(rating: str | None) -> int | None:
    """1 (buy) .. 5 (sell), or None when the label is not a rating at all.

    Unlike `direction_from_rating`, an unknown label is NOT neutral here: in a consensus, reading a
    'Results Update' or 'Daily Note' as HOLD would invent a broker opinion nobody gave."""
    return _SCORE.get(_norm(rating))


# Columns added after v1 shipped. `CREATE TABLE IF NOT EXISTS` does NOT alter a table that
# already exists, so without this every ledger written before the column existed would
# raise `no such column` and become unreadable. Additive-only by design: a column may be
# added, never renamed or dropped, because old rows are immutable and must stay readable.
# `schema_version` defaults to 0 here rather than 1 - a row written before versioning
# existed genuinely predates v1, and saying so honestly is what makes upcasting possible.
_MIGRATIONS = {
    "predictions": [("schema_version", "INTEGER NOT NULL DEFAULT 0"),
                    ("superseded_by", "TEXT")],
}


def _migrate(con: sqlite3.Connection) -> None:
    for table, columns in _MIGRATIONS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if not have:
            continue                                  # table not created yet; schema handles it
        for name, ddl in columns:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                logger.info(f"ledger migration: added {table}.{name}")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    _migrate(con)
    return con


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_prediction_id(as_of_date: str, symbol: str, payload: dict | None = None) -> str:
    """A readable, time-sortable id: `2026-09-27-HINDALCO-a3f9b2c1`.

    Date first so sorting by id sorts by time. The hash covers the payload so a broker
    and the agent - or two different brokers - can call the same stock on the same day
    without colliding.
    """
    blob = json.dumps(payload or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{as_of_date}|{symbol}|{blob}".encode()).hexdigest()[:8]
    return f"{as_of_date}-{symbol.upper()}-{digest}"


def save_predictions(preds: list[Prediction]) -> int:
    """Append predictions. A duplicate id RAISES - it is never silently overwritten.

    The plain INSERT is the whole point. An `ON CONFLICT DO UPDATE` here would let a
    later, retroactively-adjusted price quietly replace the entry price a human actually
    saw, and every score computed afterwards would be fiction.
    """
    if not preds:
        return 0
    rows = []
    for p in preds:
        p.prediction_id = p.prediction_id or make_prediction_id(
            p.as_of_date, p.symbol, {**p.payload, "source": p.source})
        p.created_at = p.created_at or _now()
        p.as_of_ts = p.as_of_ts or p.created_at
        d = asdict(p)
        for k in _JSON_COLS:
            d[k] = json.dumps(d[k], default=str)
        rows.append(tuple(d[c] for c in _PRED_COLS))
    with _connect() as con:
        con.executemany(
            f"INSERT INTO predictions ({', '.join(_PRED_COLS)}) "
            f"VALUES ({', '.join('?' * len(_PRED_COLS))})", rows)
    return len(rows)


def _to_prediction(row: sqlite3.Row) -> Prediction:
    d = dict(row)
    for k in _JSON_COLS:
        try:
            d[k] = json.loads(d[k] or "{}")
        except json.JSONDecodeError:
            d[k] = {}
    return Prediction(**d)


def load_predictions(source: str | None = None, symbol: str | None = None,
                     limit: int | None = None,
                     include_superseded: bool = False) -> list[Prediction]:
    """Live rows by default. A corrected call still exists, but a scorecard must not count
    the same judgement twice, so superseded rows are hidden unless asked for."""
    sql = f"SELECT {', '.join(_PRED_COLS)} FROM predictions"
    where, args = [], []
    if not include_superseded:
        where.append("superseded_by IS NULL")
    if source:
        where.append("source = ?")
        args.append(source)
    if symbol:
        where.append("symbol = ?")
        args.append(symbol.upper())
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY as_of_date, prediction_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [_to_prediction(r) for r in con.execute(sql, args)]


def save_outcomes(outs: list[Outcome]) -> int:
    """Upsert outcomes. Unlike predictions these are DERIVED, so recomputing is correct:
    a better abnormal-return method should be able to rebuild every score without
    touching the calls being scored."""
    if not outs:
        return 0
    rows = []
    for o in outs:
        o.computed_at = o.computed_at or _now()
        d = asdict(o)
        rows.append(tuple(d[c] for c in _OUT_COLS))
    updates = ", ".join(f"{c}=excluded.{c}" for c in _OUT_COLS
                        if c not in ("prediction_id", "horizon_days"))
    with _connect() as con:
        con.executemany(
            f"INSERT INTO outcomes ({', '.join(_OUT_COLS)}) "
            f"VALUES ({', '.join('?' * len(_OUT_COLS))}) "
            f"ON CONFLICT(prediction_id, horizon_days) DO UPDATE SET {updates}", rows)
    return len(rows)


def load_outcomes(prediction_id: str | None = None,
                  horizon_days: int | None = None) -> list[Outcome]:
    sql = f"SELECT {', '.join(_OUT_COLS)} FROM outcomes"
    where, args = [], []
    if prediction_id:
        where.append("prediction_id = ?")
        args.append(prediction_id)
    if horizon_days is not None:
        where.append("horizon_days = ?")
        args.append(horizon_days)
    if where:
        sql += " WHERE " + " AND ".join(where)
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [Outcome(**dict(r)) for r in con.execute(sql + " ORDER BY horizon_days", args)]


def calendar_days_for(horizon_days: int) -> int:
    """Rough calendar-day span of N trading days - five trading days per seven calendar.

    Deliberately slightly EAGER (20 trading days measured out at 34 calendar days; this
    returns 28). Being early costs one wasted fetch that comes back 'open' and retries
    tomorrow; being late delays grading. `outcomes.py` does the authoritative check by
    walking the real price series, so this only needs to avoid obviously pointless work.
    """
    return horizon_days * 7 // 5


def pending(horizon_days: int, today: date | None = None) -> list[Prediction]:
    """Predictions old enough to grade at this horizon and not yet conclusively graded.

    An existing outcome with label 'open' does NOT count as graded - 'open' means the
    horizon had not arrived when we last looked, so it must be retried.
    """
    today = today or date.today()
    cutoff = (today - timedelta(days=calendar_days_for(horizon_days))).isoformat()
    sql = f"""SELECT {', '.join('p.' + c for c in _PRED_COLS)} FROM predictions p
              WHERE p.as_of_date <= ?
                AND p.superseded_by IS NULL
                AND NOT EXISTS (SELECT 1 FROM outcomes o
                                WHERE o.prediction_id = p.prediction_id
                                  AND o.horizon_days = ?
                                  AND o.label != 'open')
              ORDER BY p.as_of_date, p.prediction_id"""
    with _connect() as con:
        con.row_factory = sqlite3.Row
        return [_to_prediction(r) for r in con.execute(sql, (cutoff, horizon_days))]


def supersede(old_id: str, replacement: Prediction, reason: str) -> str:
    """Correct a wrong prediction WITHOUT deleting it.

    The third invariant of an immutable ledger is non-destructive correction. Extraction
    is fallible - a model can read "we would avoid this" as a buy - and you must be able
    to fix that. But deleting the original would erase the fact that the system recorded
    the wrong call on that day, which is itself worth knowing, and a ledger you can
    quietly edit proves nothing about anything.

    So: the old row stays, stamped with what replaced it and why; the correction is a new
    append. Both are readable with `include_superseded=True`.
    """
    with _connect() as con:
        con.row_factory = sqlite3.Row
        row = con.execute(f"SELECT {', '.join(_PRED_COLS)} FROM predictions "
                          f"WHERE prediction_id = ?", (old_id,)).fetchone()
        if row is None:
            raise KeyError(f"cannot supersede unknown prediction {old_id!r}")
        old = _to_prediction(row)

    replacement.prediction_id = replacement.prediction_id or make_prediction_id(
        replacement.as_of_date, replacement.symbol,
        {**replacement.payload, "source": replacement.source, "supersedes": old_id})
    replacement.provenance = {**replacement.provenance, "supersedes": old_id}
    save_predictions([replacement])

    prov = {**old.provenance, "superseded_reason": reason,
            "superseded_at": _now(), "superseded_by": replacement.prediction_id}
    with _connect() as con:
        con.execute("UPDATE predictions SET superseded_by = ?, provenance = ? "
                    "WHERE prediction_id = ?",
                    (replacement.prediction_id, json.dumps(prov, default=str), old_id))
    logger.info(f"superseded {old_id} -> {replacement.prediction_id}: {reason}")
    return replacement.prediction_id


def counts() -> dict:
    """Headline numbers for `refresh_data.py --status` and the scorecard."""
    with _connect() as con:
        preds = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_source = dict(con.execute(
            "SELECT source, COUNT(*) FROM predictions GROUP BY source").fetchall())
        graded = con.execute(
            "SELECT COUNT(*) FROM outcomes WHERE label NOT IN ('open','unavailable')").fetchone()[0]
        awaiting = con.execute("SELECT COUNT(*) FROM outcomes WHERE label='open'").fetchone()[0]
    return {"predictions": preds, "by_source": by_source,
            "graded_outcomes": graded, "open_outcomes": awaiting}
