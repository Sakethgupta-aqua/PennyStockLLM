"""Event signals - the disclosures that move Indian stocks, dated to the moment the market learned them.

Insider trades, promoter pledges, credit-rating actions and a handful of severe announcements are
typed into `Event`s, stored, graded in the ledger (one call per company, type and day) and - for the
severe ones - kept out of the screener for a year. Every rule here was checked against real NSE data on
2026-10-05; the comments say what broke the obvious version:

  * NSE's own announcement categories are not safe to act on: all 6 "auditor resignations" in two years
    of filings for 119 companies were auditors of a SUBSIDIARY; all 33 "NCLT" hits were merger schemes;
    banks report a BORROWER's fraud under the same word. So severity needs the company's own auditor,
    its own insolvency, its own fraud.
  * Insider trades come from two feeds: the old `corporates-pit` (to April 2026) and the new
    `corporates-pit-gg`, whose trades sit in one small XBRL file per disclosure (from May 2026).
  * Rating direction comes from NSE's structured rating feed (`credit-rating-sdd-reg30`, from ~September
    2025); rating announcements carry no direction in 435 of 571 titles.

Spec: docs/superpowers/specs/2026-10-05-event-signals-design.md
"""
from __future__ import annotations

import calendar
import json
import logging
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from urllib.parse import quote

import pandas as pd

from pennystock.config import settings
from pennystock.data.ledger import Prediction, load_predictions, make_prediction_id, save_predictions
from pennystock.data.outcomes import _sessions, actionable_date, fetch_closes, history_period, scorecard
from pennystock.data.sectors import profile_for

logger = logging.getLogger(__name__)

RULES_VERSION = 3   # 2: tightened after the full 1,486-company run; 3: INC/default split, lifted insolvency

# type -> (severity, direction, conviction). "veto" also keeps the stock out of the screener.
TYPES: dict[str, tuple[str, str | None, int | None]] = {
    "insider_buy": ("signal", "bullish", 3),
    "insider_sell": ("signal", "bearish", 2),
    "pledge_created": ("signal", "bearish", 2),
    "pledge_invoked": ("signal", "bearish", 4),
    "pledge_released": ("info", None, None),
    "rating_downgrade": ("signal", "bearish", 3),
    "rating_upgrade": ("signal", "bullish", 2),
    "rating_inc": ("signal", "bearish", 2),     # mostly administrative (see _rating_type); graded, not vetoed
    "rating_default": ("veto", "bearish", 4),   # a D or C grade: the agency sees default
    "rating_other": ("info", None, None),
    "auditor_exit": ("veto", "bearish", 4),
    "auditor_exit_subsidiary": ("info", None, None),
    "insolvency": ("veto", "bearish", 4),
    "insolvency_lifted": ("info", None, None),  # stayed, set aside, withdrawn or closed - lifts the veto
    "fraud": ("veto", "bearish", 4),
    "enforcement": ("info", None, None),
}

# who counts as an insider for a buy or sell signal; only promoters pledge their holding
INSIDERS = {"promoters", "promoter", "promoter group", "director", "key managerial personnel"}
PLEDGERS = {"promoters", "promoter", "promoter group"}
MODE_TYPES = {"market purchase": "insider_buy", "market sale": "insider_sell",
              "pledge creation": "pledge_created", "invocation of pledge": "pledge_invoked",
              "revokation of pledge": "pledge_released", "revocation of pledge": "pledge_released"}

_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y")


@dataclass
class Event:
    event_key: str              # stable across re-fetches and revised filings
    symbol: str
    event_type: str
    severity: str               # "veto" | "signal" | "info"
    direction: str | None
    conviction: int | None
    signal_at: str              # exchange broadcast time - when the market learned it ("YYYY-MM-DDTHH:MM")
    event_date: str | None      # trade, rating or filing date
    source: str
    source_url: str | None = None
    details: dict = field(default_factory=dict)


def _norm(v) -> str:
    return " ".join(str(v or "").split())


def _when(v) -> datetime | None:
    s = _norm(v)
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _day(v) -> str | None:
    w = _when(v)
    return w.date().isoformat() if w else None


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _event(key: str, symbol: str, etype: str, when: datetime, event_date: str | None, source: str,
           url: str | None, details: dict) -> Event:
    severity, direction, conviction = TYPES[etype]
    return Event(event_key=key, symbol=symbol, event_type=etype, severity=severity, direction=direction,
                 conviction=conviction, signal_at=when.strftime("%Y-%m-%dT%H:%M"), event_date=event_date,
                 source=source, source_url=url, details=details)


# ------------------------------------------------------------------ insider trades and pledges
def insider_events(rows) -> list[Event]:
    """Insider rows (old-feed shape) -> events: one per person, broadcast day and mode, values summed.
    Buys and sells count only for promoters, directors and KMP, and only from Rs 10 lakh; pledges only
    for promoters. A row whose broadcast time cannot be read is skipped - never dated "today"."""
    groups: dict[tuple, list[tuple[datetime, dict]]] = defaultdict(list)
    for r in rows:
        etype = MODE_TYPES.get(_norm(r.get("acqMode")).lower())
        if not etype:
            continue
        allowed = PLEDGERS if etype.startswith("pledge") else INSIDERS
        when, sym, person = _when(r.get("date")), _norm(r.get("symbol")).upper(), _norm(r.get("acqName"))
        if _norm(r.get("personCategory")).lower() not in allowed or not (when and sym and person):
            continue
        groups[(sym, person, when.date().isoformat(), etype)].append((when, r))
    out = []
    for (sym, person, day, etype), items in groups.items():
        value = sum(_num(r.get("secVal")) or 0 for _, r in items)
        if etype in ("insider_buy", "insider_sell") and value < settings.events_min_value_rs:
            continue
        first = min(w for w, _ in items)
        r0 = items[0][1]
        details = {"person": person, "category": _norm(r0.get("personCategory")), "mode": _norm(r0.get("acqMode")),
                   "value_rs": value, "shares": sum(_num(r.get("secAcq")) or 0 for _, r in items),
                   "holding_before_pct": _num(r0.get("befAcqSharesPer")),
                   "holding_after_pct": _num(items[-1][1].get("afterAcqSharesPer")),
                   "disclosures": [str(r.get("did")) for _, r in items]}
        out.append(_event(f"pit:{sym}:{_slug(person)}:{day}:{etype}", sym, etype, first,
                          _day(r0.get("acqfromDt")) or day, str(r0.get("source") or "nse_pit"),
                          r0.get("xbrl"), details))
    return out


_XBRL_TAG = re.compile(r"<(?:[\w\-]+:)?(\w+)\s+[^>]*contextRef=\"([^\"]+)\"[^>]*>([^<]*)</", re.S)
_XBRL_FIELDS = {"acqName": "NameOfThePerson", "personCategory": "CategoryOfPerson",
                "acqMode": "ModeOfAcquisitionOrDisposal",
                "tdpTransactionType": "SecuritiesAcquiredOrDisposedTransactionType",
                "secVal": "SecuritiesAcquiredOrDisposedValueOfSecurity",
                "secAcq": "SecuritiesAcquiredOrDisposedNumberOfSecurity",
                "acqfromDt": "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate",
                "acqtoDt": "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate",
                "intimDt": "DateOfIntimationToCompany"}
_XBRL_PCT = {"befAcqSharesPer": "SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding",
             "afterAcqSharesPer": "SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding"}


def pit_xbrl_rows(xml: str, index_row: dict) -> list[dict]:
    """A new-format disclosure (`corporates-pit-gg` index row + its XBRL file) as old-feed rows, one per
    disclosed trade, so both feeds go through the same rules. XBRL holdings are fractions; old rows say
    percent, so they are scaled to percent."""
    contexts: dict[str, dict[str, str]] = defaultdict(dict)
    for tag, ctx, value in _XBRL_TAG.findall(xml or ""):
        contexts[ctx].setdefault(tag, value.strip())
    rows = []
    for ctx, tags in contexts.items():
        if "CategoryOfPerson" not in tags:
            continue
        row = {k: tags.get(v) for k, v in _XBRL_FIELDS.items()}
        for k, v in _XBRL_PCT.items():
            pct = _num(tags.get(v))
            row[k] = round(pct * 100, 4) if pct is not None else None
        row.update(symbol=index_row.get("symbol"), date=index_row.get("broadcastDateTime"),
                   did=f"{index_row.get('appId')}:{ctx}", xbrl=index_row.get("xmlFileName"),
                   source="nse_pit_xbrl")
        rows.append(row)
    return rows


def apply_clusters(events: list[Event], days: int | None = None) -> list[Event]:
    """An insider buy is a cluster (conviction 4) when at least two different insiders bought the same
    stock in the `days` ending on its signal day - several insiders putting their own money in at once
    says more than one."""
    days = days or settings.events_cluster_days
    by_symbol: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        if e.event_type == "insider_buy":
            by_symbol[e.symbol].append(e)
    bumped = set()
    for buys in by_symbol.values():
        for e in buys:
            end = date.fromisoformat(e.signal_at[:10])
            start = (end - timedelta(days=days - 1)).isoformat()
            people = {x.details.get("person") for x in buys if start <= x.signal_at[:10] <= end.isoformat()}
            if len(people) >= 2:
                bumped.add(e.event_key)
    return [replace(e, conviction=4) if e.event_key in bumped else e for e in events]


# ------------------------------------------------------------------ credit ratings
_INC = re.compile(r"not\s+co-?\s*operat", re.I)
_WITHDRAWN = re.compile(r"withdraw", re.I)
_AGENCY_PREFIX = re.compile(r"^\s*\[?(crisil|care|icra|ind|ivr|acuite|bwr|brickwork|infomerics|india\s+ratings)\]?\.?\s*", re.I)
_DEFAULT_GRADE = re.compile(r"^(D|C)(?![A-Za-z0-9+])")


def _rating_type(r: dict) -> str:
    """Measured on the live feed: most "Issuer Not Cooperating" ratings are administrative - a company that
    repaid its loans stops paying the agency, which must then move it to INC and cut it to non-investment
    grade (Vijaya Diagnostic, cash-rich, went to "Crisil B(INC)"); many are then withdrawn. Only a D or C
    grade says the agency sees default."""
    text = f"{r.get('brieFDetails') or ''} {r.get('creditRating') or ''}"
    if _WITHDRAWN.search(text):
        return "rating_other"
    if _DEFAULT_GRADE.match(_AGENCY_PREFIX.sub("", _norm(r.get("creditRating")))):
        return "rating_default"
    if _INC.search(text):
        return "rating_inc"
    action = _norm(r.get("currentAction")).lower()
    return {"downgrade": "rating_downgrade", "upgrade": "rating_upgrade"}.get(action, "rating_other")


def rating_events(rows) -> list[Event]:
    """Structured rating rows -> one event per (company, agency, broadcast day, type): one action often
    covers several instruments, which is still one piece of news. "Issuer Not Cooperating" outranks the
    action it rides on - the agency is saying it can no longer see inside the company."""
    groups: dict[tuple, list[tuple[datetime, dict]]] = defaultdict(list)
    for r in rows:
        sym, when, agency = _norm(r.get("symbol")).upper(), _when(r.get("broadcastDateTime")), _norm(r.get("creditAgencyName"))
        if sym and when and agency:
            groups[(sym, agency, when.date().isoformat(), _rating_type(r))].append((when, r))
    out = []
    for (sym, agency, day, etype), items in groups.items():
        r0 = items[0][1]
        details = {"agency": agency, "rating": _norm(r0.get("creditRating")), "outlook": _norm(r0.get("outlook")),
                   "action": _norm(r0.get("currentAction")), "text": _norm(r0.get("brieFDetails"))[:300],
                   "instruments": sorted({_norm(r.get("ratingAssigned")) for _, r in items})}
        out.append(_event(f"rating:{sym}:{_slug(agency)}:{day}:{etype}", sym, etype, min(w for w, _ in items),
                          _day(r0.get("dateOfCurrentCredit")) or day, "nse_rating_sdd",
                          r0.get("detailsOfRatingLink"), details))
    return out


# ------------------------------------------------------------------ severe announcements
# Each rule below was tightened after a full two-year run over 1,486 companies found its false alarms.
_AUDITOR = re.compile(r"(resign|remov|cessation|vacat)\w*\s+(of|by|as|from)?\b[^;()]{0,100}?statutory\s+auditor"
                      r"|statutory\s+auditors?[^;()]{0,80}?\b(resign|tendered\s+(their|its|his|her)\s+resignation"
                      r"|ceased|cessation|removed)"
                      r"|adverse\s+opinion|disclaimer\s+of\s+opinion", re.I | re.S)
# "...Statutory Auditors of Nodwin Gaming Pvt. Ltd., a material subsidiary" - the auditor of a SUBSIDIARY;
# names (with full stops) can sit between the two words, and NSE text sometimes drops the space ("ofAmara")
_SUB_AUDITOR = re.compile(r"auditors?\s*(of|in|by|for)[^;]{0,180}?subsidiar"
                          # "...a Step down Subsidiary of the Company, that the ATPL ... Statutory Auditors of ATPL"
                          r"|subsidiar[^;]{0,300}?statutory\s+auditors?\s+of\s+(?!the\s+(listed\s+)?company)", re.I | re.S)
_OWN_AUDITOR = re.compile(r"auditors?\s+of\s+the\s+(listed\s+)?(company|bank)\b(?!\s*'?s\b)", re.I)
_TERM_END = re.compile(r"(complet|expir|end)\w*\s+of\s+(their|its|the|his|her)\s+(term|tenure)", re.I)
_CIRP_DESC = re.compile(r"^corporate\s+insolvency", re.I)
_INSOLVENCY = re.compile(r"(initiat|commenc|admi)\w*.{0,80}(insolvency|CIRP).{0,60}against\s+the\s+company"
                         r"|(interim\s+)?resolution\s+professional|moratorium\s+(under|declared|has\s+been)"
                         r"|default\s+in\s+(the\s+)?(payment|repayment|servicing)\s+of\s+(interest|principal|dues|debt)"
                         r"|delay\s+in\s+(the\s+)?(payment|servicing)\s+of\s+(interest|principal)"
                         r"|winding[\s-]+up\s+(order|petition\s+(has\s+been\s+)?admitted)", re.I | re.S)
_NOT_INSOLVENCY = re.compile(r"scheme\s+of|amalgamat|\bmerger|demerger|convened\s+meeting|motion\s+petition"
                             r"|arrangement|operational\s+creditor|\bnil\b|no\s+default", re.I)
# buying an insolvent company, or claiming against one, is not being insolvent
_ACQUIRER = re.compile(r"letter\s+of\s+intent|\bLOI\b|resolution\s+(plan|applicant)|successful\s+resolution"
                       r"|filing\s+of\s+claim|claims?\s+(against|with|filed)|submitted\s+by\s+the\s+consortium"
                       r"|undergoing\s+(the\s+)?corporate\s+insolvency|financial\s+creditor", re.I)
_CIRP_CLOSED = re.compile(r"closure\s+of\s+(the\s+)?(corporate\s+insolvency|cirp)"
                          r"|discharge\s+of\s+the\s+(interim\s+)?resolution\s+professional"
                          r"|withdrawal\s+of\s+(the\s+)?(cirp|insolvency)|cirp\s+(has\s+been\s+)?(closed|withdrawn|set\s+aside)"
                          r"|grant\w*\s+(of\s+)?(a\s+)?stay|stay(ed)?\s+(on|of|against)\s+the\s+(impugned\s+)?order"
                          r"|set\s+aside", re.I)
_INSOLVENCY_CONTEXT = re.compile(r"insolvency|\bcirp\b|\bnclt\b|\bnclat\b|resolution\s+professional", re.I)
# a creditor's application that has not been heard yet is a lawsuit, not an insolvency
_PETITION_ONLY = re.compile(r"(filing|filed)\s+(of\s+)?(an\s+)?application|application\s+(has\s+been\s+)?filed"
                            r"|yet\s+to\s+come\s+up|petition\s+(has\s+been\s+)?filed", re.I)
_ADMITTED = re.compile(r"\badmi(t|ssion)", re.I)
_FRAUD = re.compile(r"forensic\s+audit|\bfraud\b|serious\s+fraud\s+investigation|\bSFIO\b"
                    r"|enforcement\s+directorate[^.;]{0,80}search|search[^.;]{0,80}enforcement\s+directorate", re.I)
# NSE's category name "Fraud/Default/Arrest" says nothing by itself - banks file borrowers' frauds under it
_CATEGORY_LABEL = re.compile(r"fraud\s*/\s*default\s*/\s*arrest", re.I)
_NOT_FRAUD = re.compile(r"borrow(al|er)|npa\s+account|loan\s+account|account\s+of\s+m/?s"
                        r"|classified\s+as\s+(a\s+)?fraud|declared\s+(as\s+)?(a\s+)?fraud"   # a lender reporting a borrower
                        r"|closure\s+of|conclu(sion|ded)"                                   # it is over
                        r"|fraud\s+(committed\s+)?by\s+(an?\s+|its\s+|the\s+|sales\s+|some\s+)?(former\s+)?(employee|staff)"
                        r"|suffered\s+a\s+(financial\s+)?loss|hack|phish|cyber|\bdupe|impersonat|spoof", re.I)
_NO_FRAUD_DESC = re.compile(r"^(press\s+release|analysts?|investor\s+meet)", re.I)
_NEWS_QUERY = re.compile(r"^news\s+verification", re.I)     # the exchange asking about a press story
_ENFORCEMENT = re.compile(r"show[\s-]+cause|adjudicat|prosecution|penalt(y|ies)\s+(imposed|levied|of)"
                          r"|(sebi|rbi)\s+(order|notice|summons|enforcement)|settlement\s+order"
                          r"|whistle[\s-]*blower|search\s+(and\s+seizure|operation|proceedings)", re.I)
_NOT_ENFORCEMENT = re.compile(r"not\s+debarred|affirmation", re.I)


def _announcement_type(desc: str, text: str) -> str | None:
    blob = f"{desc} {text}"
    if _NEWS_QUERY.search(desc):
        return None
    if _AUDITOR.search(blob) and not _TERM_END.search(blob):
        own = _OWN_AUDITOR.search(blob) or not _SUB_AUDITOR.search(blob)
        return "auditor_exit" if own else "auditor_exit_subsidiary"
    if _CIRP_CLOSED.search(blob) and (_CIRP_DESC.search(desc) or _INSOLVENCY_CONTEXT.search(blob)):
        return "insolvency_lifted"
    if _CIRP_DESC.search(desc):
        return "enforcement" if _PETITION_ONLY.search(blob) and not _ADMITTED.search(blob) else "insolvency"
    if _INSOLVENCY.search(blob) and not _NOT_INSOLVENCY.search(blob) and not _ACQUIRER.search(blob):
        return "insolvency"
    plain = _CATEGORY_LABEL.sub(" ", blob)
    if not _NO_FRAUD_DESC.search(desc) and _FRAUD.search(plain) and not _NOT_FRAUD.search(plain):
        return "fraud"
    if (_ENFORCEMENT.search(blob) or _CATEGORY_LABEL.search(blob)) and not _NOT_ENFORCEMENT.search(blob):
        return "enforcement"
    return None


def reconcile(events: list[Event], days: int = 7) -> list[Event]:
    """A bare "Resignation of Statutory Auditor" filing that names no one, in the same week as a filing
    that names a SUBSIDIARY's auditor, is the same story told twice - the subsidiary one."""
    subs = defaultdict(list)
    for e in events:
        if e.event_type == "auditor_exit_subsidiary":
            subs[e.symbol].append(date.fromisoformat(e.signal_at[:10]))
    out = []
    for e in events:
        blob = f"{e.details.get('subject', '')} {e.details.get('text', '')} {e.details.get('letter', '')}"
        if e.event_type == "auditor_exit" and not _OWN_AUDITOR.search(blob) and any(
                abs((date.fromisoformat(e.signal_at[:10]) - d).days) <= days for d in subs[e.symbol]):
            severity, direction, conviction = TYPES["auditor_exit_subsidiary"]
            e = replace(e, event_type="auditor_exit_subsidiary", severity=severity, direction=direction,
                        conviction=conviction)
        out.append(e)
    return out


def announcement_event(row: dict) -> Event | None:
    """One NSE announcement -> a severe or notable event, or None. Strict on purpose: each of these can
    keep a stock out of the screen for a year, so the measured false alarms are excluded by rule."""
    desc, text = _norm(row.get("desc")), _norm(row.get("attchmntText"))
    etype = _announcement_type(desc, text)
    when = _when(row.get("sort_date")) or _when(row.get("an_dt"))
    seq, sym = str(row.get("seq_id") or ""), _norm(row.get("symbol")).upper()
    if not (etype and when and seq.isdigit() and sym):
        return None
    return _event(f"ann:{seq}", sym, etype, when, when.date().isoformat(), "nse_announcement",
                  row.get("attchmntFile"), {"subject": desc, "text": text[:400]})


# ------------------------------------------------------------------ storage (market.db)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_signals (
    event_key     TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    severity      TEXT NOT NULL CHECK (severity IN ('veto', 'signal', 'info')),
    direction     TEXT,
    conviction    INTEGER,
    signal_at     TEXT NOT NULL,
    event_date    TEXT,
    source        TEXT NOT NULL,
    source_url    TEXT,
    details       TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    rules_version INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_event_signals_symbol ON event_signals(symbol, signal_at);
CREATE TABLE IF NOT EXISTS event_ingest_state (
    source     TEXT NOT NULL,
    period     TEXT NOT NULL,
    rows       INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (source, period)
) STRICT;
"""
_COLS = ("event_key", "symbol", "event_type", "severity", "direction", "conviction", "signal_at",
         "event_date", "source", "source_url", "details")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row(e: Event) -> tuple:
    d = asdict(e)
    d["details"] = json.dumps(d["details"], sort_keys=True, default=str)
    return tuple(d[c] for c in _COLS)


def _from_row(r) -> Event:
    d = dict(zip(_COLS, r))
    d["details"] = json.loads(d["details"] or "{}")
    return Event(**d)


def save_events(events: list[Event]) -> int:
    """Insert new events (an event already stored is left alone) and re-check insider clusters for the
    companies touched - a second insider buying today can make last week's buy part of a cluster.
    Returns how many events were new."""
    if not events:
        return 0
    with _connect() as con:
        before = con.total_changes
        con.executemany(f"INSERT OR IGNORE INTO event_signals ({', '.join(_COLS)}, ingested_at, rules_version) "
                        f"VALUES ({', '.join('?' * len(_COLS))}, ?, ?)",
                        [(*_row(e), _now(), RULES_VERSION) for e in events])
        added = con.total_changes - before
    _recluster({e.symbol for e in events if e.event_type == "insider_buy"})
    return added


def _recluster(symbols: set[str]) -> None:
    if not symbols:
        return
    for sym in symbols:
        buys = [replace(e, conviction=TYPES["insider_buy"][2]) for e in load_events(sym, types={"insider_buy"})]
        with _connect() as con:
            con.executemany("UPDATE event_signals SET conviction = ? WHERE event_key = ?",
                            [(e.conviction, e.event_key) for e in apply_clusters(buys)])


def load_events(symbol: str | None = None, since: str | None = None, types: set[str] | None = None) -> list[Event]:
    """Stored events, oldest first; `since` compares against the broadcast date (YYYY-MM-DD)."""
    sql, where, args = f"SELECT {', '.join(_COLS)} FROM event_signals", [], []
    if symbol:
        where.append("symbol = ?")
        args.append(symbol.upper())
    if since:
        where.append("signal_at >= ?")
        args.append(since)
    if types:
        where.append(f"event_type IN ({', '.join('?' * len(types))})")
        args.extend(sorted(types))
    if where:
        sql += " WHERE " + " AND ".join(where)
    with _connect() as con:
        return [_from_row(r) for r in con.execute(sql + " ORDER BY signal_at, event_key", args)]


def retype() -> int:
    """Re-derive every stored event under the current rules (no refetch: the source text is in
    `details`). An announcement that no longer qualifies is removed. Returns rows changed."""
    changed = 0
    with _connect() as con:
        rows = con.execute(f"SELECT {', '.join(_COLS)}, rules_version FROM event_signals").fetchall()
    for r in rows:
        e, version = _from_row(r[:-1]), r[-1]
        etype = e.event_type
        if e.event_key.startswith("ann:"):
            etype = _announcement_type(e.details.get("subject", ""),
                                       f"{e.details.get('text', '')} {e.details.get('letter', '')}")
        with _connect() as con:
            if etype is None:
                con.execute("DELETE FROM event_signals WHERE event_key = ?", (e.event_key,))
                changed += 1
                continue
            severity, direction, conviction = TYPES[etype]
            if etype == e.event_type == "insider_buy":
                conviction = e.conviction                 # clusters are re-derived by _recluster below
            blob = f"{e.details.get('subject', '')} {e.details.get('text', '')} {e.details.get('letter', '')}"
            if etype == "auditor_exit" and e.event_type == "auditor_exit_subsidiary" and not _OWN_AUDITOR.search(blob):
                etype, severity, direction, conviction = e.event_type, e.severity, e.direction, e.conviction
                # a bare filing reconciled with that week's subsidiary filing stays reconciled
            if (etype, severity, direction, conviction, version) != (e.event_type, e.severity, e.direction,
                                                                      e.conviction, RULES_VERSION):
                con.execute("UPDATE event_signals SET event_type = ?, severity = ?, direction = ?, conviction = ?, "
                            "rules_version = ? WHERE event_key = ?",
                            (etype, severity, direction, conviction, RULES_VERSION, e.event_key))
                changed += 1
    for sym in {e.symbol for e in load_events(types={"auditor_exit", "auditor_exit_subsidiary"})}:
        mine = load_events(sym, types={"auditor_exit", "auditor_exit_subsidiary"})
        moved = [e for e, r in zip(reconcile(mine), mine) if e.event_type != r.event_type]
        with _connect() as con:
            con.executemany("UPDATE event_signals SET event_type = ?, severity = ?, direction = ?, conviction = ? "
                            "WHERE event_key = ?",
                            [(e.event_type, e.severity, e.direction, e.conviction, e.event_key) for e in moved])
        changed += len(moved)
    _recluster({e.symbol for e in load_events(types={"insider_buy"})})
    return changed


# ------------------------------------------------------------------ ingestion
# NSE's feeds and the months each one carries (measured 2026-10-05); outside these a feed answers [].
FEEDS = {"pit": ("corporates-pit", date(2000, 1, 1), date(2026, 4, 30)),
         "pit_gg": ("corporates-pit-gg", date(2026, 5, 1), None),
         "ratings": ("credit-rating-sdd-reg30", date(2025, 9, 1), None)}


def months(since: date, until: date) -> list[tuple[int, int]]:
    out, y, m = [], since.year, since.month
    while (y, m) <= (until.year, until.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _state(source: str, period: str) -> tuple[int, str] | None:
    with _connect() as con:
        return con.execute("SELECT rows, fetched_at FROM event_ingest_state WHERE source = ? AND period = ?",
                           (source, period)).fetchone()


def _mark(source: str, period: str, rows: int) -> None:
    with _connect() as con:
        con.execute("INSERT OR REPLACE INTO event_ingest_state VALUES (?, ?, ?, ?)", (source, period, rows, _now()))


def _data(answer) -> list[dict] | None:
    """NSE answers a list or {"data": [...]}; anything else (None, an error page) is no answer."""
    if isinstance(answer, dict):
        answer = answer.get("data")
    return answer if isinstance(answer, list) else None


def _pause() -> None:
    if settings.filings_delay_s:
        time.sleep(settings.filings_delay_s)


def ingest_month(source: str, year: int, month: int, api, get=None, universe: set[str] | None = None) -> int | None:
    """One feed, one calendar month, market-wide. Returns new events stored; 0 when the month was already
    done; None when NSE did not answer - that month stays undone. The current month is never marked done."""
    period = f"{year}-{month:02d}"
    if _state(source, period):
        return 0
    path, _, _ = FEEDS[source]
    last = calendar.monthrange(year, month)[1]
    rows = _data(api(f"{path}?index=equities&from_date=01-{month:02d}-{year}&to_date={last:02d}-{month:02d}-{year}"))
    if rows is None:
        logger.warning(f"{source} {period}: no answer from NSE - will retry next run")
        return None
    if source == "ratings":
        events = rating_events(rows)
    elif source == "pit":
        events = insider_events(rows)
    else:                                         # pit_gg: one XBRL file per disclosure
        flat = []
        for r in rows:
            if universe is not None and str(r.get("symbol", "")).upper() not in universe:
                continue
            body = get(r.get("xmlFileName")) if get and r.get("xmlFileName") else None
            _pause()
            if body is None:
                logger.warning(f"pit_gg {period}: XBRL not fetched for {r.get('symbol')} - month left undone")
                return None
            flat.extend(pit_xbrl_rows(body.decode("utf-8", "replace") if isinstance(body, bytes) else body, r))
        events = insider_events(flat)
    added = save_events(events)
    if (year, month) < (date.today().year, date.today().month):
        _mark(source, period, len(rows))
    return added


def ingest_announcements(symbol: str, api, since: str) -> int | None:
    """One company's announcements (NSE returns its whole history in one answer), kept from `since`.
    The symbol is URL-encoded: M&MFIN would otherwise split the query string at its ampersand."""
    rows = _data(api(f"corporate-announcements?index=equities&symbol={quote(symbol.upper(), safe='')}"))
    if rows is None:
        return None
    events = []
    for r in rows:
        e = announcement_event({**r, "symbol": symbol})
        if e and e.signal_at[:10] >= since:
            events.append(e)
    added = save_events(reconcile(events))
    _mark("announcements", symbol.upper(), len(rows))
    return added


_BARE = re.compile(r"has\s+informed\s+the\s+exchange\s+(about|regarding)\s+resignation\s+of\s+statutory"
                   r"\s+auditors?\.?\s*$", re.I)


def _one_pdf(body: bytes) -> str:
    import io

    import pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        return " ".join((p.extract_text() or "") for p in pdf.pages[:2])


def _pdf_text(body: bytes) -> str:
    """The letter's text. NSE's auditor-resignation form arrives as a ZIP (..._ROSA_..._KMP_Doc.zip) with the
    letters inside as PDFs; members over 20 MB are skipped (no zip bombs)."""
    import io
    import zipfile
    if body[:2] != b"PK":
        return _one_pdf(body)
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        members = [i for i in z.infolist() if i.filename.lower().endswith(".pdf") and i.file_size < 20_000_000][:3]
        return " ".join(_one_pdf(z.read(i)) for i in members)


def resolve_bare_filings(get, extract=None) -> dict:
    """NSE files subsidiary-auditor resignations under the same "Resignation of Statutory Auditor" label as
    the company's own, so a filing that says nothing more is read from its letter (about 100 across the
    universe). A letter that cannot be fetched leaves the event as it is - the cautious reading."""
    extract = extract or _pdf_text
    tally: Counter = Counter()
    for e in load_events(types={"auditor_exit"}):
        if "letter" in e.details or not e.source_url or not _BARE.search(e.details.get("text", "")):
            continue
        body = get(e.source_url)
        _pause()
        try:
            letter = " ".join(extract(body).split())[:1500] if body else ""
        except Exception as err:               # a broken PDF is one failed letter, not a failed run
            logger.warning(f"{e.symbol}: auditor letter unreadable: {type(err).__name__}: {err}")
            letter = ""
        if not letter:
            tally["failed"] += 1
            continue
        etype = _announcement_type(e.details.get("subject", ""), letter) or e.event_type
        severity, direction, conviction = TYPES[etype]
        details = json.dumps({**e.details, "letter": letter}, sort_keys=True, default=str)
        with _connect() as con:
            con.execute("UPDATE event_signals SET event_type = ?, severity = ?, direction = ?, conviction = ?, "
                        "details = ? WHERE event_key = ?", (etype, severity, direction, conviction, details, e.event_key))
        tally["resolved"] += 1
    return dict(tally)


def ingest(since: date, until: date, api, get=None, universe: set[str] | None = None,
           symbols: list[str] | None = None, fresh_days: int | None = None) -> dict:
    """Everything from `since` to `until`: the market-wide feeds month by month, then each company's
    announcements (skipping companies fetched within `fresh_days`). Resumable: done months and fresh
    companies are not fetched again. One failure never stops the rest."""
    tally = Counter()
    for y, m in months(since, until):
        first = date(y, m, 1)
        for source, (_, start, end) in FEEDS.items():
            if first < start.replace(day=1) or (end and first > end):
                continue
            got = ingest_month(source, y, m, api, get, universe)
            _pause()
            tally["months_failed" if got is None else "months_done"] += 1
            tally["events"] += got or 0
    fresh = fresh_days if fresh_days is not None else settings.events_stale_days
    for sym in symbols or sorted(universe or ()):
        st = _state("announcements", sym.upper())
        if st and datetime.fromisoformat(st[1]) > datetime.now() - timedelta(days=fresh):
            tally["companies_fresh"] += 1
            continue
        got = ingest_announcements(sym, api, since.isoformat())
        _pause()
        tally["companies_failed" if got is None else "companies_done"] += 1
        tally["events"] += got or 0
    if get is not None:
        tally.update({f"letters_{k}": v for k, v in resolve_bare_filings(get).items()})
    return dict(tally)


def status() -> dict:
    """What is stored, and which feed months came back empty inside the range the feed is known to cover."""
    with _connect() as con:
        by_type = dict(con.execute("SELECT event_type, COUNT(*) FROM event_signals GROUP BY event_type"))
        companies = con.execute("SELECT COUNT(DISTINCT symbol) FROM event_signals").fetchone()[0]
        span = con.execute("SELECT MIN(signal_at), MAX(signal_at) FROM event_signals").fetchone()
        state = con.execute("SELECT source, period, rows, fetched_at FROM event_ingest_state").fetchall()
    months_done = Counter(s for s, p, _, _ in state if s != "announcements")
    empty = sorted(f"{s} {p}" for s, p, n, _ in state if s != "announcements" and n == 0)
    return {"events": sum(by_type.values()), "by_type": by_type, "companies": companies,
            "first": span[0], "last": span[1], "months_done": dict(months_done), "empty_months": empty,
            "announcement_companies": sum(1 for s, *_ in state if s == "announcements"),
            "last_fetch": max((f for *_, f in state), default=None)}


# ------------------------------------------------------------------ the ledger
def to_prediction(group: list[Event], closes=None, lag: int = 1, sector: str | None = None,
                  industry: str | None = None) -> Prediction | None:
    """One company's events of one type on one broadcast day -> one ledger call, at the strongest
    conviction among them. Dated to the first session after the broadcast day (you could not have
    traded on it before), entered at that session's close. Information-only events are never calls."""
    signals = [e for e in group if e.severity != "info" and e.direction]
    if not signals:
        return None
    lead = max(signals, key=lambda e: e.conviction or 0)
    day = min(e.signal_at[:10] for e in signals)
    as_of = actionable_date(day, closes, lag)
    entry = entry_asof = None
    if closes is not None and not closes.empty:
        idx = _sessions(closes)
        at = idx == pd.Timestamp(as_of)
        if at.any():
            entry, entry_asof = round(float(closes[at].iloc[0]), 2), as_of
    age = (date.today() - date.fromisoformat(day)).days
    etype = lead.event_type
    payload = {"event_type": etype, "signal_day": day, "events": len(signals),
               "people": sorted({e.details.get("person") for e in signals if e.details.get("person")}),
               "agencies": sorted({e.details.get("agency") for e in signals if e.details.get("agency")}),
               "value_rs": sum(e.details.get("value_rs") or 0 for e in signals) or None,
               "event_keys": [e.event_key for e in signals]}
    return Prediction(
        symbol=lead.symbol, source=f"event:{etype}", as_of_date=as_of, direction=lead.direction,
        conviction=lead.conviction or 2,
        horizon="3m",                         # the event studies behind these signals measure ~90 days
        entry_price=entry, entry_price_asof=entry_asof, sector=sector,
        rulebook=profile_for(sector, industry).name, payload=payload,
        provenance={"recorded": "live" if age <= settings.events_stale_days else "backfill",
                    "rules_version": RULES_VERSION, "entry_price_source": "feed_adjusted" if entry else "none",
                    "recorded_at": _now()},
        prediction_id=make_prediction_id(as_of, lead.symbol, {"source": f"event:{etype}", "signal_day": day}))


def record_predictions(lag: int = 1, symbols: set[str] | None = None) -> int:
    """Every stored signal-bearing event not yet in the ledger, one call per (company, type, day).
    Never rewrites a call. Prices are fetched once per company; a company with no price feed is still
    recorded (dated the day after its event) and simply grades as unavailable. Returns calls added."""
    groups: dict[tuple, list[Event]] = defaultdict(list)
    for e in load_events():
        if e.severity != "info" and e.direction and (symbols is None or e.symbol in symbols):
            groups[(e.symbol, e.event_type, e.signal_at[:10])].append(e)
    seen = {(p.symbol, p.source, p.payload.get("signal_day"))
            for p in load_predictions(include_superseded=True) if p.source.startswith("event:")}
    pending = {k: v for k, v in groups.items() if (k[0], f"event:{k[1]}", k[2]) not in seen}
    if not pending:
        return 0
    try:
        from pennystock.data.metrics import load_metrics
        sectors = {m.symbol: (m.sector, m.industry) for m in load_metrics()}
    except Exception as e:                       # a missing metrics cache only costs the rulebook
        logger.info(f"no sector data for the rulebook: {type(e).__name__}: {e}")
        sectors = {}
    oldest: dict[str, str] = {}
    for sym, _, day in pending:
        oldest[sym] = min(day, oldest.get(sym, day))
    closes = {sym: fetch_closes(sym, history_period(day)) for sym, day in oldest.items()}
    # one call per company and type per `events_call_spacing_days`: a company in insolvency files an update
    # every few days, and fifty calls on one story would swamp the type's track record. A STRONGER call
    # inside the window (a second insider turning a buy into a cluster) is still recorded.
    prior: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for p in load_predictions(include_superseded=True):
        if p.source.startswith("event:") and p.payload.get("signal_day"):
            prior[(p.symbol, p.source)].append((p.payload["signal_day"], p.conviction))
    new = []
    for (sym, _, day), group in sorted(pending.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        p = to_prediction(group, closes.get(sym), lag, *sectors.get(sym, (None, None)))
        if not p:
            continue
        since = (date.fromisoformat(day) - timedelta(days=settings.events_call_spacing_days)).isoformat()
        if any(since <= d < day and c >= p.conviction for d, c in prior[(sym, p.source)]):
            continue
        prior[(sym, p.source)].append((day, p.conviction))
        new.append(p)
    return save_predictions(new)


def track_record(event_type: str, horizon_days: int = 20) -> dict:
    """How this event type's calls have done at one horizon (hit rate over decided calls, mean
    abnormal return, sample size) and how many companies they cover - fifty calls on one company
    are one story, not fifty."""
    card = scorecard(source=f"event:{event_type}", horizon_days=horizon_days)
    return {**card, "companies": len({p.symbol for p in load_predictions(source=f"event:{event_type}")})}


# ------------------------------------------------------------------ the screener
VETO_RULE = "severe corporate event"


def veto_map(today: date | None = None) -> dict[str, list[tuple[str, str]]]:
    """{symbol: [(rule, detail)]} for every severe event (own auditor quitting, insolvency not since stayed
    or closed, company fraud, a default-grade rating) in the last `events_veto_days`. Lenders are not
    exempt: these mean the same thing in every sector."""
    today = today or date.today()
    since = (today - timedelta(days=settings.events_veto_days)).isoformat()
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    events = [e for e in load_events(since=since) if e.signal_at[:10] <= today.isoformat()]
    lifted: dict[str, str] = {}
    for e in events:
        if e.event_type == "insolvency_lifted":
            lifted[e.symbol] = max(lifted.get(e.symbol, ""), e.signal_at)
    for e in events:
        if e.event_type == "insolvency" and lifted.get(e.symbol, "") > e.signal_at:
            continue                              # stayed, set aside or closed since
        if e.severity == "veto":
            what = e.details.get("agency") or e.details.get("subject") or ""
            reason = (VETO_RULE, f"{e.event_type} on {e.signal_at[:10]}: {what}"[:120])
            if reason not in out[e.symbol]:           # one story filed twice is one reason
                out[e.symbol].append(reason)
    return dict(out)


# ------------------------------------------------------------------ the analyst's view
HOW_TO_READ = ("Dated NSE disclosures (date = when the market learned it). insider: open-market trades by "
               "promoters, directors and KMP of Rs 10 lakh or more - buying with their own money is the "
               "cleaner signal; insiders sell for many reasons. pledges: a pledge INVOKED is forced selling. "
               "severe: the company's own auditor quitting, insolvency (unless since stayed or closed), company "
               "fraud or forensic audit, or a D/C (default) rating - each keeps the stock out of the screen for "
               "a year. An Issuer Not Cooperating rating is usually administrative (the company stopped paying "
               "the agency) - a weak signal, not a verdict. "
               "track_records: how this system's ledger calls on each event type have done at 20 sessions "
               "(hit rate over decided calls, with the sample size - under ~30 means little).")


def _view(e: Event) -> dict:
    keep = ("person", "category", "value_rs", "agency", "rating", "outlook", "text", "subject")
    return {"date": e.signal_at[:10], "type": e.event_type, "severity": e.severity,
            **{k: e.details[k] for k in keep if e.details.get(k) not in (None, "")}}


def recent_summary(symbol: str, days: int = 365, today: date | None = None) -> dict | None:
    """One company's stored events from the last `days`, grouped the way an analyst reads them, with
    each signal type's track record. None when nothing is stored - never an empty "all clear"."""
    today = today or date.today()
    events = [e for e in load_events(symbol, since=(today - timedelta(days=days)).isoformat())
              if e.signal_at[:10] <= today.isoformat()]
    if not events:
        return None
    newest = sorted(events, key=lambda e: e.signal_at, reverse=True)
    buys = [e for e in newest if e.event_type == "insider_buy"]
    sells = [e for e in newest if e.event_type == "insider_sell"]
    buy_rs = sum(e.details.get("value_rs") or 0 for e in buys)
    sell_rs = sum(e.details.get("value_rs") or 0 for e in sells)
    records = {}
    for etype in sorted({e.event_type for e in events if e.severity != "info"}):
        t = track_record(etype)
        records[etype] = {"hit_rate": t["hit_rate"], "decided": t["decided_sample"], "companies": t["companies"]}
    return {
        "window_days": days,
        "insider": {"buys": len(buys), "buy_value_rs": round(buy_rs), "sells": len(sells),
                    "sell_value_rs": round(sell_rs),
                    "net": "net BUYING" if buy_rs > sell_rs else "net SELLING" if sell_rs > buy_rs else "none",
                    "recent": [_view(e) for e in (buys + sells)[:8]]},
        "pledges": [_view(e) for e in newest if e.event_type.startswith("pledge_")][:8],
        "ratings": [_view(e) for e in newest if e.event_type.startswith("rating_")][:8],
        "severe": [_view(e) for e in newest if e.severity == "veto"],
        "notable": [_view(e) for e in newest if e.event_type in ("auditor_exit_subsidiary", "enforcement")][:5],
        "track_records": records,
        "how_to_read": HOW_TO_READ,
    }
