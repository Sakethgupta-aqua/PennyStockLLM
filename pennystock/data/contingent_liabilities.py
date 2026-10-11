"""Contingent liabilities, read from the annual report and VERIFIED in code.

No structured feed carries this note (the integrated filing does not), and Indian reports print it a dozen
ways: measured over 118 latest reports (2026-10-11), headings vary ("contingent liabilities and commitments
(to the extent not provided for)", "not acknowledge as debt"), units are lakh / crore / million / thousand,
the rupee sign extracts as ₹, `, J, H or I, and banks file a different schedule. A regex-only parser read 94
notes but picked the wrong table for several (a CSR table, an ESOP valuation table). So:

  locate   code scores pages on the heading, claim language, figures and the consolidated running header
  unit     code reads lakh / crore / million / thousand from the note, the page or the 'all amounts in' line
  rows     the local model lays the table out as rows, numbers AS PRINTED (it computes nothing)
  verify   every number must occur in the note's own text; a row that fails is dropped
  totals   computed here in Rs crore by category; paid-under-protest, commitments and subtotals excluded
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pennystock.config import settings
from pennystock.data.filing_text import pages_of

HEAD = re.compile(r"contingent\s+liabilit(?:y|ies)", re.I)
STRONG = re.compile(r"not\s+acknowledged?\s+as\s+debts?", re.I)
CLAIMS = re.compile(
    r"not\s+acknowledged?\s+as\s+debts?|claims?\s+against\s+the\s+(company|group|bank|corporation)|"
    r"in\s+respect\s+of\s+(income[\s-]?tax|excise|customs|gst|goods\s+and\s+service|service\s+tax|"
    r"sales\s+tax|indirect\s+tax)|disputed\s+(tax|demand|liabilit)|income[\s-]?tax\s+(matters|demands?)|"
    r"guarantees?\s+(given|issued|provided|excluding)", re.I)
POLICY = re.compile(
    r"contingent\s+liabilit\w*\s+(is|are|exists?)\s+(a\s+possible|disclosed\s+(when|where|in)|not\s+"
    r"recogni[sz]ed)|possible\s+obligation\s+(that|which|arising)|a\s+contingent\s+liability\s+exists", re.I)
REJECT = re.compile(r"corporate\s+social\s+responsibility|\bcsr\b|dividend\s+yield|black[\s-]?scholes|"
                    r"expected\s+volatility", re.I)
CONSOLIDATED = re.compile(r"consolidated", re.I)
STANDALONE = re.compile(r"standalone|separate\s+financial\s+statements", re.I)
FIGURE = re.compile(r"\(?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\)?|\b\d+\.\d{1,2}\b")
# a table row: a label, then two amounts (MARICO FY26 prints 'Income tax 324 336' - no commas, no decimals)
_AMOUNT = r"(?:\(?\d[\d,]*(?:\.\d+)?\)?|-|–|—)"
TABLE_ROW = re.compile(rf"(?m)^.*[A-Za-z].*?\s{_AMOUNT}\s+{_AMOUNT}\s*$")
_ENDS_IN_YEAR = re.compile(r"\b(19|20)\d{2}\)?\s*$")
# the next note's heading ('47 COMMITMENTS'); note numbers this late are two or three digits - a single digit
# is a numbered item inside the note (PERSISTENT: '1 Indirect tax matters')
NEXT_NOTE = re.compile(r"^\s*\d{2,3}(?:\.\d+)?\s*[\.\)]?\s+[A-Z][A-Za-z ,&/\-()']{6,}$")
_UNIT_WORD = r"(lakhs?|lacs?|crores?|cr\b|millions?|mn\b|thousands?|'000)"
UNIT = re.compile(r"(?:₹|`|\bJ\b|\bH\b|\bI\b|rs\.?|inr|rupees|amount|amounts|figures)\s*(?:in|/|\()?\s*"
                  rf"(?:₹|`|rs\.?|inr)?\s*{_UNIT_WORD}", re.I)
ALL_AMOUNTS = re.compile(r"(?:all\s+amounts?|amounts?|figures)\s+(?:are\s+)?(?:in|stated\s+in|expressed\s+in)"
                         rf"\s+(?:₹|`|rs\.?|inr|indian\s+rupees?)?\s*{_UNIT_WORD}", re.I)
UNIT_TO_CRORE = {"lakh": 0.01, "crore": 1.0, "million": 0.1, "thousand": 1e-4}


@dataclass(frozen=True)
class Note:
    page: int
    basis: str            # consolidated | standalone | unknown
    text: str             # from the heading to the next note (onto the next page if the note runs on)
    unit: str | None


def _unit_word(word: str) -> str:
    w = word.lower()
    return ("lakh" if w.startswith(("lakh", "lac")) else "crore" if w.startswith("cr") else
            "million" if w.startswith(("million", "mn")) else "thousand")


def detect_unit(note_text: str, page_text: str, earlier: list[str]) -> str | None:
    for text in (note_text[:1500], page_text):
        m = UNIT.search(text)
        if m:
            return _unit_word(m.group(1))
    for text in reversed(earlier[-80:]):
        m = ALL_AMOUNTS.search(text)
        if m:
            return _unit_word(m.group(1))
    return None


def _basis(pages: list[dict], i: int) -> str:
    """Consolidated or standalone, from the running header or footer of this page or the nearest one
    before it."""
    for j in range(i, max(-1, i - 40), -1):
        t = pages[j]["text"] or ""
        edge = t[:400] + " " + t[-300:]
        if CONSOLIDATED.search(edge):
            return "consolidated"
        if STANDALONE.search(edge):
            return "standalone"
    return "unknown"


def _until_next_note(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if n > 2 and NEXT_NOTE.match(line.strip()) and not HEAD.search(line):
            return "\n".join(lines[:n]), True
    return text, False


def locate_note(pages: list[dict]) -> Note | None:
    best, best_key = None, None
    for i, p in enumerate(pages):
        text = p["text"] or ""
        for m in HEAD.finditer(text):
            body, ended = _until_next_note(text[m.start(): m.start() + settings.cl_note_chars])
            if not ended and i + 1 < len(pages):           # the note runs onto the next page
                body += "\n" + _until_next_note(pages[i + 1]["text"] or "")[0]
            body = body[: settings.cl_note_chars]
            head = body[:1500]
            if POLICY.search(head[:500]) or REJECT.search(head):
                continue
            claims = len(CLAIMS.findall(head))
            rows = [r for r in TABLE_ROW.findall(head) if not _ENDS_IN_YEAR.search(r)]   # dates are not rows
            figures = max(len(FIGURE.findall(head)), len(rows))
            # the canonical phrase is enough with one amount (PERSISTENT writes each claim as a paragraph)
            if not claims or figures < (1 if STRONG.search(head) else 2):
                continue
            basis = _basis(pages, i)
            key = (basis == "consolidated", bool(STRONG.search(head)), min(figures, 30), claims)
            if best_key is None or key > best_key:
                best, best_key = (i, basis, body), key
    if best is None:
        return None
    i, basis, body = best
    return Note(pages[i]["page"], basis, body,
                detect_unit(body, pages[i]["text"] or "", [p["text"] or "" for p in pages[:i]]))


CL_ROWS_PROMPT = (
    "You read ONE note from an Indian company's annual report: CONTINGENT LIABILITIES (and "
    'commitments). Return every line that carries an amount, top to bottom: the label as printed, a '
    'category, and the two amounts EXACTLY as printed - current year first, previous year second (the '
    "column order of the table); '-' or a blank cell as null. Never compute, convert, add or round "
    'anything, and never invent a line.\n'
    'Categories: disputed_tax (income tax, GST, excise, customs, service tax, sales tax / VAT, stamp '
    'duty, entry tax disputes); disputed_other (other claims against the company not acknowledged as '
    'debt: legal cases, contractors, consumer, labour, patents); guarantee_for_others (corporate '
    'guarantees, bonds or undertakings given ON BEHALF OF subsidiaries, joint ventures, associates or '
    'any other party - the line must name that party); bank_guarantee_own (guarantees issued by banks '
    "on the company's or the group's OWN behalf, performance guarantees, bonds for its own obligations "
    "- 'on behalf of the Group' is its own); letters_of_credit; bill_discounting; other_contingent (any"
    ' other contingent liability); paid_under_protest (amounts already paid or deposited against the '
    'claims above); commitment (capital or other commitments - not contingent liabilities); subtotal '
    '(any total or subtotal line).\n'
    'A guarantee given on behalf of another party that is stated in a sentence or note below the table '
    "(e.g. 'The Group had provided corporate guarantees of X lakhs to ... on behalf of ...') is a line too, "
    "unless it only breaks down a line already given ('includes ...'); a year it does not state is null.")


CATEGORIES = ("disputed_tax", "disputed_other", "guarantee_for_others", "bank_guarantee_own",
              "letters_of_credit", "bill_discounting", "other_contingent", "paid_under_protest", "commitment",
              "subtotal")
BUCKETS = {"disputed_tax": "disputed_claims", "disputed_other": "disputed_claims",
           "guarantee_for_others": "guarantees_for_others", "bank_guarantee_own": "trade_finance",
           "letters_of_credit": "trade_finance", "bill_discounting": "trade_finance",
           "other_contingent": "other"}
_NIL = {"", "-", "–", "—", "nil"}
_TOKEN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


class _Row(BaseModel):
    label: str = Field(description="the line's label as printed")
    category: Literal[CATEGORIES]
    current: str | None = Field(description="current-year amount exactly as printed, e.g. '9,443'")
    previous: str | None = Field(description="previous-year amount exactly as printed")


class _Rows(BaseModel):
    rows: list[_Row]


def model_rows(note_text: str, llm) -> list[dict]:
    structured = llm.with_structured_output(_Rows, method="json_schema")
    out = structured.invoke([SystemMessage(content=CL_ROWS_PROMPT), HumanMessage(content=note_text)])
    return [r.model_dump() for r in out.rows]


# an amount the model wrote with its unit: SRF FY26 '6.54 crores' for a sentence printing '` 6.54 crores'
_WITH_UNIT = re.compile(r"^\s*(?:₹|`|rs\.?|inr)?\s*(\(?-?\d[\d,]*(?:\.\d+)?\)?)\s*"
                        r"(lakhs?|lacs?|crores?|cr\.?|millions?|mn|thousands?)\s*$", re.I)


def _bare(printed: str) -> tuple[str, str | None]:
    """(the number as printed, the unit written beside it or None)."""
    m = _WITH_UNIT.match(printed)
    return (m.group(1), _unit_word(m.group(2))) if m else (printed, None)


def _digits(printed: str) -> str:
    return _bare(printed)[0].strip().strip("()").replace(",", "").lstrip("-").strip()


def _value(printed: str | None) -> float | None:
    if printed is None or printed.strip().lower() in _NIL:
        return 0.0
    s = _bare(printed)[0].strip()
    try:
        v = float(_digits(s))
    except ValueError:
        return None
    return -v if (s.startswith("(") and s.endswith(")")) or s.startswith("-") else v


def verify(rows: list[dict], note_text: str, unit: str | None = None) -> tuple[list[dict], list[dict]]:
    """Keep a row only if every amount it gives occurs in the note's own text - the model lays the table
    out, it never gets to supply a number. An amount written with a unit other than the note's fails: it
    would be counted in the wrong unit."""
    on_page = {_digits(t) for t in _TOKEN.findall(note_text)}
    kept, dropped = [], []

    def fits(a: str | None) -> bool:
        if a is None or a.strip().lower() in _NIL:
            return True
        written = _bare(a)[1]
        return _digits(a) in on_page and (written is None or unit is None or written == unit)

    for r in rows:
        ok = all(fits(r.get(k)) for k in ("current", "previous"))
        (kept if ok else dropped).append(r)
    return kept, dropped


# The red-flag class needs evidence in the label: a guarantee is 'for others' only when it names someone else.
# DIXON FY26, real model output: 'Guarantees issued by bankers on behalf of the Group' (its own) and its own
# customs bond were put there, inflating the class from Rs 1 Cr to Rs 322.66 Cr. Numbers can be verified
# against the page; a category cannot, so this one is checked against the label's own words.
_OWN_BEHALF = re.compile(r"on\s+behalf\s+of\s+the\s+(company|group|holding\s+company|bank|corporation)\b",
                         re.I)
# ...or 'on behalf of' anyone but the company itself (MARICO: 'on behalf of Broadcast Audience Research
# Council')
_SOMEONE_ELSE = re.compile(r"subsidiar|joint\s+venture|\bjv\b|associate|third[\s-]part|\bothers\b|"
                           r"other\s+part(?:y|ies)|"
                           r"related\s+part|group\s+compan|step[\s-]?down|fellow|promoter|directors?|"
                           r"employees?|"
                           r"on\s+behalf\s+of\s+(?!the\s+(company|group|holding|bank|corporation)\b)",
                           re.I)


# The group's slice of its joint ventures' / associates' contingents (Ind AS 112) is not a claim against the
# company itself. ASHOKLEY FY26, real model output: 'b) Share of contingent liabilities of joint ventures
# 1.60 3.76' was filed under disputed claims.
_SHARE_OF_INVESTEES = re.compile(r"share\s+of\s+(the\s+)?contingent\s+liabilit", re.I)
_CLAIM_OR_GUARANTEE = {"disputed_tax", "disputed_other", "guarantee_for_others"}
# A bank's guarantees for its customers - RBI's schedule line 'Guarantees given on behalf of constituents' -
# are its business (spec: a lender's guarantees for customers are normal business), not the red-flag class.
# AUBANK FY26, extractor v4: Rs 5,401 Cr landed there once the model kept the heading in the label.
_FOR_CONSTITUENTS = re.compile(r"on\s+behalf\s+of\s+(?:the\s+|its\s+)?constituents", re.I)


def check_categories(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        label = r.get("label") or ""
        own = _OWN_BEHALF.search(label) or not _SOMEONE_ELSE.search(label)
        if r["category"] in _CLAIM_OR_GUARANTEE and _SHARE_OF_INVESTEES.search(label):
            r = {**r, "category": "other_contingent"}
        elif r["category"] == "guarantee_for_others" and _FOR_CONSTITUENTS.search(label):
            r = {**r, "category": "other_contingent"}
        elif r["category"] == "guarantee_for_others" and own:
            kind = "bank_guarantee_own" if re.search(r"\bbank", label, re.I) else "other_contingent"
            r = {**r, "category": kind}
        out.append(r)
    return out


# The verifier checks one direction - every number in a row is printed in the note. This checks the other for
# the red-flag class: a guarantee given for someone else that the note PRINTS must be among the rows. Found
# with the gold set (2026-10-11): ASHOKLEY's 'corporate guarantee given to bank for the loan taken by a joint
# venture company ... Nil (March 31, 2025: ` 0.75 crores)' and DIXON's '(ii) The Group had provided corporate
# guarantees of ` 56,793 lakhs to two suppliers on behalf of subsidiary' sit in sentences below the table; the
# model left both out and the notes still read as verified. Checking every amount in the note instead flagged
# 55 of 87 verified notes (the text runs on into lease, investment and borrowing tables), so it is scoped to
# guarantees: 3 of 87 were flagged, DIXON's a real omission and two false alarms fixed by the rules below.
_GUARANTEE_GIVEN = re.compile(r"guarantee(?!s?\s+(?:fees?|commission|charges?))", re.I)
_PRINTED_AMOUNT = re.compile(r"(?<![\d.,])\(?(?:\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{1,2})\)?"
                             r"(?![\d%]|\.\d|\s*%)")
_HEADING_NUMBER = re.compile(r"(?m)^\s*\d+(?:\.\d+)+(?=\s+[A-Z][A-Za-z]{2})")      # '3.12 COMMITMENTS'
# a sentence ends at '. ' before a capital, or a line starts a numbered heading or an item: 'b)', '(ii)', '3]'
_SEGMENT_BREAK = re.compile(r"(?<=[.;])\s+(?=[A-Z(\[])|\n(?=\s*(?:\d+(?:\.\d+)+\s+[A-Z]|"
                            r"\(?(?:[a-z]{1,2}|[ivx]{1,4}|\d{1,2})[\)\]]))")
# a footnote that breaks a row down: ABCAPITAL 'i) Includes Corporate Guarantees given ... of ₹ 3,000 crore'
_BREAKDOWN = re.compile(r"^\W*(?:\(?[a-z0-9]{1,4}[\)\].]\s*)?(?:includes?|included|out\s+of|of\s+which)\b",
                        re.I)


def _segments(text: str) -> list[tuple[int, str]]:
    """The note as sentences, table rows and numbered items, each with its offset."""
    cuts, pos = set(), 0
    for line in text.split("\n"):
        pos += len(line) + 1
        if TABLE_ROW.match(line):                        # a table row is a segment of its own
            cuts.add(pos)
    cuts |= {m.end() for m in _SEGMENT_BREAK.finditer(text)}
    out, start = [], 0
    for cut in [*sorted(cuts), len(text)]:
        if cut > start:
            out.append((start, text[start:cut]))
            start = cut
    return out


def guarantee_gaps(rows: list[dict], note_text: str) -> list[str]:
    """Amounts the note prints beside a guarantee given for someone else that no row holds - each a warning
    that keeps the note from being 'verified'."""
    held = {_digits(a) for r in rows for a in (r.get("current"), r.get("previous")) if a}
    headings = [m.span() for m in _HEADING_NUMBER.finditer(note_text)]
    gaps = []
    for offset, seg in _segments(note_text):
        if (not _GUARANTEE_GIVEN.search(seg) or not _SOMEONE_ELSE.search(seg) or _OWN_BEHALF.search(seg)
                or _BREAKDOWN.match(seg)):
            continue
        for m in _PRINTED_AMOUNT.finditer(seg):
            at = offset + m.start()
            if any(a <= at < b for a, b in headings) or _digits(m.group()) in held:
                continue
            gaps.append(f"guarantee for another party printed in the note ({_digits(m.group())}: "
                        f"'{' '.join(seg.split())[:90]}') is not among the rows")
    return gaps


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(0.011, abs(b) * 0.001)


def _amounts(r: dict) -> tuple[float | None, float | None]:
    return _value(r.get("current")), _value(r.get("previous"))


def mark_subtotals(rows: list[dict]) -> list[dict]:
    """Relabel printed subtotals the model returned as ordinary rows: a row with no label (APL APOLLO FY26
    prints its group subtotals unlabelled), or one whose amounts for BOTH years equal the run of rows just
    before or just after it."""
    out = [dict(r) for r in rows]
    for r in out:                                     # first pass: unlabelled amounts are printed subtotals
        cur, prev = _amounts(r)
        has_amount = not (cur in (None, 0.0) and prev in (None, 0.0))
        if has_amount and not re.search(r"[A-Za-z]", r.get("label") or ""):
            r["category"] = "subtotal"
    for i, r in enumerate(out):                       # second pass: a row that adds up its neighbours
        cur, prev = _amounts(r)
        if r["category"] == "subtotal" or (cur in (None, 0.0) and prev in (None, 0.0)):
            continue
        for run in (_run(out, i, -1), _run(out, i, 1)):
            if run and _close(cur, sum(a or 0 for a, _ in run)) and _close(prev, sum(b or 0 for _, b in run)):
                r["category"] = "subtotal"
                break
    return out


def _run(rows: list[dict], i: int, step: int) -> list[tuple]:
    """The amounts of the rows next to row i (before it when step is -1), up to a subtotal or a header."""
    run, j = [], i + step
    while 0 <= j < len(rows) and rows[j]["category"] not in ("subtotal", "commitment"):
        cur, prev = _amounts(rows[j])
        if cur in (None, 0.0) and prev in (None, 0.0) and re.search(r"[A-Za-z]", rows[j].get("label") or ""):
            break                                     # a header line ends the run
        run.append((cur, prev))
        j += step
    return run


def status_of(kept: list[dict], dropped: list[dict], unit: str | None,
              warnings: list[str] | None = None) -> str:
    # a printed total the rows do not add up to means the rows are mis-structured (APL APOLLO FY26: group
    # subtotals counted twice) - every number may be on the page and the total still wrong, so it is never
    # 'verified'
    blocking = [w for w in warnings or [] if not w.startswith("note:")]
    if unit and len(kept) >= 2 and not dropped and not blocking:
        return "verified"
    return "partial" if kept else "failed"


def totals(rows: list[dict], unit: str) -> dict:
    factor = UNIT_TO_CRORE[unit]
    out = {b: {"current": 0.0, "previous": 0.0} for b in ("disputed_claims", "guarantees_for_others",
                                                          "trade_finance", "other")}
    for r in rows:
        bucket = BUCKETS.get(r["category"])
        for k in ("current", "previous"):
            v = _value(r.get(k)) if bucket else None
            if v:
                out[bucket][k] += v * factor
    return {b: {k: round(v, 2) for k, v in years.items()} for b, years in out.items()}


def reconcile(rows: list[dict]) -> list[str]:
    """A printed subtotal must equal the run of rows just above it, that run's last category, or - for a grand
    total over several groups - every row of that bucket so far. A subtotal that does not add up is only a
    'note:' when a later grand total of the same bucket does (APL APOLLO FY26 prints 1.58 for 7.27 in one
    group while its grand total matches the items); otherwise the rows are mis-structured and it blocks
    'verified'."""
    checked, run, seen = [], [], []
    for r in rows:
        if r["category"] != "subtotal":
            run.append(r)
            seen.append(r)
            continue
        v = _value(r.get("current"))
        anchor = run[-1] if run else (seen[-1] if seen else None)   # a grand total may follow a subtotal
        if v is not None and anchor is not None:
            last = anchor["category"]
            bucket = BUCKETS.get(last)
            cumulative = sum(_value(x.get("current")) or 0 for x in seen
                             if BUCKETS.get(x["category"]) == bucket)
            candidates = (sum(_value(x.get("current")) or 0 for x in run),
                          sum(_value(x.get("current")) or 0 for x in run if x["category"] == last),
                          cumulative)
            checked.append((r, any(_close(c, v) for c in candidates), bucket, _close(cumulative, v)))
        run = []
    warnings = []
    for k, (r, ok, bucket, _) in enumerate(checked):
        if ok:
            continue
        what = f"'{(r.get('label') or '')[:50]}' ({r.get('current')})"
        if any(b == bucket and grand for _, _, b, grand in checked[k + 1:]):
            warnings.append(f"note: the report's own subtotal {what} does not add up; its grand total does")
        else:
            warnings.append(f"printed subtotal {what} does not equal the rows above")
    return warnings


_SCHEMA = """CREATE TABLE IF NOT EXISTS contingent_liabilities (
    content_sha TEXT NOT NULL, extractor_version INTEGER NOT NULL, symbol TEXT NOT NULL, result TEXT NOT NULL,
    extracted_at TEXT NOT NULL, PRIMARY KEY (content_sha, extractor_version))"""


class CLStore:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or settings.market_db)
        with sqlite3.connect(self.db_path) as con:
            con.execute(_SCHEMA)

    def get(self, content_sha: str) -> dict | None:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("SELECT result FROM contingent_liabilities WHERE content_sha = ? AND "
                              "extractor_version = ?",
                              (content_sha, settings.cl_extractor_version)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, result: dict) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT OR REPLACE INTO contingent_liabilities VALUES (?,?,?,?,?)",
                        (result["content_sha"], settings.cl_extractor_version, result["symbol"],
                         json.dumps(result), datetime.now().isoformat(timespec="seconds")))


@lru_cache(maxsize=1)
def _store() -> CLStore:
    return CLStore()


def latest_report(symbol: str) -> dict | None:
    with sqlite3.connect(str(settings.filings_db)) as con:
        con.row_factory = sqlite3.Row
        # only a report whose text was extracted (NTPC's FY26 report is on file but was never read)
        row = con.execute("SELECT d.content_sha, d.doc_date, d.fiscal_period FROM filing_docs d "
                          "JOIN filing_texts t USING (content_sha) WHERE d.symbol = ? AND "
                          "d.doc_type = 'annual_report' "
                          "ORDER BY d.doc_date DESC, t.pages DESC LIMIT 1", (symbol.upper(),)).fetchone()
    return dict(row) if row else None


def extract(symbol: str, llm=None, force: bool = False) -> dict:
    """The latest annual report's note -> verified rows and totals, stored (one model call per report)."""
    report = latest_report(symbol)
    if report is None:
        return {"symbol": symbol.upper(), "status": "no_report"}
    store = _store()
    if not force and (hit := store.get(report["content_sha"])):
        return hit
    base = {"symbol": symbol.upper(), "content_sha": report["content_sha"],
            "fiscal_year": report["fiscal_period"], "doc_date": report["doc_date"]}
    note = locate_note(pages_of(report["content_sha"]))
    if note is None:
        result = {**base, "status": "no_note", "rows": [], "dropped": [], "totals": None, "warnings": []}
    else:
        if llm is None:
            from pennystock.llms import extraction_llm
            llm = extraction_llm()
        kept, dropped = verify(mark_subtotals(check_categories(model_rows(note.text, llm))), note.text,
                               note.unit)
        warnings = reconcile(kept) + guarantee_gaps(kept, note.text)
        result = {**base, "page": note.page, "basis": note.basis, "unit": note.unit,
                  "status": status_of(kept, dropped, note.unit, warnings), "rows": kept, "dropped": dropped,
                  "totals": totals(kept, note.unit) if note.unit else None, "warnings": warnings}
    store.put(result)
    return result


def stored(symbol: str) -> dict | None:
    report = latest_report(symbol)
    return _store().get(report["content_sha"]) if report else None


def summarize_cl(result: dict, net_worth_cr: float | None, lender: bool) -> dict:
    """LLM-ready view of a VERIFIED extraction: totals in Rs crore, judged against net worth."""
    t = result["totals"]

    def pct(v: float) -> float | None:
        return round(v / net_worth_cr * 100, 1) if net_worth_cr else None

    factor = UNIT_TO_CRORE[result["unit"]]
    items = sorted((r for r in result["rows"] if r["category"] in BUCKETS),
                   key=lambda r: -(_value(r.get("current")) or 0))
    out = {"machine_verified": True, "status": result["status"],
           "source": f"annual report {result['fiscal_year']}, page {result['page']}, {result['basis']}",
           "unit_as_printed": result["unit"], "totals_cr": t, "net_worth_cr": net_worth_cr,
           "disputed_claims_pct_of_net_worth": pct(t["disputed_claims"]["current"]),
           # a year with no figure printed stays None: DIXON's note (ii) gives this year only, and
           # "previous 0" would read as a new guarantee
           "largest_items": [{"label": r["label"], "category": r["category"],
                              "current_cr": round((_value(r.get("current")) or 0) * factor, 2),
                              "previous_cr": None if r.get("previous") is None else
                              round((_value(r.get("previous")) or 0) * factor, 2)}
                             for r in items[:5]],
           "warnings": result.get("warnings") or []}
    if lender:
        out["how_to_judge"] = ("This is a lender: guarantees, letters of credit and acceptances issued for "
                               "customers are its business, so only disputed claims are judged against net "
                               "worth.")
    else:
        out["guarantees_for_others_pct_of_net_worth"] = pct(t["guarantees_for_others"]["current"])
        # summed here: the live DIXON report added the four lines itself (Rs 2,084 Cr) and quoted the per-line
        # 11-12% for the sum, which is 44.6% of net worth
        every = {y: round(sum(t[b][y] for b in t), 2) for y in ("current", "previous")}
        out["all_contingents_cr"] = every
        out["all_contingents_pct_of_net_worth"] = pct(every["current"])
        out["how_to_judge"] = ("Disputed claims and guarantees given for others are the real "
                               "off-balance-sheet risk; judge them against net worth. Trade-finance "
                               "contingents (letters of credit, bank guarantees on the company's own behalf, "
                               "bill discounting) are normal operating items.")
    return out
