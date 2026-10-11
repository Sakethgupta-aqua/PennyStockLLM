"""Layout-aware extraction of Indian broker research PDFs.

These reports are not prose documents - they are DASHBOARDS. A typical ICICI Securities note puts
the recommendation, price target, market data, price performance and ESG scores in a right-hand
sidebar, the argument in a left-hand column, and the numbers in tables. Roughly half the file by
character count is legal boilerplate. Three things therefore have to happen before any of it is
worth embedding, and naive text extraction gets all three wrong:

  1. COLUMNS. `extract_text()` walks the page top-to-bottom, so the sidebar is spliced into the
     middle of sentences: "...it recently announced the / Market Cap (USD) 34,972mn / commissioning
     of its manufacturing facility...". Embedding that destroys the meaning of both halves.

  2. BOILERPLATE. Page 4 of a 5-page UltraTech note is 11,159 characters of SEBI registration
     numbers and disclaimers - 48% of the document. It is near-identical in every report from that
     broker, so keeping it fills the vector store with hundreds of near-duplicate chunks that will
     out-compete real content for any query mentioning "risk", "disclosure" or "regulation".

  3. TABLES. The forward estimates (FY27E, FY28E revenue, EBITDA, EPS, target multiples) are the
     single most valuable thing in the document, and the one kind of data the rest of this system
     has none of. They belong in STRUCTURED fields, not flattened into a sentence.

The strategy is therefore: reconstruct reading order geometrically, drop disclosure pages, pull
the cover-page facts into typed fields, and keep tables as tables.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Private-use-area codepoints are icon-font glyphs (the up/down arrows and bullets these reports
# use). They carry no text meaning and they break regexes: "HOLD \uf020\n(Maintain)" made the
# rating action unmatchable because \uf020 is not whitespace.
PUA = re.compile(r"[\ue000-\uf8ff]")

LINE_TOLERANCE = 3.0            # points; words within this vertical distance are one line
HEADER_BAND = 88.0              # running head ("UltraTech Cement | Company Update | 04 Sep 2026")
FOOTER_BAND = 40.0
MIN_GUTTER_LINES = 8            # too few lines to judge a column split reliably
GUTTER_CROSS_RATIO = 0.18       # if more than this share of lines cross, it is one column

# A page is boilerplate when several of these appear. Verified across ICICI Securities, ICICI
# Direct, Axis, Ventura, Joindre and Motilal Oswal reports - always the trailing page(s).
DISCLOSURE_MARKERS = (
    "analyst certification", "terms & conditions and other disclosures", "disclaimer",
    "investment in securities market are subject to market risks", "sebi registration",
    "research analyst", "important disclosures", "no material disciplinary action",
    "past performance is not", "registered office", "grievance", "compliance officer",
)

RATINGS = ("BUY", "SELL", "HOLD", "ADD", "REDUCE", "ACCUMULATE", "NEUTRAL",
           "OUTPERFORM", "UNDERPERFORM", "OVERWEIGHT", "UNDERWEIGHT")
MONEY = r"(?:INR|Rs\.?|₹)\s*"
NUM = r"([\d,]+(?:\.\d+)?)"

RE_RATING = re.compile(rf"\b({'|'.join(RATINGS)})\b\s*\(?\s*(Maintain|Upgrade|Downgrade|Initiat\w*|Reiterat\w*)?",
                       re.I)
RE_CMP = re.compile(rf"\bCMP\s*:?\s*{MONEY}?{NUM}", re.I)
RE_TARGET = re.compile(rf"\bTarget\s*(?:Price)?\s*:?\s*{MONEY}?{NUM}"
                       rf"(?:\s*\(\s*{MONEY}?{NUM}\s*\))?", re.I)
RE_DATE = re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+20\d\d)\b", re.I)
RE_EMAIL = re.compile(r"([A-Za-z][\w.\-]*@[\w.\-]+\.\w+)")
RE_MCAP = re.compile(rf"Market\s*Cap\s*\(?(?:INR|Rs)\)?\s*{NUM}\s*(bn|mn|cr|crore)?", re.I)

# Bumped whenever parsing changes what a PDF yields. Every stored document carries the version
# that read it, so the next ingest re-reads anything parsed by an older, buggier parser - a fix
# reaches old data without anyone remembering to pass --reparse.
PARSER_VERSION = 8


@dataclass
class Table:
    page: int
    title: str
    rows: list[list[str]]

    def as_text(self) -> str:
        """Pipe-delimited, one row per line - readable by a person AND by an embedding model."""
        body = "\n".join(" | ".join(c or "" for c in r) for r in self.rows)
        return f"{self.title}\n{body}" if self.title else body


@dataclass
class Section:
    heading: str
    text: str
    page: int


@dataclass
class BrokerReport:
    path: str
    n_pages: int = 0
    broker: str | None = None
    company: str | None = None
    sector: str | None = None
    report_date: str | None = None
    report_type: str | None = None
    rating: str | None = None
    rating_action: str | None = None          # Maintain / Upgrade / Downgrade / Initiating
    cmp: float | None = None
    target_price: float | None = None
    target_price_prev: float | None = None
    upside_pct: float | None = None
    market_cap_cr: float | None = None
    analysts: list[str] = field(default_factory=list)
    estimates: dict = field(default_factory=dict)     # {"FY28E": {"eps": 378.5, ...}}
    estimate_rows: list[EstimateRow] = field(default_factory=list)   # the same, row by row
    sections: list[Section] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    boilerplate_pages: list[int] = field(default_factory=list)
    chars_kept: int = 0
    chars_dropped: int = 0


# ---------------------------------------------------------------- geometry
def _lines(page) -> list[list[dict]]:
    """Words grouped into visual lines, each sorted left-to-right, the page furniture removed.

    The header band is stripped only from page 2 onward. On page 1 that band is not a running head
    at all - it is the recommendation banner carrying the rating, CMP and target price, i.e. the
    most important content in the document. Cutting it cost every cover-page field on the first run.
    """
    top_cut = 0.0 if (page.page_number or 1) == 1 else HEADER_BAND
    words = [w for w in page.extract_words()
             if top_cut < w["top"] < page.height - FOOTER_BAND]
    rows: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        w = {**w, "text": PUA.sub("", w["text"])}
        if w["text"].strip():
            rows[round(w["top"] / LINE_TOLERANCE)].append(w)
    return [sorted(v, key=lambda w: w["x0"]) for _, v in sorted(rows.items())]


def find_gutter(page, lines: list[list[dict]] | None = None) -> float | None:
    """The x where fewest text LINES cross - the column gutter, or None if single-column.

    Counting LINE crossings rather than looking for blank space is what makes this work on these
    reports. The cover page carries a full-width financial-summary table whose rows span the whole
    width, so every "find the empty vertical strip" approach reports no gutter at all. Full-width
    rows are a minority of lines, so a crossing count still finds the true boundary at x=380.
    """
    lines = lines or _lines(page)
    if len(lines) < MIN_GUTTER_LINES:
        return None
    width = int(page.width)
    crossing = [0] * (width + 1)
    for ln in lines:
        for x in range(int(ln[0]["x0"]), min(int(ln[-1]["x1"]) + 1, width)):
            crossing[x] += 1
    lo, hi = int(width * 0.33), int(width * 0.78)
    band = crossing[lo:hi]
    if not band:
        return None
    floor = min(band)
    if floor > len(lines) * GUTTER_CROSS_RATIO:
        return None                                    # everything crosses: one column
    xs = [lo + i for i, c in enumerate(band) if c <= floor]
    return sum(xs) / len(xs)


def page_text(page) -> str:
    """Text in true reading order, honouring a two-column layout where one exists.

    Full-width lines (table rows, banner headings) act as band separators: everything above them is
    flushed left-column-then-right-column before the full-width line is emitted. Without that a
    page would read as "all left, then all right" and any table below the columns would be hoisted
    above the prose that introduces it.
    """
    lines = _lines(page)
    if not lines:
        return ""
    split = find_gutter(page, lines)
    if split is None:
        return "\n".join(" ".join(w["text"] for w in ln) for ln in lines)

    out: list[str] = []
    left: list[str] = []
    right: list[str] = []

    def flush():
        out.extend(left)
        out.extend(right)
        left.clear()
        right.clear()

    for ln in lines:
        # A line only counts as full-width when a WORD actually straddles the gutter. Judging by
        # the line's outer bounds instead merges a left-column sentence with the sidebar entry
        # beside it - which is how "...new paint and B2B e-commerce" ended up welded to the
        # analyst's email address.
        if any(w["x0"] < split < w["x1"] for w in ln):
            flush()
            out.append(" ".join(w["text"] for w in ln))
            continue
        l_words = [w["text"] for w in ln if w["x1"] <= split]
        r_words = [w["text"] for w in ln if w["x0"] >= split]
        if l_words:
            left.append(" ".join(l_words))
        if r_words:
            right.append(" ".join(r_words))
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------- boilerplate
def is_boilerplate(text: str, page_no: int, n_pages: int) -> bool:
    """Disclosure/disclaimer page? Kept deliberately conservative - dropping a real page of
    analysis is far worse than keeping one page of legalese."""
    low = text.lower()
    hits = sum(1 for m in DISCLOSURE_MARKERS if m in low)
    if hits >= 3:
        return True
    # Two markers only counts at the very back of the document. A cover page carrying the single
    # line "Please refer to important disclosures at the end of this report" also scores two, and
    # that page holds the rating, the target price and the entire investment case.
    return hits >= 2 and page_no > n_pages * 0.8


# ---------------------------------------------------------------- cover-page facts
def _num(s: str | None) -> float | None:
    try:
        return float(str(s).replace(",", "")) if s else None
    except ValueError:
        return None


def parse_cover(text: str) -> dict:
    """The recommendation block. This is the part a human reads first and the part that makes the
    report searchable: rating, what it was before, the price, the target, and who wrote it."""
    out: dict = {}
    # Strip icon-font glyphs here too. _lines() removes them for the normal path, but this is a
    # public function and "HOLD \uf020 (Maintain)" fails to match if one survives - the glyph is
    # not whitespace, so \s* will not cross it.
    text = PUA.sub(" ", text)
    head = text[:1400]                       # the block sits at the very top of page 1

    if m := RE_RATING.search(head):
        out["rating"] = m.group(1).upper()
        if m.group(2):
            out["rating_action"] = m.group(2).title()
    if m := RE_CMP.search(head):
        out["cmp"] = _num(m.group(1))
    if m := RE_TARGET.search(head):
        out["target_price"] = _num(m.group(1))
        # "Target Price: INR 12,138 (INR 11,900)" - the bracketed figure is the PREVIOUS target.
        # A target that moved, and by how much, is the most actionable line in the whole report.
        out["target_price_prev"] = _num(m.group(2))
    if m := RE_DATE.search(head):
        out["report_date"] = m.group(1)
    if m := RE_MCAP.search(text):
        mcap, unit = _num(m.group(1)), (m.group(2) or "").lower()
        if mcap is not None:                 # normalise to Rs crore, the unit used everywhere else
            out["market_cap_cr"] = round(
                mcap * {"bn": 100.0, "mn": 0.1, "cr": 1.0, "crore": 1.0}.get(unit, 1.0), 1)

    if out.get("cmp") and out.get("target_price"):
        out["upside_pct"] = round((out["target_price"] / out["cmp"] - 1) * 100, 1)

    # Analysts are identified by their email line; the name is the line above it.
    names, lines = [], text.splitlines()
    for i, ln in enumerate(lines):
        if RE_EMAIL.search(ln) and i:
            cand = lines[i - 1].strip()
            if 2 <= len(cand.split()) <= 4 and cand.replace(" ", "").replace(".", "").isalpha():
                names.append(cand)
    out["analysts"] = list(dict.fromkeys(names))[:4]
    return out


# --------------------------------------------------------------- estimate tables
# A column header token. FY27 / FY27E / FY2027 / Q1FY27 / 1QFY27 (Motilal) / H1FY27 / 9MFY26.
# Month-style headers ("Sep-25") are deliberately NOT periods: in these reports they head
# shareholding tables, and reading them as quarters would file promoter holdings as KPIs.
RE_PERIOD = re.compile(r"(?:(?P<q>Q[1-4]|[1-4]Q|H[12]|9M)\s?)?FY\s?(?P<y>\d{4}|\d{2})\s?(?P<s>[AEP])?",
                       re.I)
# Motilal Oswal writes "Y/E March 2020 ... 2027E 2028E": bare years, which are fiscal years only
# because the label says the year ends in March. A December company's "2026" is calendar 2026.
RE_YEAR_END = re.compile(r"(?:Y/?E|Year\s*end(?:ing)?)\s*:?\s*(?P<month>[A-Za-z]{3})", re.I)
RE_BARE_YEAR = re.compile(r"(?P<y>20\d{2})(?P<s>[AEP])?", re.I)
RE_TRAILING_NUMS = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?%?$")


@dataclass(frozen=True)
class EstimateRow:
    """One number from one table row, with everything needed to judge it later.

    `label` is the row label as printed (cleaned). It is kept because it carries the BASIS:
    "nim on aum" and "nims (cal)" are both stored as nim_pct, and only the label says that one
    broker divided by AUM while another used its own calculation."""
    page: int
    label: str
    metric: str
    period: str              # normalised header token, suffix kept: FY27E, Q1FY27, FY26
    value: float


def _quarter(m: re.Match) -> str:
    q = (m.group("q") or "").upper()
    return "Q" + q[0] if len(q) == 2 and q[1] == "Q" else q       # Motilal writes 1QFY27


def _period_token(token: str) -> str | None:
    m = RE_PERIOD.fullmatch(token.strip())
    if not m:
        return None
    return f"{_quarter(m)}FY{m.group('y')[-2:]}{(m.group('s') or '').upper()}"


def _bare_year(token: str) -> str | None:
    m = RE_BARE_YEAR.fullmatch(token)
    return f"FY{m.group('y')[-2:]}{(m.group('s') or '').upper()}" if m else None


def _trailing_periods(line: str, bare_years: bool = False) -> tuple[list[str], int]:
    """The period tokens a line ENDS with, and how many label tokens precede them."""
    tokens = line.split()
    periods: list[str] = []
    while tokens and (p := _period_token(tokens[-1]) or (bare_years and _bare_year(tokens[-1]))):
        periods.insert(0, p)
        tokens.pop()
    return periods, len(tokens)


def _header_years(line: str) -> list[str] | None:
    """Is this the column header, and if so which periods does it carry?

    Returns the periods; [] for a header-shaped line whose columns cannot be mapped (a later row
    must not be pinned to the PREVIOUS header's columns); None for a line that is not a header.

    The period tokens have to be the TRAILING tokens after a short label ("Y/E March FY25A..."),
    otherwise a data row that happens to mention a year is mistaken for a new header. A live AU SF
    Bank run hit exactly that on 'EPS % change YoY 4.8 24.8 42.2 29.3 Earnings Revisions (%) FY27E
    FY28E', which silently re-pointed every following row at the wrong two columns.
    """
    periods, label_tokens = _trailing_periods(line)
    if not periods and (year_end := RE_YEAR_END.match(line)):
        bare, label_tokens = _trailing_periods(line, bare_years=True)
        if len(bare) >= 2 and label_tokens <= 6:
            if year_end.group("month").lower() != "mar":
                return []
            periods = bare
    if len(periods) < 2 or label_tokens > 6:
        return None
    if len(set(periods)) < len(periods):
        # Two tables printed side by side repeat the same years ("FY25A..FY28E FY25A..FY28E"),
        # so one set describes every row. Any other repetition is unreadable.
        half = len(periods) // 2
        if len(periods) % 2 == 0 and periods[:half] == periods[half:] and len(set(periods[:half])) == half:
            return periods[:half]
        return []
    return periods


def _paired_header(line: str) -> list[str] | None:
    """Two tables side by side, each under its own label - 'Y/E 31 Mar (Rs mn) FY25A..FY29E Y/E 31
    Mar (Rs) FY25A..FY29E' (BOB), '(Year-end March) FY25..FY28E (Year-end March) FY25..FY28E'
    (ICICI Direct). Returns one table's periods, or None."""
    tokens = line.split()
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and _period_token(tokens[j]):
            j += 1
        if j > i:
            runs.append((i, j))
        i = max(j, i + 1)
    if len(runs) != 2:
        return None
    (a, b), (c, d) = runs
    left = [_period_token(t) for t in tokens[a:b]]
    if (len(left) < 2 or left != [_period_token(t) for t in tokens[c:d]] or d != len(tokens)
            or a > 6 or not 0 < c - b <= 6):
        return None
    return left


def _split_rows(tokens: list[str], n: int) -> list[tuple[list[str], list[str]]]:
    """A line under a side-by-side header holds up to two rows: label, n numbers, label, n numbers.

    A run of numbers SHORTER than n belongs to the label ('Tier 1 (%)'). A LONGER run is a row whose
    digits the text layer split apart ('1 5,123' for 15,123): unreadable, so it is dropped together
    with its label rather than guessed at."""
    rows: list[tuple[list[str], list[str]]] = []
    label: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and RE_TRAILING_NUMS.fullmatch(tokens[j]):
            j += 1
        if j == i:
            label.append(tokens[i])
            i += 1
            continue
        run = tokens[i:j]
        if len(run) == n and label:
            rows.append((label, run))
            label = []
        elif len(run) < n:
            label += run
        else:
            label = []
        i = j
    return rows


# fiscal-year end month/day, and the SEBI LODR (Reg. 33) filing deadline after it: 45 days for
# a quarter, 60 for the full year. Until that deadline passes, an unsuffixed period is a forecast.
_PERIOD_END = {"Q1": (6, 30, -1, 45), "Q2": (9, 30, -1, 45), "Q3": (12, 31, -1, 45),
               "Q4": (3, 31, 0, 60), "H1": (9, 30, -1, 45), "H2": (3, 31, 0, 60),
               "9M": (12, 31, -1, 45), "": (3, 31, 0, 60)}


def period_info(token: str, report_date: str) -> tuple[str, str, str]:
    """('FY26A', '2026-09-04') -> ('FY26', 'actual', '2026-03-31').

    Indian fiscal years end 31 March, so FY27's Q1 ends 30 June 2026. A suffix (A / E, or P for
    projected) is the broker's own statement and is trusted. Without one, the period counts as an
    ACTUAL only once the company had to have filed it - otherwise a broker's FY26 column in an
    April note is still a forecast, and ranking it as a reported fact would be wrong.
    """
    m = RE_PERIOD.fullmatch(token.strip())
    if not m:
        raise ValueError(f"not a fiscal period: {token!r}")
    q = _quarter(m)
    fy = 2000 + int(m.group("y")[-2:])
    month, day, year_offset, deadline = _PERIOD_END[q]
    end = date(fy + year_offset, month, day)
    suffix = (m.group("s") or "").upper()
    if suffix == "A":
        kind = "actual"
    elif suffix in ("E", "P"):
        kind = "estimate"
    else:
        kind = "actual" if date.fromisoformat(report_date) >= end + timedelta(days=deadline) \
            else "estimate"
    return f"{q}FY{fy % 100:02d}", kind, end.isoformat()


# metric -> (field name, is it money?). Money fields get unit-converted to Rs crore; ratios and
# percentages are taken as-is. Labels are matched with any "(INR bn)" style suffix stripped off.
ESTIMATE_ROWS: dict[str, tuple[str, bool]] = {
    # income statement
    "net revenue": ("revenue", True), "net sales": ("revenue", True), "revenue": ("revenue", True),
    "total income": ("revenue", True),
    "ebitda": ("ebitda", True), "ebitda (%)": ("ebitda_margin_pct", False),
    "op. profit": ("operating_profit", True),
    "operating profit": ("operating_profit", True), "ppop": ("operating_profit", True),
    "pre-provision profit": ("operating_profit", True),
    "net profit": ("net_profit", True), "net income (adjusted)": ("net_profit", True),
    "net income (reported)": ("net_profit", True), "adjusted pat": ("net_profit", True),
    "pat": ("net_profit", True),
    "nii": ("net_interest_income", True), "net interest income": ("net_interest_income", True),
    "abv": ("adjusted_book_value", False), "bvps": ("book_value", False),
    # per-share and ratios
    "eps": ("eps", False), "eps (rs)": ("eps", False), "reported eps": ("eps", False),
    "adjusted eps (diluted)": ("eps", False), "adj. eps": ("eps", False),
    "adjusted eps": ("eps", False), "adjusted p/e": ("pe", False),
    "dividend per share": ("dps", False), "book value per share": ("book_value", False),
    "ebitda margin": ("ebitda_margin_pct", False),
    "ebitda margins (%)": ("ebitda_margin_pct", False),
    "ebitda margin (%)": ("ebitda_margin_pct", False),
    "p/e": ("pe", False), "p/e (x)": ("pe", False), "pe (x)": ("pe", False),
    "ev/ebitda": ("ev_ebitda", False), "ev / ebitda": ("ev_ebitda", False),
    "p/bv": ("pb", False), "p/b": ("pb", False), "p/bv (x)": ("pb", False),
    "roe": ("roe_pct", False), "roe (%)": ("roe_pct", False), "roae": ("roe_pct", False),
    "roce": ("roce_pct", False), "roce (%) (post tax)": ("roce_pct", False),
    "dividend per share (dps)": ("dps", False), "book value per share (bv)": ("book_value", False),
    "net debt / ebitda (x)": ("net_debt_ebitda", False),
    "effective tax rate": ("tax_rate_pct", False),
}

# The forecasts a reader looks for first - what get_broker_consensus reports as forward estimates.
FORWARD_ESTIMATES = ("eps", "revenue_cr", "ebitda_cr", "net_profit_cr", "net_interest_income_cr")

# SECTOR KPIs - the operating metrics each rulebook in sectors.py declares it cannot see. Matched
# by pattern rather than exact label, because brokers decorate them: a section word in front
# ("Asset Quality Credit cost"), a basis behind ("NIM on AUM", "PCR (excl. technical w/off)").
# Everything else is anchored hard: "MFI credit cost" is a SEGMENT's number, and "Passenger car
# (%)" in an auto maker's table must never become a bank's capital adequacy ratio.
_SECTION = (r"(?:(?:asset quality|margins?|balance sheet|cost efficiency|efficiency|profitability|"
            r"returns?|return ratios|key ratios|ratios|capital(?: adequacy)?|spreads?|liquidity)\s+)?")
_BASIS = r"(?:\s+(?:ratio|on\s+[a-z.\s]+?|as\s+%\s+of\s+[a-z.\s]+?))?"
_TAIL = r"(?:\s*\([^)]*\))*$"           # "(%)", "(excl. cgfmu)", "(bps)" - kept in the label


def _kpi(phrase: str, *, anchored: bool = False) -> re.Pattern:
    lead = "^" if anchored else rf"^{_SECTION}(?:total\s+)?"
    return re.compile(rf"{lead}(?:{phrase}){_BASIS}{_TAIL}")


KPI_ROWS: tuple[tuple[str, re.Pattern, bool], ...] = (
    # lenders - banks and NBFCs are judged on these, and the screen had none of them
    ("nim_pct", _kpi(r"nims?|net interest margins?"), False),
    ("gnpa_pct", _kpi(r"gnpa|gross npas?|gs-?3|gross stage[- ]?(?:3|iii)"), False),
    ("nnpa_pct", _kpi(r"nnpa|net npas?|ns-?3|net stage[- ]?(?:3|iii)"), False),
    # RBI allows provision coverage with or without technical write-offs; both are printed
    ("pcr_incl_writeoff_pct", re.compile(
        rf"^{_SECTION}(?:pcr|provision(?:ing)? coverage(?: ratio)?)\s*"
        r"\((?=[^)]*\bincl)(?=[^)]*\b(?:tech|tw|w/?off|write))[^)]*\)$"), False),
    ("pcr_pct", _kpi(r"pcr|provision(?:ing)? coverage"), False),
    ("car_pct", _kpi(r"car|crar|capital adequacy", anchored=True), False),
    ("tier1_pct", _kpi(r"tier[- ]?(?:1|i)|cet-?1"), False),
    ("casa_pct", _kpi(r"casa", anchored=True), False),
    ("credit_cost_pct", _kpi(r"credit costs?"), False),
    ("cost_to_income_pct", _kpi(r"cost[- ]?(?:to|/)[- ]?income|c/i"), False),
    ("roa_pct", _kpi(r"roaa?", anchored=True), False),
    # cement, metals
    ("ebitda_per_tonne", _kpi(r"ebitda\s*(?:/|per)\s*(?:t|tn|tonne|ton|mt)"), False),
    ("realisation_per_tonne", _kpi(r"(?:blended\s+)?reali[sz]ations?\s*(?:/|per)\s*(?:t|tn|tonne|ton|mt)"),
     False),
    ("capacity_util_pct", _kpi(r"capacity utili[sz]ation"), False),
    # IT services
    ("attrition_pct", _kpi(r"(?:ltm\s+|ttm\s+|annuali[sz]ed\s+)?attrition"), False),
    ("book_to_bill", _kpi(r"book[- ]to[- ]bill"), False),
    # power, refining, telecom, consumer
    ("plf_pct", _kpi(r"plf|plant load factor"), False),
    ("grm_usd_bbl", _kpi(r"grm|gross refining margins?"), False),
    ("arpu", _kpi(r"(?:blended\s+)?arpu", anchored=True), False),
    ("sssg_pct", _kpi(r"sssg|same[- ]store sales growth|lfl growth"), False),
    ("volume_growth_pct", _kpi(r"volume growth"), False),
    # real estate and order-book businesses - money, so a unit is mandatory
    ("presales_cr", _kpi(r"pre[- ]?sales|sales bookings?"), True),
    ("order_book_cr", _kpi(r"order ?book"), True),
    ("order_inflow_cr", _kpi(r"order (?:inflows?|intake)"), True),
)
KPI_METRICS = frozenset(k for k, _, _ in KPI_ROWS)
KPI_LABELS = {
    "eps": "EPS", "revenue_cr": "Revenue cr", "ebitda_cr": "EBITDA cr", "net_profit_cr": "Net profit cr",
    "net_interest_income_cr": "NII cr",
    "nim_pct": "NIM %", "gnpa_pct": "GNPA %", "nnpa_pct": "NNPA %",
    "pcr_incl_writeoff_pct": "PCR incl. write-offs %", "pcr_pct": "PCR %", "car_pct": "CRAR %",
    "tier1_pct": "Tier-1 %", "casa_pct": "CASA %", "credit_cost_pct": "Credit cost %",
    "cost_to_income_pct": "Cost/income %", "roa_pct": "RoA %",
    "ebitda_per_tonne": "EBITDA/tonne", "realisation_per_tonne": "Realisation/tonne",
    "capacity_util_pct": "Capacity use %", "attrition_pct": "Attrition %",
    "book_to_bill": "Book-to-bill", "plf_pct": "PLF %", "grm_usd_bbl": "GRM $/bbl", "arpu": "ARPU",
    "sssg_pct": "SSSG %", "volume_growth_pct": "Volume growth %", "presales_cr": "Pre-sales cr",
    "order_book_cr": "Order book cr", "order_inflow_cr": "Order inflow cr",
}


def _row_metric(label: str) -> tuple[str, bool] | None:
    """Canonical (metric, is_money) for a cleaned row label, or None if it is not one we keep."""
    if label in ESTIMATE_ROWS:
        return ESTIMATE_ROWS[label]
    for key, pattern, is_money in KPI_ROWS:
        if pattern.search(label):
            return key, is_money
    return None


UNIT_TO_CRORE = {"billion": 100.0, "bn": 100.0, "b": 100.0, "million": 0.1, "mn": 0.1, "m": 0.1,
                 "crores": 1.0, "crore": 1.0, "cr": 1.0, "lakhs": 0.01, "lakh": 0.01, "lacs": 0.01}
RE_UNIT = re.compile(r"\((?:INR|Rs\.?|₹)?\s*(bn|mn|cr|crore|lakh|lacs)\.?\)", re.I)
# A currency marker followed by a scale: "INR mn", "(INR m)", "₹ crore", and Joindre's squashed
# "RsinMillion". The marker is required, so "users (In m)" is not mistaken for money.
RE_UNIT_CURRENCY = re.compile(r"(?:INR|Rs\.?|₹)\s*(?:in\s*)?(billion|bn|b|million|mn|m|crores?|cr|lakhs?|lacs)\b",
                              re.I)
RE_PAREN_SUFFIX = re.compile(r"\s*\((?:INR|Rs\.?|₹)\s*[^)]*\)\s*$", re.I)


def _unit_in(text: str) -> str | None:
    m = RE_UNIT_CURRENCY.search(text) or RE_UNIT.search(text)
    return m.group(1).lower() if m else None


def _clean_label(tokens: list[str]) -> tuple[str, str | None]:
    """Normalised row label plus any unit written into it, e.g. 'Net Profit (INR bn)'."""
    raw = " ".join(tokens).strip()
    unit = _unit_in(raw)
    label = RE_PAREN_SUFFIX.sub("", raw).strip(" .:").lower()
    return re.sub(r"\s+", " ", label), unit


# Values no company can print for the metric. A row holding one has its label on another row's
# numbers (live: a text layer paired 'EBITDA Margins (%)' with EBITDA growth, incl. 194.7%), so
# every number in that row is wrong, the plausible-looking ones too.
_BOUNDS = {"ebitda_margin_pct": (-math.inf, 100.0), "gnpa_pct": (0.0, 100.0),
           "nnpa_pct": (0.0, 100.0), "casa_pct": (0.0, 100.0), "car_pct": (0.0, 100.0),
           "tier1_pct": (0.0, 100.0)}


def _is_data_row(line: str) -> bool:
    tokens = line.split()
    return bool(tokens) and bool(RE_TRAILING_NUMS.fullmatch(tokens[-1]))


def parse_estimate_rows(pages: list[tuple[int, str]]) -> list[EstimateRow]:
    """Every recognised row of every estimate table, in reading order, unit-aware.

    pdfplumber's table finder is unusable on these cover pages: ICICI draws the layout in coloured
    boxes, so `find_tables()` returns ONE 52-row "table" holding the whole page, prose included.
    The block is easy to read as text - a header line of periods, then label-plus-N-numbers rows -
    and that shape holds across every broker seen so far.

    Units MUST be honoured, and are looked for on the row label, the header line, then the caption
    line directly above the header ("Income statement (INR m)"). A money row whose unit is still
    unknown is DROPPED: assuming crore stored Trent's revenue ten times too large. Ratios and
    per-share figures carry no unit and are always kept.
    """
    out: list[EstimateRow] = []
    for page_no, text in pages:
        periods: list[str] = []
        table_unit: str | None = None
        side_by_side = False
        prev = ""
        for raw in text.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            pair = _paired_header(line)
            found = pair if pair is not None else _header_years(line)
            caption, prev = prev, line
            if found is not None:
                periods = found
                # two tables share this header: a money row cannot be told apart from a ratio
                # row of the other table, so only unit-free rows are read below it
                side_by_side = pair is not None or len(_trailing_periods(line)[0]) > len(found)
                table_unit = _unit_in(line) or (
                    _unit_in(caption) if caption and not _is_data_row(caption) else None)
                continue
            if not periods:
                continue
            tokens = line.split()
            if side_by_side:
                candidates = _split_rows(tokens, len(periods))
            else:
                nums: list[str] = []
                while tokens and RE_TRAILING_NUMS.fullmatch(tokens[-1]):
                    nums.insert(0, tokens.pop())
                candidates = [(tokens, nums)] if len(nums) == len(periods) and tokens else []
            for label_tokens, nums in candidates:
                out.extend(_row_values(page_no, label_tokens, nums, periods, table_unit, side_by_side))
    return out


def _row_values(page_no: int, label_tokens: list[str], nums: list[str], periods: list[str],
                table_unit: str | None, side_by_side: bool) -> list[EstimateRow]:
    """One table row's numbers as EstimateRows - or none, if the row is not one we keep."""
    label, row_unit = _clean_label(label_tokens)
    entry = _row_metric(label)
    if not entry:
        return []
    key, is_money = entry
    if is_money:
        unit = row_unit or table_unit
        if unit not in UNIT_TO_CRORE or side_by_side:
            return []                                 # unknown scale: never guess crore
        scale = UNIT_TO_CRORE[unit]
        key = key if key.endswith("_cr") else key + "_cr"
    else:
        # "Credit cost (bps) 80" is 0.80%: a percentage KPI quoted in basis points
        scale = 0.01 if key.endswith("_pct") and re.search(r"\bbps?\b", label) else 1.0
    values: list[tuple[str, float]] = []
    for period, raw_val in zip(periods, nums):
        v = raw_val.replace(",", "").replace("%", "")
        neg = v.startswith("(") and v.endswith(")")
        try:
            val = float(v.strip("()"))
        except ValueError:
            continue
        values.append((period, round((-val if neg else val) * scale, 4)))
    low, high = _BOUNDS.get(key, (-math.inf, math.inf))
    if any(not low <= val <= high for _, val in values):
        return []
    return [EstimateRow(page=page_no, label=label, metric=key, period=period, value=val)
            for period, val in values]


def estimates_by_period(rows: list[EstimateRow]) -> dict:
    """The rows as {period: {metric: value}} - the shape the chunk text and the ledger payload use.
    A later row for the same period and metric overwrites an earlier one."""
    est: dict[str, dict] = {}
    for r in rows:
        est.setdefault(r.period, {})[r.metric] = round(r.value, 2)
    return est


def parse_estimates_from_text(pages: list[tuple[int, str]]) -> dict:
    return estimates_by_period(parse_estimate_rows(pages))


# ---------------------------------------------------------------- tables & sections
def _table_title(page, tbl) -> str:
    """The caption sitting immediately above a table ("Exhibit 5: Profit & Loss")."""
    top, x0, x1 = tbl.bbox[1], tbl.bbox[0], tbl.bbox[2]
    above = [w for w in page.extract_words()
             if top - 26 < w["bottom"] <= top + 2 and x0 - 12 <= w["x0"] <= x1 + 12]
    return " ".join(w["text"] for w in sorted(above, key=lambda w: w["x0"]))[:110]


def extract_tables(page, page_no: int) -> list[Table]:
    """Tables as ROWS, never flattened into prose. A number is only meaningful beside its label
    and its year, and that relationship is exactly what a naive text dump destroys."""
    out = []
    for tbl in page.find_tables():
        try:
            rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in tbl.extract()]
        except Exception as e:
            logger.debug(f"table extract failed on page {page_no}: {e}")
            continue
        rows = [r for r in rows if any(c for c in r)]
        cells = [c for r in rows for c in r]
        filled = [c for c in cells if c]
        numeric = sum(1 for c in filled if re.search(r"\d", c))
        # A real data table is mostly full and mostly numeric. ICICI's cover page otherwise yields
        # a single 52-row "table" that is really the page layout, prose and all.
        if (len(rows) < 2 or not cells or len(filled) / len(cells) < 0.25
                or not filled or numeric / len(filled) < 0.3):
            continue
        out.append(Table(page=page_no, title=_table_title(page, tbl), rows=rows))
    return out


# A heading is a short line that is not a sentence: no trailing full stop, few words, and not a
# row of numbers. These reports use them heavily ("Ambitious plan to scale up the new venture").
RE_NUMERIC_LINE = re.compile(r"^[\d\s.,()%\-/x]+$")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not (3 <= len(s) <= 95) or RE_NUMERIC_LINE.match(s):
        return False
    if s.endswith((".", ";", ",", ":")) or s.count("|") > 1:
        return False
    return 1 <= len(s.split()) <= 12


RE_NUMBERISH = re.compile(r"^\(?[-+]?[\d,]*\.?\d+\)?(?:%|x|bps)?\.?$|^(?:[1-4]Q|Q[1-4]|H[12]|9M)?FY\d{2,4}[AEP]?$",
                          re.I)


def _is_chart_residue(line: str) -> bool:
    """A chart's axis labels and data labels, or an exhibit's source caption - not prose.

    pdfplumber extracts a chart as lines of bare numbers ("18.0 16.4 15.8 ... 0.0") and years
    ("FY18 FY19 ... Q1FY27"). Mixed into a section they crowd out the argument at retrieval time,
    and the numbers a reader needs are already in broker_metrics."""
    if line.lower().startswith("source:"):
        return True
    tokens = line.split()
    numeric = sum(1 for t in tokens if RE_NUMBERISH.match(t))
    # all-numeric at any length (axis labels often print one per line), or mostly numeric
    return numeric == len(tokens) or (len(tokens) >= 3 and numeric / len(tokens) >= 0.6)


def split_sections(text: str, page_no: int) -> list[Section]:
    """Narrative split on its own headings, so a chunk is a complete argument rather than an
    arbitrary window of characters that begins and ends mid-sentence."""
    sections, heading, buf = [], "", []
    lines = [raw.strip() for raw in text.splitlines()
             if raw.strip() and not _is_chart_residue(raw.strip())]
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # A wrapped line of a narrow column is short and unpunctuated too, so it only counts as a
        # heading when the paragraph before it has ENDED and what follows starts a new sentence.
        # Without this, '...RoA expanded to 1.8% by Q4FY26 and' was read as a heading and the
        # paragraph was cut mid-sentence.
        heads = (_is_heading(line) and not nxt[:1].islower()
                 and (not buf or buf[-1].endswith((".", "!", "?", ":"))))
        # a heading only counts when real prose follows it
        if heads and len(" ".join(buf)) > 120:
            sections.append(Section(heading=heading, text=" ".join(buf), page=page_no))
            heading, buf = line, []
        elif heads and not buf:
            heading = line
        else:
            buf.append(line)
    if buf:
        sections.append(Section(heading=heading, text=" ".join(buf), page=page_no))
    return [s for s in sections if len(s.text) > 80]


# ---------------------------------------------------------------- the entry point
def extract_report(path: str | Path) -> BrokerReport:
    """Parse one broker PDF into structured fields, clean sections and intact tables."""
    import pdfplumber

    path = Path(path)
    rep = BrokerReport(path=str(path))
    with pdfplumber.open(str(path)) as pdf:
        rep.n_pages = len(pdf.pages)
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(pdf.pages, 1):
            text = page_text(page)
            if is_boilerplate(text, i, rep.n_pages):
                rep.boilerplate_pages.append(i)
                rep.chars_dropped += len(text)
                continue
            rep.chars_kept += len(text)
            pages.append((i, text))
            rep.tables.extend(extract_tables(page, i))

        for page_no, text in pages:
            rep.sections.extend(split_sections(text, page_no))

    if pages:
        cover = parse_cover(pages[0][1])
        for k, v in cover.items():
            if v not in (None, [], ""):
                setattr(rep, k, v)
        rep.company, rep.sector, rep.report_type = _identity(pages[0][1])
    rep.estimate_rows = parse_estimate_rows(pages)
    rep.estimates = estimates_by_period(rep.estimate_rows)
    return rep


RE_DOCTYPE = re.compile(r"\b(Company Update|Result Update|Initiating Coverage|Event Update|"
                        r"Sector Update|IPO Note|Management Meet|Annual Report Analysis|"
                        r"Q[1-4]FY\d{2} (?:result|review))\b", re.I)


def _identity(cover: str) -> tuple[str | None, str | None, str | None]:
    """Company, sector and report type from the cover page.

    The company name is the line after the "India | Equity Research | Company Update" strapline,
    with the sector immediately beneath it - a positional rule, because the name itself is the one
    thing on the page with no label in front of it.
    """
    doctype = m.group(1) if (m := RE_DOCTYPE.search(cover)) else None
    lines = [l.strip() for l in cover.splitlines() if l.strip()]
    company = sector = None
    for i, ln in enumerate(lines):
        if "|" in ln and re.search(r"equity research|institutional", ln, re.I):
            rest = [x for x in lines[i + 1:i + 4] if 2 < len(x) < 60 and not RE_NUMERIC_LINE.match(x)]
            if rest:
                company = rest[0]
            if len(rest) > 1:
                sector = rest[1]
            break
    return company, sector, doctype


# ---------------------------------------------------------------- listing metadata
def merge_listing_metadata(rep: BrokerReport, meta: dict) -> BrokerReport:
    """Overlay the Trendlyne listing row on top of what was parsed from the PDF.

    Every broker designs its own cover page - ICICI Securities uses "CMP: INR 11,275 Target Price:
    INR 12,138", ICICI Direct "CMP: Rs 2,660 Target 3,205 (20%)", Axis scatters the words across a
    graphic - and chasing each layout is a losing game that breaks whenever a broker restyles.
    The listing row already carries stock, broker, rating, LTP and target for ALL of them, so it is
    treated as authoritative for identity and recommendation. The PDF is then left to supply what
    only the PDF has: forward estimates, the argument, the tables and the analyst names.
    """
    def num(v):
        try:
            return float(str(v).replace(",", "").strip()) or None
        except (ValueError, AttributeError):
            return None

    rep.company = (meta.get("stock") or "").strip() or rep.company
    rep.broker = (meta.get("broker") or "").strip() or rep.broker
    rep.report_date = (meta.get("date") or "").strip() or rep.report_date
    if reco := (meta.get("reco_type") or "").strip():
        rep.rating = reco.upper()
        rep.report_type = rep.report_type or (reco if "note" in reco.lower() else None)
    rep.cmp = num(meta.get("ltp")) or rep.cmp
    if tgt := num(meta.get("target")):
        # a target parsed from the PDF that disagrees with the listing is kept as the PREVIOUS one
        if rep.target_price and abs(rep.target_price - tgt) > 0.01:
            rep.target_price_prev = rep.target_price_prev or rep.target_price
        rep.target_price = tgt
    if rep.cmp and rep.target_price:
        rep.upside_pct = round((rep.target_price / rep.cmp - 1) * 100, 1)
    return rep


def load_listing_metadata(folder: str | Path) -> dict[str, dict]:
    """metadata.jsonl written by the downloader, keyed by absolute PDF path."""
    import json
    path = Path(folder) / "metadata.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("local_path"):
                out[str(Path(row["local_path"]).resolve())] = row
    return out


# ---------------------------------------------------------------- RAG chunks
def _stamp(rep: BrokerReport) -> str:
    """Every chunk is prefixed with this. A chunk pulled out of the middle of a report otherwise
    arrives with no idea who wrote it, about what, or when - and a two-year-old SELL read as
    current advice is worse than no retrieval at all."""
    bits = [rep.company or "Unknown company"]
    if rep.broker:
        bits.append(rep.broker)
    if rep.report_date:
        bits.append(rep.report_date)
    if rep.rating:
        bits.append(f"rating {rep.rating}")
    return " | ".join(bits)


def to_chunks(rep: BrokerReport, max_chars: int = 1800) -> list[dict]:
    """Embedding-ready chunks with the metadata needed to filter, cite and date them."""
    base = {
        "source": Path(rep.path).name, "company": rep.company or "", "broker": rep.broker or "",
        "report_date": rep.report_date or "", "rating": rep.rating or "",
        "report_type": rep.report_type or "", "doc_type": "broker_report",
        "target_price": rep.target_price or 0.0, "cmp": rep.cmp or 0.0,
    }
    stamp = _stamp(rep)
    chunks: list[dict] = []

    # 1. the recommendation itself, as a sentence - this is what most queries actually want
    head = [f"{stamp}."]
    if rep.cmp and rep.target_price:
        head.append(f"Current price {rep.cmp:,.0f}, target price {rep.target_price:,.0f}"
                    + (f" (previously {rep.target_price_prev:,.0f})" if rep.target_price_prev else "")
                    + (f", implying {rep.upside_pct:+.1f}% upside." if rep.upside_pct else "."))
    if rep.rating_action:
        head.append(f"Rating action: {rep.rating_action}.")
    if rep.analysts:
        head.append(f"Analyst(s): {', '.join(rep.analysts)}.")
    if rep.sections:
        head.append(rep.sections[0].text[:600])
    chunks.append({"text": " ".join(head), "metadata": {**base, "chunk_type": "recommendation"}})

    # 2. forward estimates rendered as prose so they are retrievable, not just storable
    if rep.estimates:
        lines = [f"{stamp}. Broker estimates by fiscal year (A = actual, E = estimate):"]
        for fy in sorted(rep.estimates):
            vals = ", ".join(f"{k.replace('_', ' ')} {v:,g}" for k, v in sorted(rep.estimates[fy].items()))
            lines.append(f"{fy}: {vals}")
        chunks.append({"text": "\n".join(lines),
                       "metadata": {**base, "chunk_type": "estimates"}})

    # 3. the argument, one chunk per section, split only if a section is very long
    for sec in rep.sections:
        body = f"{sec.heading}. {sec.text}" if sec.heading else sec.text
        pieces = [body[i:i + max_chars] for i in range(0, len(body), max_chars)]
        # Never DROP a short tail - fold it into the previous piece. Discarding it silently lost
        # the last few characters of every section whose length was just over a chunk boundary,
        # which is exactly the kind of quiet truncation that is impossible to notice later.
        if len(pieces) > 1 and len(pieces[-1]) < 80:
            tail = pieces.pop()        # pop FIRST: `pieces[-2] += pieces.pop()` resolves [-2] early
            pieces[-1] += tail
        for piece in pieces:
            if len(piece) < 80 and len(pieces) == 1:
                continue
            chunks.append({"text": f"{stamp}.\n{piece}",
                           "metadata": {**base, "chunk_type": "narrative",
                                        "heading": sec.heading[:120], "page": sec.page}})

    # 4. tables kept whole - a number means nothing without its row label and its year
    for t in rep.tables:
        text = t.as_text()
        if len(text) > 40:
            chunks.append({"text": f"{stamp}.\n{text[:max_chars * 2]}",
                           "metadata": {**base, "chunk_type": "table",
                                        "heading": t.title[:120], "page": t.page}})
    return chunks
