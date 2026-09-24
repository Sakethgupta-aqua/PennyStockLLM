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
import re
from collections import defaultdict
from dataclasses import dataclass, field
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
RE_FY = re.compile(r"\bFY\s?(\d{2})\s?([AEP])?\b")


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


RE_FY_TOKEN = re.compile(r"\bFY\s?\d{2}\s?[AEP]?\b", re.I)
RE_TRAILING_NUMS = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?%?$")

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
    # banks and NBFCs are measured on an entirely different set of lines
    "nii": ("net_interest_income", True), "net interest income": ("net_interest_income", True),
    "nim": ("nim_pct", False), "nim (%)": ("nim_pct", False),
    "gnpa": ("gnpa_pct", False), "gnpa (%)": ("gnpa_pct", False),
    "credit cost": ("credit_cost_pct", False), "roa": ("roa_pct", False), "roa (%)": ("roa_pct", False),
    "abv": ("adjusted_book_value", False), "bvps": ("book_value", False),
    # per-share and ratios
    "eps": ("eps", False), "eps (rs)": ("eps", False), "reported eps": ("eps", False),
    "adjusted eps (diluted)": ("eps", False), "adj. eps": ("eps", False),
    "ebitda margin": ("ebitda_margin_pct", False),
    "ebitda margins (%)": ("ebitda_margin_pct", False),
    "ebitda margin (%)": ("ebitda_margin_pct", False),
    "p/e": ("pe", False), "p/e (x)": ("pe", False), "pe (x)": ("pe", False),
    "ev/ebitda": ("ev_ebitda", False), "ev / ebitda": ("ev_ebitda", False),
    "p/bv": ("pb", False), "p/b": ("pb", False), "p/bv (x)": ("pb", False),
    "roe": ("roe_pct", False), "roe (%)": ("roe_pct", False),
    "roce": ("roce_pct", False), "roce (%) (post tax)": ("roce_pct", False),
    "dividend per share (dps)": ("dps", False), "book value per share (bv)": ("book_value", False),
    "net debt / ebitda (x)": ("net_debt_ebitda", False),
    "effective tax rate": ("tax_rate_pct", False),
}

UNIT_TO_CRORE = {"bn": 100.0, "mn": 0.1, "cr": 1.0, "crore": 1.0, "lakh": 0.01, "lacs": 0.01}
RE_UNIT = re.compile(r"\((?:INR|Rs\.?|₹)?\s*(bn|mn|cr|crore|lakh|lacs)\.?\)", re.I)
RE_PAREN_SUFFIX = re.compile(r"\s*\((?:INR|Rs\.?|₹)\s*[^)]*\)\s*$", re.I)


def _clean_label(tokens: list[str]) -> tuple[str, str | None]:
    """Normalised row label plus any unit written into it, e.g. 'Net Profit (INR bn)'."""
    raw = " ".join(tokens).strip()
    unit = m.group(1).lower() if (m := RE_UNIT.search(raw)) else None
    label = RE_PAREN_SUFFIX.sub("", raw).strip(" .:").lower()
    return re.sub(r"\s+", " ", label), unit


def _header_years(line: str) -> list[str] | None:
    """Is this the column header, and if so which fiscal years does it carry?

    The FY tokens have to be the TRAILING tokens after a short label ("Y/E March FY25A FY26A..."),
    otherwise a data row that happens to mention a year is mistaken for a new header. A live AU SF
    Bank run hit exactly that on 'EPS % change YoY 4.8 24.8 42.2 29.3 Earnings Revisions (%) FY27E
    FY28E', which silently re-pointed every following row at the wrong two columns.
    """
    tokens = line.split()
    years: list[str] = []
    while tokens and RE_FY_TOKEN.fullmatch(tokens[-1]):
        years.insert(0, tokens.pop().upper().replace(" ", ""))
    if len(years) < 2 or len(tokens) > 6:
        return None
    return years


def parse_estimates_from_text(pages: list[tuple[int, str]]) -> dict:
    """Read the financial-summary block straight out of the text, unit-aware.

    pdfplumber's table finder is unusable on these cover pages: ICICI draws the layout in coloured
    boxes, so `find_tables()` returns ONE 52-row "table" holding the whole page, prose included.
    The block is easy to read as text - a header line of fiscal years, then label-plus-N-numbers
    rows - and that shape holds across every broker seen so far.

    Units MUST be honoured. ICICI Securities quotes INR mn, its banking notes quote INR bn, and
    ICICI Direct quotes Rs crore. Assuming one of them turns a Rs 2,110 crore profit into Rs 2.11
    crore.
    """
    est: dict[str, dict] = {}
    for _, text in pages:
        years: list[str] = []
        table_unit: str | None = None
        for raw in text.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            if (found := _header_years(line)) is not None:
                years = found
                table_unit = m.group(1).lower() if (m := RE_UNIT.search(line)) else None
                continue
            if not years:
                continue
            tokens = line.split()
            nums: list[str] = []
            while tokens and RE_TRAILING_NUMS.fullmatch(tokens[-1]):
                nums.insert(0, tokens.pop())
            if len(nums) != len(years) or not tokens:
                continue
            label, row_unit = _clean_label(tokens)
            entry = ESTIMATE_ROWS.get(label)
            if not entry:
                continue
            key, is_money = entry
            scale = UNIT_TO_CRORE.get(row_unit or table_unit or "", 1.0) if is_money else 1.0
            for fy, raw_val in zip(years, nums):
                v = raw_val.replace(",", "").replace("%", "")
                neg = v.startswith("(") and v.endswith(")")
                try:
                    val = float(v.strip("()"))
                except ValueError:
                    continue
                val = -val if neg else val
                est.setdefault(fy, {})[key + ("_cr" if is_money else "")] = round(val * scale, 2)
    return est


def parse_estimates(tables: list[Table]) -> dict:
    """Forward estimates out of the financial-summary table, keyed by fiscal year.

    This is the payload. Everything else in this system is historical - filed accounts, past
    prices, past ratios. FY27E and FY28E revenue, EBITDA and EPS are the only forward-looking
    numbers available anywhere in it, and a target price is meaningless without the estimate that
    produced it.
    """
    wanted = {
        "net revenue": "revenue", "net sales": "revenue", "revenue": "revenue",
        "ebitda": "ebitda", "ebitda (%)": "ebitda_margin", "ebitda margin (%)": "ebitda_margin",
        "net profit": "net_profit", "net income (adjusted)": "net_profit",
        "adjusted eps (diluted)": "eps", "eps (rs)": "eps", "reported eps": "eps",
        "p/e (x)": "pe", "p/e": "pe", "ev/ebitda (x)": "ev_ebitda", "ev / ebitda": "ev_ebitda",
        "roe (%)": "roe", "roce (%) (post tax)": "roce", "rocE (%)".lower(): "roce",
    }
    est: dict[str, dict] = {}
    for t in tables:
        for row in t.rows:
            if not row or not row[0]:
                continue
            years = [c for c in row if c and RE_FY.fullmatch(c.strip())]
            if len(years) >= 2:                       # this row is the header: FY25A FY26A FY27E
                header = [c.strip() if c else "" for c in row]
                cols = {i: c for i, c in enumerate(header) if RE_FY.fullmatch(c)}
                for r in t.rows:
                    if not r or not r[0]:
                        continue
                    key = wanted.get(re.sub(r"\s+", " ", r[0]).strip().lower())
                    if not key:
                        continue
                    for i, fy in cols.items():
                        if i < len(r) and (v := _num((r[i] or "").replace("(", "-").replace(")", ""))) is not None:
                            est.setdefault(fy, {})[key] = v
    return est


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


def split_sections(text: str, page_no: int) -> list[Section]:
    """Narrative split on its own headings, so a chunk is a complete argument rather than an
    arbitrary window of characters that begins and ends mid-sentence."""
    sections, heading, buf = [], "", []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # a heading only counts when real prose follows it
        if _is_heading(line) and len(" ".join(buf)) > 120:
            sections.append(Section(heading=heading, text=" ".join(buf), page=page_no))
            heading, buf = line, []
        elif _is_heading(line) and not buf:
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
    rep.estimates = parse_estimates_from_text(pages)
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
            pieces[-2] += pieces.pop()
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
