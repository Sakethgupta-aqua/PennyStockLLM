# Analyst Evidence Pack and Data Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the analyst's answer to a full-company question accurate and checkable: statutory XBRL and verified
annual-report data (Step 3), a code-built evidence pack the model writes from (Step 2), and automatic checks plus a
Dixon regression test (Step 4).

**Architecture:** New data modules read NSE XBRL (shareholding pledges, integrated-filing figures, the half-yearly
related-party disclosure) and turn the annual report's contingent-liability note into verified numbers (local model
lays the table out, code verifies every figure). An `evidence_pack` graph node runs the structured tools in code and
renders a fact sheet that the analyst, fact-checker and judge all read. Two Layer-0 checks and an offline regression
test pin the failures seen on 2026-10-10/11.

**Tech Stack:** Python 3.12 (Anaconda), LangGraph 1.1, langchain-ollama (qwen3.5:35b), SQLite (market.db,
filings.db), regex XBRL parsing, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-analyst-evidence-pack-design.md` (approved 2026-10-11).

## Global Constraints

- Interpreter `/opt/anaconda3/bin/python`; suite `/opt/anaconda3/bin/python -m pytest -q` (offline: no network, no
  model). Baseline before Task 1: 691 passed.
- **No commits** - the user's standing rule. Each task ends with its test run; ledger lines record BASE..HEAD as the
  same commit.
- Lines ≤ 110 characters (`[tool.ruff] line-length = 110`); match the codebase's style: module docstrings that say
  what was measured and why; every tunable in `pennystock/config.py`, none inline.
- Money in outputs is Rs crore: INR / 1e7; lakh × 0.01; million × 0.1; thousand × 0.0001.
- A source that is down or unreadable becomes UNKNOWN with a reason - never "clean", never zero.
- NSE access only through `pennystock.data.nse_client` (`NseFeed`) or the ownership provider's primed session.
- Test runs of the agent use `PENNY_LEDGER_ENABLED=false` (no ledger call).
- Measured facts this plan relies on (2026-10-11): SHP XBRL contexts `ShareholdingOfPromoterAndPromoterGroup_ContextI`
  (promoter shares, pledged count) and `ShareholdingPattern_ContextI` (all shares), flags in context `MainI`;
  integrated-filing money in INR with segment breakdowns in dimensional contexts; related-party amounts cover the
  half-year though contexts carry the quarter's dates (Airtel: Sep 1,271.99 + Mar 1,224.67 = 2,496.66 Cr = the AGM
  notice's FY26 figure).

## Review Focus

1. A contingent-liability number the model returns that is not on the page (invented digits, swapped years) must
   never reach a total → Task 9 `test_a_number_not_on_the_page_is_dropped_and_the_note_is_not_verified`.
2. A bank's contingent liabilities are mostly customer business; only its claims are judged against net worth →
   Task 9 `test_a_lender_is_judged_on_claims_only`.
3. A company with no related-party XBRL rows (BDL) or an unreadable SHP filing gets UNKNOWN with a reason, never a
   crash or a zero → Task 4 `test_no_rows_and_no_feed_return_none`, Task 6 `test_a_pledged_flag_without_a_count_is_unknown_not_zero`,
   Task 11 `test_no_structured_source_falls_back_to_the_excerpt_and_says_so`.
4. One evidence-pack tool failing or timing out becomes a DATA GAPS line and the analyst still runs → Task 12
   `test_a_failing_tool_becomes_a_failed_call_not_a_failed_pack`, Task 13 `test_a_failed_source_is_listed_under_data_gaps`.
5. Broker estimates with no filed figure keep the old view, labelled → Task 7
   `test_without_a_filed_figure_the_view_says_so`.

---

### Task 1: Year-labelled financials (spec A6)

**Files:**
- Modify: `pennystock/data/fundamentals.py` (`_cagr` → `_growth`, `_by_year`, `_fy`; `summarize_financials`)
- Test: `tests/test_fundamentals.py`

**Interfaces:**
- Produces: `summarize_financials(fin) -> dict` with year-keyed dicts `revenue_cr`, `net_income_cr`, `eps`,
  `total_debt_cr`, `free_cash_flow_cr` ({"FY23": 12173.35, ...}), `revenue_cagr` ({"pct", "from", "to", "years"} or
  None), `latest_year`. Consumed by Task 13's renderer and Task 16.

- [ ] **Step 1: Write the failing tests** - replace `test_cagr_and_trend` and update `test_summarize_and_unavailable`:

```python
from pennystock.data.fundamentals import _growth, _trend, snapshot_warnings, summarize_financials
from pennystock.data.providers import Financials, Fundamentals


def test_growth_and_trend():
    assert _growth([2024, 2025, 2026], [100, 110, 121]) == {"pct": 10.0, "from": "FY24", "to": "FY26", "years": 2}
    assert _growth([2026], [100]) is None                     # too few points
    assert _growth([2025, 2026], [0, 50]) is None             # non-positive base guarded
    assert _growth([2025, 2026], [50, -5]) is None            # a loss at the end has no growth rate
    assert _trend([10, 20, 30]) == "rising"
    assert _trend([30, 10]) == "falling"


def test_summarize_and_unavailable():
    fin = Financials(fiscal_years=[2024, 2025], revenue_cr=[100, 121],
                     net_income_cr=[10, -5], total_debt_cr=[50, 40], equity_cr=[100, 100],
                     eps=[2.0, -1.0], operating_income_cr=[], free_cash_flow_cr=[])
    s = summarize_financials(fin)
    assert s["revenue_cagr"] == {"pct": 21.0, "from": "FY24", "to": "FY25", "years": 1}
    assert s["profitable_years"] == "1/2"
    assert s["debt_to_equity_pct_latest"] == 40.0
    assert summarize_financials(Financials()) == {"financials": "unavailable"}


def test_each_value_carries_its_year_and_growth_its_real_span():
    # DIXON 2026-10-11: Yahoo leaves FY22 empty, and parallel lists made the analyst call FY23's revenue FY22's
    fin = Financials(fiscal_years=[2022, 2023, 2024, 2025, 2026],
                     revenue_cr=[None, 12173.35, 17613.91, 38593.29, 48537.39],
                     net_income_cr=[None, 255.52, 367.75, 1095.54, 1438.64])
    s = summarize_financials(fin)
    assert s["revenue_cr"] == {"FY23": 12173.35, "FY24": 17613.91, "FY25": 38593.29, "FY26": 48537.39}
    assert s["revenue_cagr"] == {"pct": 58.6, "from": "FY23", "to": "FY26", "years": 3}
    assert s["net_income_cr"]["FY26"] == 1438.64 and "FY22" not in s["net_income_cr"]
    assert s["latest_year"] == "FY26"
```

- [ ] **Step 2: Run to see them fail**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_fundamentals.py`
Expected: ImportError `cannot import name '_growth'`.

- [ ] **Step 3: Implement** - in `fundamentals.py` replace `_cagr` with:

```python
def _fy(year: int) -> str:
    return f"FY{year % 100:02d}"


def _by_year(years: list, values: list | None) -> dict[str, float]:
    """Each value under its fiscal year, empty years dropped. Parallel lists whose first slot was empty made the
    analyst shift every year by one (DIXON, 2026-10-11: FY23's revenue reported as FY22's)."""
    return {_fy(y): v for y, v in zip(years, values or []) if v is not None}


def _growth(years: list, values: list | None) -> dict | None:
    """Compound annual growth between the first and last years that have a value, over the REAL year span (a gap
    year no longer shortens it), with both ends named so a '5-year CAGR' can never stand for three years."""
    points = [(y, v) for y, v in zip(years, values or []) if v is not None]
    if len(points) < 2:
        return None
    (y0, v0), (y1, v1) = points[0], points[-1]
    if v0 <= 0 or v1 <= 0 or y1 <= y0:
        return None
    return {"pct": round(((v1 / v0) ** (1 / (y1 - y0)) - 1) * 100, 1), "from": _fy(y0), "to": _fy(y1),
            "years": y1 - y0}
```

and make `summarize_financials` return:

```python
    years = fin.fiscal_years
    return {
        "latest_year": _fy(years[-1]),
        "revenue_cr": _by_year(years, fin.revenue_cr),
        "revenue_cagr": _growth(years, fin.revenue_cr),
        "revenue_trend": _trend(fin.revenue_cr),
        "net_income_cr": _by_year(years, ni),
        "net_income_trend": _trend(ni),
        "profitable_years": f"{profitable}/{counted}" if counted else "n/a",
        "eps": _by_year(years, fin.eps),
        "total_debt_cr": _by_year(years, fin.total_debt_cr),
        "debt_trend": _trend(fin.total_debt_cr),
        "debt_to_equity_pct_latest": de_latest,
        "roe_pct_latest": roe,          # net income / equity
        "roce_pct_latest": roce,        # EBIT / (equity + debt) - the quality bar (Coffee Can wants >=15%)
        "free_cash_flow_cr": _by_year(years, fin.free_cash_flow_cr),
    }
```

- [ ] **Step 4: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_fundamentals.py` → all pass.
- [ ] **Step 5: Commit** - skipped (user rule); run the full suite → 0 failures.

---

### Task 2: The P/E zone in words (spec A7)

**Files:**
- Modify: `pennystock/data/trendlyne_scores.py` (add `pe_zone_meaning`), `pennystock/tools/trendlyne.py`
- Test: `tests/test_trendlyne_tool.py`

**Interfaces:**
- Produces: `pe_zone_meaning(zone: str | None, pct: float | None) -> str | None`; the tool's `trendlyne_view` dict
  gains `pe_zone_meaning`. Consumed by Task 13.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_the_pe_zone_is_said_in_words(monkeypatch):
    # 'Strong Buy zone, 12%' was read as 'P/E in the top 12% of its history' (DIXON, 2026-10-10)
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("DIXON.NS", "Dixon Technologies"))
    monkeypatch.setattr(tool_mod, "view", lambda s: {"as_of": "2026-10-01T15:31", "pe_zone": "Strong Buy",
                                                     "pe_zone_pct": 12.0})
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "DIXON"}))
    assert "higher than only 12% of its own history" in out and "cheap against its past" in out


def test_pe_zone_meaning_reads_both_ends():
    from pennystock.data.trendlyne_scores import pe_zone_meaning
    assert "expensive against its past" in pe_zone_meaning("Sell", 85.0)
    assert pe_zone_meaning(None, None) is None
```

- [ ] **Step 2: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_trendlyne_tool.py` → FAIL (ImportError /
  assertion: no meaning in the output).

- [ ] **Step 3: Implement** - in `trendlyne_scores.py`:

```python
def pe_zone_meaning(zone: str | None, pct: float | None) -> str | None:
    """pe_zone_pct in words: the share of its own history the stock's P/E spent BELOW today's. Read raw, the model
    turned 'Strong Buy zone, 12%' into 'P/E in the top 12% of its history' (DIXON, 2026-10-10)."""
    if pct is None:
        return None
    verdict = "cheap" if pct <= 30 else "expensive" if pct >= 70 else "middling"
    only = "only " if pct < 50 else ""
    named = f" (Trendlyne calls this zone '{zone}')" if zone else ""
    return f"today's P/E is higher than {only}{pct:g}% of its own history - {verdict} against its past{named}"
```

In `tools/trendlyne.py` import `pe_zone_meaning` beside `view`, and before building the answer:

```python
        meaning = pe_zone_meaning(v.get("pe_zone"), v.get("pe_zone_pct"))
        if meaning:
            v = {**v, "pe_zone_meaning": meaning}
```

- [ ] **Step 4: Run** the test file → pass; full suite → 0 failures. **Commit:** skipped (user rule).

---

### Task 3: XBRL reader and integrated-filing parsers (spec A1, part 1)

**Files:**
- Create: `pennystock/data/xbrl.py`, `pennystock/data/integrated_filings.py` (parsers only in this task)
- Create fixtures: `tests/fixtures/xbrl/dixon_if_fy26_consolidated.xml`, `dixon_if_h1fy26_consolidated.xml`,
  `dixon_rpt_h2fy26.xml` (trimmed copies of the real filings in the scratchpad `research/`)
- Test: `tests/test_xbrl.py`, `tests/test_integrated_filings.py`

**Interfaces:**
- Produces: `xbrl.contexts(xml) -> dict[str, Context]`, `xbrl.facts(xml) -> list[Fact]`, `xbrl.number(fact)`;
  `Context(id, start, end, instant, dimensional)` with `.days`; `Fact(name, value, context, unit, decimals)`.
- Produces: `integrated_filings.financial_facts(xml) -> dict` = {key: {span: {"end", "value"}}, "auditor": str|None}
  with keys revenue_cr, profit_to_owners_cr, equity_to_owners_cr, equity_total_cr, basic_eps, exceptional_cr and
  spans FY / H / Q / I; `rpt_rows(xml) -> list[RptRow]`, `RptRow(entity, counterparty, relationship, type, details,
  amount_cr, approved_cr)`; `quarter_end("31-MAR-2026") -> date`; `half_label(date) -> (label, start, end)`.

- [ ] **Step 1: Create the fixtures** from the real files with this one-off script (scratchpad, not the repo):

```python
import re
from pathlib import Path
R = Path("/private/tmp/claude-501/-Users-saketh-PennyStockLLM/23edcb08-bd7a-4fda-8176-7dace827f966/scratchpad/research")
F = Path("tests/fixtures/xbrl"); F.mkdir(parents=True, exist_ok=True)
HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" ' \
       'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:in-capmkt="http://www.sebi.gov.in/xbrl/in-capmkt">\n'

def trim(src: str, keep_facts, keep_contexts=None) -> str:
    x = (R / src).read_text()
    ctx = dict(re.findall(r'(<xbrli:context id="([^"]+)">.*?</xbrli:context>)', x, re.S)[::-1] and
               [(m.group(2), m.group(1)) for m in re.finditer(r'(<xbrli:context id="([^"]+)">.*?</xbrli:context>)', x, re.S)])
    facts = [m.group(0) for m in re.finditer(r'<in-capmkt:(\w+) contextRef="([^"]+)"[^>]*>[^<]*</in-capmkt:\1>', x)
             if keep_facts(m.group(1), m.group(2))]
    used = {re.search(r'contextRef="([^"]+)"', f).group(1) for f in facts} | set(keep_contexts or ())
    return HEAD + "\n".join(ctx[c] for c in sorted(used) if c in ctx) + "\n" + "\n".join(facts) + "\n</xbrli:xbrl>\n"

FIN = {"RevenueFromOperations", "ProfitOrLossAttributableToOwnersOfParent", "ProfitLoss", "Equity",
       "EquityAttributableToOwnersOfParent", "BasicEarningsLossPerShareFromContinuingOperations",
       "ExceptionalItemsBeforeTax", "AuditorsFirmName"}
(F / "dixon_if_fy26_consolidated.xml").write_text(trim("dixon_if_q4fy26_0.xml", lambda n, c: n in FIN))
# the Sep-2025 consolidated file is fetched again (it was not saved): see the plan's Task 3 note
RPT_KEEP = ("Padget Electronics Pvt. Ltd", "Ismartu India Private Limited", "KUNSHAN Q TECHNOLOGY",
            "BHARTI AIRTEL LTD", "Lightanium", "Director")
```

The RPT fixture keeps the rows (all facts of their `D_RelatedPartyTransactionN` context) whose counterparty or
relationship contains one of `RPT_KEEP`, at most 10 rows; the H1 fixture is fetched with `NseFeed("DIXON")` from the
`30-SEP-2025` Consolidated row's `xbrl` URL and trimmed to `RevenueFromOperations` (+ its contexts). Each fixture
keeps real values unchanged and one dimensional context (a segment revenue) so the "dimensional facts are ignored"
test has something to ignore.

- [ ] **Step 2: Write the failing tests**

`tests/test_xbrl.py`:

```python
"""Just enough XBRL for NSE's filings - contexts, facts, and the trap that segment rows reuse the same names."""
from pennystock.data import xbrl

DOC = """<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance">
<xbrli:context id="FY"><xbrli:period><xbrli:startDate>2025-04-01</xbrli:startDate>
<xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="Seg"><xbrli:period><xbrli:startDate>2025-04-01</xbrli:startDate>
<xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period><xbrli:scenario><xbrldi:explicitMember
dimension="a:SegmentsAxis">a:MobilesMember</xbrldi:explicitMember></xbrli:scenario></xbrli:context>
<xbrli:context id="END"><xbrli:period><xbrli:instant>2026-03-31</xbrli:instant></xbrli:period></xbrli:context>
<in-capmkt:RevenueFromOperations contextRef="FY" decimals="-5" unitRef="INR">488728000000</in-capmkt:RevenueFromOperations>
<in-capmkt:RevenueFromOperations contextRef="Seg" unitRef="INR">99000000000</in-capmkt:RevenueFromOperations>
<in-capmkt:AuditorsFirmName contextRef="FY">S.N. DHAWAN &amp; CO LLP</in-capmkt:AuditorsFirmName>
</xbrli:xbrl>"""


def test_contexts_carry_period_length_and_whether_they_are_a_breakdown():
    ctx = xbrl.contexts(DOC)
    assert ctx["FY"].days == 365 and not ctx["FY"].dimensional
    assert ctx["Seg"].dimensional and ctx["END"].instant == "2026-03-31"


def test_facts_are_read_without_their_prefix_and_unescaped():
    facts = xbrl.facts(DOC)
    revenue = [f for f in facts if f.name == "RevenueFromOperations"]
    assert len(revenue) == 2 and xbrl.number(revenue[0]) == 488728000000.0 and revenue[0].unit == "INR"
    assert next(f.value for f in facts if f.name == "AuditorsFirmName") == "S.N. DHAWAN & CO LLP"
```

`tests/test_integrated_filings.py` (parsers part):

```python
"""NSE integrated filings: audited figures and the half-yearly related-party disclosure, from real Dixon XBRL."""
from datetime import date
from pathlib import Path

from pennystock.data import integrated_filings as itf

FIX = Path(__file__).parent / "fixtures" / "xbrl"


def test_year_quarter_and_balance_sheet_figures_come_from_undimensioned_contexts():
    fin = itf.financial_facts((FIX / "dixon_if_fy26_consolidated.xml").read_text())
    assert fin["revenue_cr"]["FY"] == {"end": "2026-03-31", "value": 48872.8}
    assert fin["revenue_cr"]["Q"]["value"] == 10510.51
    assert fin["profit_to_owners_cr"]["FY"]["value"] == 1438.64
    assert fin["equity_to_owners_cr"]["I"] == {"end": "2026-03-31", "value": 4676.67}
    assert fin["basic_eps"]["FY"]["value"] == 271.59 and fin["exceptional_cr"]["FY"]["value"] == 0.0
    assert fin["auditor"] == "S.N. DHAWAN & CO LLP"


def test_related_party_rows_are_read_in_crore():
    rows = itf.rpt_rows((FIX / "dixon_rpt_h2fy26.xml").read_text())
    airtel = next(r for r in rows if "AIRTEL" in r.counterparty.upper())
    assert airtel.amount_cr == 1224.67 and airtel.relationship == "Entity having significant influence"
    assert airtel.entity.startswith("Dixon Electro Appliances")


def test_quarter_ends_name_their_half_year():
    assert itf.quarter_end("31-MAR-2026") == date(2026, 3, 31)
    assert itf.half_label(date(2026, 3, 31)) == ("H2 FY26", "2025-10-01", "2026-03-31")
    assert itf.half_label(date(2025, 9, 30)) == ("H1 FY26", "2025-04-01", "2025-09-30")
```

- [ ] **Step 3: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_xbrl.py tests/test_integrated_filings.py` →
  FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implement `pennystock/data/xbrl.py`:**

```python
"""Just enough XBRL to read NSE's filings: each fact's name, context (period, and whether the context is a
breakdown along some dimension), value, unit and decimals.

NSE's instances are flat and small (15-650 KB): a regex pass reads one in milliseconds and needs no taxonomy
download. Names are matched WITHOUT their prefix (in-bse-shp:, in-capmkt:), which differs by taxonomy and year.
Segment and line-item breakdowns reuse the same concept names in dimensional contexts, so callers that want the
company's own figure filter on `context.dimensional` (an integrated filing's segment revenue is not its revenue).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date

_CONTEXT = re.compile(r"<(?:xbrli:)?context\b[^>]*?\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S)
_START = re.compile(r"<(?:xbrli:)?startDate>\s*([^<\s]+)")
_END = re.compile(r"<(?:xbrli:)?endDate>\s*([^<\s]+)")
_INSTANT = re.compile(r"<(?:xbrli:)?instant>\s*([^<\s]+)")
_DIMENSION = re.compile(r"explicitMember|typedMember")
_FACT = re.compile(r"<([A-Za-z][\w\-]*):([A-Za-z]\w*)\b([^>]*?\bcontextRef=\"([^\"]+)\"[^>]*?)"
                   r"(?:/>|>(.*?)</\1:\2\s*>)", re.S)
_UNIT = re.compile(r"\bunitRef=\"([^\"]+)\"")
_DECIMALS = re.compile(r"\bdecimals=\"([^\"]+)\"")


@dataclass(frozen=True)
class Context:
    id: str
    start: str | None
    end: str | None
    instant: str | None
    dimensional: bool

    @property
    def days(self) -> int | None:
        if not (self.start and self.end):
            return None
        return (date.fromisoformat(self.end[:10]) - date.fromisoformat(self.start[:10])).days + 1


@dataclass(frozen=True)
class Fact:
    name: str
    value: str
    context: Context
    unit: str | None
    decimals: str | None


def contexts(xml: str) -> dict[str, Context]:
    out = {}
    for cid, body in _CONTEXT.findall(xml):
        s, e, i = _START.search(body), _END.search(body), _INSTANT.search(body)
        out[cid] = Context(cid, s.group(1) if s else None, e.group(1) if e else None,
                           i.group(1) if i else None, bool(_DIMENSION.search(body)))
    return out


def facts(xml: str) -> list[Fact]:
    ctx = contexts(xml)
    out = []
    for _prefix, name, attrs, cref, value in _FACT.findall(xml):
        if cref in ctx:
            u, d = _UNIT.search(attrs), _DECIMALS.search(attrs)
            out.append(Fact(name, html.unescape((value or "").strip()), ctx[cref],
                            u.group(1) if u else None, d.group(1) if d else None))
    return out


def number(fact: Fact | None) -> float | None:
    if fact is None:
        return None
    try:
        return float(fact.value.replace(",", ""))
    except ValueError:
        return None
```

- [ ] **Step 5: Implement the parsers in `pennystock/data/integrated_filings.py`** (store and fetch come in Task 4):

```python
"""NSE integrated filings - a company's audited figures and its half-yearly related-party disclosure, as XBRL.

Since December 2024 every listed company files "Integrated Filing - Financials" with NSE as XBRL
(/api/integrated-filing-results). Measured on DIXON (2026-10-11): the March consolidated filing carries the year's
revenue (Rs 48,872.80 Cr), profit to owners (Rs 1,438.64 Cr), equity to owners (Rs 4,676.67 Cr), basic EPS and the
auditor; the September and March filings carry the SEBI Reg 23(9) related-party disclosure - itemised, with each
counterparty's relationship. Two traps, both measured: segment figures reuse the same names in dimensional contexts,
and related-party amounts cover the HALF-YEAR although their contexts carry the quarter's dates (Airtel: Sep
1,271.99 + Mar 1,224.67 = 2,496.66 Cr, exactly the AGM notice's FY26 figure).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

from pennystock.data.xbrl import Context, facts, number

logger = logging.getLogger(__name__)

FINANCIALS = "Integrated Filing- Financials"
CRORE = 1e7
# the first concept a filing carries wins: standalone filings have no owners/NCI split
_CONCEPTS = {
    "revenue_cr": ("RevenueFromOperations",),
    "profit_to_owners_cr": ("ProfitOrLossAttributableToOwnersOfParent", "ProfitLossForPeriod", "ProfitLoss"),
    "equity_to_owners_cr": ("EquityAttributableToOwnersOfParent",),
    "equity_total_cr": ("Equity",),
    "basic_eps": ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                  "BasicEarningsLossPerShareFromContinuingOperations", "BasicEarningsLossPerShare"),
    "exceptional_cr": ("ExceptionalItemsBeforeTax", "ExceptionalItems"),
}
_RPT_FIELDS = {
    "NameOfListedEntityOrSubsidiaryEnteringIntoTheTransaction": "entity",
    "NameOfCounterParty": "counterparty",
    "RelationshipOfTheCounterpartyWithTheListedEntityOrItsSubsidiary": "relationship",
    "TypeOfRelatedPartyTransaction": "type",
    "DetailsOfOtherRelatedPartyTransaction": "details",
    "AmountOfRelatedPartyTransactionDuringTheReportingPeriod": "amount",
    "ValueOfTheRelatedPartyTransactionAsApprovedByTheAuditCommittee": "approved",
}


@dataclass(frozen=True)
class RptRow:
    entity: str
    counterparty: str
    relationship: str
    type: str
    details: str
    amount_cr: float | None
    approved_cr: float | None


def _span(ctx: Context) -> str | None:
    if ctx.instant:
        return "I"
    d = ctx.days
    if d is None:
        return None
    return "FY" if 330 <= d <= 380 else "H" if 170 <= d <= 190 else "Q" if 80 <= d <= 100 else None


def financial_facts(xml: str) -> dict:
    """{key: {span: {"end", "value"}}} for the company's own figures (undimensioned contexts only; the newest end
    per span wins), money in Rs crore, EPS in rupees; plus "auditor"."""
    found = facts(xml)
    out: dict = {}
    for key, names in _CONCEPTS.items():
        for name in names:
            picks: dict = {}
            for f in found:
                if f.name != name or f.context.dimensional:
                    continue
                span, v = _span(f.context), number(f)
                if span is None or v is None:
                    continue
                end = (f.context.instant or f.context.end)[:10]
                if span not in picks or end > picks[span]["end"]:
                    picks[span] = {"end": end, "value": v if key == "basic_eps" else round(v / CRORE, 2)}
            if picks:
                out[key] = picks
                break
    out["auditor"] = next((f.value for f in found if f.name == "AuditorsFirmName" and f.value), None)
    return out


def _money(raw: str | None) -> float | None:
    try:
        return round(float(str(raw).replace(",", "")) / CRORE, 2)
    except (TypeError, ValueError):
        return None


def rpt_rows(xml: str) -> list[RptRow]:
    """Every related-party transaction in a filing, in document order, amounts in Rs crore."""
    by_ctx: dict[str, dict] = {}
    for f in facts(xml):
        key = _RPT_FIELDS.get(f.name)
        if key:
            by_ctx.setdefault(f.context.id, {})[key] = f.value
    return [RptRow(d.get("entity", ""), d["counterparty"], d.get("relationship", ""), d.get("type", ""),
                   d.get("details", ""), _money(d.get("amount")), _money(d.get("approved")))
            for d in by_ctx.values() if d.get("counterparty")]


def quarter_end(qe: str) -> date | None:
    try:
        return datetime.strptime(str(qe).strip().title(), "%d-%b-%Y").date()
    except ValueError:
        return None


def half_label(end: date) -> tuple[str, str, str] | None:
    """The half-year a related-party disclosure covers, named from the filing's quarter end."""
    if end.month == 3:
        return f"H2 FY{end.year % 100:02d}", f"{end.year - 1}-10-01", end.isoformat()
    if end.month == 9:
        return f"H1 FY{(end.year + 1) % 100:02d}", f"{end.year}-04-01", end.isoformat()
    return None
```

- [ ] **Step 6: Run** both test files → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 4: Integrated filings - store, fetch and warm-up script (spec A1, part 2)

**Files:**
- Modify: `pennystock/data/integrated_filings.py` (add `FiledYear`, `RptDisclosure`, `IntegratedFilings`,
  `filed_year`, `rpt_disclosure`), `pennystock/config.py` (`integrated_filings_max_age_days = 7`,
  `integrated_parser_version = 1`)
- Create: `scripts/ingest_integrated_filings.py`
- Test: `tests/test_integrated_filings.py` (append)

**Interfaces:**
- Consumes: Task 3 parsers.
- Produces: `filed_year(symbol) -> FiledYear | None` (symbol, fiscal_year "FY26", period_end, basis, revenue_cr,
  profit_to_owners_cr, equity_to_owners_cr, equity_total_cr, basic_eps, exceptional_cr, auditor, broadcast_at, url);
  `rpt_disclosure(symbol) -> RptDisclosure | None` (symbol, half, period_start, period_end, rows, broadcast_at, url,
  revenue_cr, revenue_note). Consumed by Tasks 5, 7, 11, 12.

- [ ] **Step 1: Write the failing tests** (append; `FakeFeed` serves the listing and the fixture files):

```python
import json

LISTING = {"data": [
    {"qe_Date": "31-MAR-2026", "consolidated": "Consolidated", "type": itf.FINANCIALS,
     "broadcast_Date": "12-May-2026 23:38:17", "xbrl": "u/fy26c"},
    {"qe_Date": "31-MAR-2026", "consolidated": "Standalone", "type": itf.FINANCIALS,
     "broadcast_Date": "12-May-2026 23:38:14", "xbrl": "u/fy26s"},
    {"qe_Date": "30-SEP-2025", "consolidated": "Consolidated", "type": itf.FINANCIALS,
     "broadcast_Date": "17-Oct-2025 21:15:27", "xbrl": "u/h1c"},
    {"qe_Date": "31-MAR-2026", "consolidated": None, "type": "Integrated Filing- Governance", "xbrl": "u/gov"}]}
FILES = {"u/fy26c": "dixon_if_fy26_consolidated.xml", "u/fy26s": "dixon_rpt_h2fy26.xml",
         "u/h1c": "dixon_if_h1fy26_consolidated.xml"}


class FakeFeed:
    def __init__(self, listing=LISTING):
        self.listing, self.gets = listing, []

    def api(self, path):
        return self.listing

    def get(self, url):
        self.gets.append(url)
        return (FIX / FILES[url]).read_bytes() if url in FILES else None


def store(tmp_path, feed):
    return itf.IntegratedFilings(db_path=tmp_path / "m.db", feed_factory=lambda symbol: feed)


def test_the_filed_year_comes_from_the_march_consolidated_filing(tmp_path):
    fy = store(tmp_path, FakeFeed()).filed_year("DIXON")
    assert (fy.fiscal_year, fy.basis, fy.revenue_cr, fy.profit_to_owners_cr, fy.equity_to_owners_cr) == \
           ("FY26", "consolidated", 48872.8, 1438.64, 4676.67)
    assert fy.broadcast_at.startswith("12-May-2026")


def test_the_disclosure_is_the_half_year_with_its_own_revenue(tmp_path):
    d = store(tmp_path, FakeFeed()).rpt_disclosure("DIXON")
    assert (d.half, d.period_start, d.period_end) == ("H2 FY26", "2025-10-01", "2026-03-31")
    assert d.revenue_cr == 21182.1 and "minus" in d.revenue_note        # 48,872.80 - 27,690.70
    assert any("AIRTEL" in r.counterparty.upper() for r in d.rows)


def test_a_parsed_file_is_never_downloaded_twice(tmp_path):
    feed = FakeFeed()
    s = store(tmp_path, feed)
    s.filed_year("DIXON"), s.filed_year("DIXON")
    assert feed.gets.count("u/fy26c") == 1


def test_no_rows_and_no_feed_return_none(tmp_path):
    s = store(tmp_path, FakeFeed(listing=None))
    assert s.filed_year("BDL") is None and s.rpt_disclosure("BDL") is None
```

- [ ] **Step 2: Run** → FAIL (`IntegratedFilings` missing).

- [ ] **Step 3: Implement** (append to `integrated_filings.py`; add `import json`, `import sqlite3`,
  `from dataclasses import asdict`, `from datetime import timedelta`, `from functools import lru_cache`,
  `from urllib.parse import quote`, `from pennystock.config import settings`,
  `from pennystock.data.nse_client import NseFeed`):

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS integrated_listings (
    symbol TEXT PRIMARY KEY, rows TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS integrated_xbrl (
    url TEXT PRIMARY KEY, symbol TEXT NOT NULL, qe_date TEXT, basis TEXT, broadcast_at TEXT,
    financials TEXT NOT NULL, rpt_rows TEXT NOT NULL, parser_version INTEGER NOT NULL, parsed_at TEXT NOT NULL);
"""


@dataclass(frozen=True)
class FiledYear:
    symbol: str
    fiscal_year: str
    period_end: str
    basis: str
    revenue_cr: float | None
    profit_to_owners_cr: float | None
    equity_to_owners_cr: float | None
    equity_total_cr: float | None
    basic_eps: float | None
    exceptional_cr: float | None
    auditor: str | None
    broadcast_at: str | None
    url: str


@dataclass(frozen=True)
class RptDisclosure:
    symbol: str
    half: str
    period_start: str
    period_end: str
    rows: list
    broadcast_at: str | None
    url: str
    revenue_cr: float | None
    revenue_note: str


def _value(fin: dict, key: str, span: str) -> float | None:
    return ((fin.get(key) or {}).get(span) or {}).get("value")


class IntegratedFilings:
    """The cache in market.db: a listing per company (refreshed weekly) and each XBRL parsed once - a filing's URL
    never changes (a revision is a new URL)."""

    def __init__(self, db_path=None, feed_factory=None):
        self.db_path = str(db_path or settings.market_db)
        self._feed_factory = feed_factory or (lambda symbol: NseFeed(symbol))
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def listing(self, symbol: str, refresh: bool = False) -> list[dict]:
        symbol = symbol.upper()
        with self._connect() as con:
            row = con.execute("SELECT rows, fetched_at FROM integrated_listings WHERE symbol = ?",
                              (symbol,)).fetchone()
        if row and not refresh and datetime.now() - datetime.fromisoformat(row[1]) < timedelta(
                days=settings.integrated_filings_max_age_days):
            return json.loads(row[0])
        answer = self._feed_factory(symbol).api(f"integrated-filing-results?index=equities&symbol={quote(symbol)}")
        data = answer.get("data") if isinstance(answer, dict) else None
        if data is None:                                  # NSE down: a stale listing beats none
            return json.loads(row[0]) if row else []
        rows = [r for r in data if r.get("type") == FINANCIALS and r.get("xbrl")]
        with self._connect() as con:
            con.execute("INSERT INTO integrated_listings VALUES (?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                        "rows = excluded.rows, fetched_at = excluded.fetched_at",
                        (symbol, json.dumps(rows), datetime.now().isoformat(timespec="seconds")))
        return rows

    def parsed(self, symbol: str, row: dict) -> dict | None:
        url = row["xbrl"]
        with self._connect() as con:
            hit = con.execute("SELECT financials, rpt_rows FROM integrated_xbrl WHERE url = ? AND parser_version = ?",
                              (url, settings.integrated_parser_version)).fetchone()
        if hit:
            return {"financials": json.loads(hit[0]), "rpt_rows": [RptRow(**r) for r in json.loads(hit[1])]}
        body = self._feed_factory(symbol).get(url)
        if body is None:
            logger.warning(f"integrated filing unavailable: {url}")
            return None
        xml = body.decode("utf-8", "replace")
        fin, rows = financial_facts(xml), rpt_rows(xml)
        with self._connect() as con:
            con.execute("INSERT OR REPLACE INTO integrated_xbrl VALUES (?,?,?,?,?,?,?,?,?)",
                        (url, symbol.upper(), row.get("qe_Date"), row.get("consolidated"), row.get("broadcast_Date"),
                         json.dumps(fin), json.dumps([asdict(r) for r in rows]), settings.integrated_parser_version,
                         datetime.now().isoformat(timespec="seconds")))
        return {"financials": fin, "rpt_rows": rows}

    def _pick(self, rows: list[dict], end: date, basis: str) -> dict | None:
        return next((r for r in rows if quarter_end(r.get("qe_Date")) == end and r.get("consolidated") == basis), None)

    def _ends(self, rows: list[dict], months: tuple[int, ...]) -> list[date]:
        ends = {quarter_end(r.get("qe_Date")) for r in rows}
        return sorted((e for e in ends if e and e.month in months), reverse=True)

    def filed_year(self, symbol: str) -> FiledYear | None:
        rows = self.listing(symbol)
        for end in self._ends(rows, (3,)):
            for basis in ("Consolidated", "Standalone"):
                row = self._pick(rows, end, basis)
                fin = ((self.parsed(symbol, row) or {}).get("financials") or {}) if row else {}
                if _value(fin, "revenue_cr", "FY") is None:
                    continue
                return FiledYear(symbol.upper(), f"FY{end.year % 100:02d}", end.isoformat(), basis.lower(),
                                 _value(fin, "revenue_cr", "FY"), _value(fin, "profit_to_owners_cr", "FY"),
                                 _value(fin, "equity_to_owners_cr", "I"), _value(fin, "equity_total_cr", "I"),
                                 _value(fin, "basic_eps", "FY"), _value(fin, "exceptional_cr", "FY"),
                                 fin.get("auditor"), row.get("broadcast_Date"), row["xbrl"])
        return None

    def _half_revenue(self, symbol: str, rows: list[dict], end: date) -> tuple[float | None, str]:
        """The same half-year's consolidated revenue: a September filing's six-month figure; for March, the year
        minus that September half; half the year when only the year is on file (said so)."""
        def revenue(at: date, span: str) -> float | None:
            row = self._pick(rows, at, "Consolidated") or self._pick(rows, at, "Standalone")
            return _value(((self.parsed(symbol, row) or {}).get("financials") or {}), "revenue_cr", span) if row else None
        if end.month == 9:
            h = revenue(end, "H")
            return (h, f"six months to {end.isoformat()}") if h else (None, "half-year revenue not on file")
        fy, h1 = revenue(end, "FY"), revenue(date(end.year - 1, 9, 30), "H")
        if fy and h1:
            return round(fy - h1, 2), f"FY{end.year % 100:02d} revenue minus the April-September half"
        if fy:
            return round(fy / 2, 2), "approximate: half of the full year's revenue"
        return None, "half-year revenue not on file"

    def rpt_disclosure(self, symbol: str) -> RptDisclosure | None:
        rows = self.listing(symbol)
        for end in self._ends(rows, (3, 9))[:2]:          # the latest half-year, else the one before it
            for basis in ("Standalone", "Consolidated"):
                row = self._pick(rows, end, basis)
                items = ((self.parsed(symbol, row) or {}).get("rpt_rows") or []) if row else []
                if items:
                    label, start, stop = half_label(end)
                    revenue, note = self._half_revenue(symbol, rows, end)
                    return RptDisclosure(symbol.upper(), label, start, stop, items, row.get("broadcast_Date"),
                                         row["xbrl"], revenue, note)
        return None


@lru_cache(maxsize=1)
def _store() -> IntegratedFilings:
    return IntegratedFilings()


def filed_year(symbol: str) -> FiledYear | None:
    try:
        return _store().filed_year(symbol)
    except Exception as e:                                  # a yardstick is a bonus, never a crash
        logger.warning(f"filed year unavailable for {symbol}: {type(e).__name__}: {e}")
        return None


def rpt_disclosure(symbol: str) -> RptDisclosure | None:
    try:
        return _store().rpt_disclosure(symbol)
    except Exception as e:
        logger.warning(f"related-party disclosure unavailable for {symbol}: {type(e).__name__}: {e}")
        return None
```

- [ ] **Step 4: Create `scripts/ingest_integrated_filings.py`:**

```python
"""Warm the NSE integrated-filing cache (audited figures + the half-yearly related-party disclosure) for every
company in the documents library, and report coverage.

  python scripts/ingest_integrated_filings.py --status
  python scripts/ingest_integrated_filings.py                 # every company; a parsed file is never re-downloaded
  python scripts/ingest_integrated_filings.py --symbol DIXON
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pennystock.config import settings                                      # noqa: E402
from pennystock.data.integrated_filings import IntegratedFilings            # noqa: E402


def companies() -> list[str]:
    with sqlite3.connect(str(settings.filings_db)) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM filing_docs ORDER BY symbol")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol", nargs="*")
    args = ap.parse_args()
    store = IntegratedFilings()
    symbols = [s.upper() for s in args.symbol] if args.symbol else companies()
    years = rpts = 0
    for n, sym in enumerate(symbols, 1):
        fy, rpt = store.filed_year(sym), store.rpt_disclosure(sym)
        years += fy is not None
        rpts += rpt is not None
        if not args.status:
            print(f"[{n}/{len(symbols)}] {sym:<12} filed year: {fy.fiscal_year if fy else '-':<5} "
                  f"related parties: {f'{rpt.half}, {len(rpt.rows)} rows' if rpt else '-'}", flush=True)
            time.sleep(settings.refresh_delay_s)
    print(f"\n{len(symbols)} companies: filed year {years} ({years / max(len(symbols), 1):.0%}), "
          f"related-party disclosure {rpts} ({rpts / max(len(symbols), 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(`--status` still reads the listing, which is cached for a week, so it is offline after a warm run.)

- [ ] **Step 5: Run** the test file → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 5: Related-party summary (spec A2)

**Files:**
- Create: `pennystock/data/related_parties.py`
- Test: `tests/test_related_parties.py`

**Interfaces:**
- Consumes: `RptDisclosure`, `RptRow` (Task 4).
- Produces: `classify(relationship) -> str` (one of `CLASSES`); `summarize_rpt(d: RptDisclosure) -> dict` with keys
  source, period, filed, transactions, half_year_revenue_cr, revenue_note, by_class_cr, outside_subsidiaries_cr,
  outside_subsidiaries_pct_of_revenue, promoter_kmp_entities_cr, promoter_kmp_entities_pct_of_revenue,
  largest_outside_subsidiaries, loans_guarantees_outside_subsidiaries, how_to_judge. Consumed by Tasks 11, 13.

- [ ] **Step 1: Write the failing tests:**

```python
"""Related-party transactions from the XBRL disclosure: classified by relationship, judged against revenue."""
import pytest

from pennystock.data.integrated_filings import RptDisclosure, RptRow
from pennystock.data.related_parties import classify, summarize_rpt


@pytest.mark.parametrize("text, cls", [
    ("Subsidiary", "subsidiary"),
    ("Wholly Owned Subsidiary of Lightanium technologies Private Limited", "subsidiary"),
    ("Joint Venture", "jv_associate"),
    ("Entities over which holding company are able to exercise control/significant influence", "jv_associate"),
    ("Entity having significant influence", "group_or_partner"),
    ("Entity under common control", "group_or_partner"),
    ("Fellow subsidiary", "group_or_partner"),
    ("Enterprises over which KMP and their relatives exercise significant influence", "promoter_kmp_entity"),
    ("Promoter group company", "promoter_kmp_entity"),
    ("Executive Director", "people"),
    ("KMP", "people"),
    ("Trust for employees", "other"),
])
def test_relationships_are_classified(text, cls):
    assert classify(text) == cls


def _row(entity, counterparty, relationship, kind, amount, details=""):
    return RptRow(entity, counterparty, relationship, kind, details, amount, None)


DIXON = RptDisclosure("DIXON", "H2 FY26", "2025-10-01", "2026-03-31", [
    _row("Padget Electronics Private Limited", "Ismartu India Private Limited",
         "Entities over which holding company are able to exercise control/significant influence",
         "Sale of goods or services", 3209.9),
    _row("Dixon Electro Appliances Private Limited", "BHARTI AIRTEL LTD", "Entity having significant influence",
         "Sale of goods or services", 1224.67),
    _row("Dixon Technologies (India) Limited", "Padget Electronics Pvt. Ltd", "Subsidiary",
         "Any other transaction", 567.9, "Corporate guarantee given"),
    _row("Dixon Technologies (India) Limited", "Ms X", "Executive Director", "Remuneration", 5.2),
    _row("Dixon Technologies (India) Limited", "Lightanium Technologies Private Limited", "Joint Venture",
         "Any other transaction", 142.8, "Loan given")],
    "12-May-2026 23:38:14", "u/fy26s", 21182.1, "FY26 revenue minus the April-September half")


def test_outside_the_subsidiaries_is_judged_against_the_half_years_revenue():
    s = summarize_rpt(DIXON)
    assert s["period"] == "H2 FY26 (2025-10-01 to 2026-03-31)" and s["filed"] == "12-May-2026"
    assert s["outside_subsidiaries_cr"] == 4582.57                    # 3209.9 + 1224.67 + 5.2 + 142.8
    assert s["outside_subsidiaries_pct_of_revenue"] == 21.6
    assert s["promoter_kmp_entities_cr"] == 0.0 and s["by_class_cr"]["subsidiary"] == 567.9
    top = s["largest_outside_subsidiaries"][0]
    assert top["counterparty"] == "Ismartu India Private Limited" and top["class"] == "jv_associate"


def test_loans_and_guarantees_to_non_subsidiaries_are_named():
    loans = summarize_rpt(DIXON)["loans_guarantees_outside_subsidiaries"]
    assert [x["counterparty"] for x in loans] == ["Lightanium Technologies Private Limited"]
    assert loans[0]["type"] == "Loan given"
```

- [ ] **Step 2: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_related_parties.py` → FAIL (no module).

- [ ] **Step 3: Implement `pennystock/data/related_parties.py`:**

```python
"""Related-party transactions from the company's own half-yearly disclosure to NSE (SEBI LODR Reg 23(9), XBRL).

The classic Indian governance check is value leaving a listed company for entities its promoters own. The
disclosure itemises every transaction - entity, counterparty, relationship (free text), type, amount - so it can be
totalled by kind of relationship instead of scraped from a PDF table. Subsidiaries are eliminated on consolidation,
so what matters is what flows OUTSIDE them, against the same half-year's revenue (~10% is the long-standing
threshold). The relationship text cannot always say whose group a counterparty belongs to: "common control" is a
promoter's group in a family conglomerate and a joint-venture partner's group at DIXON (Bharti Airtel holds 47.58%
of a Dixon subsidiary), so that class is named for what it is and the counterparties are always shown.
"""
from __future__ import annotations

import re

from pennystock.data.integrated_filings import RptDisclosure, RptRow

CLASSES = ("subsidiary", "jv_associate", "promoter_kmp_entity", "group_or_partner", "people", "other")
_PEOPLE_WORDS = r"(kmp|key\s+manageri\w*|directors?|relatives?|promoters?|individuals?|whole[\s-]?time)"
_OWNED_BY_PEOPLE = re.compile(r"(enterprise|entit|compan|firm|trust|concern|llp|huf)\w*\b.{0,60}?\b(owned|controlled|"
                              rf"influenced|over\s+which|in\s+which|where)\b.{{0,100}}?\b{_PEOPLE_WORDS}", re.I)
_PROMOTER_ENTITY = re.compile(r"promoter['’]?s?\s+(group|compan|entit|firm|enterprise)", re.I)
_OWN_CONTROL = re.compile(r"over\s+which\s+(the\s+)?(holding\s+|listed\s+)?(company|entity)\b.{0,40}"
                          r"\b(control|influence)", re.I)
_FELLOW = re.compile(r"fellow\s+subsidiar", re.I)
_SUBSIDIARY = re.compile(r"subsidiar", re.I)
_JV = re.compile(r"joint\s+venture|\bjv\b|associate", re.I)
_GROUP = re.compile(r"holding\s+compan|common\s+control|significant\s+influence|investing\s+part|venturer|"
                    r"group\s+compan", re.I)
_PEOPLE = re.compile(rf"\b{_PEOPLE_WORDS}|managing|chief\s+(executive|financial)|company\s+secretary|\bceo\b|\bcfo\b",
                     re.I)
_LOANLIKE = re.compile(r"\bloan|guarantee|advance|deposit|investment|security\s+given|inter[\s-]?corporate", re.I)
HOW_TO_JUDGE = (
    "Subsidiaries are inside the group (eliminated on consolidation). Judge what flows OUTSIDE them against the "
    "half-year's revenue (~10% is the long-standing threshold) and say who the counterparties are: "
    "promoter_kmp_entity is the classic channel for value to leave; group_or_partner can be a promoter's group OR a "
    "joint-venture partner's group - the names tell which. Amounts are as disclosed and can include taxes.")


def classify(relationship: str) -> str:
    r = relationship or ""
    if _OWNED_BY_PEOPLE.search(r) or _PROMOTER_ENTITY.search(r):
        return "promoter_kmp_entity"
    if _OWN_CONTROL.search(r):                     # entities the listed company itself controls or influences
        return "jv_associate"
    if _FELLOW.search(r):
        return "group_or_partner"
    if _SUBSIDIARY.search(r):
        return "subsidiary"
    if _JV.search(r):
        return "jv_associate"
    if _GROUP.search(r):
        return "group_or_partner"
    if _PEOPLE.search(r):
        return "people"
    return "other"


def _describe(cls: str, r: RptRow) -> dict:
    kind = r.details if r.type.lower().startswith("any other") and r.details else r.type
    return {"entity": r.entity, "counterparty": r.counterparty, "class": cls, "relationship": r.relationship,
            "type": kind, "amount_cr": r.amount_cr}


def summarize_rpt(d: RptDisclosure) -> dict:
    rows = [(classify(r.relationship), r) for r in d.rows if r.amount_cr]
    by_class: dict[str, float] = {}
    for cls, r in rows:
        by_class[cls] = by_class.get(cls, 0.0) + r.amount_cr
    outside = [(c, r) for c, r in rows if c != "subsidiary"]
    outside_total = round(sum(r.amount_cr for _, r in outside), 2)
    pk_total = round(by_class.get("promoter_kmp_entity", 0.0), 2)

    def pct(x: float) -> float | None:
        return round(x / d.revenue_cr * 100, 1) if d.revenue_cr else None

    loans = [(c, r) for c, r in outside if c != "people" and _LOANLIKE.search(f"{r.type} {r.details}")]
    return {
        "source": "NSE Reg 23(9) XBRL",
        "period": f"{d.half} ({d.period_start} to {d.period_end})",
        "filed": (d.broadcast_at or "")[:11],
        "transactions": len(rows),
        "half_year_revenue_cr": d.revenue_cr,
        "revenue_note": d.revenue_note,
        "by_class_cr": {c: round(v, 2) for c, v in by_class.items()},
        "outside_subsidiaries_cr": outside_total,
        "outside_subsidiaries_pct_of_revenue": pct(outside_total),
        "promoter_kmp_entities_cr": pk_total,
        "promoter_kmp_entities_pct_of_revenue": pct(pk_total),
        "largest_outside_subsidiaries": [_describe(c, r) for c, r in
                                         sorted(outside, key=lambda cr: -cr[1].amount_cr)[:6]],
        "loans_guarantees_outside_subsidiaries": [_describe(c, r) for c, r in
                                                  sorted(loans, key=lambda cr: -cr[1].amount_cr)[:5]],
        "how_to_judge": HOW_TO_JUDGE,
    }
```

- [ ] **Step 4: Run** → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 6: Promoter pledge from the statutory shareholding filing (spec A4)

**Files:**
- Modify: `pennystock/data/ownership_providers.py` (`parse_shp_pledge`, `NseOwnershipProvider.fetch`, store column
  `encumbrance_note`, manual pledge protected), `pennystock/data/ownership.py` (`Quarter.encumbrance_note`, flags),
  `pennystock/tools/market_data.py` (`get_shareholding` adds `pledge_filings_since_quarter`)
- Create fixtures: `tests/fixtures/xbrl/dixon_shp_jun2026.xml`, `tests/fixtures/xbrl/jaynecoind_shp_jun2026.xml`
  (trimmed: contexts `MainI`, `ShareholdingOfPromoterAndPromoterGroup_ContextI`, `ShareholdingPattern_ContextI`;
  facts NumberOfShares, the six `WhetherAnySharesHeldByPromotersAreEncumbered*` flags, NumberOfSharesEncumberedUnderPledged)
- Test: `tests/test_shareholding_xbrl.py`, `tests/test_ownership.py` (append)

**Interfaces:**
- Produces: `parse_shp_pledge(xml) -> dict | None` (promoter_shares, total_shares, pledged_shares, pledge_pct,
  pledge_pct_of_total, encumbrance_note); quarters with `pledge_source == "nse_shp_xbrl"`; tool key
  `pledge_filings_since_quarter`; summary key `encumbrance_note`. Consumed by Tasks 13, 15.

- [ ] **Step 1: Write the failing tests** - `tests/test_shareholding_xbrl.py`:

```python
"""Promoter pledge from each quarter's statutory shareholding filing (XBRL) - the NSE pledge API returns nothing."""
from pathlib import Path

from pennystock.data.ownership import Ownership, Quarter, summarize_ownership
from pennystock.data.ownership_providers import NseOwnershipProvider, OwnershipStore, parse_shp_pledge

FIX = Path(__file__).parent / "fixtures" / "xbrl"


def test_no_pledge_is_a_statutory_zero():
    p = parse_shp_pledge((FIX / "dixon_shp_jun2026.xml").read_text())
    assert p["pledge_pct"] == 0.0 and p["pledged_shares"] == 0.0 and p["promoter_shares"] == 17438673.0
    assert p["encumbrance_note"] is None


def test_a_pledge_is_a_share_of_the_promoters_own_holding():
    p = parse_shp_pledge((FIX / "jaynecoind_shp_jun2026.xml").read_text())
    assert p["pledge_pct"] == 99.87 and p["pledge_pct_of_total"] == 55.08       # 534,804,690 / 535,519,366


def test_a_pledged_flag_without_a_count_is_unknown_not_zero():
    xml = (FIX / "jaynecoind_shp_jun2026.xml").read_text()
    xml = "\n".join(l for l in xml.splitlines() if "NumberOfSharesEncumberedUnderPledged" not in l)
    assert parse_shp_pledge(xml) is None


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if "share-holdings-master" in url:
            return _Resp([{"date": "30-JUN-2026", "pr_and_prgrp": "28.55", "public_val": "71.45",
                           "xbrl": "https://x/SHP_jun.xml"},
                          {"date": "31-MAR-2026", "pr_and_prgrp": "28.68", "public_val": "71.32", "xbrl": None}])
        if url.endswith("SHP_jun.xml"):
            return _Resp(text=(FIX / "dixon_shp_jun2026.xml").read_text())
        raise AssertionError(f"unexpected {url}")


def test_the_provider_reads_pledge_from_the_filing_and_never_calls_the_dead_api(tmp_path, monkeypatch):
    session = _Session()
    prov = NseOwnershipProvider(store=OwnershipStore(tmp_path / "m.db"))
    monkeypatch.setattr(prov, "_session", lambda symbol: session)
    own = prov.fetch("DIXON")
    jun = next(q for q in own.quarters if q.quarter == "Jun 2026")
    assert (jun.pledge_pct, jun.pledge_source, jun.promoter_pct) == (0.0, "nse_shp_xbrl", 28.55)
    assert not any("pledgedata" in u for u in session.urls)


def test_a_quarter_already_read_from_its_filing_is_not_downloaded_again(tmp_path, monkeypatch):
    st = OwnershipStore(tmp_path / "m.db")
    st.save(Ownership("DIXON", [Quarter("Jun 2026", pledge_pct=0.0, pledge_source="nse_shp_xbrl")]))
    session = _Session()
    prov = NseOwnershipProvider(store=st)
    monkeypatch.setattr(prov, "_session", lambda symbol: session)
    prov.fetch("DIXON")
    assert not any(u.endswith("SHP_jun.xml") for u in session.urls)


def test_a_manual_pledge_is_never_overwritten(tmp_path):
    st = OwnershipStore(tmp_path / "m.db")
    st.set_pledge("ABC", "Jun 2026", 12.5)
    st.save(Ownership("ABC", [Quarter("Jun 2026", promoter_pct=40.0, pledge_pct=0.0, pledge_source="nse_shp_xbrl")]))
    q = st.load("ABC").quarters[0]
    assert (q.pledge_pct, q.pledge_source, q.promoter_pct) == (12.5, "manual", 40.0)


def test_a_statutory_zero_is_reported_as_no_pledge_not_unknown():
    own = Ownership("DIXON", [Quarter("Jun 2026", promoter_pct=28.55, pledge_pct=0.0, pledge_pct_of_total=0.0,
                                      pledge_source="nse_shp_xbrl")])
    s = summarize_ownership(own)
    assert s["pledge_band"] == "none (ideal)"
    assert any(f.startswith("NO PROMOTER PLEDGE") for f in s["risk_flags"])
    assert not any("UNKNOWN" in f for f in s["risk_flags"])
```

- [ ] **Step 2: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_shareholding_xbrl.py` → FAIL (ImportError).

- [ ] **Step 3: Implement.** `ownership.py`: `Quarter` gains `encumbrance_note: str | None = None`; in
  `summarize_ownership` replace the UNKNOWN text and add the zero and encumbrance flags:

```python
    if latest.pledge_pct is None:
        flags.append("PLEDGE UNKNOWN - the statutory shareholding filing could not be read for this quarter. Do NOT "
                     "assume it is zero; state that it is unverified.")
    else:
        if latest.pledge_pct == 0 and latest.pledge_source == "nse_shp_xbrl":
            flags.append(f"NO PROMOTER PLEDGE - the statutory shareholding filing for {latest.quarter} reports no "
                         "promoter shares pledged.")
        # (existing SEVERE / HIGH RISK / CAUTION / RISING branches unchanged)
    if latest.encumbrance_note:
        flags.append(f"ENCUMBRANCE: {latest.encumbrance_note} ({latest.quarter} filing).")
```

and add `"encumbrance_note": latest.encumbrance_note,` to the returned dict.

`ownership_providers.py`: add `from pennystock.data.xbrl import facts, number`, then

```python
_PROMOTER_CTX = "ShareholdingOfPromoterAndPromoterGroup_ContextI"
_TOTAL_CTX = "ShareholdingPattern_ContextI"
_ENCUMBERED = "WhetherAnySharesHeldByPromotersAreEncumbered"


def parse_shp_pledge(xml: str) -> dict | None:
    """Promoter pledge from one quarter's statutory shareholding pattern (SEBI LODR Reg 31, XBRL).

    NSE's corporate-pledgedata API returned no rows for ANY company on 2026-10-11 (including ones with 39 pledge
    filings on record), while every quarter's SHP XBRL carries yes/no encumbrance flags and the pledged share count.
    Pledge is a % of the promoters' OWN holding (the market convention) and, beside it, of all shares."""
    fs = facts(xml)

    def at(name: str, ctx: str):
        return next((f for f in fs if f.name == name and f.context.id == ctx), None)

    def flag(suffix: str) -> bool | None:
        f = next((f for f in fs if f.name == _ENCUMBERED + suffix), None)
        return None if f is None else f.value.strip().lower() == "true"

    promoter, total = number(at("NumberOfShares", _PROMOTER_CTX)), number(at("NumberOfShares", _TOTAL_CTX))
    pledged = flag("UnderPledged")
    if promoter is None or pledged is None:
        return None
    count = number(at("NumberOfSharesEncumberedUnderPledged", _PROMOTER_CTX))
    if count is None:
        if pledged:                                    # says pledged, gives no count: unreadable, never zero
            return None
        count = 0.0
    other = [name for name, on in (("a non-disposal undertaking", flag("UnderNonDisposalUndertaking")),
                                   ("another encumbrance", flag("OtherThanByWayOfPledgeOrNDU"))) if on]
    return {"promoter_shares": promoter, "total_shares": total, "pledged_shares": count,
            "pledge_pct": round(count / promoter * 100, 2) if promoter else 0.0,
            "pledge_pct_of_total": round(count / total * 100, 2) if total else None,
            "encumbrance_note": ("promoter shares are also encumbered by " + " and ".join(other)) if other else None}
```

`NseOwnershipProvider`: add `def __init__(self, store: "OwnershipStore | None" = None): self.store = store`, update the
class docstring (the pledge now comes from each quarter's SHP XBRL; corporate-pledgedata is dead), and replace
`fetch` with:

```python
    def fetch(self, symbol: str) -> Ownership:
        symbol = symbol.upper()
        own = Ownership(symbol=symbol, fetched_at=datetime.now().isoformat(timespec="seconds"))
        by_quarter: dict[str, Quarter] = {}
        try:
            s = self._session(symbol)
            filings: list[dict] = []
            try:                                        # --- promoter / public history, 20-25 quarters ---
                filings = s.get(f"{self.BASE}/api/corporate-share-holdings-master?index=equities&symbol={symbol}",
                                timeout=settings.ownership_timeout_s).json() or []
                for r in filings:
                    q = by_quarter.setdefault(norm_quarter(r.get("date")), Quarter(quarter=norm_quarter(r.get("date"))))
                    q.promoter_pct, q.public_pct = _num(r.get("pr_and_prgrp")), _num(r.get("public_val"))
            except Exception as e:
                logger.warning(f"NSE shareholding fetch failed for {symbol}: {type(e).__name__}: {e}")

            # --- pledge, from each recent quarter's statutory XBRL, read once per quarter ---
            done = {q.quarter for q in (self.store or _store()).load(symbol).quarters
                    if q.pledge_source == "nse_shp_xbrl"}
            recent = sorted((r for r in filings if r.get("xbrl")),
                            key=lambda r: quarter_ord(norm_quarter(r.get("date"))), reverse=True)
            for r in recent[: settings.ownership_lookback_quarters + 1]:
                quarter = norm_quarter(r.get("date"))
                if quarter in done:
                    continue
                try:
                    pledge = parse_shp_pledge(s.get(r["xbrl"], timeout=settings.ownership_timeout_s).text)
                except Exception as e:
                    logger.warning(f"SHP XBRL unreadable for {symbol} {quarter}: {type(e).__name__}: {e}")
                    continue
                if pledge is not None:
                    q = by_quarter[quarter]
                    q.pledge_pct, q.pledge_pct_of_total = pledge["pledge_pct"], pledge["pledge_pct_of_total"]
                    q.pledge_source, q.encumbrance_note = "nse_shp_xbrl", pledge["encumbrance_note"]
            own.quarters = sorted(by_quarter.values(), key=lambda q: quarter_ord(q.quarter))
            logger.info(f"NSE: {symbol} -> {len(own.quarters)} quarter(s)")
        except Exception as e:
            logger.warning(f"NSE ownership unavailable for {symbol}: {type(e).__name__}: {e}")
        return own
```

`OwnershipStore`: migrate `ALTER TABLE shareholding ADD COLUMN encumbrance_note TEXT` (same try/except pattern),
`load`/`save` carry it, and `save` protects a manual pledge:

```python
                    "   pledge_pct=CASE WHEN shareholding.pledge_source = 'manual' THEN shareholding.pledge_pct"
                    "                   ELSE COALESCE(excluded.pledge_pct, shareholding.pledge_pct) END,"
                    "   pledge_source=CASE WHEN shareholding.pledge_source = 'manual' THEN 'manual'"
                    "                   ELSE COALESCE(excluded.pledge_source, shareholding.pledge_source) END,"
                    "   pledge_pct_of_total=CASE WHEN shareholding.pledge_source = 'manual'"
                    "                   THEN shareholding.pledge_pct_of_total"
                    "                   ELSE COALESCE(excluded.pledge_pct_of_total, shareholding.pledge_pct_of_total) END,"
                    "   encumbrance_note=COALESCE(excluded.encumbrance_note, shareholding.encumbrance_note),"
```

`_MERGE_FIELDS` gains `"encumbrance_note"`. In `get_shareholding` (market_data.py), after `summary` is built:

```python
        try:                                   # pledges created / invoked since the quarter the filing covers
            ev = await asyncio.to_thread(recent_summary, base)
            q_end = _quarter_end(summary.get("as_of_quarter"))
            summary["pledge_filings_since_quarter"] = [p for p in (ev or {}).get("pledges", [])
                                                       if q_end and p.get("date", "") > q_end]
        except Exception as e:
            logger.warning(f"pledge events unavailable for {symbol}: {e}")
```

with `_quarter_end("Jun 2026") -> "2026-06-30"` (month-end of the quarter label; None if unparseable) defined in
market_data.py.

- [ ] **Step 4: Run** the new file and `tests/test_ownership.py` → pass (update any existing assertion on the old
  UNKNOWN text to the new wording); full suite → 0 failures. **Commit:** skipped.

---

### Task 7: Broker estimates on one basis (spec A5)

**Files:**
- Modify: `pennystock/data/broker_metrics.py` (`_latest_votes`, `anchored_estimates`), `pennystock/tools/broker_research.py`
  (`filed_figures`, tool output, HOW_TO_READ), `pennystock/config.py` (`broker_filed_tolerance_pct = 3.0`)
- Test: `tests/test_broker_metrics.py` (append), `tests/test_broker_tools.py` (update `_wire`)

**Interfaces:**
- Consumes: `filed_year` (Task 4), `get_provider().get_financials`.
- Produces: `anchored_estimates(symbol, filed: dict[tuple[str, str], float], forecasts=2) -> list[dict]` - reported
  view {kpi, period, kind="reported", filed, brokers_on_filed_basis, brokers_adjusted?, meaning?} or, without a filed
  figure, {kpi, period, kind, filed=None, brokers_median, brokers, note}; forecast view {kpi, period,
  kind="forecast", basis, median, low, high, brokers, other_basis?}. `filed_figures(ticker) -> (dict, note)`.
  Consumed by Task 13.

- [ ] **Step 1: Write the failing tests** (test_broker_metrics.py; `load_observations` is replaced by fixture rows):

```python
from pennystock.data import broker_metrics as bm


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
```

In `tests/test_broker_tools.py::_wire` add:

```python
    monkeypatch.setattr(br, "filed_figures", lambda ticker: ({}, None))
    monkeypatch.setattr(br, "anchored_estimates", lambda sym, filed, forecasts=2: [])
```

- [ ] **Step 2: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_broker_metrics.py tests/test_broker_tools.py`
  → FAIL (`anchored_estimates` missing).

- [ ] **Step 3: Implement** in `broker_metrics.py` (refactor `summarize` to call `_latest_votes`; import
  `FORWARD_ESTIMATES, KPI_LABELS` from `broker_reports` if not already imported):

```python
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
```

In `broker_research.py`: import `anchored_estimates`, `filed_year` (integrated_filings), `settings`; add

```python
def filed_figures(ticker: str) -> tuple[dict[tuple[str, str], float], str | None]:
    """{(metric, 'FY26'): value} from the company's own accounts: the NSE integrated filing for its year (audited,
    point-in-time), Yahoo's statements for earlier years."""
    from pennystock.data.providers import get_provider
    out: dict[tuple[str, str], float] = {}
    note = None
    try:
        fin = get_provider().get_financials(ticker, settings.financials_years)
        for y, rev, ni in zip(fin.fiscal_years, fin.revenue_cr or [], fin.net_income_cr or []):
            fy = f"FY{y % 100:02d}"
            if rev is not None:
                out[("revenue_cr", fy)] = rev
            if ni is not None:
                out[("net_profit_cr", fy)] = ni
        note = "Yahoo statements"
    except Exception as e:                         # anchoring is a bonus; the estimates still come back
        logger.warning(f"filed figures unavailable for {ticker}: {type(e).__name__}: {e}")
    fy = filed_year(ticker.rsplit(".", 1)[0])
    if fy:
        if fy.revenue_cr is not None:
            out[("revenue_cr", fy.fiscal_year)] = fy.revenue_cr
        if fy.profit_to_owners_cr is not None:
            out[("net_profit_cr", fy.fiscal_year)] = fy.profit_to_owners_cr
        note = f"NSE integrated filing {fy.fiscal_year} ({(fy.broadcast_at or '')[:11]}); Yahoo for earlier years"
    return out, note
```

and in the tool: `filed, filed_note = await asyncio.to_thread(filed_figures, ticker)`;
`estimates = await asyncio.to_thread(anchored_estimates, base, filed)`; answer keys
`"forward_estimates": estimates, "filed_figures_source": filed_note`; drop the old `_view(v) for v in estimates`
(keep `_view` for sector KPIs). Append to HOW_TO_READ: "forward_estimates: for a reported year `filed` is the
company's own figure; brokers that print a different figure are on an ADJUSTED basis - never mix the two, and
compare a broker's forecast with its own basis (I/B/E/S majority basis)."

- [ ] **Step 4: Run** both files → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 8: Contingent-liability note - locate and unit (spec A3, steps 1-2)

**Files:**
- Create: `pennystock/data/contingent_liabilities.py` (locator + unit in this task)
- Modify: `pennystock/config.py` (`cl_note_chars = 7000`)
- Create fixtures: `tests/fixtures/notes/` page texts from filings.db: `dixon_p474.txt`, `dixon_p475.txt`,
  `astral_p211.txt`, `irctc_p344.txt`, `heromotoco_p238.txt`, plus synthetic `policy.txt` (the accounting-policy
  paragraph) and `csr.txt` (a CSR table under a "contingent liabilities" mention)
- Test: `tests/test_contingent_liabilities.py`

**Interfaces:**
- Produces: `Note(page, basis, text, unit)`; `locate_note(pages: list[dict]) -> Note | None` (pages are
  `filing_text.pages_of` dicts: page, text); `detect_unit(note_text, page_text, earlier: list[str]) -> str | None`
  returning one of "lakh", "crore", "million", "thousand"; `UNIT_TO_CRORE`. Consumed by Task 9.

- [ ] **Step 1: Create the fixtures** with `sqlite3 filings.db "SELECT text FROM filing_pages WHERE content_sha=? AND
  page=?"` for each (symbol's latest annual report), written verbatim.

- [ ] **Step 2: Write the failing tests:**

```python
"""Contingent liabilities: the annual report's note, located, read into rows and VERIFIED against the page."""
from pathlib import Path

import pytest

from pennystock.data import contingent_liabilities as cl

NOTES = Path(__file__).parent / "fixtures" / "notes"


def pages(*names, start=100):
    return [{"page": start + i, "text": (NOTES / n).read_text()} for i, n in enumerate(names)]


def test_the_consolidated_note_is_found_with_its_unit():
    note = cl.locate_note(pages("policy.txt", "dixon_p474.txt", "dixon_p475.txt"))
    assert note.page == 101 and note.unit == "lakh"
    assert "Excise, custom duty and service tax" in note.text


@pytest.mark.parametrize("name, unit", [("astral_p211.txt", "million"), ("irctc_p344.txt", "lakh")])
def test_variant_headings_and_rupee_glyphs_are_read(name, unit):
    # Astral: no "not acknowledged as debt"; IRCTC: "not acknowledge as debt" and a rupee sign extracted as "J"
    note = cl.locate_note(pages(name))
    assert note is not None and note.unit == unit


def test_policy_prose_and_a_csr_table_are_not_the_note():
    assert cl.locate_note(pages("policy.txt", "csr.txt")) is None


def test_the_unit_falls_back_to_the_reports_all_amounts_line():
    assert cl.detect_unit("Claims not acknowledged as debt 1,234 1,100", "no unit here",
                          ["(All amounts in ₹ crores unless otherwise stated)"]) == "crore"
```

- [ ] **Step 3: Run** → FAIL (no module).

- [ ] **Step 4: Implement the locator and unit detector:**

```python
"""Contingent liabilities, read from the annual report and VERIFIED in code.

No structured feed carries this note (the integrated filing does not), and Indian reports print it a dozen ways:
measured over 118 latest reports (2026-10-11), headings vary ("contingent liabilities and commitments (to the
extent not provided for)", "not acknowledge as debt"), units are lakh / crore / million / thousand, the rupee sign
extracts as ₹, `, J or H, and banks file a different schedule. A regex-only parser read 94 notes but picked the
wrong table for several (a CSR table, an ESOP valuation table). So:

  locate   code scores pages on the heading, claim language, figures and the consolidated running header
  unit     code reads lakh / crore / million / thousand from the note, the page or the 'all amounts in' line
  rows     the local model lays the table out as rows, numbers AS PRINTED (it computes nothing)
  verify   every number must occur in the note's own text; a row that fails is dropped (Task 9)
  totals   computed here in Rs crore by category; paid-under-protest, commitments and subtotals excluded (Task 9)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pennystock.config import settings

HEAD = re.compile(r"contingent\s+liabilit(?:y|ies)", re.I)
STRONG = re.compile(r"not\s+acknowledged?\s+as\s+debts?", re.I)
CLAIMS = re.compile(r"not\s+acknowledged?\s+as\s+debts?|claims?\s+against\s+the\s+(company|group|bank|corporation)|"
                    r"in\s+respect\s+of\s+(income[\s-]?tax|excise|customs|gst|goods\s+and\s+service|service\s+tax|"
                    r"sales\s+tax|indirect\s+tax)|disputed\s+(tax|demand|liabilit)|income[\s-]?tax\s+(matters|demands?)|"
                    r"guarantees?\s+(given|issued|provided|excluding)", re.I)
POLICY = re.compile(r"contingent\s+liabilit\w*\s+(is|are|exists?)\s+(a\s+possible|disclosed\s+(when|where|in)|not\s+"
                    r"recogni[sz]ed)|possible\s+obligation\s+(that|which|arising)|a\s+contingent\s+liability\s+exists",
                    re.I)
REJECT = re.compile(r"corporate\s+social\s+responsibility|\bcsr\b|dividend\s+yield|black[\s-]?scholes|"
                    r"expected\s+volatility", re.I)
CONSOLIDATED = re.compile(r"consolidated", re.I)
STANDALONE = re.compile(r"standalone|separate\s+financial\s+statements", re.I)
FIGURE = re.compile(r"\(?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\)?|\b\d+\.\d{1,2}\b")
NEXT_NOTE = re.compile(r"^\s*\d{1,3}(?:\.\d+)?\s*[\.\)]?\s+[A-Z][A-Za-z ,&/\-()']{6,}$")
_UNIT_WORD = r"(lakhs?|lacs?|crores?|cr\b|millions?|mn\b|thousands?|'000)"
UNIT = re.compile(r"(?:₹|`|\bJ\b|\bH\b|rs\.?|inr|rupees|amount|amounts|figures)\s*(?:in|/|\()?\s*(?:₹|`|rs\.?|inr)?"
                  rf"\s*{_UNIT_WORD}", re.I)
ALL_AMOUNTS = re.compile(r"(?:all\s+amounts?|amounts?|figures)\s+(?:are\s+)?(?:in|stated\s+in|expressed\s+in)\s+"
                         rf"(?:₹|`|rs\.?|inr|indian\s+rupees?)?\s*{_UNIT_WORD}", re.I)
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
    """Consolidated or standalone, from the running header or footer of this page or the nearest page before it."""
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
            claims, figures = len(CLAIMS.findall(head)), len(FIGURE.findall(head))
            if not claims or figures < 2:
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
```

- [ ] **Step 5: Run** → pass (adjust only the synthetic fixtures if a regex choice needs a ruling; ledger it). Full
  suite → 0 failures. **Commit:** skipped.

---

### Task 9: Contingent-liability rows - model, verify, totals, store, summary (spec A3, steps 3-8)

**Files:**
- Modify: `pennystock/data/contingent_liabilities.py`, `pennystock/llms.py` (`extraction_llm`),
  `pennystock/agents/prompts.py` (`CL_ROWS_PROMPT`), `pennystock/config.py` (`cl_extractor_version = 1`,
  `extraction_num_ctx = 16384`, `extraction_keep_alive = "10m"`)
- Create fixture: `tests/fixtures/notes/dixon_note46_rows.json` (the model's rows for note 46, as printed)
- Test: `tests/test_contingent_liabilities.py` (append), `tests/test_llms.py` (append)

**Interfaces:**
- Consumes: Task 8 `locate_note`, `filing_text.pages_of`.
- Produces: `model_rows(note_text, llm) -> list[dict]`; `verify(rows, note_text) -> (kept, dropped)`;
  `totals(rows, unit) -> dict` ({bucket: {"current", "previous"}} in Rs crore); `reconcile(rows) -> list[str]`;
  `extract(symbol, llm=None, force=False) -> dict`; `stored(symbol) -> dict | None`;
  `summarize_cl(result, net_worth_cr, lender) -> dict` (machine_verified, status, source, unit_as_printed, totals_cr,
  net_worth_cr, disputed_claims_pct_of_net_worth, guarantees_for_others_pct_of_net_worth, largest_items, warnings,
  how_to_judge). Consumed by Tasks 10, 11, 13, 16.

- [ ] **Step 1: Write the fixture** `dixon_note46_rows.json` - the rows exactly as printed on p.474-475:

```json
[{"label": "Income tax", "category": "disputed_tax", "current": "9,443", "previous": "4,125"},
 {"label": "Sales tax", "category": "disputed_tax", "current": "146", "previous": "146"},
 {"label": "Goods and service tax", "category": "disputed_tax", "current": "2,354", "previous": "3,282"},
 {"label": "Excise, custom duty and service tax", "category": "disputed_tax", "current": "39,841", "previous": "8,716"},
 {"label": "Consumer court", "category": "disputed_other", "current": "4", "previous": "4"},
 {"label": "Commercial patent rights", "category": "disputed_other", "current": "1,000", "previous": "202"},
 {"label": "Other disputes", "category": "disputed_other", "current": "239", "previous": "11"},
 {"label": "Paid under protest: Sales tax", "category": "paid_under_protest", "current": "49", "previous": "49"},
 {"label": "Paid under protest: Excise, custom duty and service tax", "category": "paid_under_protest",
  "current": "834", "previous": "1,607"},
 {"label": "Paid under protest: Goods and service tax", "category": "paid_under_protest", "current": "220",
  "previous": "114"},
 {"label": "Paid under protest: Income tax", "category": "paid_under_protest", "current": "469", "previous": "76"},
 {"label": "Total paid under protest", "category": "subtotal", "current": "1,572", "previous": "1,846"},
 {"label": "Guarantees issued by bankers on behalf of the Group", "category": "bank_guarantee_own",
  "current": "13,611", "previous": "2,647"},
 {"label": "Letters of credit (outstanding)", "category": "letters_of_credit", "current": "66,324",
  "previous": "46,633"},
 {"label": "Bill discounting with banks", "category": "bill_discounting", "current": "-", "previous": "49,358"},
 {"label": "Bond given to custom department on behalf of the joint venture company",
  "category": "guarantee_for_others", "current": "100", "previous": "100"},
 {"label": "Bond given to custom department under authorised economic operator", "category": "other_contingent",
  "current": "18,555", "previous": "18,756"}]
```

- [ ] **Step 2: Write the failing tests:**

```python
import json

DIXON_ROWS = json.loads((NOTES / "dixon_note46_rows.json").read_text())
DIXON_TEXT = (NOTES / "dixon_p474.txt").read_text() + "\n" + (NOTES / "dixon_p475.txt").read_text()


def test_every_row_on_the_page_survives_verification():
    kept, dropped = cl.verify(DIXON_ROWS, DIXON_TEXT)
    assert len(kept) == len(DIXON_ROWS) and dropped == []


def test_a_number_not_on_the_page_is_dropped_and_the_note_is_not_verified():
    bad = DIXON_ROWS + [{"label": "Invented", "category": "disputed_tax", "current": "77,777", "previous": None}]
    kept, dropped = cl.verify(bad, DIXON_TEXT)
    assert [r["label"] for r in dropped] == ["Invented"]
    assert cl.status_of(kept, dropped, "lakh") == "partial"


def test_totals_are_in_crore_by_bucket_with_protest_payments_and_subtotals_left_out():
    t = cl.totals(DIXON_ROWS, "lakh")
    assert t["disputed_claims"] == {"current": 530.27, "previous": 164.86}
    assert t["trade_finance"] == {"current": 799.35, "previous": 987.38}      # guarantees + LCs + bill discounting
    assert t["guarantees_for_others"] == {"current": 1.0, "previous": 1.0}
    assert t["other"] == {"current": 185.55, "previous": 187.56}


def test_a_printed_subtotal_is_reconciled():
    assert cl.reconcile(DIXON_ROWS) == []
    broken = [dict(r, current="1,999") if r["category"] == "subtotal" else r for r in DIXON_ROWS]
    assert len(cl.reconcile(broken)) == 1


def _result(status="verified"):
    kept, dropped = cl.verify(DIXON_ROWS, DIXON_TEXT)
    return {"symbol": "DIXON", "fiscal_year": "FY26", "page": 474, "basis": "consolidated", "unit": "lakh",
            "status": status, "rows": kept, "dropped": dropped, "totals": cl.totals(kept, "lakh"),
            "warnings": cl.reconcile(kept)}


def test_the_summary_judges_claims_against_net_worth():
    s = cl.summarize_cl(_result(), 4676.67, lender=False)
    assert s["machine_verified"] and s["disputed_claims_pct_of_net_worth"] == 11.3
    assert s["source"] == "annual report FY26, page 474, consolidated"
    assert s["largest_items"][0]["label"] == "Letters of credit (outstanding)"


def test_a_lender_is_judged_on_claims_only():
    s = cl.summarize_cl(_result(), 4676.67, lender=True)
    assert "lender" in s["how_to_judge"] and "guarantees_for_others_pct_of_net_worth" not in s


def test_extract_runs_the_pipeline_and_stores_the_result(tmp_path, monkeypatch):
    class StubLLM:                          # the model's part is stubbed; everything after it is real
        pass
    monkeypatch.setattr(cl, "latest_report", lambda symbol: {"content_sha": "sha", "doc_date": "2026-09-03",
                                                              "fiscal_period": "FY26"})
    monkeypatch.setattr(cl, "pages_of", lambda sha: pages("policy.txt", "dixon_p474.txt", "dixon_p475.txt",
                                                            start=473))
    monkeypatch.setattr(cl, "model_rows", lambda text, llm: [dict(r) for r in DIXON_ROWS])
    monkeypatch.setattr(cl, "_store", lambda: cl.CLStore(tmp_path / "m.db"))
    r = cl.extract("DIXON", llm=StubLLM())
    assert r["status"] == "verified" and r["page"] == 474 and r["totals"]["disputed_claims"]["current"] == 530.27
    assert cl.stored("DIXON")["status"] == "verified"
```

and in `tests/test_llms.py`:

```python
def test_the_extraction_role_runs_the_analyst_model_without_thinking():
    llm = llms.extraction_llm()
    assert llm.model == llms.settings.analyst_model and llm.reasoning is False
    assert llm.num_ctx == llms.settings.extraction_num_ctx and llm.temperature == 0.0
```

- [ ] **Step 3: Run** → FAIL (missing functions).

- [ ] **Step 4: Implement.** `prompts.py`:

```python
CL_ROWS_PROMPT = (
    "You read ONE note from an Indian company's annual report: CONTINGENT LIABILITIES (and commitments). Return "
    "every line that carries an amount, top to bottom: the label as printed, a category, and the two amounts EXACTLY "
    "as printed - current year first, previous year second (the column order of the table); '-' or a blank cell as "
    "null. Never compute, convert, add or round anything, and never invent a line.\n"
    "Categories: disputed_tax (income tax, GST, excise, customs, service tax, sales tax / VAT, stamp duty, entry tax "
    "disputes); disputed_other (other claims against the company not acknowledged as debt: legal cases, contractors, "
    "consumer, labour, patents); guarantee_for_others (corporate guarantees, bonds or undertakings given ON BEHALF OF "
    "subsidiaries, joint ventures, associates or any other party); bank_guarantee_own (guarantees issued by banks on "
    "the company's own behalf, performance guarantees, bonds for its own obligations); letters_of_credit; "
    "bill_discounting; other_contingent (any other contingent liability); paid_under_protest (amounts already paid "
    "or deposited against the claims above); commitment (capital or other commitments - not contingent "
    "liabilities); subtotal (any total or subtotal line).")
```

`llms.py`:

```python
def extraction_llm() -> ChatOllama:
    """The analyst's model, offline: lays a contingent-liability table out as rows. It computes nothing - every
    number it returns is verified against the page in code - so thinking is off; warm across a batch."""
    return _make(settings.analyst_model, temperature=0.0, keep_alive=settings.extraction_keep_alive,
                 reasoning=False, num_ctx=settings.extraction_num_ctx)
```

`contingent_liabilities.py` (append; imports `json`, `sqlite3`, `datetime`, `Literal`, `pydantic.BaseModel, Field`,
`langchain_core.messages.HumanMessage, SystemMessage`, `pennystock.agents.prompts.CL_ROWS_PROMPT`,
`pennystock.data.filing_text.pages_of`, `functools.lru_cache`):

```python
CATEGORIES = ("disputed_tax", "disputed_other", "guarantee_for_others", "bank_guarantee_own", "letters_of_credit",
              "bill_discounting", "other_contingent", "paid_under_protest", "commitment", "subtotal")
BUCKETS = {"disputed_tax": "disputed_claims", "disputed_other": "disputed_claims",
           "guarantee_for_others": "guarantees_for_others", "bank_guarantee_own": "trade_finance",
           "letters_of_credit": "trade_finance", "bill_discounting": "trade_finance", "other_contingent": "other"}
_NIL = {"", "-", "–", "—", "nil"}
_TOKEN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


class _Row(BaseModel):
    label: str = Field(description="the line's label as printed")
    category: Literal[CATEGORIES]
    current: str | None = Field(description="current-year amount exactly as printed, e.g. '9,443' or '(1,234)'")
    previous: str | None = Field(description="previous-year amount exactly as printed")


class _Rows(BaseModel):
    rows: list[_Row]


def model_rows(note_text: str, llm) -> list[dict]:
    structured = llm.with_structured_output(_Rows, method="json_schema")
    out = structured.invoke([SystemMessage(content=CL_ROWS_PROMPT), HumanMessage(content=note_text)])
    return [r.model_dump() for r in out.rows]


def _digits(printed: str) -> str:
    return printed.strip().strip("()").replace(",", "").lstrip("-").strip()


def _value(printed: str | None) -> float | None:
    if printed is None or printed.strip().lower() in _NIL:
        return 0.0
    s = printed.strip()
    try:
        v = float(_digits(s))
    except ValueError:
        return None
    return -v if (s.startswith("(") and s.endswith(")")) or s.startswith("-") else v


def verify(rows: list[dict], note_text: str) -> tuple[list[dict], list[dict]]:
    """Keep a row only if every amount it gives occurs in the note's own text - the model lays the table out, it
    never gets to supply a number."""
    on_page = {_digits(t) for t in _TOKEN.findall(note_text)}
    kept, dropped = [], []
    for r in rows:
        amounts = [r.get(k) for k in ("current", "previous")]
        ok = all(a is None or a.strip().lower() in _NIL or _digits(a) in on_page for a in amounts)
        (kept if ok else dropped).append(r)
    return kept, dropped


def status_of(kept: list[dict], dropped: list[dict], unit: str | None) -> str:
    if unit and len(kept) >= 2 and not dropped:
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
    """A printed subtotal must equal the run of rows just above it (or that run's last category)."""
    warnings, run = [], []
    for r in rows:
        if r["category"] != "subtotal":
            run.append(r)
            continue
        v = _value(r.get("current"))
        if v is not None and run:
            full = sum(_value(x.get("current")) or 0 for x in run)
            same = sum(_value(x.get("current")) or 0 for x in run if x["category"] == run[-1]["category"])
            if min(abs(full - v), abs(same - v)) > 1:
                warnings.append(f"printed subtotal '{r['label'][:50]}' ({r['current']}) does not equal the rows above")
        run = []
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
                              "extractor_version = ?", (content_sha, settings.cl_extractor_version)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, result: dict) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("INSERT OR REPLACE INTO contingent_liabilities VALUES (?,?,?,?,?)",
                        (result["content_sha"], settings.cl_extractor_version, result["symbol"], json.dumps(result),
                         datetime.now().isoformat(timespec="seconds")))


@lru_cache(maxsize=1)
def _store() -> CLStore:
    return CLStore()


def latest_report(symbol: str) -> dict | None:
    with sqlite3.connect(str(settings.filings_db)) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT content_sha, doc_date, fiscal_period FROM filing_docs WHERE symbol = ? AND "
                          "doc_type = 'annual_report' ORDER BY doc_date DESC LIMIT 1", (symbol.upper(),)).fetchone()
    return dict(row) if row else None


def extract(symbol: str, llm=None, force: bool = False) -> dict:
    """The latest annual report's note -> verified rows and totals, stored (one model call per report)."""
    report = latest_report(symbol)
    if report is None:
        return {"symbol": symbol.upper(), "status": "no_report"}
    store = _store()
    if not force and (hit := store.get(report["content_sha"])):
        return hit
    base = {"symbol": symbol.upper(), "content_sha": report["content_sha"], "fiscal_year": report["fiscal_period"],
            "doc_date": report["doc_date"]}
    note = locate_note(pages_of(report["content_sha"]))
    if note is None:
        result = {**base, "status": "no_note", "rows": [], "dropped": [], "totals": None, "warnings": []}
    else:
        if llm is None:
            from pennystock.llms import extraction_llm
            llm = extraction_llm()
        kept, dropped = verify(model_rows(note.text, llm), note.text)
        result = {**base, "page": note.page, "basis": note.basis, "unit": note.unit,
                  "status": status_of(kept, dropped, note.unit), "rows": kept, "dropped": dropped,
                  "totals": totals(kept, note.unit) if note.unit else None, "warnings": reconcile(kept)}
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
           "largest_items": [{"label": r["label"], "category": r["category"],
                              "current_cr": round((_value(r.get("current")) or 0) * factor, 2),
                              "previous_cr": round((_value(r.get("previous")) or 0) * factor, 2)} for r in items[:5]],
           "warnings": result.get("warnings") or []}
    if lender:
        out["how_to_judge"] = ("This is a lender: guarantees, letters of credit and acceptances issued for customers "
                               "are its business, so only disputed claims are judged against net worth.")
    else:
        out["guarantees_for_others_pct_of_net_worth"] = pct(t["guarantees_for_others"]["current"])
        out["how_to_judge"] = ("Disputed claims and guarantees given for others are the real off-balance-sheet risk; "
                               "judge them against net worth. Trade-finance contingents (letters of credit, bank "
                               "guarantees on the company's own behalf, bill discounting) are normal operating items.")
    return out
```

- [ ] **Step 5: Run** both test files → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 10: Batch extraction, gold set and evaluation (spec A3, evaluation)

**Files:**
- Create: `scripts/extract_contingent_liabilities.py`, `scripts/eval_contingent.py`, `eval/contingent_gold.jsonl`

**Interfaces:**
- Consumes: `extract`, `latest_report`, `CLStore` (Task 9), `sectors.profile_for`, `metrics.load_metrics`.

- [ ] **Step 1: Write `scripts/extract_contingent_liabilities.py`:**

```python
"""Read every company's contingent-liability note into verified numbers: the local model lays the table out, code
verifies every figure against the page. Resumable - a report already read at this extractor version is skipped.

  python scripts/extract_contingent_liabilities.py --status
  python scripts/extract_contingent_liabilities.py                  # every company with an annual report
  python scripts/extract_contingent_liabilities.py --symbol DIXON --force
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pennystock.config import settings                                      # noqa: E402
from pennystock.data import contingent_liabilities as cl                     # noqa: E402


def companies() -> list[str]:
    with sqlite3.connect(str(settings.filings_db)) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT symbol FROM filing_docs WHERE doc_type = "
                                          "'annual_report' ORDER BY symbol")]


def lenders() -> set[str]:
    from pennystock.data.metrics import load_metrics
    from pennystock.data.sectors import profile_for
    return {m.symbol for m in load_metrics() if profile_for(m.sector, m.industry).name == "financials"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--symbol", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    symbols = [s.upper() for s in args.symbol] if args.symbol else companies()
    results = {}
    for n, sym in enumerate(symbols, 1):
        if args.status:
            results[sym] = cl.stored(sym) or {"status": "not_read"}
            continue
        t0 = time.time()
        results[sym] = r = cl.extract(sym, force=args.force)
        print(f"[{n}/{len(symbols)}] {sym:<12} {r['status']:<9} page {r.get('page', '-')} unit {r.get('unit', '-')} "
              f"rows {len(r.get('rows') or [])} dropped {len(r.get('dropped') or [])} ({time.time() - t0:.0f}s)",
              flush=True)
    by = Counter(r["status"] for r in results.values())
    banks = lenders()
    others = [s for s in results if s not in banks]
    verified = sum(results[s]["status"] == "verified" for s in others)
    print(f"\n{len(results)} reports: {dict(by)}")
    print(f"verified coverage of non-lenders: {verified}/{len(others)} ({verified / max(len(others), 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the batch** (real model, ~20-30 min, in the background):
  `/opt/anaconda3/bin/python scripts/extract_contingent_liabilities.py > $SCRATCH/cl_batch.log 2>&1`.

- [ ] **Step 3: Build the gold set by hand** - 15 reports spanning lakh / crore / million / thousand, a PSU (BDL,
  IRCTC), variant headings (HEROMOTOCO, ASTRAL), a bank (AUBANK), new-age (SWIGGY), DIXON, KALYANKJIL, LUPIN, NTPC
  and three random others. For each, read the located page text and write one line to `eval/contingent_gold.jsonl`:
  `{"symbol", "content_sha", "page", "unit", "disputed_claims_cr": [current, previous],
  "guarantees_for_others_cr": [current, previous], "note"}` computed by hand from the page (or `"no_table": true`
  when the report prints no quantified note).

- [ ] **Step 4: Write `scripts/eval_contingent.py`** - for each gold line, load `cl.stored(symbol)`; a VERIFIED
  extraction matches when both totals agree within 0.5 Cr for both years; print per-symbol PASS / WRONG / NOT-VERIFIED,
  then `verified accuracy`, `verified-but-wrong`, `coverage`; exit 1 if any verified-but-wrong.

- [ ] **Step 5: Run the evaluation.** Accept: verified-but-wrong = 0; ≥ 90% of verified gold rows match; verified
  coverage ≥ 80% of non-lenders (from Step 2's summary). Below target → ruling + locator / prompt fix with a test
  reproducing the miss (TDD), re-run the affected symbols with `--force`, re-evaluate. **Commit:** skipped.

---

### Task 11: get_annual_report_notes on verified sources (spec A2 + A3 wiring)

**Files:**
- Modify: `pennystock/tools/market_data.py` (`get_annual_report_notes`), `pennystock/data/annual_report.py`
  (`AGM_NOTICE` skip in `_section`, `notes_from_pages`), `pennystock/agents/prompts.py` (ANNUAL-REPORT FIGURE RULE)
- Test: `tests/test_annual_report.py` (replace the Step-1 tool test, add new ones)

**Interfaces:**
- Consumes: `stored` + `summarize_cl` (Task 9), `rpt_disclosure` + `summarize_rpt` (Tasks 4-5), `filed_year`
  (Task 4), `profile_for`.
- Produces: tool output {symbol, company, contingent_liabilities, related_party_transactions, yardsticks{net_worth_cr,
  revenue_cr, source}}; fallback sections carry `machine_verified: False` and `why`. Consumed by Tasks 13, 15, 16.

- [ ] **Step 1: Write the failing tests:**

```python
def test_the_agm_notices_proposed_rpt_is_not_the_related_party_note():
    # DIXON FY26: the AGM notice (pp. 4-55) is bound in front and its 'proposed RPT' tables won the search
    agm = ("Dixon Technologies (India) Limited_33rd AGM Notice 2025-26\nPart A: Minimum information of the "
           "proposed RPT\nNature of transaction Sale of goods 2,496.66 1,159.78")
    note = "32. Related party disclosures\nTransactions with related parties\nNature of transaction 4,500 3,210"
    picked = _section([agm, note], RELATED_PARTY, RELATED_DETAIL, WINDOW, RELATED_STRONG)
    assert "4,500" in picked and "2,496.66" not in picked


def _wire_tool(monkeypatch, cl_result=None, rpt=None, filed=None):
    import asyncio, ast
    from pennystock.tools import market_data as md
    monkeypatch.setattr(md, "_resolve_ticker", lambda s, e: ("DIXON.NS", "Dixon Technologies"))
    monkeypatch.setattr(md, "stored_cl", lambda sym: cl_result)
    monkeypatch.setattr(md, "rpt_disclosure", lambda sym: rpt)
    monkeypatch.setattr(md, "filed_year", lambda sym: filed)
    monkeypatch.setattr(md, "_is_lender", lambda ticker: False)
    monkeypatch.setattr(md, "_excerpt_notes", lambda sym: AnnualReportNotes(
        symbol="DIXON", from_yr="2025", to_yr="2026", url="http://x/ar.pdf",
        contingent_text="not acknowledged as debt 39,841", related_party_text="Nature of Transaction 2,496"))
    return lambda: ast.literal_eval(asyncio.run(md.get_annual_report_notes.ainvoke({"symbol": "DIXON"})))


def test_verified_claims_and_the_xbrl_disclosure_are_returned_with_filed_yardsticks(monkeypatch):
    from tests.test_contingent_liabilities import _result
    from tests.test_related_parties import DIXON
    from pennystock.data.integrated_filings import FiledYear
    fy = FiledYear("DIXON", "FY26", "2026-03-31", "consolidated", 48872.8, 1438.64, 4676.67, 5386.67, 271.59, 0.0,
                   "S.N. DHAWAN & CO LLP", "12-May-2026 23:38:17", "u")
    out = _wire_tool(monkeypatch, cl_result=_result(), rpt=DIXON, filed=fy)()
    assert out["contingent_liabilities"]["machine_verified"] is True
    assert out["contingent_liabilities"]["disputed_claims_pct_of_net_worth"] == 11.3
    assert out["related_party_transactions"]["source"] == "NSE Reg 23(9) XBRL"
    assert out["yardsticks"] == {"net_worth_cr": 4676.67, "revenue_cr": 48872.8,
                                 "source": "NSE integrated filing FY26 (12-May-2026)"}


def test_no_structured_source_falls_back_to_the_excerpt_and_says_so(monkeypatch):
    out = _wire_tool(monkeypatch)()
    assert out["contingent_liabilities"]["machine_verified"] is False
    assert "figures_mentioned" in out["contingent_liabilities"] and out["contingent_liabilities"]["why"]
    assert out["related_party_transactions"]["machine_verified"] is False
```

(Replace the Step-1 test `test_the_tool_judges_the_notes_against_the_latest_net_worth_and_revenue`: its behaviour now
lives in `_yardsticks`'s Yahoo fallback; keep a variant that sets `filed=None` and asserts the yardsticks come from
the fake provider's LATEST year with `source == "Yahoo statements"`.)

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** `annual_report.py`:

```python
# the AGM notice is often bound in front of the annual report; its proposed-RPT resolutions read like the
# related-party note and won the search (DIXON FY26: pp. 4-55, 'Part A: Minimum information of the proposed RPT')
AGM_HEADER = re.compile(r"\bAGM\s+Notice\b|notice\s+of\s+(the\s+)?\d*\w*\s*annual\s+general\s+meeting", re.I)
AGM_BODY = re.compile(r"minimum\s+information\s+of\s+the\s+proposed\s+rpt|notice\s+is\s+hereby\s+given", re.I)
```

and in `_section`'s loop, first line: `if AGM_HEADER.search(text[:400]) or AGM_BODY.search(text): continue`. Add:

```python
def notes_from_pages(symbol: str, pages: list[str], url: str | None, from_yr: str | None,
                     to_yr: str | None) -> AnnualReportNotes:
    """The two notes from page texts already extracted (filings.db) - no download."""
    return AnnualReportNotes(symbol=symbol, from_yr=from_yr, to_yr=to_yr, url=url, pages=len(pages),
                             contingent_text=_section(pages, CONTINGENT, CONTINGENT_DETAIL,
                                                      settings.annual_report_window, CONTINGENT_STRONG),
                             related_party_text=_section(pages, RELATED_PARTY, RELATED_DETAIL,
                                                         settings.annual_report_window, RELATED_STRONG))
```

`market_data.py`: imports `stored as stored_cl, summarize_cl, latest_report` (contingent_liabilities),
`filed_year, rpt_disclosure` (integrated_filings), `summarize_rpt` (related_parties), `notes_from_pages`,
`pages_of` (filing_text), `profile_for` (sectors). Helpers:

```python
def _is_lender(ticker: str) -> bool:
    f = get_provider().get_fundamentals(ticker)
    return profile_for(f.sector, f.industry).name == "financials"


def _excerpt_notes(base: str) -> AnnualReportNotes:
    """The two notes as text: from the report already in filings.db, else NSE's PDF (the old download path)."""
    report = latest_report(base)
    if report:
        texts = [p["text"] for p in pages_of(report["content_sha"])]
        return notes_from_pages(base, texts, None, None, report.get("fiscal_period"))
    return fetch_notes(base)


async def _yardsticks(ticker: str, fy) -> tuple[float | None, float | None, str]:
    if fy and fy.equity_to_owners_cr is not None:
        return fy.equity_to_owners_cr, fy.revenue_cr, f"NSE integrated filing {fy.fiscal_year} " \
                                                      f"({(fy.broadcast_at or '')[:11]})"
    fin = await asyncio.to_thread(get_provider().get_financials, ticker, settings.financials_years)
    latest = lambda xs: next((v for v in reversed(xs or []) if v is not None), None)   # noqa: E731
    return latest(fin.equity_cr), latest(fin.revenue_cr), "Yahoo statements"
```

Tool body (docstring rewritten per spec A2/A3; keeps the UNCHECKED instruction):

```python
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        fy = await asyncio.to_thread(filed_year, base)
        net_worth, revenue, source = await _yardsticks(ticker, fy)
        cl = await asyncio.to_thread(stored_cl, base)
        rpt = await asyncio.to_thread(rpt_disclosure, base)
        verified = bool(cl and cl.get("status") == "verified")
        notes = None if (verified and rpt) else await asyncio.to_thread(_excerpt_notes, base)
        excerpt = summarize_notes(notes, net_worth, revenue) if notes else {}
        if verified:
            lender = await asyncio.to_thread(_is_lender, ticker)
            cl_view = summarize_cl(cl, net_worth, lender)
        else:
            cl_view = {**(excerpt.get("contingent_liabilities") or {}), "machine_verified": False,
                       "why": (f"the note was read but not verified ({cl['status']})" if cl else
                               "the note has not been machine-read yet (scripts/extract_contingent_liabilities.py)")}
        rpt_view = summarize_rpt(rpt) if rpt else {
            **(excerpt.get("related_party_transactions") or {}), "machine_verified": False,
            "why": "no related-party disclosure found in the company's NSE integrated filings"}
        return str({"symbol": base, "company": name, "contingent_liabilities": cl_view,
                    "related_party_transactions": rpt_view,
                    "yardsticks": {"net_worth_cr": net_worth, "revenue_cr": revenue, "source": source},
                    **({"caveat": excerpt["caveat"]} if not (verified and rpt) and excerpt.get("caveat") else {})})
    except Exception as e:
        logger.warning(f"annual report notes failed for {symbol}: {e}")
        return f"Error fetching annual report notes for {symbol}."
```

`prompts.py` ANNUAL-REPORT FIGURE RULE becomes: "contingent_liabilities with machine_verified True and the XBRL
related_party_transactions (source 'NSE Reg 23(9) XBRL') are exact - quote them with their page or period. Anything
with machine_verified False is a PDF excerpt: quote its figures only as indicative, name the source, never as exact
totals. If nothing is returned, write that off-balance-sheet exposure is UNCHECKED - never that it is clean."

- [ ] **Step 4: Run** `tests/test_annual_report.py` + full suite → 0 failures. **Commit:** skipped.

---

### Task 12: Evidence pack - gather (spec Part B)

**Files:**
- Create: `pennystock/agents/evidence.py`
- Modify: `pennystock/config.py` (`evidence_concurrency = 4`, `evidence_tool_timeout_s = 300.0`,
  `evidence_max_chars = 14000`, `evidence_alternatives = 6`)
- Test: `tests/test_evidence.py`

**Interfaces:**
- Produces: `PackCall(name, args, output, data, ok, seconds)`; `EvidencePack(symbol, company, fact_sheet, calls,
  built_at)`; `parse_output(text) -> dict | None`; `wants_alternatives(question) -> bool`;
  `gather(symbol, question) -> list[PackCall]` (12 tools + `nse_integrated_filing` + `screen_stocks` when asked);
  `build_pack(symbol, question) -> EvidencePack`. Consumed by Tasks 13-14.

- [ ] **Step 1: Write the failing tests:**

```python
"""The evidence pack: the structured tools run in code, in parallel, before the analyst writes a word."""
import asyncio

from pennystock.agents import evidence as ev


class Tool:
    def __init__(self, name, output, fail=False):
        self.name, self.output, self.fail, self.calls = name, output, fail, []

    async def ainvoke(self, args):
        self.calls.append(args)
        if self.fail:
            raise TimeoutError("NSE did not answer")
        return self.output


def _tools(monkeypatch, fail=()):
    tools = [Tool(t.name, str({"company": "Dixon Technologies", "sector": "Technology"}), t.name in fail)
             for t in ev.PACK_TOOLS]
    monkeypatch.setattr(ev, "PACK_TOOLS", tuple(tools))
    monkeypatch.setattr(ev, "filed_year", lambda symbol: None)
    return tools


def test_every_structured_tool_runs_for_the_resolved_company(monkeypatch):
    tools = _tools(monkeypatch)
    calls = asyncio.run(ev.gather("DIXON", "Is Dixon a buy?"))
    assert all(t.calls == [{"symbol": "DIXON"}] for t in tools)
    assert {c.name for c in calls} >= {t.name for t in tools} and all(c.ok for c in calls if c.name != "nse_integrated_filing")


def test_a_failing_tool_becomes_a_failed_call_not_a_failed_pack(monkeypatch):
    _tools(monkeypatch, fail=("get_forensics",))
    calls = {c.name: c for c in asyncio.run(ev.gather("DIXON", "q"))}
    assert not calls["get_forensics"].ok and "TimeoutError" in calls["get_forensics"].output
    assert calls["get_financials"].ok


def test_alternatives_are_screened_in_the_companys_sector_only_when_asked(monkeypatch):
    _tools(monkeypatch)
    screen = Tool("screen_stocks", str({"candidates": []}))
    monkeypatch.setattr(ev, "screen_stocks", screen)
    asyncio.run(ev.gather("DIXON", "Is Dixon a buy?"))
    assert screen.calls == []
    asyncio.run(ev.gather("DIXON", "if not, which stocks would you shortlist instead?"))
    assert screen.calls == [{"sector": "Technology", "top_n": ev.settings.evidence_alternatives}]


def test_tool_output_is_parsed_back_into_a_dict():
    assert ev.parse_output("{'a': 1, 'b': [('2021-03-18', 5.0)]}") == {"a": 1, "b": [("2021-03-18", 5.0)]}
    assert ev.parse_output("Error fetching x") is None
```

- [ ] **Step 2: Run** → FAIL (no module).

- [ ] **Step 3: Implement `pennystock/agents/evidence.py`:**

```python
"""The evidence pack: the analyst's data, gathered and computed by code before the model writes a word.

Why (measured): one 16-tool question produced 23 raw tool results - 22,724 tokens, more than the window - and even
with room to spare the model converted units, lined up years and read percentiles wrongly (DIXON, 2026-10-11).
Context engineering says give the model the smallest set of high-signal tokens, so code runs the structured tools
(the same tools the analyst has: one source of truth), and `fact_sheet.render_fact_sheet` writes one sheet with
every comparison made. The model explains and judges; it calls tools only for text (filings, broker reasoning,
books) or for what the sheet lists under DATA GAPS.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime

from pennystock.agents.fact_sheet import render_fact_sheet
from pennystock.config import settings
from pennystock.data.integrated_filings import filed_year
from pennystock.tools import (get_annual_report_notes, get_broker_consensus, get_corporate_actions,
                              get_corporate_events, get_financials, get_forensics, get_management_commentary,
                              get_price_history, get_shareholding, get_stock_fundamentals, get_trendlyne_view,
                              get_valuation_context, screen_stocks)

logger = logging.getLogger(__name__)

PACK_TOOLS = (get_stock_fundamentals, get_price_history, get_financials, get_forensics, get_valuation_context,
              get_shareholding, get_corporate_events, get_corporate_actions, get_annual_report_notes,
              get_broker_consensus, get_trendlyne_view, get_management_commentary)
ALTERNATIVES = re.compile(r"\b(instead|alternatives?|shortlist|other\s+stocks|peers?\s+to\s+consider|"
                          r"similar\s+stocks|which\s+stocks)\b", re.I)


@dataclass
class PackCall:
    name: str
    args: dict
    output: str
    data: dict | None
    ok: bool
    seconds: float


@dataclass
class EvidencePack:
    symbol: str
    company: str | None
    fact_sheet: str
    calls: list[PackCall]
    built_at: str


def parse_output(text: str) -> dict | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def wants_alternatives(question: str) -> bool:
    return bool(ALTERNATIVES.search(question or ""))


async def _timed(name: str, args: dict, run, sem: asyncio.Semaphore) -> PackCall:
    async with sem:
        started = time.time()
        try:
            out = str(await asyncio.wait_for(run(), timeout=settings.evidence_tool_timeout_s))
        except Exception as e:                       # one slow or broken source never sinks the pack
            out = f"Error: {name} failed ({type(e).__name__}: {e})"
        data = parse_output(out)
        return PackCall(name, args, out, data, data is not None and not out.lstrip().startswith("Error"),
                        round(time.time() - started, 1))


async def gather(symbol: str, question: str) -> list[PackCall]:
    sem = asyncio.Semaphore(settings.evidence_concurrency)
    jobs = [_timed(t.name, {"symbol": symbol}, lambda t=t: t.ainvoke({"symbol": symbol}), sem) for t in PACK_TOOLS]

    async def filed() -> str:
        fy = await asyncio.to_thread(filed_year, symbol)
        return str(asdict(fy)) if fy else "Error: no integrated filing on record"
    jobs.append(_timed("nse_integrated_filing", {"symbol": symbol}, filed, sem))
    calls = list(await asyncio.gather(*jobs))
    if wants_alternatives(question):
        fund = next((c.data for c in calls if c.name == "get_stock_fundamentals" and c.data), None) or {}
        args = {"sector": fund.get("sector") or "", "top_n": settings.evidence_alternatives}
        calls.append(await _timed("screen_stocks", args, lambda: screen_stocks.ainvoke(args), sem))
    return calls


async def build_pack(symbol: str, question: str) -> EvidencePack:
    calls = await gather(symbol, question)
    company = next((c.data.get("company") for c in calls if c.data and c.data.get("company")), None)
    sheet = render_fact_sheet(symbol, company, {c.name: c for c in calls}, today=date.today())
    logger.info(f"Evidence pack for {symbol}: {sum(c.ok for c in calls)}/{len(calls)} sources, "
                f"{len(sheet):,} chars, slowest {max(c.seconds for c in calls):.0f}s")
    return EvidencePack(symbol, company, sheet, calls, datetime.now().isoformat(timespec="seconds"))
```

(Task 12 imports `fact_sheet`; create `pennystock/agents/fact_sheet.py` with a stub
`def render_fact_sheet(symbol, company, calls, today): return ""` so the import resolves - Task 13 replaces it.)

- [ ] **Step 4: Run** → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 13: The fact sheet renderer (spec Part B)

**Files:**
- Modify: `pennystock/agents/fact_sheet.py`
- Test: `tests/test_fact_sheet.py`

**Interfaces:**
- Consumes: `PackCall` (Task 12) and the tool output shapes of Tasks 1, 2, 5-7, 9, 11 plus the unchanged tools
  (forensics, valuation, events, corporate actions, price history, management commentary, screen).
- Produces: `render_fact_sheet(symbol, company, calls: dict[str, PackCall], today: date) -> str`: header, DATA GAPS,
  then sections PRICE & VALUATION, GROWTH, CAN THE NUMBERS BE TRUSTED, OFF THE BALANCE SHEET, OWNERS, STREET VIEW,
  TRENDLYNE, TECHNICALS, DIVIDENDS & CORPORATE ACTIONS, MANAGEMENT, ALTERNATIVES (when screened); ≤
  `settings.evidence_max_chars`. **Rule: the renderer formats numbers the tools returned; it computes no new ones**
  (so every figure in the sheet round-matches a tool number). Consumed by Tasks 14, 16.

- [ ] **Step 1: Write the failing tests** (fixtures: the Step-1 run's real outputs in
  `tests/fixtures/dixon/tool_outputs.json`, written from `traces/20261011T033115-70acfb.json` for the unchanged
  tools, and the new-format outputs built in the test from Tasks 1, 5, 6, 7, 9):

```python
"""The fact sheet: one line per fact, with its source, every comparison already made, gaps listed first."""
import json
from datetime import date
from pathlib import Path

from pennystock.agents.evidence import PackCall, parse_output
from pennystock.agents.fact_sheet import render_fact_sheet

FIX = Path(__file__).parent / "fixtures" / "dixon"
RAW = json.loads((FIX / "tool_outputs.json").read_text())


def call(name, output=None, ok=True):
    out = RAW[name] if output is None else output
    return PackCall(name, {"symbol": "DIXON"}, out, parse_output(out) if ok else None, ok, 1.0)


def sheet(**over):
    calls = {n: call(n) for n in RAW}
    calls.update(over)
    return render_fact_sheet("DIXON", "Dixon Technologies (India) Limited", calls, today=date(2026, 10, 11))


def test_the_header_and_gaps_come_first_and_the_sheet_fits_its_budget():
    s = sheet()
    assert s.startswith("EVIDENCE PACK - Dixon Technologies (India) Limited (DIXON)")
    assert s.index("### DATA GAPS") < s.index("### PRICE & VALUATION")
    from pennystock.config import settings
    assert len(s) <= settings.evidence_max_chars


def test_valuation_is_worded_against_its_own_history_and_its_peers():
    s = sheet()
    assert "BELOW its historical range" in s and "67.4% PREMIUM to its sector median P/E" in s


def test_forensic_scores_are_stated_with_their_caveats():
    s = sheet()
    assert "Piotroski F-score 6/9 (MIDDLING)" in s and "Altman Z''/EM 5.03 (SAFE)" in s
    assert "Beneish M-score -2.13 (no manipulation signal" in s


def test_technicals_say_which_side_of_each_average_the_price_is():
    s = sheet()
    assert "ABOVE its 200-day average Rs 12,097" in s and "BELOW its 50-day Rs 13,844" in s


def test_a_failed_source_is_listed_under_data_gaps():
    s = sheet(get_forensics=call("get_forensics", "Error: get_forensics failed (TimeoutError)", ok=False))
    gaps = s.split("### DATA GAPS")[1].split("###")[0]
    assert "trust checks (Piotroski, Altman, Beneish): UNKNOWN" in gaps


def test_the_sheet_formats_tool_numbers_and_invents_none():
    import re
    s, blob = sheet(), " ".join(RAW.values())
    tool_numbers = {float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", blob) if x.strip("-,.")}
    for x in re.findall(r"(?<![\w.])-?\d[\d,]*\.\d+", s):          # every decimal figure in the sheet
        v = float(x.replace(",", ""))
        assert any(abs(v - t) <= max(0.01, abs(t) * 0.0005) for t in tool_numbers), x
```

- [ ] **Step 2: Run** → FAIL (stub returns "").

- [ ] **Step 3: Implement `pennystock/agents/fact_sheet.py`.** One function per section, each `(calls, gaps) -> list
  of lines`, reading the parsed dicts defensively (`.get` everywhere, a missing field skipped, a missing source added
  to `gaps` with the topic name from `_TOPICS`). The formatting helpers and sections:

```python
"""The fact sheet: one line per fact, each with its source, every comparison already made.

Pure: takes the evidence pack's parsed tool outputs and returns text. It formats the numbers the tools returned and
computes no new ones, so every figure in the sheet traces to a tool (the numeric-provenance check relies on it).
Units are converted, years labelled and verdicts worded upstream, in the tools, before the analyst reads them.
"""
from __future__ import annotations

from datetime import date

from pennystock.config import settings

_TOPICS = {
    "get_stock_fundamentals": "price, P/E and market value", "get_price_history": "price and technicals",
    "get_financials": "five-year financials", "get_forensics": "trust checks (Piotroski, Altman, Beneish)",
    "get_valuation_context": "valuation against peers and own history", "get_shareholding": "shareholding and pledge",
    "get_corporate_events": "insider trades, ratings and severe events", "get_corporate_actions": "dividends and splits",
    "get_annual_report_notes": "contingent liabilities and related parties", "get_broker_consensus": "broker views",
    "get_trendlyne_view": "Trendlyne's scores", "get_management_commentary": "earnings-call guidance",
    "nse_integrated_filing": "filed annual figures (NSE integrated filing)", "screen_stocks": "alternatives screen"}


def _d(calls: dict, name: str) -> dict:
    c = calls.get(name)
    return c.data if c is not None and c.ok and c.data else {}


def _cr(v) -> str:
    return "UNKNOWN" if v is None else f"Rs {v:,.2f} Cr"


def _rs(v, places: int = 0) -> str:
    return "UNKNOWN" if v is None else f"Rs {v:,.{places}f}"


def _years(d: dict | None) -> str:
    return " | ".join(f"{fy} {v:,.2f}" for fy, v in (d or {}).items()) or "UNKNOWN"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"
```

Section bodies (exact wording the tests pin; everything else follows the same pattern):

```python
def _price_valuation(calls, gaps):
    f, ph, v = _d(calls, "get_stock_fundamentals"), _d(calls, "get_price_history"), _d(calls, "get_valuation_context")
    out = []
    if ph.get("last_close") is not None:
        line = f"- Price {_rs(ph['last_close'], 2)} (last close)"
        if f.get("fifty_two_week_low") is not None and f.get("fifty_two_week_high") is not None:
            line += f"; 52-week range {_rs(f['fifty_two_week_low'])}-{_rs(f['fifty_two_week_high'])[3:]}"
        if ph.get("pct_from_peak") is not None:
            line += f"; {ph['pct_from_peak']}% from its 5-year peak {_rs(ph.get('peak_5y'), 2)}"
        out.append(line + ". [Yahoo]")
    if f:
        out.append(f"- P/E {f.get('pe_trailing')} trailing, {f.get('pe_forward')} forward; P/B {f.get('price_to_book')}; "
                   f"dividend yield {f.get('dividend_yield')}%; debt/equity {f.get('debt_to_equity_pct')}% "
                   "(a percentage of equity, not a multiple). [Yahoo]")
    own, peers = v.get("vs_own_history") or {}, v.get("vs_peers") or {}
    if own.get("verdict"):
        out.append(f"- Own history: {own['verdict']} [get_valuation_context]")
    if peers.get("headline"):
        line = f"- Peers: {peers['headline']} Compared against {peers.get('compared_against')}."
        pe_peer = ((peers.get("metrics") or {}).get("P/E") or {}).get("value")
        if pe_peer and f.get("pe_trailing") and abs(pe_peer - f["pe_trailing"]) > 0.1 * f["pe_trailing"]:
            line += (f" (the peer table uses the screening cache's P/E {pe_peer}; Yahoo's trailing P/E is "
                     f"{f['pe_trailing']})")
        out.append(line)
    return out


def _trust(calls, gaps):
    fo = _d(calls, "get_forensics")
    if not fo:
        return []
    p, a, b = fo.get("piotroski") or {}, fo.get("altman_emerging_markets") or {}, fo.get("beneish") or {}
    out = []
    if p:
        failed = ", ".join(t.replace("_", " ") for t in p.get("failed_tests") or [])
        out.append(f"- Piotroski F-score {p.get('f_score')}/{p.get('out_of')} ({p.get('verdict')})"
                   + (f"; failed: {failed}" if failed else "") + ". [get_forensics]")
    if a:
        out.append(f"- Altman Z''/EM {a.get('z_em_score')} ({a.get('zone')}) - {a.get('note')}")
    if b:
        out.append(f"- Beneish M-score {b.get('m_score')} ({b.get('verdict')}; {b.get('note')})")
    ic, de, cc = fo.get("interest_coverage") or {}, fo.get("debt_to_ebitda") or {}, fo.get("cash_conversion") or {}
    wc = fo.get("working_capital") or {}
    out.append(f"- Interest cover {ic.get('interest_coverage')}x ({ic.get('verdict')}); debt/EBITDA "
               f"{de.get('debt_to_ebitda')}x ({de.get('verdict')}); cash from operations / profit {cc.get('cfo_to_pat')} "
               f"over {cc.get('years')} years ({cc.get('verdict')}); cash conversion cycle "
               f"{wc.get('cash_conversion_cycle_days')} days.")
    out += [f"- Red flag: {x}" for x in fo.get("red_flags") or [] if not str(x).lower().startswith("no forensic")]
    return out


def _technicals(calls, gaps):
    ph = _d(calls, "get_price_history")
    last, s200, s50 = ph.get("last_close"), ph.get("sma_200"), ph.get("sma_50")
    if None in (last, s200, s50):
        return []
    rsi = ph.get("rsi_14")
    mood = "overbought" if rsi and rsi >= 70 else "oversold" if rsi and rsi <= 30 else "neutral"
    return [f"- Price {_rs(last)} is {'ABOVE' if last > s200 else 'BELOW'} its 200-day average {_rs(s200)} and "
            f"{'ABOVE' if last > s50 else 'BELOW'} its 50-day {_rs(s50)}; RSI(14) {rsi} ({mood}); trend "
            f"{ph.get('trend')}; support {_rs(ph.get('support'))}, resistance {_rs(ph.get('resistance'))}. "
            "[get_price_history]"]
```

The remaining sections follow the spec Part B list with the same helpers: GROWTH (`_years` of the Task-1 dicts, the
`revenue_cagr` span, ROE/ROCE/D-E, then the `nse_integrated_filing` line "Filed FY26 (consolidated, NSE integrated
filing 12-May-2026): revenue from operations Rs 48,872.80 Cr, profit to shareholders Rs 1,438.64 Cr, net worth Rs
4,676.67 Cr, basic EPS Rs 271.59, exceptional items Rs 0.00 Cr; auditor ..."), OFF THE BALANCE SHEET (Task 9/11
`summarize_cl` totals, % of net worth, largest items, warnings; or "NOT machine-verified (why)" + gap; Task 5
`summarize_rpt` period, outside-subsidiary total and %, promoter/KMP total and %, top counterparties with class and
raw relationship, loans/guarantees), OWNERS (promoter %, change, FII/DII; pledge line: "NONE - no promoter shares
pledged (statutory shareholding filing, Jun 2026)" / "x% of the promoters' own holding (y% of all shares; band)" /
"UNKNOWN - reason" + gap; encumbrance note; pledge filings since the quarter; insider trades, ratings, severe events
from `stored_events`), STREET VIEW (each call, target median / range / upside / dispersion / flags, then each Task-7
estimate view), TRENDLYNE (scores and labels, classification, checklist, analyst aggregate, `pe_zone_meaning`),
DIVIDENDS & CORPORATE ACTIONS ("Dividends per share (Yahoo, ex-date): 2021-09 Rs 1 | ..."; "Splits / bonus issues
(Yahoo records a bonus as a split ratio): 5-for-1 on 2021-03-18"), MANAGEMENT (latest call date, questions, tone
latest vs previous, hedging, explicit non-answers; up to 6 `guidance_latest` and 4 `guidance_previous` sentences,
each clipped to 220 characters), ALTERNATIVES (funnel counts, then up to 6 candidates other than the subject: company,
symbol, score, ROCE, P/E, debt/EBITDA, red flags). Then:

```python
_SECTIONS = (("PRICE & VALUATION", _price_valuation), ("GROWTH", _growth), ("CAN THE NUMBERS BE TRUSTED", _trust),
             ("OFF THE BALANCE SHEET", _off_balance_sheet), ("OWNERS", _owners), ("STREET VIEW", _street),
             ("TRENDLYNE", _trendlyne), ("TECHNICALS", _technicals),
             ("DIVIDENDS & CORPORATE ACTIONS", _corporate_actions), ("MANAGEMENT", _management),
             ("ALTERNATIVES", _alternatives))


def render_fact_sheet(symbol: str, company: str | None, calls: dict, today: date) -> str:
    gaps = [f"- {_TOPICS.get(n, n)}: UNKNOWN - {_clip(c.output, 120)}" for n, c in calls.items() if not c.ok]
    body: list[str] = []
    for title, section in _SECTIONS:
        lines = section(calls, gaps)
        if lines:
            body += [f"### {title}", *lines]
    head = [f"EVIDENCE PACK - {company or symbol} ({symbol}), computed by code on {today:%Y-%m-%d} from the tools in "
            "brackets. Figures are exact as stated (money in Rs crore unless marked; percentages already in %). "
            "Use them as given: do not recompute, re-label years or convert units. Every UNKNOWN below is unknown - "
            "say so; never fill it from memory.", "### DATA GAPS", *(gaps or ["- none: every source answered"])]
    text = "\n".join(head + body)
    if len(text) > settings.evidence_max_chars:
        text = text[: settings.evidence_max_chars - 40].rsplit("\n", 1)[0] + "\n[... cut to fit the evidence budget]"
    return text
```

- [ ] **Step 4: Run** `tests/test_fact_sheet.py` → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 14: Graph wiring and prompt rules (spec Part B + A8)

**Files:**
- Modify: `pennystock/agents/nodes.py` (`evidence_pack_node`, routing, held pack calls, grounding + judge inputs),
  `pennystock/agents/graph.py`, `pennystock/agents/state.py` (`evidence`), `pennystock/tracing.py` (pack calls +
  fact sheet), `pennystock/evals/assertions.py` (`_tool_blob` includes the fact sheet), `pennystock/agents/prompts.py`
  (EVIDENCE PACK RULE, PLEDGE RULE, BROKER RULE, tool 11 wording), `pennystock/config.py` (`judge_evidence_chars = 6000`)
- Test: `tests/test_agent_flow.py`, `tests/test_tracing.py`

**Interfaces:**
- Consumes: `build_pack`, `EvidencePack` (Task 12).
- Produces: node `evidence_pack`; `EVIDENCE = "evidence_pack"` (the fact-sheet message's name); state
  `evidence = {fact_sheet, call_keys, built_at, calls: [{name, args, ok, seconds, result}]}`; trace keys
  `evidence` and pack entries in `tool_calls` with `source: "evidence_pack"`.

- [ ] **Step 1: Write the failing tests** (test_agent_flow.py):

```python
def _pack(monkeypatch, sheet="EVIDENCE PACK - Abc (ABC) ... Piotroski F-score 6/9", ok_tools=("get_forensics",)):
    from pennystock.agents.evidence import EvidencePack, PackCall
    calls = [PackCall(n, {"symbol": "ABC"}, "{'company': 'Abc'}", {"company": "Abc"}, True, 1.0) for n in ok_tools]

    async def build(symbol, question):
        return EvidencePack(symbol, "Abc", sheet, calls, "2026-10-11T10:00:00")
    monkeypatch.setattr(nodes, "build_pack", build)


def test_a_resolved_company_gets_its_evidence_pack_before_the_analyst(monkeypatch):
    _pack(monkeypatch)
    analyst = FakeLLM("REPORT")
    monkeypatch.setattr(nodes, "resolve_from_query", lambda q: {"symbol": "ABC", "company_name": "Abc"})
    path, final = run_graph(monkeypatch, analyst)
    assert path[:3] == ["resolve_entity", "MarketDataAgent", "evidence_pack"]
    sent = analyst.payloads[0]
    assert any(getattr(m, "name", None) == nodes.EVIDENCE and "Piotroski" in m.content for m in sent)


def test_re_calling_a_pack_tool_points_to_the_pack(monkeypatch):
    _pack(monkeypatch)
    monkeypatch.setattr(nodes, "resolve_from_query", lambda q: {"symbol": "ABC", "company_name": "Abc"})
    forensics = FakeTool("{'piotroski': 6}")
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_forensics", (forensics, "FinancialAnalystAgent"))
    analyst = FakeLLM(tool_call("get_forensics", symbol="ABC"), "REPORT")
    _, final = run_graph(monkeypatch, analyst)
    reply = next(m for m in final["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c1")
    assert forensics.calls == [] and "Already in the EVIDENCE PACK" in reply.content


def test_the_fact_check_and_the_judge_read_the_fact_sheet(monkeypatch):
    _pack(monkeypatch, sheet="EVIDENCE PACK - sheet text 4,676.67")
    monkeypatch.setattr(nodes, "resolve_from_query", lambda q: {"symbol": "ABC", "company_name": "Abc"})
    monkeypatch.setitem(nodes.TOOL_REGISTRY, "get_price_history", (FakeTool("last close 100"), "FinancialAnalystAgent"))
    checker, judge = FakeLLM("GROUNDED"), FakeLLM("Faithful. Score: 5")
    analyst = FakeLLM(tool_call("get_price_history", symbol="ABC"), "REPORT")
    run_graph(monkeypatch, analyst, grounding=checker, judge=judge)
    assert "sheet text 4,676.67" in checker.payloads[0][0].content
    assert "sheet text 4,676.67" in judge.payloads[0][0].content
```

(`run_graph` gains `grounding`/`judge` parameters that accept a `FakeLLM` instance as well as a reply string, and
stops patching `resolve_from_query` when a test already did - add `resolve=None` parameter; the first routing test's
expected path gains `"evidence_pack"`.)

test_tracing.py:

```python
def test_the_trace_records_the_pack_calls_and_the_fact_sheet():
    state = {"messages": [AIMessage(content="REPORT")],
             "evidence": {"fact_sheet": "EVIDENCE PACK - x", "built_at": "t", "call_keys": [],
                          "calls": [{"name": "get_forensics", "args": {"symbol": "ABC"}, "ok": True, "seconds": 2.0,
                                     "result": "{'piotroski': 6}"}]}}
    t = build_trace("q", state)
    assert t["tool_calls"][0] == {"seq": 1, "name": "get_forensics", "args": {"symbol": "ABC"},
                                  "result": "{'piotroski': 6}", "ok": True, "trimmed": False,
                                  "source": "evidence_pack", "seconds": 2.0}
    assert t["evidence"]["fact_sheet"] == "EVIDENCE PACK - x"
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement.** `state.py`: `evidence: dict | None`. `nodes.py`:

```python
from pennystock.agents.evidence import build_pack

EVIDENCE = "evidence_pack"        # the name of the message that carries the fact sheet


async def evidence_pack_node(state: AgentState) -> Command[Literal["FinancialAnalystAgent"]]:
    """Gather the structured data in code and hand the analyst one fact sheet (pennystock/agents/evidence.py).
    Discovery questions (no company) keep the tool-calling path."""
    symbol = state.get("resolved_symbol")
    if not symbol:
        return Command(goto="FinancialAnalystAgent")
    try:
        pack = await build_pack(symbol, str(state["messages"][0].content))
    except Exception as e:                            # the analyst still has every tool
        logger.warning(f"Evidence pack failed for {symbol} ({type(e).__name__}: {e}); the analyst will call tools.")
        return Command(goto="FinancialAnalystAgent")
    keys = [_call_key(c.name, TOOL_REGISTRY[c.name][0], c.args) for c in pack.calls
            if c.ok and c.name in TOOL_REGISTRY]
    evidence = {"fact_sheet": pack.fact_sheet, "call_keys": keys, "built_at": pack.built_at,
                "calls": [{"name": c.name, "args": c.args, "ok": c.ok, "seconds": c.seconds, "result": c.output}
                          for c in pack.calls]}
    return Command(goto="FinancialAnalystAgent",
                   update={"messages": [HumanMessage(content=pack.fact_sheet, name=EVIDENCE)], "evidence": evidence})
```

`market_data_agent` routes `goto="evidence_pack"` when it makes no tool call (Literal updated). In `execute_tools`:
`pack_keys = set((state.get("evidence") or {}).get("call_keys") or [])`; a key in `pack_keys` (and not in `held`)
answers `f"Already in the EVIDENCE PACK above ({name}) - use those figures as given."` with `meta = {"call_key": key,
"repeat": True}`. `_evidence(messages)` puts the fact sheet first and applies the budget to the tool results:

```python
    sheet = next((str(m.content) for m in messages if isinstance(m, HumanMessage) and m.name == EVIDENCE), "")
    ...  # (existing newest-first loop, with the budget reduced by len(sheet))
    return "\n\n".join(([sheet] if sheet else []) + list(reversed(kept)))
```

`evaluator_node`: after building `instruction`,

```python
    sheet = (state.get("evidence") or {}).get("fact_sheet")
    if sheet:
        instruction += ("\n\nThe analyst was given this EVIDENCE PACK of verified facts; a faithful report agrees "
                        "with it:\n" + sheet[: settings.judge_evidence_chars])
```

`graph.py`: `builder.add_node("evidence_pack", evidence_pack_node)`. `tracing.py` `build_trace`: pack calls first,
then message calls, renumbered:

```python
    evidence = (final_state or {}).get("evidence") or {}
    pack = [{"seq": 0, "name": c["name"], "args": c["args"], "result": _truncate(c["result"]), "ok": c["ok"],
             "trimmed": False, "source": "evidence_pack", "seconds": c["seconds"]}
            for c in evidence.get("calls") or [] if c["name"] != "nse_integrated_filing"]
    calls = [{**c, "seq": i} for i, c in enumerate(pack + _extract_tool_calls(messages), 1)]
```

and `"evidence": {"fact_sheet": _truncate(evidence.get("fact_sheet") or ""), "built_at": evidence.get("built_at")} if
evidence else None`. `assertions._tool_blob` appends `(trace.get("evidence") or {}).get("fact_sheet") or ""`.

`prompts.py` (keep "SIXTEEN tools" and the numbering):
- After the tool list, insert: "EVIDENCE PACK RULE: when the conversation carries an EVIDENCE PACK, it already holds the
  results of the data tools, computed by code. Use its figures as given - never recompute, convert or re-label them.
  Call tools only for text (search_filings, search_broker_research, penny_stock_researcher) or for what the pack lists
  under DATA GAPS; every DATA GAPS line is UNKNOWN and must be said to be unknown."
- Tool 11 line: "screen_stocks - rank the cached universe ... Use it for discovery, and when the user asks for
  ALTERNATIVES to a named company (the evidence pack runs it for the company's sector). Alternatives come ONLY from
  the screen - never from memory."
- PLEDGE RULE: "... pledge_source 'nse_shp_xbrl' with 0.0 means the statutory shareholding filing reports NO promoter
  pledge - say so with the quarter. If it is None the pledge is UNKNOWN ..." (rest unchanged).
- BROKER RULE gains: "For a reported year quote the FILED figure; brokers' ADJUSTED figures are a different basis -
  never mix them, and compare a forecast only with its own basis."

- [ ] **Step 4: Run** `tests/test_agent_flow.py tests/test_tracing.py tests/test_prompts.py` → pass; full suite → 0
  failures. **Commit:** skipped.

---

### Task 15: Layer-0 checks - no false "missing", no cut-off report (spec Part C)

**Files:**
- Modify: `pennystock/evals/assertions.py` (`check_no_false_missing`, `check_report_complete`, ALL_CHECKS → 12)
- Test: `tests/test_assertions.py`

**Interfaces:**
- Produces: two checks named `no_false_missing` and `report_complete`. Consumed by `scripts/check_traces.py`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_saying_data_is_missing_when_the_tool_returned_it_fails():
    t = trace(report="No Piotroski F-Score or Altman Z-Score was returned in the provided data.",
              tool_calls=[{"name": "get_forensics", "result": "{'piotroski': {'f_score': 6}}"}])
    r = check_no_false_missing(t)
    assert not r["passed"] and "get_forensics" in r["detail"]


def test_saying_it_is_missing_when_it_really_is_passes():
    t = trace(report="The pledge could not be verified.",
              tool_calls=[{"name": "get_shareholding", "result": "{'pledge_pct_of_promoter_holding': None}"}])
    assert check_no_false_missing(t)["passed"]


def test_a_report_cut_mid_sentence_fails():
    cut = "## Verdict\n*   **System View:** The system views the stock as a mid-range performer (Durability: High"
    assert not check_report_complete(trace(report=cut))["passed"]
    assert check_report_complete(trace(report="## Verdict\nA growth stock, not a value stock."))["passed"]
    stopped = trace(report="Fine.", llm_calls=[{"role": "analyst", "done_reason": "length"}])
    assert not check_report_complete(stopped)["passed"]
```

and bump `test_run_all_returns_every_check` to 12.

- [ ] **Step 2: Run** → FAIL (ImportError).

- [ ] **Step 3: Implement:**

```python
_MISSING = re.compile(r"\b(not\s+(?:been\s+)?(?:returned|provided|available|included|disclosed|reported|found)|"
                      r"unavailable|no\s+data|data\s+gaps?|missing|unverified|unchecked|could\s+not\s+be\s+"
                      r"(?:verified|confirmed|assessed|quantified|extracted))\b", re.I)
# topic words in a sentence -> the tool whose result settles it, and how to tell the result really carries it
_HAVE = [
    (("piotroski", "altman", "beneish", "forensic"), "get_forensics", lambda r: "'piotroski'" in r),
    (("promoter holding", "shareholding", "fii", "dii"), "get_shareholding", lambda r: "'promoter_pct': None" not in r),
    (("pledge", "pledging"), "get_shareholding",
     lambda r: "'pledge_pct_of_promoter_holding'" in r and "'pledge_pct_of_promoter_holding': None" not in r),
    (("revenue", "net profit", "net income"), "get_financials", lambda r: "'revenue_cr': {" in r),
    (("historical p/e", "own history", "p/e band", "peer median"), "get_valuation_context",
     lambda r: "'vs_own_history'" in r),
    (("contingent liabilit",), "get_annual_report_notes", lambda r: "'machine_verified': True" in r),
    (("related-party", "related party"), "get_annual_report_notes", lambda r: "'source': 'NSE Reg 23(9) XBRL'" in r),
]


def check_no_false_missing(trace: dict) -> dict:
    """A report must not call data missing that a tool returned - the DIXON report of 2026-10-10 said the forensic
    scores and shareholding were 'not returned' because the context had dropped them."""
    results: dict[str, list[str]] = {}
    for c in trace.get("tool_calls", []):
        results.setdefault(c.get("name"), []).append(c.get("result") or "")
    wrong = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", trace.get("report") or ""):
        low = sentence.lower()
        if not _MISSING.search(low):
            continue
        for words, tool, has in _HAVE:
            if any(w in low for w in words) and any(has(r) for r in results.get(tool, [])):
                wrong.append(f"{tool}: '{sentence.strip()[:90]}'")
    if not trace.get("report"):
        return _ok("no_false_missing", True, "no report", applicable=False)
    return _ok("no_false_missing", not wrong, "; ".join(wrong[:3]) if wrong else "no false 'missing' claims")


def check_report_complete(trace: dict) -> dict:
    """The report must end as a finished text - not stopped at the window or cut mid-sentence (the DIXON report of
    2026-10-10 ended '...a mid-range performer (Durability: High' and still scored 4/5)."""
    report = (trace.get("report") or "").rstrip()
    if not report:
        return _ok("report_complete", False, "no report")
    analyst = [c for c in trace.get("llm_calls") or [] if c.get("role") == "analyst"]
    if analyst and analyst[-1].get("done_reason") == "length":
        return _ok("report_complete", False, "the final draft stopped at its length limit")
    last = report.splitlines()[-1].strip()
    if len(last.split()) >= 6 and not re.search(r"[.!?)\]\"'”’*_|%:]$", last):
        return _ok("report_complete", False, f"ends mid-sentence: '...{last[-60:]}'")
    return _ok("report_complete", True, f"ends: '...{last[-40:]}'")
```

Register both in `ALL_CHECKS`.

- [ ] **Step 4: Run** → pass; full suite → 0 failures. **Commit:** skipped.

---

### Task 16: Dixon regression test (spec Part C)

**Files:**
- Create: `tests/test_dixon_regression.py`; fixtures already present from Tasks 3, 6, 8, 9, 13 plus
  `tests/fixtures/dixon/financials.json` (Yahoo Financials fields for DIXON, fetched once) and
  `tests/fixtures/dixon/broker_obs.json` (the DIXON FORWARD_ESTIMATES rows from market.db)

**Interfaces:**
- Consumes: every producer above; network and model stubbed.

- [ ] **Step 1: Write the test** - build each changed tool's output with the real code on the fixtures, keep the
  unchanged tools' recorded outputs, render the sheet, and assert the four Section-2 errors cannot recur:

```python
"""The Dixon question of 2026-10-10/11, offline: the four errors the live runs made must not come back."""
import json
from datetime import date
from pathlib import Path

from pennystock.agents.evidence import PackCall, parse_output
from pennystock.agents.fact_sheet import render_fact_sheet
from pennystock.data import broker_metrics as bm
from pennystock.data import contingent_liabilities as cl
from pennystock.data import integrated_filings as itf
from pennystock.data.fundamentals import summarize_financials
from pennystock.data.ownership import Ownership, Quarter, summarize_ownership
from pennystock.data.ownership_providers import parse_shp_pledge
from pennystock.data.providers import Financials
from pennystock.data.related_parties import summarize_rpt

FIX = Path(__file__).parent / "fixtures"
RAW = json.loads((FIX / "dixon" / "tool_outputs.json").read_text())


def _call(name, data):
    return PackCall(name, {"symbol": "DIXON"}, str(data), parse_output(str(data)), True, 1.0)


def build_sheet(monkeypatch) -> str:
    fin = summarize_financials(Financials(**json.loads((FIX / "dixon" / "financials.json").read_text())))
    pledge = parse_shp_pledge((FIX / "xbrl" / "dixon_shp_jun2026.xml").read_text())
    own = summarize_ownership(Ownership("DIXON", [Quarter("Jun 2026", promoter_pct=28.55, pledge_pct=pledge["pledge_pct"],
                                                          pledge_pct_of_total=pledge["pledge_pct_of_total"],
                                                          pledge_source="nse_shp_xbrl")]))
    rows = json.loads((FIX / "notes" / "dixon_note46_rows.json").read_text())
    text = (FIX / "notes" / "dixon_p474.txt").read_text() + "\n" + (FIX / "notes" / "dixon_p475.txt").read_text()
    kept, dropped = cl.verify(rows, text)
    cl_view = cl.summarize_cl({"fiscal_year": "FY26", "page": 474, "basis": "consolidated", "unit": "lakh",
                               "status": cl.status_of(kept, dropped, "lakh"), "rows": kept,
                               "totals": cl.totals(kept, "lakh"), "warnings": cl.reconcile(kept)}, 4676.67, False)
    disclosure = itf.RptDisclosure("DIXON", "H2 FY26", "2025-10-01", "2026-03-31",
                                   itf.rpt_rows((FIX / "xbrl" / "dixon_rpt_h2fy26.xml").read_text()),
                                   "12-May-2026 23:38:14", "u", 21182.1, "FY26 revenue minus the April-September half")
    notes = {"symbol": "DIXON", "contingent_liabilities": cl_view,
             "related_party_transactions": summarize_rpt(disclosure)}
    obs = [bm.Observation(**o) for o in json.loads((FIX / "dixon" / "broker_obs.json").read_text())]
    monkeypatch.setattr(bm, "load_observations", lambda symbol, metrics=None, since=None: obs)
    consensus = {**parse_output(RAW["get_broker_consensus"]),
                 "forward_estimates": bm.anchored_estimates("DIXON", {("net_profit_cr", "FY26"): 1438.64,
                                                                      ("revenue_cr", "FY26"): 48872.8})}
    calls = {n: _call(n, parse_output(o)) for n, o in RAW.items()}
    calls.update(get_financials=_call("get_financials", fin), get_shareholding=_call("get_shareholding", own),
                 get_annual_report_notes=_call("get_annual_report_notes", notes),
                 get_broker_consensus=_call("get_broker_consensus", consensus))
    return render_fact_sheet("DIXON", "Dixon Technologies (India) Limited", calls, today=date(2026, 10, 11))


def test_contingent_liabilities_are_in_crore_and_against_net_worth(monkeypatch):
    s = build_sheet(monkeypatch)
    assert "disputed claims Rs 530.27 Cr" in s and "11.3% of net worth" in s


def test_years_are_labelled_and_growth_has_its_real_span(monkeypatch):
    s = build_sheet(monkeypatch)
    assert "FY23 12,173.35" in s and "FY22" not in s and "over FY23->FY26 (3 years)" in s


def test_pledge_is_a_statutory_none_not_unknown(monkeypatch):
    s = build_sheet(monkeypatch)
    assert "NONE - no promoter shares pledged (statutory shareholding filing, Jun 2026)" in s


def test_related_parties_are_classified_and_brokers_never_mix_bases(monkeypatch):
    s = build_sheet(monkeypatch)
    assert "BHARTI AIRTEL LTD [group_or_partner" in s
    assert "filed Rs 1,438.64 Cr" in s and "ADJUSTED" in s
```

- [ ] **Step 2: Run** `/opt/anaconda3/bin/python -m pytest -q tests/test_dixon_regression.py` → pass (it exercises
  finished code; to see it fail first, temporarily set `UNIT_TO_CRORE["lakh"] = 0.001` and watch
  `test_contingent_liabilities_are_in_crore_and_against_net_worth` fail, then restore). Full suite → 0 failures.
  **Commit:** skipped.

---

### Task 17: Live verification and docs

**Files:**
- Modify: `CLAUDE.md` (commands block), memory `agent-context-overflow.md`

- [ ] **Step 1: Warm the data:** `ingest_integrated_filings.py` (all companies; coverage line), refresh ownership for
  all companies (`scripts/ingest_ownership.py`) and measure SHP pledge coverage (share of companies whose latest
  quarter has `pledge_source = 'nse_shp_xbrl'`), the contingent-liability batch from Task 10 done.
- [ ] **Step 2: Live Dixon run:** `PENNY_LEDGER_ENABLED=false /opt/anaconda3/bin/python demo.py`; then
  `python scripts/check_traces.py --last 1` and the scratchpad `check_run.py`. Accept (spec §10): every Layer-0 check
  passes including tool_coverage, no_false_missing, report_complete, context_fits; the four Section-2 errors absent;
  ≤ 4 analyst calls; duration ≤ 365.9 s.
- [ ] **Step 3: Docs:** add to CLAUDE.md under a new "# analyst evidence (statutory XBRL + verified notes)" block:
  `ingest_integrated_filings.py [--status]`, `extract_contingent_liabilities.py [--status|--symbol X --force]`,
  `eval_contingent.py`, and `PENNY_LEDGER_ENABLED=false python demo.py` for test runs. Update memory.
- [ ] **Step 4:** full suite → 0 failures; final self-review per executing-plans; finishing-a-development-branch
  (present the options; nothing is committed without the user).
