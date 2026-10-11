# Stock Report Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse every Trendlyne Stock Report into a `trendlyne_snapshots` row, show it beside our own
score (screener + AI analyst), and grade Trendlyne's view in the ledger.

**Architecture:** `pennystock/data/trendlyne_scores.py` owns parsing (pure text functions + a thin
pdfplumber edge), storage, bands/lean and ledger conversion. `pennystock/tools/trendlyne.py` is the
analyst tool. `scripts/parse_stock_reports.py` is the CLI; the two Stock Report scripts call it after
filing. `actionable_date` moves from the broker backfill script into `data/outcomes.py` so both
sources date predictions the same way.

**Tech Stack:** Python 3, pdfplumber 0.11, SQLite STRICT tables, LangChain `@tool`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-stock-report-reader-design.md`

## Global Constraints

- Bands: good when D > 55, V > 50, M ≥ 60; bad when D < 35, V < 30, M < 35; medium otherwise.
- Lean: bullish = no readable score bad AND ≥ 2 good; bearish = ≥ 2 bad; else neutral. Conviction 4 (three agree), 3 (two), 2 (neutral).
- Overprint rule: same glyph text, same font, same x0 (rounded to 0.1), |Δtop| < 2.0 pt → duplicate.
- Unreadable fields are NULL, never guessed. Analyst split stored only when five counts are printed.
- Snapshots: one row per report `doc_key`; re-parse replaces; `PARSER_VERSION` bump re-parses all.
- Ledger: source `"trendlyne"`, as_of_date = first session after the report day (lag 1), entry = printed price, one prediction per (symbol, report as_of), never rewritten.
- No Trendlyne PDF is committed; tests are offline and use text fixtures / synthetic glyphs.
- Commits only when the user asks; each task ends at a green full suite.

## Review Focus

- A report whose page 1 lacks the consensus block (no analyst coverage) → consensus fields NULL, scores still parsed. (Task 1)
- Durability "Not Eligible" → D NULL, the two printed numbers read as V and M, lean uses V and M only. (Tasks 1, 2)
- A layout change that breaks page-1 scores → the plain score sentences on later pages still fill them. (Task 1)
- Re-running the parser or the ledger step → no duplicate snapshots, no duplicate predictions. (Tasks 2, 3)
- A symbol with no report → panel says so, tool says UNKNOWN (never "neutral"). (Task 4)

---

### Task 1: Parsing a report (pure functions)

**Files:** Create `pennystock/data/trendlyne_scores.py`; Test `tests/test_trendlyne_scores.py`

**Interfaces — Produces:** `PARSER_VERSION`, `Snapshot` (frozen dataclass: `as_of, price, durability,
valuation, momentum, durability_label, valuation_label, momentum_label, classification,
checklist_pass_pct, checklist_passed, checklist_failed, checklist_groups, consensus, target_price,
upside_pct, est_low, est_avg, est_high, analysts, n_strong_buy, n_buy, n_hold, n_sell, n_strong_sell,
pe_zone, pe_zone_pct`), `overprint_duplicates(chars, dy=2.0) -> set[tuple]`, `clean_page_text(page) -> str`,
`parse_snapshot(pages: list[str]) -> Snapshot`.

- [ ] **Step 1: failing tests** — fixtures are the cleaned page-1 text of AU SF Bank and Fortis (as
  measured), the score sentences of pages 2/4/5, and the wrapped zone wording:

```python
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
FORTIS_P1 = AU_P1.replace("Durability Score (D) Medium Financial Strength", "Durability Score (D) Not Eligible") \
    .replace("45 38 47", "31 37").replace("13 4 4 4 3 Financials 3 5", "16 2 2 Financials 5 2 1") \
    .replace("Total 12 14", "Total 13 9 1").replace("46.2%", "59.1%").replace("BUY 1121 (13.59%)", "STRONG BUY 1114 (41.92%)") \
    .replace("Mid Valuation", "Mid Valuation").replace("based on 28", "based on 20")
ZONE_PAGE = "below the current P/E 76.1. This puts it in the PE Strong Buy\nZone\nStrong Buy Zone 3.1% into P/E buy sell zone This is based on"
SENTENCES = ["X Ltd. has a durability score of 45, which indicates Medium Financial Strength.",
             "X Ltd. has a valuation score of 38, which indicates an Mid Valuation.",
             "X Ltd. has a Momentum score of 47, which indicates that it is Technically Neutral."]

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

def g(text, x0, top, font="F-Bold"):
    return {"text": text, "x0": x0, "x1": x0 + 3.6, "top": top, "fontname": font}

def test_drop_shadow_copies_are_removed_and_real_double_letters_kept():
    chars = [g("4", 108.45, 251.67), g("5", 112.04, 251.67), g("4", 108.45, 250.11), g("5", 112.04, 250.11),
             g("D", 115.45, 169.24), g("D", 115.45, 168.20),
             g("l", 40.0, 300.0, "F-Reg"), g("l", 41.8, 300.0, "F-Reg")]       # "Sell": adjacent, kept
    drop = ts.overprint_duplicates(chars)
    assert drop == {("4", 108.45, 250.11), ("5", 112.04, 250.11), ("D", 115.45, 168.2)}
```

- [ ] **Step 2: run, expect ModuleNotFoundError.**
- [ ] **Step 3: implement** (regexes: `RE_ASOF`, `RE_PRICE`, `RE_LABELS`, the score line directly above
  the gauge marks `0 35 55 100`, `RE_SENTENCE`, `RE_CLASS` rejecting "D V M"/digits, `RE_REC`,
  the percentage on the line before "Passed in checklist", `RE_ANALYSTS`, the estimate line before
  "Low Est.", `RE_GROUP` with optional third count, the five-bucket line, `RE_ZONE` on whitespace-joined
  text) and the overprint rule exactly as in Global Constraints.
- [ ] **Step 4: run, expect all pass; full suite.**

### Task 2: Storage, bands, lean, the CLI

**Files:** Modify `pennystock/data/trendlyne_scores.py`; Create `scripts/parse_stock_reports.py`; Test `tests/test_trendlyne_scores.py`

**Interfaces — Produces:** `BANDS`, `band(kind, score) -> str | None`, `lean(snap) -> tuple[str, int]`,
`save_snapshot(doc_key, symbol, snap) -> None`, `latest(symbol) -> tuple[str, Snapshot] | None`,
`history(symbol) -> list[tuple[str, Snapshot]]` (oldest first), `changes(prev, cur) -> dict`,
`parse_report(path) -> Snapshot`, `parse_all(reparse=False) -> dict` (tally; reads
`documents.list_documents(kind="stock_report")`, skips rows already at `PARSER_VERSION`).

- [ ] **Step 1: failing tests**

```python
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

def test_a_reparse_replaces_the_snapshot_and_history_keeps_months_apart(db):
    ts.save_snapshot("k1", "AUBANK", snap(45, 38, 47, as_of="2026-10-01T15:31"))
    ts.save_snapshot("k1", "AUBANK", snap(46, 38, 47, as_of="2026-10-01T15:31"))
    ts.save_snapshot("k2", "AUBANK", snap(50, 30, 30, as_of="2026-11-02T15:31"))
    hist = ts.history("AUBANK")
    assert [(k, s.durability) for k, s in hist] == [("k1", 46), ("k2", 50)]
    assert ts.latest("AUBANK")[0] == "k2"
    assert ts.changes(hist[0][1], hist[1][1])["durability"] == 4
```
(`snap(d, v, m, **kw)` builds a `Snapshot` with defaults; `db` patches `settings.market_db`.)

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (STRICT table per spec §5, JSON for
  `checklist_groups`; the CLI with `--status`, `--symbol`, `--reparse`, `--no-ledger`).
- [ ] **Step 4: run; full suite; `python scripts/parse_stock_reports.py --symbol AUBANK` on the real file.**

### Task 3: Trendlyne's view in the ledger

**Files:** Modify `pennystock/data/outcomes.py` (receive `actionable_date`, `_sessions`),
`scripts/backfill_broker_predictions.py` (import them), `pennystock/data/trendlyne_scores.py`;
Test `tests/test_trendlyne_scores.py`, existing `tests/test_backfill.py` must stay green.

**Interfaces — Produces:** `to_prediction(symbol, doc_key, snap, closes=None, lag=1, sector=None,
industry=None) -> Prediction | None`, `record_predictions(lag=1) -> int`.

- [ ] **Step 1: failing tests**

```python
def test_a_snapshot_becomes_a_graded_prediction_dated_to_the_next_session():
    closes = pd.Series([100.0, 101.0, 102.0], index=pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-03"]))
    p = ts.to_prediction("APARINDS", "k9", snap(65, 31, 67, as_of="2026-10-01T15:31", price=987.0), closes)
    assert (p.source, p.direction, p.conviction, p.as_of_date) == ("trendlyne", "bullish", 3, "2026-10-03")
    assert (p.entry_price, p.entry_price_asof) == (987.0, "2026-10-01")
    assert p.payload["durability"] == 65 and p.provenance["doc_key"] == "k9"

def test_recording_twice_adds_nothing_the_second_time(db, monkeypatch):
    ts.save_snapshot("k1", "AUBANK", snap(45, 38, 47, as_of="2026-10-01T15:31", price=987.0))
    monkeypatch.setattr(ts, "fetch_closes", lambda *a, **k: None)
    assert ts.record_predictions() == 1
    assert ts.record_predictions() == 0
```

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (move `actionable_date`/`_sessions`
  verbatim; natural key = (symbol, payload["as_of"]) over `load_predictions(source="trendlyne",
  include_superseded=True)`). **Step 4: run; full suite (test_backfill included).**

### Task 4: Panel, analyst tool, wiring

**Files:** Modify `scripts/screen.py`, `pennystock/agents/nodes.py`, `pennystock/agents/prompts.py`,
`pennystock/evals/assertions.py`, `pennystock/tools/__init__.py`, `scripts/file_stock_reports.py`,
`scripts/fetch_stock_reports.py`; Create `pennystock/tools/trendlyne.py`;
Test `tests/test_trendlyne_scores.py`, `tests/test_trendlyne_tool.py`, existing `tests/test_assertions.py`.

**Interfaces — Produces:** `agreement(ours: dict[str, float | None], snap) -> list[dict]`,
`view(symbol) -> dict | None`, tool `get_trendlyne_view(symbol, exchange="NSE") -> str`.

- [ ] **Step 1: failing tests**

```python
def test_strong_disagreement_is_flagged():
    rows = ts.agreement({"quality": 0.8, "valuation": 0.2, "momentum": 0.5}, snap(20, 70, 47))
    assert [(r["ours"], r["theirs"], r["verdict"]) for r in rows] == [
        ("quality", "durability", "DISAGREE"), ("valuation", "valuation", "DISAGREE"),
        ("momentum", "momentum", "agree")]

# tests/test_trendlyne_tool.py
def test_no_report_on_file_is_unknown_never_neutral(monkeypatch):
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("XYZ.NS", "Xyz Ltd"))
    monkeypatch.setattr(tool_mod, "view", lambda s: None)
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "XYZ"}))
    assert "UNKNOWN" in out and "neutral" not in out.lower()

def test_the_view_is_attributed_and_carries_how_to_read(monkeypatch):
    monkeypatch.setattr(tool_mod, "_resolve_ticker", lambda s, e: ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(tool_mod, "view", lambda s: {"as_of": "2026-10-01T15:31", "durability": 45})
    out = asyncio.run(tool_mod.get_trendlyne_view.ainvoke({"symbol": "AUBANK"}))
    assert "Trendlyne" in out and "how_to_read" in out and "45" in out
```
  plus: `test_assertions` already asserts registry ⊆ known tools and stock-tool parity - extend the
  assertion sets and add `("trendlyne", "dvm", "durability score", "momentum score", "valuation score")`
  → `get_trendlyne_view` to the coverage table, with one test that a "what is the DVM score" question
  expects the tool.

- [ ] **Step 2: run, expect failures.** **Step 3: implement** panel (`print_trendlyne_panel`, after
  the broker consensus panel, using the explained stock's pillar scores), tool, wiring, prompt line
  `15. get_trendlyne_view` + THIRD-PARTY rule, and the parse/record hook in both Stock Report scripts.
  **Step 4: run; full suite.**

### Task 5: Real run, docs

- [ ] Parse all reports (`parse_stock_reports.py`), check `--status` (expect ≥ 117 with scores),
  record predictions, `screen.py --explain AUBANK / FORTIS / APARINDS`, call the tool once,
  CLAUDE.md commands, CI lint gate, full suite.
