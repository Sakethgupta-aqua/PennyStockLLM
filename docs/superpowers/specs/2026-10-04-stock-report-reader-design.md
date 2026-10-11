# Stock Report Reader — Design Spec (sub-project 5 of 6)

**Date:** 2026-10-04
**Status:** design approved by the user ("Yes, as proposed"); this written spec awaits review
**Builds on:** `2026-10-04-company-documents-library-design.md` (Stock_Reports/ + the `documents` catalog)
**Programme:** filings & stock-report intelligence, built in the user's chosen order
5 Stock Report reader → 4 event signals → 1 filings search → 3 results + earnings surprise →
2 guidance tracker → 6 annual-report forensics. Each has its own spec, plan and build.

---

## 1. Goal

Turn every Trendlyne Stock Report PDF into a structured snapshot and use it the way the research says
a third-party score should be used: **shown beside our own analysis, attributed, tracked over time,
and graded** - never trusted on the vendor's word.

## 2. Research that shapes it

- Trendlyne scores every stock 0-100 on **Durability** (financial strength: revenue, cash flow, debt),
  **Valuation** (driven by P/E, P/BV) and **Momentum** (30+ technical indicators, recomputed daily).
  Trendlyne itself calls Momentum a short-term score and Durability/Valuation long-term ones.
  Sources: trendlyne.com/score-details, Trendlyne help centre (DVM FAQ, stock report structure).
- Bands (gauge marks printed on every report: D 35/55, V 30/50, M 35/60; Trendlyne's site agrees on
  the upper marks): **good** when D > 55, V > 50, M ≥ 60; **bad** when D < 35, V < 30, M < 35;
  medium otherwise. (The site gives Momentum's lower mark as 30; the report's own gauge, 35, is used.)
- Named classifications follow from the band mix - Strong Performer (good on all three), Value Play,
  Expensive Rocket, Turnaround Potential, Value Trap, Momentum Trap, Falling Comet (bad on all three) -
  plus in-between names (Mid-range Performer, Expensive Performer, Slowing Down Stock ...).
- "Stocks that score high ... tend to outperform the index" is **Trendlyne's own claim**; no
  independent evidence was found. Hence grading (§7), the same discipline the ledger applies to brokers.
- The report's analyst consensus covers far more analysts than our broker notes (AU SF Bank: 28
  analysts in Trendlyne's forecaster vs 2 broker PDFs on file) - genuinely new information.

## 3. Ground truth measured on the 119 reports on disk

- Bold text is drawn twice, the copy shifted 1.0-1.6 pt **vertically at the same x** (a drop shadow),
  so naive extraction reads "45 38 47" as "4455 3388 4477". pdfplumber's position-tolerance dedupe
  cannot separate it from genuine double letters ("Sell", "Technically") at any tolerance. The exact
  rule - same glyph, same font, same x (±0.05), within 2.0 pt vertically → duplicate - parses
  **117 of 119** page-1 blocks cleanly; the other 2 print no classification line.
- Durability can be **"Not Eligible"** (Fortis): only Valuation and Momentum numbers are printed.
- Checklist group lines carry pass, fail and sometimes a third "not applicable" count
  (Fortis "Total 13 9 1" → 13/(13+9) = 59.1%, matching its printed pass rate).
- The analyst split omits zero buckets (Fortis prints "16 2 2" for five buckets), so the split is only
  unambiguous when five numbers are printed.
- The P/E zone sentence ("Strong Buy Zone 16.7% into P/E buy sell zone") sits on page 4 or 5 and may
  wrap; it is found on the joined text of all pages.
- Observed classifications: Mid-range Performer 49, Falling Comet 15, Expensive Performer 14, Slowing
  Down Stock 6, Strong Performer Under Radar 6, Momentum Trap 4, Strong Performer Getting Expensive 4,
  Await Turnaround 4, Expensive Rocket 4, Value Stock Under Radar 3, Expensive Underperformer 3,
  Expensive Star 3, Turnaround Potential 2. Consensus: Buy 57, Strong Buy 35, Hold 24, Sell 1.

## 4. What is read (and what is not)

Per report: data timestamp (`as_of`), price; Durability, Valuation, Momentum and each label; the
classification; checklist pass rate, total passed/failed and each group's pass/fail; Trendlyne's
consensus recommendation, average target, implied upside, low/avg/high estimate, number of analysts,
and the five-bucket split when unambiguous; the P/E zone and how far into it.

Not read: key statistics, peers, shareholding, insider/SAST tables - Trendlyne's copies of data the
project already takes from NSE, the primary source.

Any field that cannot be read is NULL - never guessed. Score fallback: the plain sentences on pages
2/4/5 ("... has a durability score of 45, which indicates Medium Financial Strength.").

## 5. Storage: `trendlyne_snapshots` (market.db, STRICT)

One row per report file, keyed by the catalog's `doc_key`: `symbol`, `as_of`, `price`,
`durability`/`valuation`/`momentum` (INTEGER, NULL allowed) and their labels, `classification`,
`checklist_pass_pct`, `checklist_passed`, `checklist_failed`, `checklist_groups` (JSON text),
`consensus`, `target_price`, `upside_pct`, `est_low`, `est_avg`, `est_high`, `analysts`,
`n_strong_buy` … `n_strong_sell`, `pe_zone`, `pe_zone_pct`, `parser_version`, `parsed_at`.
A snapshot is a pure function of the PDF bytes and the parser: re-parsing replaces its row, and a
`PARSER_VERSION` bump re-parses every report on the next run (the broker-pipeline rule).

## 6. Where it is used

1. **`screen.py --explain` - "TRENDLYNE'S VIEW (third-party scores - an opinion, not this
   analysis)"**: scores with labels, classification, checklist, consensus of N analysts with target and
   upside, P/E zone; changes since the previous snapshot; and agreement with our own pillars -
   quality ↔ Durability, valuation ↔ Valuation, momentum ↔ Momentum. Our pillar percentile ≥ 0.67 is
   "high", ≤ 0.33 "low"; a high/good-vs-low/bad pair is flagged DISAGREE (the interesting case).
2. **AI analyst tool `get_trendlyne_view(symbol)`** returning the latest snapshot, the change since the
   previous one, the bands and a how-to-read note; "UNKNOWN" when no report is on file. Prompt rule:
   third-party scores - attribute them, Momentum is short-term, Durability/Valuation long-term, never
   present them as this system's view. Layer-0 assertions learn the tool name and its coverage words
   ("trendlyne", "dvm", "durability score", "momentum score", "valuation score").
3. **Ledger** (§7).
4. **Monthly flow**: `file_stock_reports.py` and `fetch_stock_reports.py` parse what they file;
   `scripts/parse_stock_reports.py` parses everything on disk (`--status`, `--symbol`, `--no-ledger`).

## 7. Grading Trendlyne's view

Each snapshot becomes a ledger prediction, `source="trendlyne"`:
- direction from the published bands - **bullish** when no readable score is bad and at least two are
  good; **bearish** when at least two are bad; otherwise **neutral**;
- conviction 4 when all three scores agree on the direction, 3 for two, 2 for neutral;
- `as_of_date` = the first session after the report's data timestamp (the broker rule: you could not
  have traded on it before), entry price = the price printed in the report (point-in-time);
- payload: the three scores, classification, checklist rate, consensus, P/E zone, `doc_key`;
- idempotent: one prediction per (symbol, source, as_of), never rewritten.
The nightly grader then scores it against the Nifty 500 at 1/5/20/60 sessions like every other source.

## 8. Testing (offline; no Trendlyne PDF is ever committed)

Dedupe on synthetic glyph lists (shadow copies at +1.04 and +1.56 pt removed; adjacent "ll" kept);
page-1 parsing on text fixtures shaped like AU SF Bank and Fortis (Not Eligible, three-count checklist
lines, five vs three analyst buckets); score-sentence fallback; wrapped P/E-zone wording; the band →
direction table; storage replace-on-reparse and history/changes; the panel's agreement rule; the tool's
UNKNOWN path; ledger idempotence and the as-of/entry rule.

## 9. Limits

- Trendlyne can change the report layout; unreadable fields become NULL and `--status` counts them.
- Two reports print no classification; Durability is sometimes "Not Eligible" - both are NULL, and the
  direction rule then uses the scores that exist.
- Grading needs time: a verdict on Trendlyne's edge needs several monthly snapshots per company.
