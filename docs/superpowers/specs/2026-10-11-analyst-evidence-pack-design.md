# Analyst Evidence Pack and Data Fixes — Design Spec (Steps 2–4 of the Dixon diagnosis)

**Date:** 2026-10-11
**Status:** approved by the user ("yes", 2026-10-11): go straight from the plan to inline execution. The user asked:
"Execute steps 2, 3 and 4 after performing thorough deep dive research … production level code changes … use all
the required superpower skills".
**Builds on:** Step 1 (context plumbing, done 2026-10-11, uncommitted: 64K window, fixed routing, latest draft only,
repeat-call pointers, full fact-check evidence, token log + `context_fits`, budget guard).
**Diagnosis:** memory `agent-context-overflow`; Step-1 live run `traces/20261011T033115-70acfb.json`.

## 1. Goal

Step 1 made the analyst *see* every tool result. It still *misreads* some of them. This spec makes the answers to a
full-company question accurate, complete and checkable:

- **Step 3: data sources.** Fix the four sources that produce wrong or "unknown" values, using the statutory,
  machine-readable filings where they exist.
- **Step 2: evidence pack.** Code gathers the structured data, computes the comparisons and writes a compact fact
  sheet. The model explains and judges; it no longer converts units, lines up years or reads percentiles.
- **Step 4: keep it fixed.** Automatic checks for the two failure classes seen, plus the Dixon question as an
  offline regression test.

Implementation order is 3 → 2 → 4: the pack is only as good as the facts it carries.

## 2. What the Step-1 run still got wrong (measured, Dixon, 2026-10-11)

| Report said | Truth | Cause |
|---|---|---|
| Contingent liabilities "around ₹50 Cr", "negligible" | disputed claims ₹530.27 Cr = 11.3% of net worth (note 46, p.474, ₹ lakhs); trade-finance contingents besides: bank guarantees ₹136.11 Cr, letters of credit ₹663.24 Cr, customs bonds ₹185.55 Cr, and a ₹1 Cr customs bond given for the joint venture | the model did lakh→crore maths on raw note text |
| "FY22 ₹12,173 Cr", "5-year CAGR 58.6%" | ₹12,173 Cr is FY23; the CAGR spans FY23→FY26 (3 years) | `get_financials` prints parallel lists whose first slot is empty |
| Pledging "none reported" and "unknown" | statutory filing (SHP, Jun 2026): no promoter shares pledged, no NDU, no other encumbrance | NSE `corporate-pledgedata` returns 0 rows for every company |
| Alternatives Voltas, Blue Star, Havells, Sona BLW | not from any tool | prompt forbids `screen_stocks` when a company is named |

## 3. Research that shapes it

- **Context engineering** (Anthropic, *Effective context engineering for AI agents*, 2025-09-29): give the model
  the smallest set of high-signal tokens; models weigh salient, recent tokens more and dilute details in long
  inputs. → a compact, pre-computed fact sheet, not 23 raw tool dumps.
- **LLM extraction needs algorithmic validation** (arXiv 2511.10659; 2508.05201): most errors in LLM extraction
  from financial tables are hallucinated numbers; high accuracy comes from multi-stage pipelines that validate
  against the source and reconcile totals. → contingent liabilities are structured by the model but every number is
  verified verbatim against the page in code.
- **Consensus on one basis** (I/B/E/S "majority basis"): estimates on a different accounting basis from the
  majority are removed from the consensus. → broker figures are anchored to the filed number and never mixed
  across adjusted and reported bases.
- **Statutory XBRL is already filed** (measured below): SEBI shareholding patterns (Reg 31) carry promoter pledge
  counts; NSE "Integrated Filing – Financials" (since Dec 2024) carries audited consolidated figures and the
  half-yearly related-party disclosure (Reg 23(9), XBRL-only per NSE circular 06-09-2021).
- **Indian annual reports are heterogeneous** (measured below): the contingent-liability note uses many headings,
  units (₹ lakhs / crores / millions / '000), rupee glyphs that extract as `, J or H, and banks file a different
  schedule. A regex-only parser read 94 of 118 notes but several of those were the wrong table.

## 4. Ground truth measured (2026-10-11)

- **Shareholding XBRL** (`/api/corporate-share-holdings-master` → `xbrl` per quarter): present for 5/5 recent
  quarters at all 7 sampled companies. Tags: `WhetherAnySharesHeldByPromotersAreEncumberedUnderPledged`
  (+ NDU / other), and for pledged companies `NumberOfSharesEncumberedUnderPledged` in the promoter-group context
  (JAYNECOIND 534,804,690; VPRPL 45,150,000). DIXON Jun 2026: all `false`.
- **Integrated Filing – Financials** (`/api/integrated-filing-results?index=equities&symbol=X`, 16–20 rows each):
  DIXON FY26 consolidated (filed 2026-05-12): revenue ₹48,872.80 Cr, profit to owners ₹1,438.64 Cr, equity to
  owners ₹4,676.67 Cr, basic EPS ₹271.59, exceptional items 0, auditor S.N. Dhawan & Co LLP. Each fact carries
  its XBRL period (quarter and year-to-date), so a half-year is a year-to-date figure (Sep) or year minus H1 (Mar).
- **Related-party XBRL** in the half-year filing (Sep / Mar), in the standalone file: DIXON 159 rows, NTPC 243,
  KALYANKJIL 85, SWIGGY 55, ASTRAL 55, IRCTC 30, AUBANK 9; BDL 0 (fallback needed). Fields: entity, counterparty,
  relationship (free text: "Subsidiary", "Joint Venture", "Entity having significant influence", "KMP", …), type,
  amount in the period (INR), audit-committee-approved value.
- **Contingent-liability notes** in `filings.db` (118 latest annual reports): 103 contain "not acknowledged as
  debt(s)", 116 mention contingent liabilities. Regex prototype: 94 parsed to rows, units found in 85 (crore 46,
  million 22, lakh 14, thousand 3); wrong tables picked for SWIGGY (ESOP valuation) and AUBANK (CSR); 24 missed
  (variant headings: HEROMOTOCO, ASTRAL; a typo and a "J" rupee glyph: IRCTC; banks' schedules).

## 5. Approaches considered

**Evidence delivery (Step 2).** (a) *Recommended:* an evidence-pack node runs the structured tools in code and
renders a fact sheet; the analyst keeps its tools for text (filings, broker reasoning, books) and narrow follow-ups.
(b) Pack only, no tools: most deterministic and fastest, but unusual questions lose flexibility. (c) Keep the
tool-calling agent and summarise each result as it arrives: smallest change, but the model still decides which
data to fetch, so a missed tool stays missed.

**Contingent liabilities (Step 3).** (a) *Recommended:* code locates the note and its unit, the local model turns
the table into rows, code verifies every number against the page and computes the totals. (b) Regex parser only:
prototype quality ~60–70%, long tail of formats. (c) PDF table libraries (camelot / pdfplumber): Indian notes are
mostly borderless, so it still needs label, unit and category logic on top.

**Related parties.** (a) *Recommended:* the Reg 23(9) XBRL rows. (b) Parse the annual-report note's wide tables.

**Pledge.** (a) *Recommended:* the quarterly SHP XBRL for the level, the event store's SAST filings for changes
since. (b) Event store only: tells what changed, not how much is pledged.

## 6. Part A — data sources (Step 3)

### A1. Filed financials and the related-party disclosure from NSE integrated filings
New `pennystock/data/integrated_filings.py`.
- List a company's filings; parse "Integrated Filing- Financials" XBRL: consolidated preferred, standalone
  fallback. Facts with their XBRL period: `RevenueFromOperations`, `ProfitOrLossAttributableToOwnersOfParent`
  (fallback `ProfitLoss`), `EquityAttributableToOwnersOfParent` (and `Equity`), basic EPS, `ExceptionalItemsBeforeTax`,
  `AuditorsFirmName`. Money stored in ₹ crore (INR / 1e7); each value keeps period, basis and the filing's
  broadcast time (point-in-time).
- Related-party rows from the half-year filings (Sep, Mar), whichever basis carries them. **Measured:** the
  amounts cover the HALF-YEAR although the XBRL stamps them with the quarter's dates (DIXON → Bharti Airtel: Sep
  ₹1,271.99 Cr + Mar ₹1,224.67 Cr = ₹2,496.66 Cr, exactly the AGM notice's FY26 figure). Periods are labelled
  H1 (Apr–Sep) / H2 (Oct–Mar) from the filing's quarter-end; the context dates are not used. Amounts are as
  disclosed and can include taxes.
- Cache: each XBRL URL is immutable → parsed once into `market.db` (`integrated_filings`: url PK, symbol, period,
  basis, kind, broadcast_at, facts JSON, rpt_rows JSON, parsed_at, parser_version). A listing is refreshed when
  older than `integrated_filings_max_age_days` (7).
- Gives the **yardsticks** used everywhere below: latest FY revenue, profit to owners, net worth (equity to owners),
  half-year revenue (Sep: the six-month year-to-date figure; Mar: the year minus the Sep half). Yahoo
  (`get_financials`) remains the fallback.

### A2. Related-party transactions
`summarize_rpt(rows, half_year_revenue_cr)`:
- Classify each relationship by keyword: `subsidiary` (eliminated on consolidation; loans, guarantees and
  investments into them still listed), `jv_associate` (JVs, associates, entities the company controls or
  influences), `promoter_kmp_entity` (entities owned or controlled by promoters, KMP, directors or relatives - the
  classic channel), `group_or_partner` (holding company, fellow subsidiaries, "common control", "significant
  influence": a promoter's group OR a JV partner's group - the text alone cannot tell, so the counterparty names
  are shown), `people` (KMP, directors, relatives), `other`. Raw text is kept beside every class.
- Totals by class and by type for the half-year; top counterparties outside subsidiaries; `promoter_kmp_entity`
  and all outside-subsidiary flows as % of the half-year's consolidated revenue (10% threshold), and loans /
  guarantees given to anyone outside the subsidiaries flagged by name.
- `get_annual_report_notes` takes its related-party section from here; the annual-report excerpt is used only when
  no XBRL rows exist, and its page search skips the AGM notice bound in front of the report.

### A3. Contingent liabilities, verified
New `pennystock/data/contingent_liabilities.py` + `scripts/extract_contingent_liabilities.py`.
1. **Locate** in the latest annual report in `filings.db`: score pages on the heading (all variants incl.
   "contingent liabilities and commitments", "not acknowledge(d) as debt(s)"), claim/guarantee language, number
   density, and the consolidated running header; reject policy prose ("a contingent liability exists when").
   Take the best page and the next.
2. **Unit** in code: "₹/`/J/H/Rs/INR … lakh(s)/lac(s)/crore(s)/cr/million/mn/thousand/'000", near the heading,
   else the page, else the nearest earlier "all amounts in … unless otherwise stated". No unit → no totals.
3. **Structure** with the local model (`qwen3.5:35b`, thinking off, temperature 0, JSON schema): rows of
   {label, category, current, previous} with the numbers **as printed**; category one of disputed_tax,
   disputed_other, guarantee_for_others, bank_guarantee_own, letters_of_credit, bill_discounting, other_contingent,
   paid_under_protest, commitment, subtotal.
4. **Verify** in code: every printed number must appear verbatim in the note text; failing rows are dropped; a
   printed subtotal is reconciled where present. Status `verified` (all rows pass, ≥ 2 rows), `partial`, `failed`.
5. **Totals** in ₹ crore for both years: disputed claims, guarantees for others (the red-flag class), trade-finance
   contingents (LCs, bill discounting, own bank guarantees), other; paid-under-protest, commitments and subtotals are
   excluded. Compared with net worth from A1 (same fiscal year-end) or Yahoo.
6. **Lenders** (`profile_for(...).name == "financials"`): only claims not acknowledged as debt are judged against
   net worth; guarantees and acceptances for customers are shown as "normal business for a lender".
7. **Store** in `market.db` (`contingent_liabilities`: content_sha + extractor_version PK, symbol, fiscal year,
   page, basis, unit, status, rows JSON, totals JSON, extracted_at). The batch script is resumable and has
   `--status`; the analyst tool only reads the store (no model call at question time).
8. `get_annual_report_notes` returns verified totals as exact figures with page and unit; otherwise the excerpt
   with the current "indicative only" caveat and a note that the note was not machine-verified.

### A4. Promoter pledge from the statutory shareholding filing
In `NseOwnershipProvider`: replace the dead `corporate-pledgedata` call. For the latest
`ownership_lookback_quarters + 1` quarters, parse each SHP XBRL: promoter-group shares, pledged shares, and the
pledge / NDU / other-encumbrance flags. Flags all `false` → pledge 0.0 with source `nse_shp_xbrl`; pledged → count
÷ promoter shares. A quarter's XBRL is fetched once. `summarize_ownership` then reports UNKNOWN only when no
filing could be read, and adds pledge filings from the event store since the quarter end.

### A5. Broker estimates on one basis
For reported fiscal years (revenue, net profit, EPS), compare each broker's figure with the filed number (A1; Yahoo
fallback): within `broker_filed_tolerance_pct` (3%) → "reported basis", else "adjusted basis". The headline value
for a reported year is the **filed** figure; brokers' adjusted figures are shown separately with their count. A
forecast consensus uses the majority basis (inferred per broker from its reported-year figure); minority-basis
forecasts are listed apart. EBITDA has no filed anchor and stays "as each broker prints it".

### A6. Year-labelled financials
`summarize_financials` returns year-keyed values with empty years dropped, e.g. `revenue_cr: {"FY23": 12173.35,
…, "FY26": 48537.39}` and `revenue_cagr: {"pct": 58.6, "from": "FY23", "to": "FY26", "years": 3}`; the same
for net income, EPS, debt and free cash flow.

### A7. Plain-language verdicts in code
Fields the model misread get a `*_meaning` sentence beside them, e.g. Trendlyne `pe_zone_meaning`: "today's P/E is
higher than only 12% of its own history — cheap against its past (Trendlyne calls this zone 'Strong Buy')".

### A8. Alternatives from the screen
When the question asks for alternatives ("instead", "alternative(s)", "shortlist", "other stocks", "peers to
consider"), the evidence pack runs `screen_stocks` for the company's sector and includes the top names with their
scores; the prompt rule changes from "never with a named company" to "alternatives come only from the screen,
never from memory".

## 7. Part B — the evidence pack (Step 2)

- **Node** `evidence_pack`, after the data agent, when a company was resolved (no company → today's path).
- **Runs in code** (bounded concurrency `evidence_concurrency` = 4, per-tool timeout): get_stock_fundamentals,
  get_financials, get_forensics, get_valuation_context, get_shareholding, get_corporate_events,
  get_corporate_actions, get_annual_report_notes, get_broker_consensus, get_trendlyne_view, get_price_history,
  get_management_commentary, + screen_stocks when A8 applies. The same tools the analyst has, so there is one
  source of truth; outputs are parsed back into dicts (unparseable output is kept as text and flagged).
- **Fact sheet** (`render_fact_sheet`, ≤ `evidence_max_chars` = 14,000 ≈ 4K tokens): one line per fact with its
  source and date, values pre-computed and labelled, verdict words in code ("BELOW its own 4-year range"). Sections:
  price & valuation; growth (year-labelled); can the numbers be trusted; off the balance sheet (contingent
  liabilities, related parties); owners (pledge, promoter trend, insiders, ratings); street view (filed vs adjusted);
  Trendlyne; technicals; dividends and corporate actions; management guidance; alternatives (A8); **data gaps**
  (every UNKNOWN with its reason). Raw outputs stay in state, not in the conversation.
- **Placement:** one `HumanMessage` named `evidence_pack`, right after the user's question, opening "EVIDENCE PACK
  (computed by code from the tools below; figures are exact unless marked)". Human messages are never trimmed by the
  budget guard.
- **Analyst:** prompt gains an EVIDENCE PACK RULE (use the pack first; call tools for text — search_filings,
  search_broker_research, penny_stock_researcher — or for what the pack marks missing). Re-calling a pack tool gets
  "already in the evidence pack" instead of a second copy (the pack's call keys join the Step-1 held set).
- **Fact-check and judge** read the same fact sheet (the judge's instruction carries it); the trace stores the fact
  sheet and records each pack call in `tool_calls` with `source: "evidence_pack"`, so every Layer-0 check sees it.

## 8. Part C — keeping it fixed (Step 4)

- **`no_false_missing`** (Layer 0): fails when the report says a topic's data is missing / not returned /
  unavailable / unverified while that topic's tool returned it (forensics, shareholding, financials, valuation,
  pledge, contingent liabilities, related parties).
- **`report_complete`** (Layer 0): fails when the final analyst call stopped for length or the report ends
  mid-sentence.
- **Dixon regression** (offline, deterministic): fixtures from the real 2026-10-11 tool outputs, the SHP and
  integrated-filing XBRL, and the note-46 page text; the model step of A3 is stubbed with its recorded JSON. The test
  builds the fact sheet and asserts the four Section-2 errors cannot recur: disputed claims ₹530.27 Cr (FY26) =
  11.3% of net worth with bank guarantees ₹136.11 Cr in trade finance; FY23→FY26 labels and a 3-year CAGR; pledge 0%
  per the Jun-2026 SHP; related parties classified (Bharti Airtel = group_or_partner, Padget = subsidiary).
- **Eval:** the Dixon question joins the eval questions; `check_traces.py` reports the new checks.

## 9. Error handling

A source that is down or unparseable never becomes "clean": the fact sheet says UNKNOWN with the reason and the
analyst must repeat it. XBRL / NSE failures are logged with the URL; cached data is served with its as-of date. A
verification failure in A3 falls back to the excerpt, never to unverified numbers. NSE calls go through the existing
session handling and `refresh_delay_s` pacing.

## 10. Evaluation and acceptance

- **Contingent liabilities:** a hand-verified gold set (`eval/contingent_gold.jsonl`, ~15 reports chosen to span
  lakhs / crores / millions, PSU, bank, new-age, variant headings). Accept: ≥ 90% of `verified` extractions match the
  gold totals to the rupee crore; **zero** verified-but-wrong totals; `verified` coverage ≥ 80% of non-lender reports.
- **Pledge:** SHP XBRL read for ≥ 95% of the 119 companies' latest quarter.
- **Related parties:** XBRL rows found for ≥ 80% of companies' latest half-year.
- **Live Dixon run (ledger off):** all Layer-0 checks pass including tool_coverage, no_false_missing,
  report_complete and context_fits; the four Section-2 errors absent; ≤ 4 analyst calls; time ≤ the Step-1 run
  (365.9 s).

## 11. Testing (offline)

Unit tests per module with recorded fixtures (no network, no model): XBRL parsers (SHP pledged / not pledged,
integrated financials, RPT rows), relationship classifier, note locator and unit detector (lakhs, crores, millions,
"J" glyph, variant headings, policy prose rejected, bank schedule), verifier (hallucinated number dropped, subtotal
reconciliation), totals and exclusions, broker basis anchoring (Dixon FY26: Keynote reported, ABML and MOSL
adjusted), year-labelled financials, fact-sheet rendering and size cap, graph wiring (pack node, held calls, judge
and fact-check inputs, trace), the two new checks, and the Dixon regression.

## 12. Limits

- Integrated filings exist from Dec 2024; older periods use Yahoo and the annual report.
- The relationship classifier is keyword-based on free text; raw labels always travel with the class.
- A3 needs the local model once per annual report (batch, ~10–15 s each); a report not yet extracted falls back to
  the excerpt.
- Point-in-time: live runs use the latest filings; every value carries its broadcast date so a later backtest can
  filter by it, but no backtest path is built here.
- Not in scope: the earnings-surprise work of sub-project 3, MCA (AOC-4) filings, a UI.
