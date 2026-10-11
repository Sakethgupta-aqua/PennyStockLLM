# Company Documents Library — Design Spec

**Date:** 2026-10-04
**Status:** approved by the user and built 2026-10-04 (plan: `docs/superpowers/plans/2026-10-04-company-documents-library.md`)
**Builds on:** `2026-10-02-broker-intelligence-layers-design.md` (Broker_Reports and its pipeline stay as they are)

---

## 1. Goal

Every company in the broker folder (≈120: the 99 Midcap-100 names, 12 option-2 lenders, 9 from the
flat Trendlyne downloads) gets three kinds of documents, kept in an orderly local library:

| Kind | What it is | Source |
|---|---|---|
| Broker reports | sell-side research notes | Trendlyne / brokers' own sites (exists today) |
| Stock reports | Trendlyne's own scored PDF snapshot of the company | Trendlyne only |
| Corporate filings | the company's statutory documents | NSE (primary source) |

Success means: each file sits in a predictable place, every file is recorded once in one catalog
(what, whose, when, from where, fingerprint), re-runs never duplicate, and every quota is respected.

This spec covers **acquisition, layout and catalog** only. Using the new documents in the screener
and the analyst (§10) is a later, separate change.

## 2. Decisions taken with the user

| Question | Decision |
|---|---|
| Layout | three separate libraries + one catalog table (approach A of three) |
| Filings scope | key documents, last 2 years: results, investor presentations, earnings-call transcripts, annual reports, credit ratings, material events |
| Stock-report cadence | one snapshot now, then one per month; old snapshots are kept |
| Filings source | NSE directly (free, primary); Trendlyne's filings quota only as a fallback |

## 3. Ground truth measured on 2026-10-03/04

- **Trendlyne quotas** (its `/member/daily-runout/` page, this account): research reports 70/day,
  300/month; stock reports 70/day, 1,000/month; corporate-filing PDFs 100/day, 800/month. Daily
  limits reset at midnight IST, monthly ones on the 1st. The page says the caps exist to stop
  crawlers and downloaders.
- **Stock report.** `GET /equity/stock-report/get-pdf/<tl_id>/` returns `application/pdf` inline
  (no redirect), 1.5–2.6 MB, `Content-Disposition` filename `<SYM>-StockReport-<YYYYMMDD>-<HHMM>.pdf`
  (the generation time). `tl_id` comes free from the redirect of `/research-reports/stock/<SYMBOL>/`.
  Contents: DVM (durability/valuation/momentum) scores, SWOT + checklist, financials, valuation and
  buy/sell zone, technicals, peer comparison, forecaster estimates, shareholding and deals,
  management pay. Fetching needs a logged-in session; in the browser the PDF viewer page blocks
  storage, so files are captured as downloads.
- **119 of 120 stock reports are already on disk** (`Stock_Reports/<FOLDER>/`, 243 MB, `index.csv`).
  Pranav Constructions returned no PDF (fresh IPO).
- **Trendlyne's "corporate filings" are BSE announcements** served through
  `/get-document/post/pdf/<id>/` — copies of exchange documents, each costing filings quota.
- **NSE** `corporate-announcements?index=equities&symbol=X` returns ~1,400 rows per company (AUBANK
  1,383; 1,207 with a PDF on `nsearchives.nseindia.com`). Most are routine: AUBANK has 520
  "Analysts/Institutional Investor Meet/Con. Call Updates", 60 newspaper copies, 50 trading-window
  notices. Annual reports come from `annual-reports?index=equities&symbol=X` (already used by
  `data/annual_report.py`). `data/nse_client.py` holds the header recipe NSE requires.
- Research-report rows on Trendlyne also link `/get-document/post/pdf/` and `/cached_pdf/` copies.
- Disk: 654 GB free.

## 4. Rules that hold everywhere

1. **Never work around a quota.** Stop at Trendlyne's daily-runout page. Never fetch a research
   report through a `post/pdf` or `cached_pdf` link (that counts it against a different bucket to get
   past the research-report cap). No other accounts, no cookie resets.
2. **Bytes are verified, never trusted.** A file is kept only if it starts with `%PDF-` and is larger
   than a minimum size; HTML error pages and login pages are rejected.
3. **Writes are atomic.** Write to a temp file in the target folder, fsync, rename. A crash never
   leaves a half-written PDF under a real name.
4. **Records are immutable.** A catalog row is inserted once per document key and never rewritten.
   A newer stock report is a new row and a new file; nothing is overwritten.
5. **Everything stays local.** `Stock_Reports/` and `Corporate_Filings/` are git-ignored (done),
   like `Broker_Reports/`: third-party copyright, for reading only.

## 5. Layout

```
Broker_Reports/<FOLDER>/manifest.csv, *.pdf          unchanged
Stock_Reports/<FOLDER>/<FOLDER>-StockReport-<YYYYMMDD>-<HHMM>.pdf
Stock_Reports/index.csv                              human index (the catalog is authoritative)
Corporate_Filings/<FOLDER>/<category>/<YYYY-MM-DD>_<category>_<slug>_<id>.pdf
Corporate_Filings/<FOLDER>/manifest.csv              human index for that company
```

- `FOLDER` = the symbol made filesystem-safe by the same rule Broker_Reports uses (`M&MFIN` → `M_MFIN`,
  `GVT&D` → `GVT_D`).
- `category` ∈ `results | investor_presentation | earnings_call | annual_report | credit_rating |
  material_event`.
- `slug` = the first ~6 words of NSE's subject, lower-case, hyphenated; `id` = NSE's `seq_id` (a
  stable numeric id on every announcement row), so two filings on one day never collide. Annual
  reports have no `seq_id`; their id is `AR<fromYr>-<toYr>`.
- Dates come first so a folder lists oldest → newest.

## 6. Catalog: `documents` table in `market.db` (STRICT)

| Column | Meaning |
|---|---|
| `doc_key` TEXT PK | `filing:nse:<seq_id>` (annual reports `filing:nse:AR:<symbol>:<fromYr>-<toYr>`), `stock_report:trendlyne:<tl_id>:<YYYYMMDD-HHMM>`, `broker_report:trendlyne:<report_id>` (or `broker_report:sha:<hash>`) |
| `symbol` TEXT NOT NULL | NSE symbol |
| `kind` TEXT NOT NULL | `broker_report` \| `stock_report` \| `filing` (CHECK) |
| `category` TEXT | filing category; NULL for the other kinds |
| `doc_date` TEXT NOT NULL | ISO date the document is dated (filing date, report date, generation date) |
| `title` TEXT | NSE subject / broker headline / "Trendlyne Stock Report" |
| `source` TEXT NOT NULL | `nse` \| `bse` \| `trendlyne` \| `broker_site` |
| `source_url` TEXT | where it came from |
| `local_path` TEXT NOT NULL | relative to the project root |
| `bytes` INTEGER NOT NULL, `sha256` TEXT NOT NULL | integrity |
| `fetched_at` TEXT NOT NULL | transaction time (valid time is `doc_date`) — same bitemporal rule as the ledger |

- `INSERT … ON CONFLICT(doc_key) DO NOTHING`: a re-run is a no-op.
- A document already catalogued with an existing, hash-matching file is skipped before any download.
- `documents.py` exposes: `has(doc_key)`, `save_file(...)` (atomic write + hash + insert),
  `list_documents(symbol, kind=None)`, `counts()`, `verify()` (re-hash files, report missing/changed).

## 7. Corporate filings (NSE)

**Discovery.** One NSE session per company (existing `nse_session`), then:
- `corporate-announcements` → rows with `seq_id`, `sort_date` (ISO), `desc`, `attchmntText`,
  `attchmntFile`, `attFileSize`;
- `annual-reports` → `fileName`, `fromYr`, `toYr`, `attFileSize`.

**Window.** Filing date within the last 730 days; annual reports for the last two fiscal years.

**Classification** (first match wins; anything unmatched is not downloaded):

| Category | Rule |
|---|---|
| `earnings_call` | `transcripts.TRANSCRIPT_HINT` matches AND the attachment name contains "transcript" |
| `investor_presentation` | `desc` is "Investor Presentation", or text says investor/earnings presentation |
| `results` | `desc` is "Financial Result Updates", or an "Outcome of Board Meeting"/"Integrated Filing" whose text mentions financial results |
| `credit_rating` | `desc` is "Credit Rating", or `events.classify()` returns "credit rating action" |
| `material_event` | `events.classify()` returns any other material type (auditor exit, default, fraud/enforcement, pledge, senior-management exit, litigation, fund raise …) |
| `annual_report` | from the annual-reports endpoint |

Routine filings are excluded by `events.py`'s existing routine patterns before any of the above
(except `earnings_call`, which those patterns deliberately call routine).

**Download.** Same NSE session, ~1 request/second, one retry on a transient error, PDF verified,
saved via `documents.save_file`. Non-PDF attachments (some annual reports are ZIPs) are recorded as
skipped, not downloaded. Expected volume: ~50 files per company, ~6,000 in all.

**CLI** `scripts/fetch_filings.py`: `--status`, `--dry-run`, `--symbol X`, `--limit N`,
`--days 730`. Company list = every symbol with a Broker_Reports folder or a parsed broker document.

## 8. Stock reports

- **Filing** (`scripts/file_stock_reports.py`, productionised from tonight's helper): picks up
  `<SYM>-StockReport-<stamp>[__TL<id>].pdf` downloads (and legacy `.bin` bundles), verifies them,
  writes `Stock_Reports/<FOLDER>/…`, inserts catalog rows, refreshes `index.csv`, moves originals out
  of `~/Downloads`.
- **Fetching** (`scripts/fetch_stock_reports.py --cookies cookies.txt`): resolves `tl_id` through the
  research-page redirect (page views, no quota), fetches `get-pdf` (1 open each), at most 70 opens per
  run, stops at the runout page, skips a company that already has a snapshot this calendar month.
  Without a cookie it prints the browser procedure instead.
- **Monthly refresh**: running the fetcher (or the browser procedure) once a month adds one dated
  snapshot per company (~120 of the 1,000 monthly quota).

## 9. Broker reports in the catalog

`scripts/catalog_documents.py --backfill` registers every Broker_Reports PDF found by
`broker_corpus.discover()` as `kind=broker_report` rows, and every stock report already on disk. No
files move; the broker pipeline is untouched.

## 10. Later (not in this build)

- Filings → a symbol-pinned search over management's own words (presentations, transcripts),
  alongside Layer 3's broker search; `get_management_commentary` reading local transcripts instead of
  re-downloading them.
- Stock reports → parse Trendlyne's DVM scores into a table, shown next to the screener's score as an
  independent second opinion.

## 11. Testing

All offline: recorded NSE response fixtures, temporary database and folders, no model loads.
- catalog: atomic write (no partial file on failure), idempotent insert, hash recorded, CHECK on
  `kind`, `verify()` catches a changed file;
- filings: classification table (real NSE subjects → category), the 730-day window, routine filtering,
  file naming (date first, collision-free), non-PDF rejection;
- stock reports: name parsing (with and without `__TL<id>`), monthly skip rule, catalog rows, runout
  stop, 70-open cap.

## 12. Limits, stated up front

- NSE sometimes answers 403 or rate-limits; the fetcher backs off and resumes (it is idempotent).
- Some filings are scans with no text layer; they are stored, and flagged for OCR only when §10 needs
  their text.
- Stock-report fetching needs the user's logged-in session (browser or exported cookie).
