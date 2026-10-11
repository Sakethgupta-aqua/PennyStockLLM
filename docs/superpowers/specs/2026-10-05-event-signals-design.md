# Event Signals — Design Spec (sub-project 4 of 6)

**Date:** 2026-10-05
**Status:** design approved by the user ("yes go ahead": stored + graded events; severe events exclude a
stock for 12 months; all 1,486 companies in the screening cache); this written spec awaits review
**Builds on:** the prediction ledger (`ledger.py`, `outcomes.py`), the documents library, the screener
**Programme:** 5 Stock Report reader (done) → **4 event signals** → 1 filings search → 3 results + earnings
surprise → 2 guidance tracker → 6 annual-report forensics. Each has its own spec, plan and build.

---

## 1. Goal

Store the disclosures that move Indian stocks - insider buying and selling, promoter pledges, credit-rating
actions, a company's own auditor quitting, defaults, fraud - **dated to the moment the market learned
them**. Show them to the user and the AI analyst, exclude the worst cases from the screener, and grade
every signal-bearing type in the ledger, so the system learns which events actually move *these* stocks
rather than trusting the literature.

## 2. Research that shapes it

- **Insider trades.** Indian insider purchase portfolios earn positive abnormal returns after the
  disclosure and sale portfolios negative ones (NITK, SEBI PIT disclosures to BSE 2007-2015,
  idr.nitk.ac.in/handle/123456789/14150). Purchases are the cleaner signal: insiders sell for many reasons.
- **Credit ratings.** Downgrades bring significant negative abnormal returns in India; upgrades show
  little or no reaction, and reactions weakened after 2008 (IIM Bangalore WP 134,
  repository.iimb.ac.in/bitstream/123456789/487/1/WP.IIMB.134.pdf; research.iimb.ac.in/fac_pubs/333).
- **Issuer Not Cooperating (INC).** A rating agency marks a rating INC when the issuer stops supplying
  information or fees; INC on all ratings for six months forces a downgrade to non-investment grade
  (SEBI circular on disclosure of INC issuers; taxguru.in summary).
- **Promoter pledging.** Pledging is significantly associated with future crash risk in India
  (257 S&P BSE 500 firms, 2011-2020; Emerald JAMR-01-2023-0003). An invoked pledge is forced selling.
- **Auditor resignations.** SEBI's circular CIR/CFD/CMD1/114/2019 (18 Oct 2019) followed a surge of
  mid-term resignations; the reasons most often cited were non-cooperation and non-receipt of
  information. It concerns the listed company's **own** statutory auditor.
- **Enforcement news** (show-cause notices, penalties): no significant market reaction was found for
  insider-trading enforcement announcements (The Leap Journal, 2026) - shown, not graded or vetoed.

## 3. Ground truth measured (2026-10-05)

**NSE feeds.**
- `corporates-pit` (insider trades, old format): reliable only as a **market-wide** query by month
  (March 2025: 1,980 disclosures, 368 companies; March 2026: 2,057). Per-company queries are not: the date
  filter is inconsistent (ZYDUSLIFE: 2-year window 1 row, 4-year window 32) and the default returns the
  last 20 rows however old (YESBANK: 2021). Populated **up to April 2026**, empty from May 2026.
  Modes seen: Market Purchase, Market Sale, Off Market, ESOP, Pledge Creation, Invocation of pledge,
  Revokation of Pledge, Preferential Offer, Inter-se-Transfer, Gift, Scheme...; person categories:
  Promoters, Promoter Group, Director, Key Managerial Personnel, Immediate relative, Employees.
  `date` is the exchange broadcast time.
- `corporates-pit-gg` (insider trades, new format) starts **May 2026** (347 disclosures; September 2026:
  895). Each row is an index entry with `broadcastDateTime` and an XBRL file (~5 KB) holding the trade:
  `CategoryOfPerson`, `NameOfThePerson`, `SecuritiesAcquiredOrDisposedTransactionType` (Buy/Sell/...),
  `ModeOfAcquisitionOrDisposal`, `...ValueOfSecurity`, holdings before/after (%), trade and intimation dates.
  Old + new feeds together give continuous insider history.
- `credit-rating-sdd-reg30` (structured rating actions) starts **~September 2025** (ZYDUSLIFE: 14 actions,
  2025-09-10 → 2026-09-10; nothing for 2023-2025 windows). Per instrument: agency, rating, outlook,
  `currentAction` ∈ {Re-affirm, Other, Upgrade, New, Downgrade}, free-text `brieFDetails` (e.g.
  "IVR B /Negative; ISSUER NOT COOPERATING* (Rating Downgraded...)", "Affirmed; Outlook revised to
  Positive"), `broadcastDateTime`. September 2026: 4,103 rows, 180 upgrades, 43 downgrades, 7 INC.
- `corporate-credit-rating` (old) is a market-wide list of the last ~3 days with the symbol mostly empty -
  **unusable**, and it is what `events.py` reads today (§6.3).
- `corporate-pledgedata` returned nothing for every company tried; promoter pledges come from the insider
  feeds (March 2025 alone: 91 pledge events).

**Announcements (6,014 filings of 119 companies, 2 years).** The current keyword screen is unsafe for
exclusion: all 6 "auditor resignation" hits were auditors of a **subsidiary**; all 33 "NCLT/insolvency"
hits were **merger-scheme** hearings; "fraud" hits included banks **reporting a borrower's** fraud;
"pledge" hits were **company** financing (Mankind pledging a subsidiary's shares), and one enforcement hit
was the boilerplate "not debarred by any SEBI order". Rating announcements carry no direction in 435 of
571 titles and none says "downgrade" - direction comes from the rating feed, not announcements.

## 4. Event types

| type | source | rule | ledger call | screener |
|---|---|---|---|---|
| `insider_buy` | insider feeds | Market Purchase by Promoters, Promoter Group, Director or KMP; one event per person per broadcast day; value ≥ ₹10 lakh | bullish 3; **4** when ≥ 2 different insiders bought in the 30 days ending on its signal day | panel |
| `insider_sell` | insider feeds | Market Sale, same people and threshold | bearish 2 | panel |
| `pledge_created` | insider feeds | Pledge Creation by Promoters / Promoter Group | bearish 2 | panel |
| `pledge_invoked` | insider feeds | Invocation of pledge | bearish 4 | panel |
| `pledge_released` | insider feeds | Revocation of pledge | - | panel |
| `rating_downgrade` | rating feed | `currentAction` Downgrade (any instrument) | bearish 3 | panel |
| `rating_upgrade` | rating feed | `currentAction` Upgrade | bullish 2 | panel |
| `rating_inc` | rating feed | "ISSUER NOT COOPERATING" in the details | bearish 3 | **veto** |
| `rating_other` | rating feed | reaffirm / new / withdrawn / watch / outlook change | - | panel |
| `auditor_exit` | announcements | the company's **own** statutory auditor resigns or is removed, or an adverse opinion / disclaimer; never "of a (material) subsidiary" | bearish 4 | **veto** |
| `auditor_exit_subsidiary` | announcements | a subsidiary's auditor resigns | - | panel (low severity) |
| `insolvency` | announcements | CIRP / insolvency petition admitted, default or delay in paying interest or principal, winding-up admitted; never a scheme, merger, convened meeting or motion petition | bearish 4 | **veto** |
| `fraud` | announcements | forensic audit appointed or reported, fraud by the company or its officials, search and seizure; never a lender reporting a **borrower's** fraud | bearish 4 | **veto** |
| `enforcement` | announcements | show-cause notice, penalty, SEBI/RBI order; never "not debarred" boilerplate | - | panel |

Several rows on one day (instruments of one rating, trades by one person) collapse into one event per
(company, type, person or agency, day); the ledger gets **one call per (company, type, signal day)** with
the strongest conviction. Thresholds and windows (₹10 lakh, 30 days, 12 months) are settings.

## 5. Storage: `event_signals` (market.db, STRICT)

One row per event: `event_key` (stable id from the source - NSE's disclosure id / XBRL file / announcement
`seq_id` / rating ISIN+agency+broadcast), `symbol`, `event_type`, `severity` (`veto` / `signal` / `info`),
`direction` (NULL for info), `conviction`, `signal_at` (exchange broadcast time - when the market learned
it), `event_date` (trade or rating date), `source`, `source_url`, `details` (JSON: person, category, mode,
value, holdings before/after, agency, rating, outlook, text), `ingested_at`, `rules_version`. The source row is kept in `details`, so `--retype` re-derives type,
severity and direction under new rules without refetching. Insert-or-ignore by `event_key`: re-running
never duplicates. Market-wide feeds store every listed company they return; announcements are fetched for
the 1,486 companies in the screening cache. Ledger calls already recorded are never rewritten by a re-type.

## 6. Where it is used

1. **Ledger.** Each signal-bearing event → `Prediction(source="event:<type>")`; as-of date = first
   session after the broadcast day (`outcomes.actionable_date`, lag 1 - as for brokers and Trendlyne);
   entry = that session's close from the feed (provenance `feed_adjusted`); payload = the event's details;
   provenance records whether it was recorded live or backfilled. Idempotent by (company, type, signal
   day). Graded nightly at 1/5/20/60 sessions vs Nifty 500 like every other source, so each type earns a
   **track record** (`scorecard(source="event:insider_buy")`).
2. **Screener.** A veto gate *severe corporate event in the last 12 months* (`auditor_exit`,
   `insolvency`, `fraud`, `rating_inc`), applied to lenders too; `--explain` gets an **EVENTS** panel:
   the last 12 months of insider buying/selling (who, how much), pledges, rating actions and severe
   events with dates and links, plus each type's track record (n graded, hit rate at 20 sessions).
3. **AI analyst.** `get_corporate_events` reads the stored events for insider trades, pledges and rating
   actions (dated, typed, with track records) instead of today's live calls - which read a market-wide
   rating list (another company's rating can be reported as this one's) and insider rows that can be
   years old. Live announcements stay for the newest days. Prompt and Layer-0 coverage learn "pledge"
   and "insider buying".
4. **Ingestion.** `scripts/ingest_events.py`: insider and rating feeds by month, market-wide (old insider
   feed to April 2026, new feed from May 2026 with one XBRL fetch per disclosure of a company in the
   screening cache, rating feed from September 2025); announcements per company for the 1,486 companies over 2 years; resumable, ~1
   request/s, no Trendlyne quota; `--status`, `--symbol`, `--since`, `--no-ledger`. A full first run is
   ~1.5-2 hours in the background; later runs fetch only the new days. Registered with
   `refresh_data.py --status` as a source that goes stale after 7 days.

## 7. Testing (offline)

Real-shaped fixtures for each feed (old insider rows, a new-format XBRL file, rating rows, announcement
texts). Every false positive measured in §3 becomes a negative test (subsidiary auditor, NCLT merger
scheme, borrower fraud, company pledge of a subsidiary's shares, "not debarred" boilerplate) beside true
positives (own auditor resignation, CIRP admission, forensic audit, INC rating). Plus: the ₹10 lakh
threshold and the 30-day cluster rule; one call per (company, type, day); lag-1 dating; the 12-month veto
window (in at 364 days, out at 366); idempotent ingestion and recording; the panel; the tool's stored path
and its UNKNOWN path.

## 8. Limits

- Rating actions exist structurally only from ~September 2025; older ones are not backfilled.
- New-format insider details need one small XBRL fetch per disclosure.
- Announcement rules are English keyword rules: precision is checked on 6,014 real filings; recall is
  unknown, so a missing severe event means "not detected", never "clean".
- A track record needs time: a type's hit rate is shown with its sample size and means little below ~30
  graded calls.
- NSE can change these private endpoints (the insider feed already moved once, in May 2026);
  `--status` reports a feed that returns nothing for a month.

## 9. Amendments after the full run (2026-10-05, approved design unchanged in intent)

The first full run (10,752 events, 2,059 companies) showed where the approved rules would exclude good
companies; each change below has a test built from the real filing that exposed it.

- **Auditor exits.** A subsidiary's auditor is recognised even when the subsidiary's name sits between
  "auditor" and "subsidiary", or is named first; "auditors of the Company" always counts as the company's
  own. A filing that says only "Resignation of Statutory Auditor" is settled by reading its attached letter
  (a ZIP of PDFs); one filed in the same week as a subsidiary filing is that same story.
- **Fraud.** Press releases, the exchange's news queries, closures, fraud committed *against* the company
  (by employees or hackers) and NSE's bare "Fraud/Default/Arrest" label are not company fraud. Income-tax
  and GST searches are shown (`enforcement`), not vetoed; Enforcement Directorate searches and SFIO
  investigations still veto.
- **Insolvency.** Buying or claiming against an insolvent company is not insolvency; a petition that has
  not been admitted is shown only. A stay, set-aside, withdrawal or closure (`insolvency_lifted`) lifts the
  veto when it is later than the last insolvency filing.
- **Ratings.** "Issuer Not Cooperating" is usually administrative (a company that repaid its loans stops
  paying the agency, which must then cut it to non-investment grade), so `rating_inc` is a graded signal
  (bearish 2), not a veto. A D or C grade is the veto (`rating_default`, bearish 4); a withdrawn INC is
  information.
- **Ledger.** One call per company and type per 30 days unless the conviction rises, so a company filing
  insolvency updates every week, or a promoter buying every few days, does not swamp a type's record.
