# Broker Intelligence, Layers 1–3 — Design Spec

**Date:** 2026-10-02
**Status:** built on the user's instruction to implement without an approval stop
**Builds on:** `2026-09-27-prediction-ledger-design.md` (the ledger is Layer 2's store of broker calls)

---

## 1. Goal

The broker-report corpus is currently text the agent cannot reach and numbers that exist for
40 milliseconds before being flattened into a sentence. Turn it into three things:

| Layer | Question it answers | Consumer |
|---|---|---|
| **1. Sector KPIs** | "What are this lender's NIM, GNPA, CRAR…?" | screener (`--explain`, and the score once coverage allows), agent |
| **2. Consensus** | "What do the brokers think, how much do they disagree, which way are they moving?" | `--explain`, agent |
| **3. Narrow RAG** | "*Why* does this broker rate it Buy?" | agent |

Rule that splits the layers: **numbers come from structured tables (1, 2); arguments come from
retrieval (3).** Research on financial RAG is consistent that tables embed badly and that
numeric questions should be answered by tools, narrative ones by retrieval.

## 2. Ground truth measured on 2026-10-02

- **Corpus.** 12 flat PDFs (all dated 4 Sep 2026) plus the Midcap-100 manifest: 297 reports,
  99 companies, 20 brokers, dated 2020-07-21 → 2026-09-29. **1 of 297 PDFs is downloaded**:
  Trendlyne serves them only to a logged-in session.
- **The screener is blind on lenders.** All 186 lenders score exactly 0.50 on safety: every
  safety input is (correctly) skipped for them, so 20% of a bank's score is a constant.
- **The parser misses the best KPI table.** AU SF Bank's "Key Ratios" table (NIM 5.7%, CRAR
  18.7%, GNPA 2.0%, NNPA 0.7%, PCR 64/85%, cost-to-income 57.4%) is headed
  `Q1FY27 Q1FY26 Q4FY26 FY26`; quarter tokens are not recognised, so the table is skipped.
- **Unit bug, 10×.** When the unit sits in the caption above a table ("(INR m)", "Rs in Million")
  rather than on the header line, money is stored as crore: Trent FY25 revenue is stored as
  1,71,346 cr (true ≈ 17,135 cr); Physicswallah is 10× high in one table and 100× low in another.
- **A call was silently lost.** Trendlyne calls AU Small Finance Bank "AU SF Bank", which does not
  resolve, so its BUY never reached the ledger. The manifest carries the ticker; the flat
  listing does not.
- **The ingest only sees one folder** (`glob("*.pdf")`), so the 99 per-company folders are
  invisible; the vector ingest uses random ids, so re-running it duplicates every chunk.
- **Latent grading bug.** `grade()` fetches two years of prices; for an older call the first
  available session is used as the start, so the call is silently graded on the wrong window.
  Harmless today (every call is from 2026); wrong the moment the manifest's 2020–2025 calls land.
- **Latent screener crash.** Four `stock_metrics` rows hold the *text* `'Infinity'` in `pe`;
  `screen.py --no-vetoes --min-market-cap 0 --min-turnover 0` raises `TypeError`.

## 3. Research → rules

| Finding | Rule here |
|---|---|
| I/B/E/S builds a consensus only from estimates on the **same accounting basis**, and reviews estimates older than ~100 days | store the basis (the row label as printed); a broker's newer report replaces its own older one; calls older than `broker_max_age_days` (180) are excluded from "current" |
| Brokers disagree on settled facts (Muthoot FY26 NIM: 11.5–12.7%) because NIM has no single definition; GNPA/CRAR are regulator-defined | KPI observations are facts about *what a broker said*, never about the company; conflicts are resolved at read time (median, range, n), never at write time |
| Fiscal-year columns mix actuals (A) and forecasts (E) | `period_kind` is stored; forecasts are never ranked as facts. Unsuffixed periods: actual once SEBI LODR's filing deadline has passed (60 days after year-end, 45 after a quarter) |
| Point-in-time / bitemporal discipline (valid time vs transaction time) | `report_date` (valid) and `parsed_at` (transaction) are both stored; re-parsing a document replaces only that document's rows |
| Dispersion of forecasts predicts lower returns (Diether, Malloy & Scherbina 2002); consensus target upside is informative when dispersion is low and **contrarian when high**, often because some targets are stale (Palley, Steffen & Zhang 2025) | dispersion is a first-class output; above `broker_high_dispersion` the panel says the consensus upside is unreliable |
| The **change** in consensus recommendation predicts returns better than its level (Jegadeesh, Kim, Krische & Lee 2004); sell calls drift more than buys (Womack 1996); forecast revisions predict returns (Chan, Jegadeesh & Lakonishok 1996) | 90-day drift in the consensus, rating up/downgrades, target raises/cuts, EPS revision breadth |
| A Buy from a broker that rates most stocks Buy carries less information (Barber, Lehavy, McNichols & Trueman 2006) | each broker's share of bullish calls is shown once it has ≥10 calls |
| Only 38% of targets are met at the 12-month horizon (Bradshaw, Brown & Huang 2013) | targets are opinions; the agent must attribute them and never present one as its own forecast |
| Financial RAG improves most from **metadata pre-filtering** by company, layout-aware chunks and routing numeric questions to structured data | the broker retriever filters by symbol and research chunk types before ranking; numbers route to `get_broker_consensus` |

## 4. Architecture

```
broker_corpus.discover()          one record per report: flat PDFs + per-company manifests,
   │                              de-duplicated (Trendlyne id / content hash), ticker resolved
   ├── ingest_broker_reports.py   parse PDF → broker_documents + broker_metrics      (Layer 1)
   │                              research chunks → Chroma, tagged with symbol       (Layer 3)
   └── backfill_broker_predictions.py
                                  listing facts → ledger `predictions` (source=broker) (Layer 2)

read side:  broker_metrics.kpi_snapshot / screening_kpis      → screener + agent  (Layer 1)
            consensus.broker_consensus                         → screener + agent  (Layer 2)
            retrieval.broker_search.search_broker_passages     → agent             (Layer 3)
```

## 5. Data model (SQLite, `market.db`, STRICT tables)

`broker_documents` — one row per parsed PDF (the ingest catalogue): `doc_id` (sha256 of the
bytes, 16 hex) PK, `report_id` (Trendlyne id), `symbol`, `company`, `broker`, `report_date`,
`rating`, `source_file`, `pages`, `metric_rows`, `parser_version`, `parsed_at`, `embedded_at`.

`broker_metrics` — one row per (document, metric, period): `metric`, `period` (`FY27`,
`Q1FY27`), `period_kind` (`actual`|`estimate`), `period_end`, `value`, `label` (the basis, as
printed), `page`, plus `symbol`/`broker`/`report_date` copied from the document.

- Re-parsing a document **replaces that document's rows** in one transaction: the numbers are a
  pure function of the PDF bytes and the parser, so this corrects extraction without ever letting
  a later report overwrite an earlier one.
- `PARSER_VERSION` is stamped on every document; a parser change re-parses old documents
  automatically on the next ingest.
- Within one document the first occurrence of a (metric, period) wins — the cover summary is the
  broker's headline table; later pages are detail.
- A money row whose unit cannot be determined is **dropped**, never assumed to be crore.

Layer 2 adds no table: broker calls are ledger predictions (immutable, graded nightly).

## 6. Layer 1 — reduction and screening

- Per (metric, period): latest report per broker → `median`, `low`, `high`, `n`, `labels`,
  `status` (`single` | `agree` | `differ`; agree = equal within rounding,
  `max(0.15, 1% of |median|)`).
- Snapshot per metric: the latest **actual** period (by period end, quarters included) plus the
  next two forecast years.
- **Score admission** (financials only — the proven gap): the latest fiscal-year actual, median
  across brokers, from reports within `broker_max_age_days`, and only once
  `broker_kpi_min_companies` (20) lenders in the screen have that metric. A lender without broker
  coverage is unaffected (a missing KPI is not a data gap and does not lower completeness).
- Blind spots become computed: a blind spot is "covered" when a KPI mapped to it has data.

## 7. Layer 2 — consensus panel

From the ledger's live broker predictions for one symbol: current calls (latest per broker,
≤180 days), rating mix and mean score (I/B/E/S 1–5, 1 = Buy), target median/range/dispersion,
implied upside when a live price is supplied, 90-day revisions per broker against its own
previous call, 90-day consensus drift, EPS-estimate revision breadth from `broker_metrics`, and
flags (single broker, high dispersion, stale calls excluded, a sell call present).

## 8. Layer 3 — narrow RAG

Ingest only documents resolved to a listed symbol (daily notes and macro letters fall out
naturally), tag chunks with `symbol`/`doc_id`/ISO `report_date`, deterministic ids, skip documents
already embedded. Two text-quality rules were added after reading real retrieval output: a short
wrapped line only starts a new section when the previous paragraph has ended and the next line
starts a sentence (it was cutting paragraphs mid-sentence), and chart residue - lines of bare axis
numbers or years, and "Source:" captions - is dropped from prose (it was the top hit for "why will
credit cost fall"). 881 fragmentary chunks became 372 coherent ones on the same 9 reports. Query: filter `{symbol, chunk_type ∈ {narrative, recommendation}}` → dense +
BM25 on the filtered set → RRF → cross-encoder → ≤2 chunks per report → newest first, each stamped
with broker, date and rating.

## 9. Agent integration

Two symbol-pinned tools — `get_broker_consensus` (Layers 1+2, numbers) and
`search_broker_research` (Layer 3, arguments) — a STREET VIEW section and BROKER RULE in the
analyst prompt, Layer-0 assertion updates (known tools, tool-coverage keywords), and trace
parsing so quotes from broker passages are checked by `citation_provenance`.

## 10. Testing

Every rule above has a unit test; parsing tests use text shaped like the real reports; storage
tests run against a temporary database; nothing in the suite touches the network or loads a model.

## 11. Limits, stated up front

- The midcap PDFs arrive in daily rounds: Trendlyne allows this account 70 report opens a day
  and 300 a month (49 of 297 on disk after 2026-10-02). Until enough lenders are covered, Layer 1
  stays thin and Layer 2 leans on listing facts (ratings, targets, dates).
- Month-style quarter headers (`Sep-25`) are not parsed — they are mostly shareholding tables.
- Headers added after the first live batch (PARSER_VERSION 6–7): Motilal's bare years under
  "Y/E March" (`2026 2027E`) and two labelled tables on one line (BOB, ICICI Direct). A row whose
  digits the text layer split apart (`1 5,123`) is dropped, and so is a row holding an impossible
  value (a 194.7% margin means the label sits on another row's numbers). Money in side-by-side
  tables is still never stored: the two halves can carry different units.
- PARSER_VERSION 8 (2026-10-03): for a percentage KPI, a row whose label says it is a percentage
  (`GNPA Ratio (%)`) beats a bare label in the same document. Motilal's asset-quality tables print
  `GNPA 45.6` - the amount in INR b - above the ratio, which had stored IDFC First's GNPA as 45.6%.
- KPI score admission is inert until 20 lenders have broker KPIs; that is the gate working.
