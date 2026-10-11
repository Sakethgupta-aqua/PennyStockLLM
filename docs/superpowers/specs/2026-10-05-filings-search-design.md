# Filings Search — Design Spec (sub-project 1 of 6)

**Date:** 2026-10-05
**Status:** design approved by the user ("approved": pages + passages; annual reports embed key sections
only, keyword search on every page); this written spec awaits review
**Builds on:** the documents library (`Corporate_Filings/`, the `documents` catalog), the retrieval stack
(bge-m3, bm25s, bge-reranker-v2-m3, reciprocal-rank fusion), the eval tooling (`build_eval_set.py`,
`review_eval_set.py`, `eval_rag.py`)
**Programme:** 5 Stock Report reader (done) → 4 event signals (done) → **1 filings search** → 3 results +
earnings surprise → 2 guidance tracker → 6 annual-report forensics.

---

## 1. Goal

Make every NSE filing on disk searchable and quotable, one company at a time, by meaning and by exact
words, so the analyst and the user can cite what a company said or disclosed - **document type, date and
page** - and never see a document filed after an "as of" date. Words only: numbers from results tables
belong to tools (sub-project 3 reads them from the exchange's structured XBRL).

## 2. Research that shapes it

- **Scope by company.** On FinanceBench, a shared vector store across companies answered 19% correctly
  against 50% for a per-document store (Islam et al. 2023, arXiv 2311.11944). Every search here is
  pre-filtered on one company, as the broker search already is.
- **Find the page.** Retrievers usually find the right document (recall 0.93) but miss the answer page
  (0.46); narrowing to pages before passages helps, and ~1,024-token chunks with 128 overlap work well
  for financial filings (arXiv 2602.17981).
- **Chunk by structure.** Element-based chunking improves financial RAG and halves indexing cost
  (Yepes et al., arXiv 2402.05131); simple parsers with recursive/sentence splitting and ~25% overlap lead,
  and tiny "semantic" chunks fall 8-12 MRR points behind (El Bachyr et al., ICSE-SEIP 2026, arXiv 2604.12047).
- **Context in the chunk.** Prepending a short context to each chunk cut failed top-20 retrievals by 35%
  (embeddings), 49% (with BM25) and 67% (with reranking) (Anthropic, Contextual Retrieval, 2024). Here the
  context is written deterministically from the catalog - company, type, date, period, page, speaker - at
  no model cost.
- **Earnings calls by speaker turn,** with a size cap, keeping a question with its answer (deepset;
  transcript-chunking practice). The project's own `transcripts.parse` already splits turns.
- **Keyword search at scale.** SQLite FTS5 ranks with BM25 natively, persists, and is the common partner
  of vector search under reciprocal-rank fusion.
- **Time.** Standard RAG ranks by relevance, not temporal validity; an as-of filter keeps retrieval
  point-in-time (look-ahead bias literature, e.g. arXiv 2601.13770).
- **Extractor.** pypdfium2 is the fastest Python extractor with a permissive licence; pdfplumber garbles
  multi-column reading order (public extraction benchmarks).

## 3. Ground truth measured (2026-10-05)

| Type | Files | Pages each | Text-less pages | Chars/page |
|---|---|---|---|---|
| annual_report | 229 | 292 | 0.5% | 4,462 |
| results | 1,213 | 24 | 1.4% | 2,302 |
| investor_presentation | 774 | 32 | 5.7% (image slides) | 954 |
| earnings_call | 747 | 19 | 0% | 2,521 |
| material_event | 2,480 | 6 | 0% | 2,487 |
| credit_rating | 571 | 5 | 3.1% | 2,004 |

- ~152,600 pages; annual reports hold ~64% of the text (~470M characters in all).
- **19 files are truncated** (no `%%EOF`; 13 annual reports, 4 results, 1 presentation, 1 material
  event) - round sizes such as 4,030,464 bytes show downloads cut off; the library checked only `%PDF-`.
- **294 files are exact duplicates** (same sha256 filed under two announcements).
- 6,013 of 6,014 source URLs carry the exchange broadcast stamp (`_DDMMYYYYHHMMSS_`).
- pypdfium2 reads a page in ~2 ms (pdfplumber 30-100 ms): all pages in ~5 minutes.
- bge-m3 (fp16, Apple GPU) embeds ~37 passages/s of ~1,200 characters.
- Annual-report section headings at page tops (10 sampled): MD&A 8, key audit matters 8, contingent
  liabilities 7, risk 10, related party 10; "Board's/Auditor's Report" need curly-apostrophe (’) folding.
- `transcripts.parse` yields >= 10 speaker turns for 13 of 20 sampled calls; the rest need a fallback.
- Results pages: digit share p50 0.06, p90 0.22 - statement tables are separable from commentary.

## 4. Storage

1. **Completeness.** A filing counts as stored only when it starts with `%PDF-`, carries `%%EOF` in its
   last 2 KB and opens in pypdfium2. `documents.verify` reports the incomplete; a repair pass re-downloads
   them (streamed, longer timeout) and replaces them atomically.
2. **`filings.db`** (a separate SQLite file beside `market.db`, ~1.5 GB, so the market cache stays small):
   - `filing_texts` - one row per unique file content (`content_sha` PK): symbol, doc_type, doc_date,
     `available_at` (broadcast time, else the filing date), pages, extractor and version, the doc_keys that
     point to it (duplicates indexed once).
   - `filing_pages` - one row per page: content_sha, page number, text, chars, `ocr` flag, `kind`
     (`prose` / `table` / `boilerplate`), `section` (annual reports, e.g. `mdna`, `auditor`).
   - `pages_fts` - FTS5 over page text (porter + unicode61 tokenizer) with symbol, doc_type and
     available_at as filter columns. Every page of every filing is keyword-searchable.
3. **Chroma collection `filings`** (in `penny_stock_vector_db/`, separate from books and brokers):
   passages with metadata symbol, doc_type, doc_date, available_at, fiscal period, page_from/page_to,
   section or speaker, content_sha. Deterministic ids; re-indexing a file replaces its passages.

Everything is keyed by content and versioned (`EXTRACTOR_VERSION`, `CHUNKER_VERSION`): a version bump
re-processes only what changed; a re-run of an unchanged file is a no-op.

## 5. Extraction

pypdfium2 text per page, normalised (whitespace, ligatures, curly quotes folded). A page under 40
characters is OCR'd with Tesseract at 300 dpi - except in investor presentations, where text-less pages
are chart images and OCR returns noise. Page kinds: `table` when numeric tokens exceed 30% of a page's tokens (the
default; checked against real results pages during the build), `boilerplate` for annual-report pages outside the
key sections, else `prose`.

## 6. Passages (what gets embedded)

The fiscal period comes from the filing date on India's April-March year: results, calls and
presentations filed within ~75 days of a quarter end belong to that quarter (Q1 = Apr-Jun); annual
reports belong to the year they cover; otherwise it is left blank.

Each passage starts with a context line built from the catalog, e.g.
`AU Small Finance Bank · earnings call · 2026-07-25 · Q1 FY27 · p.7 · Q&A - analyst: <name>`.

| Type | Unit | Embedded |
|---|---|---|
| earnings_call | speaker turns, a question kept with its answer, capped at ~1,000 tokens; recursive split when parsing fails | all |
| investor_presentation | one slide (page) | prose slides |
| credit_rating | the rationale page(s) | all |
| material_event | the filing (pages merged up to the cap) | all |
| results | press release / commentary pages | prose pages only; tables stay keyword-only |
| annual_report | key sections: MD&A, Board's/Directors' report, risk management, independent auditor's report incl. key audit matters, notes on contingent liabilities and related parties | key sections only; the rest keyword-only |

Recursive splits use ~1,000 tokens with 25% overlap. Estimated 150-200k passages, ~1-1.5 hours of
background embedding on first run; afterwards only new filings.

## 7. Retrieval

`search_filings(symbol, question, doc_types=None, as_of=None, top_n=8)`:
1. filter: the company (required), document types (optional), `available_at <= as_of` (optional);
2. keyword: FTS5 BM25 over pages -> best pages; meaning: dense search over passages;
3. pages found by keyword map to their passages (keyword-only pages return an FTS5 snippet);
4. reciprocal-rank fusion -> bge-reranker -> at most 2 passages per document -> top N;
5. each result: text, company, type, date, fiscal period, page(s), section/speaker, file path.
Results read newest first after ranking, as the broker search does.

## 8. Where it is used

1. **Analyst tool `search_filings`** - for "what did management say about…", "why was the rating cut",
   "what did the company disclose". Prompt rule: quote with type and date; never take a number from a
   passage (numbers come from tools); say "not found in the filings on file" rather than guess. Layer-0
   learns the tool and its coverage words.
2. **CLI**: `scripts/index_filings.py` (extract, index, embed; `--status`, `--symbol`, `--repair`,
   `--reindex`) and `scripts/search_filings.py "question" --symbol X [--type earnings_call] [--as-of DATE]`.
3. **Freshness**: `refresh_data.py` gains a "filings index" dataset - stale when catalogued filings have no
   extracted text or passages.

## 9. Evaluation

A filings golden set (80-100 questions): passages are sampled across types and companies, the local LLM
writes a question each answers (the `build_eval_set.py` pattern), and each row is reviewed against its
passage before it is marked verified (`review_eval_set.py`). The eval reports page recall@5/@10 and MRR for
four configurations - keyword only, meaning only, fused, fused + reranker - and the best one becomes the
default, with the numbers recorded.

## 10. Testing (offline)

Completeness check (truncated, header-only, valid); duplicate content indexed once; page kinds (table vs
prose) on real-shaped text; annual-report section detection with curly apostrophes; speaker-turn passages
and the fallback split; context lines; FTS5 search with company/type/as-of filters; fusion, the
per-document cap and newest-first order; the as-of filter never leaking a later filing; idempotent
re-index and version bumps; the tool's "not found" path.

## 11. Limits

- Table numbers are not answered from text; sub-project 3 reads results from XBRL.
- OCR quality on scans varies; chart-only slides are not OCR'd.
- Annual reports outside the key sections are keyword-searchable only (sub-project 6 goes deeper).
- The golden set is model-written and reviewed, not exhaustive; scores are relative, for A/B choices.
