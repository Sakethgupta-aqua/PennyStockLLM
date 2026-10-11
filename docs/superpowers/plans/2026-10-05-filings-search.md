# Filings Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every NSE filing on disk searchable by exact words and by meaning, one company at a time, with
type, date and page on every result and an as-of filter that never leaks a later filing.

**Architecture:** `documents.py` gains a completeness check and `filings.py` a repair pass.
`pennystock/data/filing_text.py` extracts pages (pypdfium2, Tesseract fallback) into `filings.db` with an
FTS5 index. `pennystock/retrieval/filing_passages.py` turns pages into structure-aware passages with a
context line. `pennystock/retrieval/filing_search.py` embeds passages into the Chroma collection `filings`
and runs the hybrid search. The analyst tool, CLIs, maintenance dataset and eval sit on top.

**Tech Stack:** pypdfium2, pytesseract, SQLite FTS5, Chroma + bge-m3, bge-reranker-v2-m3, LangChain, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-filings-search-design.md`

## Global Constraints

- Text: pypdfium2 per page; OCR (Tesseract, 300 dpi) only for pages under 40 characters, never for investor presentations.
- Completeness: a stored PDF starts with `%PDF-` and carries `%%EOF` in its last 2 KB; `verify` also requires it to open.
- `filings.db` beside `market.db`; FTS5 `tokenize='porter unicode61'`; duplicate content (same sha256) indexed once.
- Passages: at most ~1,000 tokens (4,000 characters), 25% overlap on recursive splits, a context line first.
- Table pages: numeric tokens > 30% of tokens - keyword-only. Annual reports: only key sections embedded.
- Search: company required; `available_at <= as_of` on both paths; at most 2 passages per document; newest first after ranking.
- The analyst never takes a number from a passage. Tests offline. No commits unless the user asks.

## Review Focus

- A question full of FTS5 syntax (quotes, AND/OR/NOT, `-`, `*`, `:`) never crashes keyword search. (Task 2)
- An as-of date before a filing's broadcast time never returns it, keyword or dense. (Task 4)
- A results filing whose pages are tables is findable by keyword and never embedded as prose. (Tasks 2-3)
- A crash mid-embedding resumes without duplicates (deterministic ids; the version is set after success). (Task 4)
- A company with no indexed filings is reported as such, never as "nothing disclosed". (Task 5)

---

### Task 1: Complete files only, and a repair pass

**Files:** Modify `pennystock/data/documents.py`, `pennystock/data/filings.py`, `pennystock/config.py`;
Test `tests/test_documents.py`, `tests/test_filings.py` (fixtures gain a `%%EOF` trailer).

**Interfaces — Produces:** `documents.is_complete_pdf(data) -> bool`, `documents.opens_cleanly(path) -> bool`,
`documents.replace_file(doc_key, data) -> Path | None`, `verify()` gains `"incomplete"`,
`filings.repair(get, keys=None) -> dict`, `settings.filings_repair_timeout_s = 300`.

- [ ] **Step 1: failing tests**

```python
def test_a_pdf_cut_off_mid_download_is_not_complete():
    assert docs.is_complete_pdf(b"%PDF-1.7\n" + b"x" * 6000 + b"\n%%EOF\n")
    assert not docs.is_complete_pdf(b"%PDF-1.7\n" + b"x" * 6000)          # the 4,030,464-byte kind
    assert not docs.is_complete_pdf(b"<html>error</html>")

def test_a_truncated_download_is_never_stored(project):
    assert save(project, data=b"%PDF-1.7\n" + b"x" * 6000) is None

def test_verify_lists_files_that_are_incomplete_on_disk(project):
    save(project)
    path = project / "Corporate_Filings" / "AUBANK" / "a.pdf"
    path.write_bytes(path.read_bytes()[:-10])                                # cut the trailer off
    assert docs.verify()["incomplete"] == ["filing:nse:1"]

def test_replacing_a_broken_file_updates_its_size_and_hash(project):
    save(project)
    good = b"%PDF-1.7\n" + b"y" * 6000 + b"\n%%EOF\n"
    assert docs.replace_file("filing:nse:1", b"%PDF-1.7\n truncated") is None
    path = docs.replace_file("filing:nse:1", good)
    d = docs.get("filing:nse:1")
    assert path.read_bytes() == good and d.bytes == len(good) and d.sha256 == hashlib.sha256(good).hexdigest()

# tests/test_filings.py
def test_repair_refetches_only_incomplete_filings(project, monkeypatch):
    monkeypatch.setattr(fl.documents, "verify", lambda: {"missing": [], "changed": [], "incomplete": ["filing:nse:1"]})
    monkeypatch.setattr(fl.documents, "get", lambda k: type("D", (), {"source_url": "https://x/a.pdf", "doc_key": k})())
    replaced = []
    monkeypatch.setattr(fl.documents, "replace_file", lambda k, data: replaced.append(k) or project / "a.pdf")
    assert fl.repair(lambda url: b"%PDF-1.7\nok\n%%EOF\n") == {"repaired": 1}
    assert replaced == ["filing:nse:1"]
```
  (The test fixtures' `PDF` constants gain `b"\n%%EOF\n"`; existing tests otherwise unchanged.)

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (`save_file` requires `is_complete_pdf`; `verify`
  adds `incomplete` = no trailer or does not open in pypdfium2; `replace_file` writes atomically and updates
  bytes/sha256/fetched_at; `repair` uses a getter with `filings_repair_timeout_s`). **Step 4: run; full suite.**

### Task 2: Pages, `filings.db`, keyword search

**Files:** Create `pennystock/data/filing_text.py`; Modify `pennystock/config.py`; Test `tests/test_filing_text.py`

**Interfaces — Produces:** `EXTRACTOR_VERSION`, `KEY_SECTIONS`, `normalize(text)`, `page_kind(text, doc_type, section)`,
`ar_sections(pages: list[str]) -> list[str | None]`, `fiscal_period(doc_type, doc_date) -> str | None`,
`available_at(source_url, doc_date) -> str`, `fts_query(question) -> str | None`,
`extract_pages(path, doc_type, ocr=None) -> list[tuple[str, bool]]`, `index_texts(docs=None, reextract=False, extract=None) -> dict`,
`keyword_pages(symbol, question, doc_types=None, as_of=None, limit=50) -> list[dict]`, `doc_meta(content_sha) -> dict`,
`pages_of(content_sha) -> list[dict]`, `indexed_count(symbol) -> int`, `status() -> dict`.
Settings: `filings_db`, `filings_ocr_min_chars=40`, `filings_table_numeric_share=0.30`.

- [ ] **Step 1: failing tests**

```python
def test_curly_quotes_and_ligatures_are_folded():
    assert ft.normalize("Board’s Report ﬁnancial") == "Board's Report financial"

def test_a_statement_page_is_a_table_and_commentary_is_prose():
    table = "Revenue from operations 12,345.67 11,234.56 45,678.90 Other income 123.45 98.76 456.78 " * 6
    prose = "During the quarter the bank grew advances by 22% while asset quality held steady. " * 6
    assert ft.page_kind(table, "results", None) == "table"
    assert ft.page_kind(prose, "results", None) == "prose"
    assert ft.page_kind(prose, "annual_report", None) == "boilerplate"     # outside the key sections
    assert ft.page_kind(prose, "annual_report", "mdna") == "prose"

def test_annual_report_sections_run_from_their_heading_to_the_next_major_heading():
    pages = ["Notice of the Annual General Meeting", "Management Discussion and Analysis\nIndustry overview",
             "Our margins improved", "Board’s Report\nDear Members", "Dividend", "Report on Corporate Governance",
             "Independent Auditor’s Report\nKey Audit Matters", "Balance Sheet as at 31 March 2026",
             "Note 38 Contingent liabilities and commitments", "Note 39 Related party disclosures"]
    assert ft.ar_sections(pages) == [None, "mdna", "mdna", "board_report", "board_report", None,
                                     "auditor", None, "contingent", "related_party"]

def test_the_fiscal_period_follows_indias_april_to_march_year():
    assert ft.fiscal_period("results", "2026-07-25") == "Q1 FY27"
    assert ft.fiscal_period("earnings_call", "2026-02-10") == "Q3 FY26"
    assert ft.fiscal_period("annual_report", "2026-07-20") == "FY26"
    assert ft.fiscal_period("material_event", "2026-07-25") is None

def test_the_broadcast_stamp_in_the_url_is_when_the_market_saw_it():
    url = "https://nsearchives.nseindia.com/corporate/BDL_24122024171434_BDLIDCessation24Sep2024.pdf"
    assert ft.available_at(url, "2024-12-24") == "2024-12-24T17:14"
    assert ft.available_at("https://x/a.pdf", "2024-12-24") == "2024-12-24T00:00"

def test_a_question_full_of_search_syntax_becomes_a_safe_query():
    assert ft.fts_query('What did "management" say -- about NIM AND CASA* (Q1)?') == \
        '"management" OR "say" OR "nim" OR "casa" OR "q1"'
    assert ft.fts_query("what is the") is None

def test_duplicate_files_are_indexed_once_and_keyword_search_honours_company_type_and_date(db):
    pages = {"s1": [("The bank's NIM expanded to 5.9% on CASA growth.", False)],
             "s2": [("Credit cost guidance for FY27 is 1.5%.", False)]}
    docs = [doc("filing:nse:1", "AUBANK", "earnings_call", "s1", "2026-07-25", "_25072026183000_"),
            doc("filing:nse:2", "AUBANK", "earnings_call", "s1", "2026-07-25", "_25072026183000_"),   # same file again
            doc("filing:nse:3", "AUBANK", "investor_presentation", "s2", "2026-08-02", "_02082026100000_")]
    tally = ft.index_texts(docs, extract=lambda path, doc_type: pages[path.stem])
    assert tally["extracted"] == 2 and tally["duplicates"] == 1
    assert [h["page"] for h in ft.keyword_pages("AUBANK", "NIM and CASA")] == [1]
    assert ft.keyword_pages("AUBANK", "credit cost", doc_types=["earnings_call"]) == []
    assert ft.keyword_pages("AUBANK", "credit cost", as_of="2026-08-01T00:00") == []
    assert len(ft.keyword_pages("AUBANK", "credit cost", as_of="2026-08-02T10:00")) == 1
    assert ft.keyword_pages("OTHERCO", "NIM") == []
    assert ft.index_texts(docs, extract=lambda *a: 1 / 0)["extracted"] == 0               # unchanged: a no-op
```
  (`doc(key, symbol, type, sha, date, stamp)` builds a `documents.Document` whose `local_path` stem is the
  sha and whose `source_url` carries the stamp; `db` points `filings_db` at a temp file.)

- [ ] **Step 2: run, expect ImportError.** **Step 3: implement** (STRICT `filing_texts`, `filing_docs`,
  `filing_pages` + FTS5 external-content `pages_fts` kept in sync by triggers; extraction skips content already
  at `EXTRACTOR_VERSION`; section propagation stops at other major headings; OCR default renders the page with
  pypdfium2 and runs pytesseract). **Step 4: run; full suite.**

### Task 3: Structure-aware passages

**Files:** Create `pennystock/retrieval/filing_passages.py`; Modify `pennystock/config.py`; Test `tests/test_filing_passages.py`

**Interfaces — Produces:** `CHUNKER_VERSION`, `Passage` (text, page_from, page_to, section, speaker), `context_line(...)`,
`recursive_split(text, max_chars, overlap) -> list[str]`, `passages_for(meta: dict, pages: list[dict]) -> list[Passage]`.
Settings: `filings_passage_chars=4000`, `filings_overlap=0.25`.

- [ ] **Step 1: failing tests**

```python
META = {"symbol": "AUBANK", "company": "AU Small Finance Bank", "doc_type": "earnings_call",
        "doc_date": "2026-07-25", "fiscal_period": "Q1 FY27"}

def page(n, text, kind="prose", section=None):
    return {"page": n, "text": text, "kind": kind, "section": section}

def test_every_passage_opens_with_where_it_came_from():
    line = fp.context_line("AU Small Finance Bank", "earnings_call", "2026-07-25", "Q1 FY27", 7, 7, speaker="Q&A - Rahul Jain")
    assert line == "AU Small Finance Bank · earnings call · 2026-07-25 · Q1 FY27 · p.7 · Q&A - Rahul Jain"

def test_a_long_text_splits_on_paragraphs_with_overlap_and_never_beyond_the_cap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 150 for i in range(12))
    parts = fp.recursive_split(text, 2000, 0.25)
    assert all(len(p) <= 2000 for p in parts) and len(parts) > 3
    assert parts[1].split()[0:2] != ["Paragraph", "0"] and any(w in parts[1] for w in parts[0].split()[-20:])

def test_an_earnings_call_keeps_each_question_with_its_answer():
    call = ("Moderator: Ladies and gentlemen, welcome. We will now begin the question-and-answer session. "
            "The first question is from the line of Rahul Jain from Goldman Sachs.\n"
            "Rahul Jain: What is the NIM guidance for FY27?\n"
            "Sanjay Agarwal: We expect NIM of 5.8% to 6.0% as deposit costs ease.\n"
            "Moderator: The next question is from the line of Priya Shah from Kotak.\n"
            "Priya Shah: How is credit cost trending?\nVivek Tripathi: Credit cost should stay near 1.5%.\n") * 3
    ps = fp.passages_for(META, [page(1, call)])
    qa = [p for p in ps if "NIM guidance" in p.text]
    assert qa and "5.8% to 6.0%" in qa[0].text and qa[0].speaker.startswith("Q&A")

def test_a_call_the_speaker_parser_cannot_read_falls_back_to_plain_splits():
    ps = fp.passages_for(META, [page(1, "Our deposits grew strongly this quarter. " * 200)])
    assert ps and all(p.speaker is None for p in ps)

def test_presentations_give_one_passage_per_slide_and_skip_image_slides():
    ps = fp.passages_for({**META, "doc_type": "investor_presentation"},
                         [page(1, "Q1 FY27 highlights: advances up 22% " * 5), page(2, "", "prose"), page(3, "CASA ratio 31% " * 5)])
    assert [(p.page_from, p.page_to) for p in ps] == [(1, 1), (3, 3)]

def test_table_pages_and_annual_report_boilerplate_are_never_embedded():
    res = fp.passages_for({**META, "doc_type": "results"},
                          [page(1, "Press release: profit rose 18% on loan growth " * 10), page(2, "12,345 6,789 " * 50, "table")])
    assert [p.page_from for p in res] == [1]
    ar = fp.passages_for({**META, "doc_type": "annual_report"},
                         [page(1, "CSR spend details " * 30, "boilerplate"), page(2, "Margins improved " * 30, "prose", "mdna")])
    assert [p.section for p in ar] == ["mdna"]
```

- [ ] **Step 2: run, expect ImportError.** **Step 3: implement** (speaker turns via `transcripts.parse`: an analyst
  turn opens a Q&A passage that takes the following management turns; prepared remarks pass through
  `recursive_split`; fewer than 10 parsed turns -> plain splits; passage pages tracked by locating each turn's
  opening words). **Step 4: run; full suite.**

### Task 4: The vector index and the hybrid search

**Files:** Create `pennystock/retrieval/filing_search.py`; Modify `pennystock/config.py`; Test `tests/test_filing_search.py`

**Interfaces — Produces:** `filings_store()`, `index_passages(store=None, reindex=False, symbols=None) -> dict`,
`search_filings(symbol, question, doc_types=None, as_of=None, top_n=None, mode=None, store=None) -> list[Document]`
(`mode` in `keyword | dense | fused | fused_rerank`). Settings: `filings_collection="filings"`, `filings_search_top_n=8`,
`filings_per_doc=2`, `filings_fts_k=50`, `filings_dense_k=50`, `filings_search_mode="fused_rerank"`.

- [ ] **Step 1: failing tests** (a `MemoryStore` like the broker tests' plus `similarity_search` ranking by word
  overlap; `rerank` monkeypatched to keep order)

```python
def test_indexing_twice_writes_each_passage_once_and_resumes_after_a_crash(db, store):
    seed_two_calls(db)                                   # two AUBANK calls through index_texts
    assert fs.index_passages(store=store)["indexed"] > 0
    n = len(store.rows)
    assert fs.index_passages(store=store)["indexed"] == 0 and len(store.rows) == n

def test_search_stays_on_one_company_and_cites_type_date_and_page(db, store):
    seed_two_calls(db); fs.index_passages(store=store)
    hits = asyncio.run(fs.search_filings("AUBANK", "NIM guidance", store=store))
    assert hits and all(h.metadata["symbol"] == "AUBANK" for h in hits)
    m = hits[0].metadata
    assert {"doc_type", "doc_date", "page_from", "page_to", "path"} <= set(m)

def test_an_as_of_date_never_lets_a_later_filing_through(db, store):
    seed_two_calls(db); fs.index_passages(store=store)        # calls broadcast 2026-04-25 and 2026-07-25
    for mode in ("keyword", "dense", "fused", "fused_rerank"):
        hits = asyncio.run(fs.search_filings("AUBANK", "NIM guidance", as_of="2026-05-01T00:00", mode=mode, store=store))
        assert all(h.metadata["available_at"] <= "2026-05-01T00:00" for h in hits), mode

def test_no_document_fills_the_results_and_they_read_newest_first(db, store):
    seed_two_calls(db); fs.index_passages(store=store)
    hits = asyncio.run(fs.search_filings("AUBANK", "NIM", top_n=8, store=store))
    per_doc = Counter(h.metadata["content_sha"] for h in hits)
    assert max(per_doc.values()) <= 2
    assert [h.metadata["available_at"] for h in hits] == sorted((h.metadata["available_at"] for h in hits), reverse=True)

def test_a_crash_mid_indexing_leaves_no_file_marked_done_and_the_rerun_has_no_duplicates(db, store, monkeypatch):
    seed_two_calls(db)
    real, calls = store.add_documents, []
    def flaky(docs, ids):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        real(docs, ids)
    monkeypatch.setattr(store, "add_documents", flaky)
    with pytest.raises(RuntimeError):
        fs.index_passages(store=store)
    monkeypatch.setattr(store, "add_documents", real)
    fs.index_passages(store=store)
    ids = list(store.rows)
    assert len(ids) == len(set(ids)) and fs.index_passages(store=store)["indexed"] == 0

def test_a_table_page_is_found_by_keyword_but_never_embedded(db, store):
    seed_results_with_a_table_page(db); fs.index_passages(store=store)
    assert all(m.get("kind") != "table" for m in (d.metadata for d in store.rows.values()))
    hits = asyncio.run(fs.search_filings("AUBANK", "revenue from operations", mode="keyword", store=store))
    assert hits and hits[0].metadata["kind"] == "table"
```

  (`seed_two_calls(db)` runs `index_texts` with injected extraction for two AUBANK earnings calls broadcast
  2026-04-25 and 2026-07-25, each mentioning NIM guidance; `seed_results_with_a_table_page(db)` adds a results
  filing with one prose page and one statement-table page.)

- [ ] **Step 2: run, expect ImportError.** **Step 3: implement** (Chroma metadata carries `available_ts` as an
  integer YYYYMMDDHHMM so `$lte` filters work; keyword hits map to the passages covering their page, else become
  a page-level document from the page text; RRF -> reranker -> cap per `content_sha` -> newest first;
  `passages_version` is written only after a file's passages are stored). **Step 4: run; full suite.**

### Task 5: The analyst tool, CLIs, freshness

**Files:** Create `pennystock/tools/filings.py`, `scripts/index_filings.py`, `scripts/search_filings.py`; Modify
`pennystock/tools/__init__.py`, `pennystock/agents/nodes.py`, `pennystock/agents/prompts.py`,
`pennystock/evals/assertions.py`, `pennystock/data/maintenance.py`; Test `tests/test_filings_tool.py`,
`tests/test_assertions.py`, `tests/test_maintenance.py`.

**Interfaces — Produces:** tool `search_filings(symbol, question, doc_types="", exchange="NSE") -> str`,
`Dataset("filings index", ...)`.

- [ ] **Step 1: failing tests**

```python
def test_a_company_with_nothing_indexed_is_reported_as_such(monkeypatch):
    monkeypatch.setattr(tool, "_resolve_ticker", lambda s, e: ("XYZ.NS", "Xyz Ltd"))
    monkeypatch.setattr(tool, "indexed_count", lambda s: 0)
    out = asyncio.run(tool.search_filings.ainvoke({"symbol": "XYZ", "question": "capex plans"}))
    assert "NOT INDEXED" in out and "nothing disclosed" not in out.lower()

def test_passages_come_back_attributed_with_the_numbers_rule(monkeypatch):
    monkeypatch.setattr(tool, "_resolve_ticker", lambda s, e: ("AUBANK.NS", "AU Small Finance Bank"))
    monkeypatch.setattr(tool, "indexed_count", lambda s: 12)
    async def fake(*a, **k):
        return [Document(page_content="AU Small Finance Bank · earnings call · 2026-07-25 · Q1 FY27 · p.7\nNIM 5.8-6.0%",
                         metadata={"doc_type": "earnings_call", "doc_date": "2026-07-25", "page_from": 7, "page_to": 7})]
    monkeypatch.setattr(tool, "_search", fake)
    out = asyncio.run(tool.search_filings.ainvoke({"symbol": "AUBANK", "question": "NIM guidance"}))
    assert "2026-07-25" in out and "p.7" in out and "how_to_read" in out

# tests/test_assertions.py
def test_a_disclosure_question_needs_the_filings_search():
    r = check_tool_coverage(trace(query="What did AU Small Finance Bank disclose about the merger?", tool_calls=[]))
    assert not r["passed"] and "search_filings" in r["detail"]
```
  plus: `test_maintenance` lists "filings index"; the registry-parity test covers the new tool.

- [ ] **Step 2: run, expect failures.** **Step 3: implement** (prompt line 16 + FILINGS RULE: quote with type and
  date, never a number from a passage, say "not found in the filings on file"; coverage words "filing",
  "filings", "disclose", "disclosed", "disclosure", "investor presentation", "rating rationale"; CLIs with
  `--status`, `--repair`, `--symbol`, `--reextract`, `--reindex`, `--no-embed` / `--type`, `--as-of`, `--top`).
  **Step 4: run; full suite.**

### Task 6: The filings golden set and the A/B eval

**Files:** Create `pennystock/evals/filings_eval.py`, `scripts/build_filings_eval.py`, `scripts/eval_filings.py`;
Modify `scripts/review_eval_set.py` (`--file`); Test `tests/test_filings_eval.py`.

**Interfaces — Produces:** `page_hit(meta, row) -> bool`, `score(results: dict[str, list[list[dict]]], rows, ks=(5, 10)) -> dict`.

- [ ] **Step 1: failing tests**

```python
ROW = {"content_sha": "s1", "page_from": 7, "page_to": 8}

def test_a_result_counts_when_it_covers_the_answer_page_of_the_same_file():
    assert fe.page_hit({"content_sha": "s1", "page_from": 8, "page_to": 9}, ROW)
    assert not fe.page_hit({"content_sha": "s2", "page_from": 7, "page_to": 7}, ROW)
    assert not fe.page_hit({"content_sha": "s1", "page_from": 10, "page_to": 11}, ROW)

def test_recall_and_mrr_per_configuration():
    hit, miss = {"content_sha": "s1", "page_from": 7, "page_to": 7}, {"content_sha": "s9", "page_from": 1, "page_to": 1}
    out = fe.score({"fused": [[miss, hit], [miss] * 12]}, [ROW, ROW], ks=(1, 5))
    assert out["fused"] == {"recall@1": 0.0, "recall@5": 0.5, "mrr": 0.25, "n": 2}
```

- [ ] **Step 2: run, expect ImportError.** **Step 3: implement** (`build_filings_eval.py` samples passages across
  types and companies and has the local LLM write one question per passage into `eval/filings_golden.jsonl`;
  `eval_filings.py` runs the four modes on verified rows, prints and saves the table under `eval/reports/`).
  **Step 4: run; full suite.**

### Task 7: Real run, defaults, docs

- [ ] `index_filings.py --repair` (expect the 19 truncated files fixed), full extraction + FTS, embedding in the
  background, golden set built and each row reviewed against its passage, `eval_filings.py` -> the best mode
  becomes `filings_search_mode`, spot searches on three companies, one tool call, CLAUDE.md, CI lint, full suite.
