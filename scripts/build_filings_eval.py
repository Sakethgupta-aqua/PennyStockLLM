"""
Build the GOLDEN set for filings search: questions a local LLM writes from real passages of real filings.

For passages spread over document types and companies (filings_eval.sample_passages), the local model
writes one investor question + reference answer per passage. Each row records the passage's file
(content_sha) and pages - the place the search has to find (filings_eval.page_hit). Then review the rows,
mark the good ones verified, and A/B the search modes:

  python scripts/build_filings_eval.py --n 100
  python scripts/review_eval_set.py --file eval/filings_golden.jsonl --show 3
  python scripts/review_eval_set.py --file eval/filings_golden.jsonl --verify 1,2,5
  python scripts/eval_filings.py

The model sees the passage text and its document type only - never the company, date or period line, so
a question cannot win by quoting the date the dense passages carry and the keyword pages do not.

Resumable: re-running keeps the rows already written and adds until the file holds --n usable rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                                                # noqa: E402
from pennystock.data.text_quality import junk_reason                                  # noqa: E402
from pennystock.evals.filings_eval import GEN_PROMPT, sample_passages, usable_question  # noqa: E402

OUT = ROOT / "eval" / "filings_golden.jsonl"
MIN_BODY_CHARS = 600            # shorter passages make thin, guessable questions
CHUNK = 40                      # passages fetched from Chroma at a time
log = logging.getLogger("build_filings_eval")


class QA(BaseModel):
    question: str
    answer: str


def all_metadatas(col, page: int = 5000) -> tuple[list[str], list[dict]]:
    """Every passage's id and metadata, read in pages: one get() over ~90,000 passages exceeds SQLite's
    limit on query variables."""
    ids: list[str] = []
    metas: list[dict] = []
    for offset in range(0, col.count(), page):
        got = col.get(include=["metadatas"], limit=page, offset=offset)
        ids += got["ids"]
        metas += got["metadatas"]
    return ids, metas


def body(text: str) -> str:
    """A passage without its context line (company · type · date · period · pages)."""
    return text.partition("\n")[2] or text


def generate(llm, doc_type: str, text: str) -> tuple[str, str] | None:
    passage = f"[{doc_type.replace('_', ' ')}]\n{text[:3000]}"
    try:
        r = llm.with_structured_output(QA, method="json_schema").invoke(GEN_PROMPT.format(passage=passage))
        q, a = " ".join(r.question.split()), " ".join(r.answer.split())
        return (q, a) if len(q) > 10 and len(a) > 5 else None
    except Exception as e:
        log.warning(f"generation failed: {type(e).__name__}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="usable rows the file should hold")
    ap.add_argument("--model", default="qwen3.5:35b", help="Ollama model that writes the questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    for noisy in ("httpx", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
    done = {r["source_id"] for r in rows}
    if len(rows) >= args.n:
        log.info(f"{OUT.name} already holds {len(rows)} rows (>= --n {args.n}); nothing to do.")
        return 0

    import chromadb
    client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
    col = client.get_collection(settings.filings_collection)
    items = [(pid, meta) for pid, meta in zip(*all_metadatas(col)) if pid not in done]
    if not items:
        log.info("no passages to sample - run scripts/index_filings.py first")
        return 1
    order = sample_passages(items, len(items), seed=args.seed)
    log.info(f"{len(rows)} rows on file; writing up to {args.n - len(rows)} more from {len(items):,} "
             f"passages with {args.model}")

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.model, temperature=0.3, num_ctx=8192, reasoning=False)

    skipped: Counter = Counter()
    need = args.n - len(rows)
    with OUT.open("a") as f:                               # append + flush per line = crash-safe
        for start in range(0, len(order), CHUNK):
            if need <= 0:
                break
            chunk = order[start:start + CHUNK]
            fetched = col.get(ids=chunk, include=["documents", "metadatas"])
            by_id = {pid: (doc, meta) for pid, doc, meta in zip(fetched["ids"], fetched["documents"],
                                                                fetched["metadatas"])}
            for pid in chunk:
                if need <= 0:
                    break
                text, meta = by_id[pid]
                passage = body(text)
                if len(passage) < MIN_BODY_CHARS:
                    skipped["short passage"] += 1
                    continue
                if junk_reason(passage):
                    skipped["not prose"] += 1
                    continue
                qa = generate(llm, meta["doc_type"], passage)
                if not qa:
                    skipped["generation failed"] += 1
                    continue
                ok, why = usable_question(qa[0])
                if not ok:
                    skipped[why.split(":")[0]] += 1
                    log.info(f"  REJECTED ({why}): {qa[0][:80]}")
                    continue
                # the fields a reviewer reads first, the long passage last (scripts/review_eval_set.py)
                f.write(json.dumps({
                    "verified": False, "symbol": meta["symbol"], "doc_type": meta["doc_type"],
                    "question": qa[0], "reference_answer": qa[1], "content_sha": meta["content_sha"],
                    "page_from": meta["page_from"], "page_to": meta["page_to"], "doc_date": meta["doc_date"],
                    "source_id": pid, "source_text": text}, ensure_ascii=False) + "\n")
                f.flush()
                need -= 1
                log.info(f"  [{args.n - need}/{args.n}] {meta['symbol']:<11} {meta['doc_type']:<21} "
                         f"{qa[0][:70]}")
    total = sum(1 for l in OUT.read_text().splitlines() if l.strip())
    log.info(f"Done: {OUT} holds {total} rows; skipped {sum(skipped.values())} {dict(skipped)}")
    log.info(f"Next: python scripts/review_eval_set.py --file {OUT.relative_to(ROOT)} --pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
