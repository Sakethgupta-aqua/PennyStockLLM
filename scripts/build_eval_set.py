"""
Build a GOLDEN RAG evaluation set from the book corpus.

For each of a stratified sample of real chunks in the vector DB, a local LLM writes a question +
reference answer the chunk answers. Output is JSONL you then HAND-REVIEW (keep the good rows, set
"verified": true). This fixed set is the yardstick scripts/eval_rag.py measures the retriever against.

  python scripts/build_eval_set.py --n 60 --model qwen3.5:35b
  # then open eval/golden_set.jsonl and set "verified": true on the good rows

Idempotent + resumable: re-running skips chunks already turned into questions.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.data.text_quality import junk_reason  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("build_eval")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "golden_set.jsonl"
DB_DIR = ROOT / "penny_stock_vector_db"
COLLECTION = "knowledge_base"

GEN_PROMPT = (
    "You are writing ONE retrieval-test question from an investing-book passage.\n\n"
    "The question will later be shown to a search engine ON ITS OWN, with NO passage attached. So "
    "it MUST make complete sense in isolation and must contain enough distinctive subject matter "
    "for the right passage to be findable.\n\n"
    "HARD RULES - a question breaking any of these is useless:\n"
    "  * NEVER refer to the source: no 'the passage', 'the text', 'the author', 'the book', "
    "'the table', 'the exhibit', 'mentioned above'.\n"
    "  * NEVER use a bare pointer with no antecedent: 'the period', 'the company', 'the portfolio', "
    "'this strategy'. Name the actual concept or company instead.\n"
    "  * Ask about a CONCEPT, principle, definition, rule of thumb or framework - not about a "
    "specific figure in a table.\n"
    "  * It must be answerable from this passage alone.\n\n"
    "GOOD:  'What minimum return on capital employed does the coffee-can approach require?'\n"
    "BAD:   'What was the value of the portfolio at the start of the period?'\n\n"
    "Produce JSON with:\n"
    "  'question': the self-contained question\n"
    "  'answer': a concise 1-2 sentence answer grounded ONLY in the passage\n\n"
    "PASSAGE:\n{passage}"
)

# Auto-reject obviously unusable questions so hand-verification is spent on real candidates.
VAGUE = re.compile(
    # references to the source document itself
    r"\b(the (passage|text|author|book|chapter|table|exhibit|figure|article|excerpt|above|study)"
    r"|according to the|as (mentioned|described|stated|shown|discussed)"
    # narrative pointers - caught after review: "In THE INVESTMENT STORY, what did THE PARTNERS..."
    r"|the (story|investment story|example|case|scenario|illustration|discussion|anecdote)"
    r"|the (partners|narrator|speaker|investor mentioned)"
    # dangling determiners with no antecedent
    r"|this (strategy|approach|portfolio|company|period|table|example|case)"
    r"|the (period|portfolio|company|firm|author's))\b", re.I)


def is_self_contained(question: str) -> tuple[bool, str]:
    """Reject questions that cannot stand alone in front of a search engine."""
    q = " ".join(str(question).split())
    if len(q.split()) < 6:
        return False, "too short"
    if not q.rstrip().endswith("?"):
        return False, "not a question"
    m = VAGUE.search(q)
    if m:
        return False, f"vague reference: '{m.group(0)}'"
    return True, ""


class QA(BaseModel):
    question: str
    answer: str


def load_chunks():
    import chromadb
    col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    got = col.get(include=["documents", "metadatas"])
    return list(zip(got["ids"], got["documents"], got["metadatas"]))


def stratified_sample(chunks, n: int, seed: int):
    """Sample ~evenly across books so every book is represented, not just the biggest one."""
    random.seed(seed)
    by_book: dict[str, list] = {}
    for cid, text, meta in chunks:
        if junk_reason(text) is None:
            by_book.setdefault((meta or {}).get("source", "?"), []).append((cid, text, meta))
    books = list(by_book)
    if not books:
        return []
    per = max(1, n // len(books))
    picked = []
    for b in books:
        random.shuffle(by_book[b])
        picked += by_book[b][:per]
    random.shuffle(picked)
    return picked[:n]


def generate_qa(llm, text: str):
    try:
        r = llm.with_structured_output(QA, method="json_schema").invoke(GEN_PROMPT.format(passage=text[:2000]))
        q, a = r.question.strip(), r.answer.strip()
        if len(q) > 10 and len(a) > 5:
            return q, a
    except Exception as e:
        log.warning(f"generation failed: {e}")
    return None


def main():
    ap = argparse.ArgumentParser(description="Generate a golden RAG eval set from the book corpus.")
    ap.add_argument("--n", type=int, default=60, help="how many Q&A to generate")
    ap.add_argument("--model", default="qwen3.5:35b", help="Ollama model (use an 8b for a faster first pass)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=args.model, temperature=0.3, num_ctx=8192)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():                                       # resume: don't re-question the same chunk
        done = {json.loads(l)["source_id"] for l in OUT.read_text().splitlines() if l.strip()}
        log.info(f"{len(done)} questions already exist; resuming.")

    sample = [c for c in stratified_sample(load_chunks(), args.n, args.seed) if c[0] not in done]
    log.info(f"Generating {len(sample)} Q&A pairs with {args.model} ...")

    written = 0
    rejected: dict[str, int] = {}
    with OUT.open("a") as f:                               # append + flush per line = crash-safe
        for i, (cid, text, meta) in enumerate(sample, 1):
            qa = generate_qa(llm, text)
            if not qa:
                rejected["generation failed"] = rejected.get("generation failed", 0) + 1
                continue
            q, a = qa
            ok, why = is_self_contained(q)
            if not ok:
                rejected[why.split(":")[0]] = rejected.get(why.split(":")[0], 0) + 1
                log.info(f"  [{i}/{len(sample)}] REJECTED ({why}): {q[:70]}")
                continue
            # key order matters for human review: the fields you read and edit come FIRST on the
            # line, the long source_text last (see scripts/review_eval_set.py)
            f.write(json.dumps({
                "verified": False,
                "source_book": (meta or {}).get("source", "?"),
                "question": q,
                "reference_answer": a,
                "source_id": cid,
                "source_text": text,
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            log.info(f"  [{i}/{len(sample)}] {(meta or {}).get('source','?')[:28]} :: {q[:70]}")
    log.info(f"Done. Wrote {written} usable rows to {OUT}; rejected {sum(rejected.values())} "
             f"({rejected}).")
    log.info("Next: open the file, keep the good ones, set \"verified\": true, then run "
             "scripts/eval_rag.py --verified-only")


if __name__ == "__main__":
    main()
