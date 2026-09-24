"""
Evaluate the RETRIEVAL pipeline against the golden set.

For each question, run the REAL advanced_retrieve() and check whether the chunk the question was
written from is retrieved, and at what rank. Reports hit@k, MRR, and a per-book breakdown, and saves
a timestamped report stamped with the current retrieval settings so you can compare runs before/after
any change (reranker on/off, more sub-queries, etc.).

  python scripts/eval_rag.py --verified-only   # the baseline you should quote
  python scripts/eval_rag.py --label "reranker-off"   # tag a run so you can find it later
  python scripts/eval_rag.py --compare         # diff the two most recent runs

The point of this file is A/B discipline: change ONE thing, re-run, and let the number decide.

Needs Ollama running (for query decomposition) and downloads/loads the reranker on first use.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("eval_rag")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5)


def _norm(t: str) -> str:
    return " ".join(t.split()).strip()


def compare_runs() -> int:
    """Diff the two most recent result files - the whole point of keeping them."""
    files = sorted(RESULTS.glob("eval_*.json"))
    if len(files) < 2:
        print(f"Need at least 2 result files to compare; found {len(files)} in {RESULTS}.")
        return 0
    old, new = (json.loads(f.read_text()) for f in files[-2:])
    print(f"\n  BEFORE  {files[-2].name}  {old.get('label') or '(no label)'}  n={old['n_questions']}")
    print(f"  AFTER   {files[-1].name}  {new.get('label') or '(no label)'}  n={new['n_questions']}\n")
    print(f"  {'METRIC':<12}{'BEFORE':>9}{'AFTER':>9}{'CHANGE':>10}")
    for k in new["metrics"]:
        a, b = old["metrics"].get(k), new["metrics"][k]
        if a is None:
            continue
        d = b - a
        arrow = "better" if d > 0 else ("worse" if d < 0 else "same")
        print(f"  {k:<12}{a:>9.3f}{b:>9.3f}{d:>+10.3f}  {arrow}")
    changed = {k: (old["retrieval_config"].get(k), v)
               for k, v in new["retrieval_config"].items() if old["retrieval_config"].get(k) != v}
    print(f"\n  config changed: {changed or 'nothing - the difference is run-to-run noise'}")
    return 0


def rank_of_source(source_text: str, retrieved) -> int | None:
    """Position (1-based) of the source chunk in the reranked list, or None if it wasn't retrieved.
    We sort by rerank_score to recover the true relevance order (the pipeline reorders for the LLM)."""
    ranked = sorted(retrieved, key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
    src = _norm(source_text)
    for i, d in enumerate(ranked):
        if _norm(d.page_content) == src:
            return i + 1
    key = src[:200]                                        # fallback: strong prefix overlap
    for i, d in enumerate(ranked):
        if key and key in _norm(d.page_content):
            return i + 1
    return None


async def main():
    ap = argparse.ArgumentParser(description="Score the retrieval pipeline against the golden set.")
    ap.add_argument("--verified-only", action="store_true", help="only score rows you marked verified")
    ap.add_argument("--label", default="", help="tag this run (e.g. 'reranker-off') for later comparison")
    ap.add_argument("--compare", action="store_true", help="diff the two most recent result files and exit")
    ap.add_argument("--limit", type=int, help="score only the first N questions (quick smoke test)")
    args = ap.parse_args()

    if args.compare:
        return compare_runs()

    from pennystock.config import settings
    from pennystock.retrieval.store import advanced_retrieve

    if not GOLDEN.exists():
        log.error(f"No golden set at {GOLDEN}. Run scripts/build_eval_set.py first.")
        return
    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    if args.verified_only:
        rows = [r for r in rows if r.get("verified")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        log.error("No eval rows found. Run build_eval_set.py first (and verify some rows).")
        return
    log.info(f"Evaluating {len(rows)} questions against advanced_retrieve ...")

    ranks: list[int | None] = []
    per_book = defaultdict(lambda: {"n": 0, "hits": 0})
    for i, r in enumerate(rows, 1):
        docs = await advanced_retrieve(r["question"])
        rank = rank_of_source(r["source_text"], docs)
        ranks.append(rank)
        b = per_book[r["source_book"]]
        b["n"] += 1
        b["hits"] += 1 if rank else 0
        log.info(f"  [{i}/{len(rows)}] rank={rank}  {r['question'][:64]}")

    n = len(ranks)
    hit_at = {k: round(sum(1 for rk in ranks if rk and rk <= k) / n, 3) for k in KS}
    hit_any = round(sum(1 for rk in ranks if rk) / n, 3)
    mrr = round(sum(1 / rk for rk in ranks if rk) / n, 3)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "n_questions": n,
        "verified_only": args.verified_only,
        "metrics": {**{f"hit@{k}": hit_at[k] for k in KS}, "hit_any": hit_any, "mrr": mrr},
        "per_book": {b: {"n": v["n"], "hit_rate": round(v["hits"] / v["n"], 3)} for b, v in per_book.items()},
        "retrieval_config": {                              # so this run is reproducible & comparable
            "research_subqueries": settings.research_subqueries, "candidate_k": settings.candidate_k,
            "rerank_pool": settings.rerank_pool, "rerank_top_n": settings.rerank_top_n,
            "reranker_model": settings.reranker_model, "dense_k": settings.dense_k,
            "bm25_k": settings.bm25_k, "embed_model": settings.embed_model, "collection": settings.collection,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 50)
    print(f"  RAG EVAL  ({n} questions)")
    print("=" * 50)
    for k in KS:
        print(f"  hit@{k:<2}: {hit_at[k]:.3f}")
    print(f"  hit_any (in top-{settings.rerank_top_n}): {hit_any:.3f}")
    print(f"  MRR    : {mrr:.3f}")
    print("  per book:")
    for b, v in report["per_book"].items():
        print(f"     {v['hit_rate']:.2f}  ({v['n']:>2})  {b[:44]}")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
