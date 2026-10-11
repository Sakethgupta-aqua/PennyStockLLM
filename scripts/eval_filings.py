"""
A/B the filings search: page recall@k and MRR of each retrieval mode on the verified golden set.

  python scripts/eval_filings.py                                  # verified rows, all four modes
  python scripts/eval_filings.py --modes keyword,fused_rerank     # just these
  python scripts/eval_filings.py --include-unverified --limit 10  # a quick smoke test

A hit is a result from the answer's own file that covers one of its pages (pennystock/evals/filings_eval.py).
Results are read in ranked order, not the newest-first order the analyst sees. The report - with every
question's first-hit rank per mode, for reading the misses - is saved under eval/results/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.evals.filings_eval import page_hit, score         # noqa: E402
from pennystock.retrieval.filing_search import search_filings     # noqa: E402

GOLDEN = ROOT / "eval" / "filings_golden.jsonl"
RESULTS = ROOT / "eval" / "results"
MODES = ("keyword", "dense", "fused", "fused_rerank")
KS = (1, 5, 10)


async def run(rows: list[dict], modes: list[str], top: int) -> tuple[dict, dict]:
    results: dict[str, list[list[dict]]] = {m: [] for m in modes}
    seconds: dict[str, float] = defaultdict(float)
    for i, row in enumerate(rows, 1):
        for m in modes:
            t = time.perf_counter()
            docs = await search_filings(row["symbol"], row["question"], mode=m, top_n=top, newest_first=False)
            seconds[m] += time.perf_counter() - t
            results[m].append([d.metadata for d in docs])
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} questions", flush=True)
    return results, {m: round(s / max(len(rows), 1), 3) for m, s in seconds.items()}


def table(scored: dict, latency: dict | None = None) -> str:
    cols = [f"recall@{k}" for k in KS] + ["mrr"]
    head = f"  {'mode':<14}" + "".join(f"{c:>11}" for c in cols) + f"{'n':>6}"
    lines = [head + ("   s/query" if latency else "")]
    for mode, s in scored.items():
        lines.append(f"  {mode:<14}" + "".join(f"{s[c]:>11.3f}" for c in cols) + f"{s['n']:>6}"
                     + (f"{latency[mode]:>10.2f}" if latency else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--include-unverified", action="store_true", help="score rows not yet reviewed too")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--top", type=int, default=10, help="results read per question")
    ap.add_argument("--label", default="", help="tag this run in the saved report")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := set(modes) - set(MODES):
        print(f"unknown mode(s) {sorted(unknown)}; choose from {MODES}")
        return 1
    if not GOLDEN.exists():
        print(f"No filings golden set at {GOLDEN}. Run scripts/build_filings_eval.py first.")
        return 1
    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    if not args.include_unverified:
        rows = [r for r in rows if r.get("verified")]
    rows = rows[:args.limit] if args.limit else rows
    if not rows:
        print("No rows to score - verify some (scripts/review_eval_set.py --file eval/filings_golden.jsonl) "
              "or pass --include-unverified.")
        return 1

    print(f"Scoring {len(rows)} questions x {len(modes)} modes (top {args.top}) ...")
    results, latency = asyncio.run(run(rows, modes, args.top))
    overall = score(results, rows, ks=KS)
    print("\nFILINGS SEARCH - page recall@k and MRR\n" + table(overall, latency))
    by_type: dict[str, dict] = {}
    for doc_type in sorted({r["doc_type"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["doc_type"] == doc_type]
        sub = {m: [results[m][i] for i in idx] for m in modes}
        by_type[doc_type] = score(sub, [rows[i] for i in idx], ks=KS)
        print(f"\n  {doc_type}\n" + table(by_type[doc_type]))

    ranks = [{"question": r["question"], "symbol": r["symbol"], "doc_type": r["doc_type"],
              "answer_pages": f"{r['page_from']}-{r['page_to']}", "source_id": r.get("source_id"),
              "first_hit": {m: next((k for k, meta in enumerate(results[m][i], 1) if page_hit(meta, r)), None)
                            for m in modes}}
             for i, r in enumerate(rows)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"filings_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"label": args.label, "verified_only": not args.include_unverified,
                               "top": args.top, "n": len(rows), "seconds_per_query": latency,
                               "overall": overall, "by_doc_type": by_type, "questions": ranks}, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
