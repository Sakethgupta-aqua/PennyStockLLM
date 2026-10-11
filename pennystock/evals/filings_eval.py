"""Scoring filings search against a golden set: page recall@k and MRR, per retrieval mode.

A golden row is a question written from one passage of one filing (content_sha, page_from-page_to). A
result counts when it comes from the same file and covers any page of that range - page-level grading,
because the retrieval research found the right DOCUMENT is usually found (0.93) while the answer PAGE is
missed half the time (0.46): page recall is the number worth moving.
"""
from __future__ import annotations

import random
import re
from collections import deque

GEN_PROMPT = (
    "You are writing ONE test question for a search engine over a single company's stock-exchange "
    "filings (earnings calls, investor presentations, rating rationales, results releases, annual "
    "reports). The search already knows which company - the question does not need its name.\n\n"
    "HARD RULES:\n"
    "  * Ask what an investor would ask: what management said, guided, explained or disclosed.\n"
    "  * NEVER refer to the source: no 'the passage', 'the text', 'this document', 'the above'.\n"
    "  * Do not ask for a bare number read off a table; ask about a statement, reason, plan or risk.\n"
    "  * It must be answerable from this passage alone, and specific enough to find it again.\n\n"
    "GOOD: 'What NIM range did management guide for FY27, and why?'\n"
    "BAD:  'What is mentioned in the passage about margins?'\n\n"
    "Produce JSON with 'question' and 'answer' (1-2 sentences grounded ONLY in the passage).\n\n"
    "PASSAGE:\n{passage}"
)
_VAGUE = re.compile(r"\b(the (passage|text|excerpt|document|above|table|exhibit)|this (document|passage|text)"
                    r"|according to the|as (mentioned|described|stated|shown))\b", re.I)


def usable_question(question: str) -> tuple[bool, str]:
    q = " ".join(str(question).split())
    if len(q.split()) < 6:
        return False, "too short"
    if not q.endswith("?"):
        return False, "not a question"
    m = _VAGUE.search(q)
    return (False, f"points at the source: '{m.group(0)}'") if m else (True, "")


def sample_passages(items: list[tuple[str, dict]], n: int, seed: int = 42) -> list[str]:
    """Ids of n passages spread evenly over document types and, within a type, over companies - so the
    eval is not just the longest annual reports of the biggest filers. items = (id, metadata) pairs."""
    rng = random.Random(seed)
    by_type: dict[str, dict[str, list[str]]] = {}
    for pid, meta in items:
        by_type.setdefault(meta.get("doc_type", "?"), {}).setdefault(meta.get("symbol", "?"), []).append(pid)
    queues: dict[str, deque[str]] = {}
    for doc_type, by_company in sorted(by_type.items()):
        companies = sorted(by_company)
        rng.shuffle(companies)
        for c in companies:
            rng.shuffle(by_company[c])
        depth = max(len(v) for v in by_company.values())
        queues[doc_type] = deque(by_company[c][i] for i in range(depth) for c in companies
                                 if i < len(by_company[c]))
    picked: list[str] = []
    while len(picked) < n and any(queues.values()):
        for doc_type in sorted(queues):
            if queues[doc_type] and len(picked) < n:
                picked.append(queues[doc_type].popleft())
    return picked


def page_hit(meta: dict, row: dict) -> bool:
    """Same file, and the result's pages overlap the answer's pages."""
    return (meta.get("content_sha") == row["content_sha"]
            and int(meta.get("page_from", 0)) <= int(row["page_to"])
            and int(meta.get("page_to", 0)) >= int(row["page_from"]))


def score(results: dict[str, list[list[dict]]], rows: list[dict], ks: tuple[int, ...] = (5, 10)) -> dict:
    """results[mode][i] = the ranked result metadata for rows[i]. Recall@k = share of questions with a
    hit in the top k; MRR = mean of 1/rank of the first hit (0 when none)."""
    out = {}
    for mode, ranked in results.items():
        first = []
        for metas, row in zip(ranked, rows):
            rank = next((i + 1 for i, m in enumerate(metas) if page_hit(m, row)), None)
            first.append(rank)
        n = len(first)
        out[mode] = {**{f"recall@{k}": round(sum(1 for r in first if r and r <= k) / n, 3) if n else 0.0
                        for k in ks},
                     "mrr": round(sum(1 / r for r in first if r) / n, 3) if n else 0.0, "n": n}
    return out
