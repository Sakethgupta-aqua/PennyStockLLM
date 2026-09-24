"""Layer-0 assertions: deterministic, binary pass/fail checks over a saved trace.

No LLM is involved - these are code assertions, so they are free, instant, and never
disagree with themselves. Each check encodes a REAL bug this system has shipped, so the
suite is a permanent regression net.

Every check returns:
    {"name": str, "passed": bool, "applicable": bool, "detail": str}
`applicable=False` means the check did not apply to this run (e.g. no data_warning was
produced), so it is excluded from failure-rate statistics rather than counted as a free pass.
"""
from __future__ import annotations

import re

from pennystock.config import settings

# Mirrors nodes.STOCK_TOOLS - duplicated deliberately so evaluation never imports the runtime
# graph (importing nodes would construct model clients as a side effect).
_STOCK_TOOLS = {"get_indian_stock_data", "get_stock_fundamentals", "get_price_history",
                "get_financials", "get_corporate_actions", "get_shareholding",
                "get_forensics", "get_valuation_context", "get_corporate_events",
                "get_management_commentary", "get_annual_report_notes"}
# screen_stocks is universe-level, so it is known but NOT symbol-scoped
_KNOWN_TOOLS = _STOCK_TOOLS | {"penny_stock_researcher", "screen_stocks"}
# these return no 'symbol': 'X.NS' field, so they cannot be checked for entity drift
_NO_SYMBOL_TOOLS = {"get_shareholding", "get_forensics", "get_valuation_context",
                    "get_corporate_events", "get_management_commentary",
                    "get_annual_report_notes"}

_SYMBOL_RE = re.compile(r"'symbol':\s*'([A-Z0-9&\-]+)\.(?:NS|BO)'")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{40,})[\"“”]|^>\s?(.+)$", re.MULTILINE)
_WARN_WORDS = ("warning", "caution", "unreliable", "corporate action", "distort",
               "demerger", "inconsist", "treat with care")


def _ok(name: str, passed: bool, detail: str = "", applicable: bool = True) -> dict:
    return {"name": name, "passed": bool(passed), "applicable": applicable, "detail": detail}


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split())


def _tool_blob(trace: dict) -> str:
    return "\n".join(c.get("result", "") or "" for c in trace.get("tool_calls", []))


# ------------------------------------------------------------------ checks
def check_entity_consistency(trace: dict) -> dict:
    """Every market-data tool must have fetched the SAME company (the TMCV/TMPV bug)."""
    syms = set()
    for c in trace.get("tool_calls", []):
        if c.get("name") in _STOCK_TOOLS and c.get("name") not in _NO_SYMBOL_TOOLS:
            syms.update(_SYMBOL_RE.findall(c.get("result", "") or ""))
    if not syms:
        return _ok("entity_consistency", True, "no market-data results", applicable=False)
    pinned = (trace.get("resolved_symbol") or "").upper()
    if len(syms) > 1:
        return _ok("entity_consistency", False, f"tools fetched DIFFERENT companies: {sorted(syms)}")
    got = syms.pop()
    if pinned and got != pinned:
        return _ok("entity_consistency", False, f"fetched {got} but entity was pinned to {pinned}")
    return _ok("entity_consistency", True, f"all tools used {got}")


def check_no_unknown_tools(trace: dict) -> dict:
    """The model must not invent tool names."""
    names = {c.get("name") for c in trace.get("tool_calls", [])} - {"?", None}
    bad = sorted(n for n in names if n not in _KNOWN_TOOLS)
    return _ok("no_unknown_tools", not bad,
               f"hallucinated tools: {bad}" if bad else f"{len(names)} valid tool name(s)")


def check_task_completion(trace: dict) -> dict:
    """The run must actually produce a report (catches the empty/truncated-output bug)."""
    n = len(trace.get("report", "") or "")
    if n < settings.min_report_chars:
        return _ok("task_completion", False, f"report only {n} chars (min {settings.min_report_chars})")
    return _ok("task_completion", True, f"report {n} chars")


def _windows(words: list[str], size: int) -> list[str]:
    return [" ".join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))]


def check_citation_provenance(trace: dict) -> dict:
    """Long quoted spans must appear in a retrieved chunk (the fabricated-Graham-quote bug)."""
    chunks = trace.get("retrieved_chunks", [])
    if not chunks:
        return _ok("citation_provenance", True, "nothing retrieved", applicable=False)
    corpus = _norm(" ".join(c.get("text", "") for c in chunks))
    quotes = []
    for m in _QUOTE_RE.finditer(trace.get("report", "") or ""):
        q = (m.group(1) or m.group(2) or "").strip()
        if len(q.split()) >= settings.min_quote_words:
            quotes.append(q)
    if not quotes:
        return _ok("citation_provenance", True, "no long quotes to verify", applicable=False)
    unsupported = [q[:70] for q in quotes
                   if not any(w in corpus for w in _windows(_norm(q).split(), settings.min_quote_words))]
    return _ok("citation_provenance", not unsupported,
               f"{len(unsupported)}/{len(quotes)} quotes NOT in retrieved text: {unsupported[:2]}"
               if unsupported else f"all {len(quotes)} quote(s) traced to retrieved text")


def _report_numbers(text: str, min_decimals: int = 2) -> list[tuple[float, int, str]]:
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        if "." not in raw:
            continue
        dec = len(raw.split(".")[-1])
        if dec < min_decimals:
            continue
        try:
            out.append((float(raw.replace(",", "")), dec, raw))
        except ValueError:
            pass
    return out


def _tool_numbers(text: str) -> set[float]:
    vals = set()
    for m in _NUM_RE.finditer(text):
        try:
            vals.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return vals


def check_numeric_provenance(trace: dict) -> dict:
    """Precise figures in the report must round-match a number the tools returned.

    Only numbers with >=2 decimals are checked - those are copied from data, whereas computed
    values (percent changes, sums) are usually rounder. Matching rounds the tool value to the
    report's own precision, so '54.35' correctly matches a tool's 54.34993.
    """
    blob = _tool_blob(trace)
    if not blob.strip():
        return _ok("numeric_provenance", True, "no tool output", applicable=False)
    tool_vals = _tool_numbers(blob)
    nums = _report_numbers(trace.get("report", "") or "")
    if not nums:
        return _ok("numeric_provenance", True, "no precise figures to verify", applicable=False)
    missing = [raw for val, dec, raw in nums if not any(round(t, dec) == val for t in tool_vals)]
    return _ok("numeric_provenance", not missing,
               f"{len(missing)}/{len(nums)} figures not in tool data: {missing[:5]}"
               if missing else f"all {len(nums)} precise figure(s) traced to tool data")


def check_source_diversity(trace: dict) -> dict:
    """Retrieval must not collapse onto one book (the vanished-Coffee-Can bug)."""
    chunks = trace.get("retrieved_chunks", [])
    if not chunks:
        return _ok("source_diversity", True, "researcher not used", applicable=False)
    srcs = {c.get("source") for c in chunks}
    return _ok("source_diversity", len(srcs) >= settings.min_distinct_sources,
               f"{len(srcs)} distinct book(s): {sorted(str(s)[:22] for s in srcs)}")


_TOOL_RULES = [
    (("five year", "5 year", "revenue", "earnings", "profit", "cagr", "track record", "debt level"),
     "get_financials"),
    (("200-day", "200 day", "rsi", "support", "resistance", "moving average", "momentum", "the tape", "chart"),
     "get_price_history"),
    (("p/e", "book value", "valuation", "margin of safety", "price to book", "leveraged", "debt-to-equity"),
     "get_stock_fundamentals"),
    (("dividend", "split", "bonus", "corporate action", "rewarded shareholders", "capital discipline"),
     "get_corporate_actions"),
    (("promoter", "pledge", "pledging", "shareholding", "governance", "holding pattern",
      "fii", "dii", "institutional", "stake", "skin in the game"), "get_shareholding"),
    (("forensic", "manipulation", "accounting", "trust the numbers", "cash conversion",
      "bankruptcy", "distress", "solvency", "working capital", "earnings quality"), "get_forensics"),
    (("peer", "peers", "sector median", "industry median", "versus its history", "expensive",
      "cheap", "overvalued", "undervalued", "re-rating", "rerating"), "get_valuation_context"),
    (("auditor", "announcement", "insider", "credit rating", "downgrade", "litigation",
      "news flow", "regulatory", "default"), "get_corporate_events"),
    (("management", "guidance", "commentary", "concall", "earnings call", "conference call",
      "what management said", "tone", "credibility", "promised"), "get_management_commentary"),
    (("contingent liabilit", "related party", "related-party", "off balance sheet",
      "off-balance-sheet", "annual report", "guarantees given", "group company",
      "disputed tax", "promoter group dealing"), "get_annual_report_notes"),
    (("screen", "shortlist", "find me stocks", "which stocks", "candidates", "universe",
      "best stocks in"), "screen_stocks"),
    (("graham", "philosophy", "principle", "behavioral", "behavioural", "trap", "wisdom", "framework",
      "psychology", "classic error", "old-timer"), "penny_stock_researcher"),
    (("quote", "current price", "pull its", "pull me"), "get_indian_stock_data"),
]


def check_tool_coverage(trace: dict) -> dict:
    """If the question clearly needs a tool, that tool must have been called."""
    q = (trace.get("query") or "").lower()
    used = {c.get("name") for c in trace.get("tool_calls", [])}
    expected, missing = 0, []
    # word-boundary match: plain `in` made "stake" fire on "mistakes" and "fii" on any word
    def mentions(kw: str) -> bool:
        return re.search(rf"\b{re.escape(kw)}\b", q) is not None
    for keywords, tool in _TOOL_RULES:
        if any(mentions(k) for k in keywords):
            expected += 1
            if tool not in used:
                missing.append(tool)
    if not expected:
        return _ok("tool_coverage", True, "no tool-specific intent detected", applicable=False)
    return _ok("tool_coverage", not missing,
               f"expected but never called: {missing}" if missing
               else f"all {expected} expected tool(s) called")


def check_data_warning_surfaced(trace: dict) -> dict:
    """If a tool flagged bad data, the report must say so (the DATA-QUALITY RULE)."""
    flagged = [c.get("name") for c in trace.get("tool_calls", []) if "data_warning" in (c.get("result") or "")]
    if not flagged:
        return _ok("data_warning_surfaced", True, "no data warnings raised", applicable=False)
    report = (trace.get("report") or "").lower()
    hit = any(w in report for w in _WARN_WORDS)
    return _ok("data_warning_surfaced", hit,
               f"tools flagged {flagged}; report {'mentions' if hit else 'IGNORES'} it")


def check_pledge_disclosed(trace: dict) -> dict:
    """If ownership data says the promoter pledge is UNKNOWN, the report must SAY so.

    Silence reads as "no pledge" to a reader, which is exactly the wrong inference for an Indian
    small-cap. Enforces the PLEDGE RULE in ANALYST_PROMPT.
    """
    sh = [c for c in trace.get("tool_calls", []) if c.get("name") == "get_shareholding"]
    if not sh:
        return _ok("pledge_disclosed", True, "shareholding not fetched", applicable=False)
    unknown = any("'pledge_pct': None" in (c.get("result") or "") for c in sh)
    if not unknown:
        return _ok("pledge_disclosed", True, "pledge value was available", applicable=False)
    report = (trace.get("report") or "").lower()
    said = ("pledge" in report and
            any(w in report for w in ("unknown", "unverified", "not available", "could not",
                                      "unavailable", "not disclosed", "no data")))
    return _ok("pledge_disclosed", said,
               "report states the pledge is unverified" if said
               else "pledge is UNKNOWN but the report never says so - reads as 'no pledge'")


ALL_CHECKS = [check_entity_consistency, check_no_unknown_tools, check_task_completion,
              check_citation_provenance, check_numeric_provenance, check_source_diversity,
              check_tool_coverage, check_data_warning_surfaced, check_pledge_disclosed]


def run_all(trace: dict) -> list[dict]:
    """Run every check; a broken check reports as a failure instead of killing the suite."""
    results = []
    for fn in ALL_CHECKS:
        try:
            results.append(fn(trace))
        except Exception as e:
            results.append(_ok(fn.__name__.replace("check_", ""), False, f"check errored: {e}"))
    return results
