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
                "get_management_commentary", "get_annual_report_notes",
                "get_broker_consensus", "search_broker_research", "get_trendlyne_view",
                "search_filings"}
# screen_stocks is universe-level, so it is known but NOT symbol-scoped
_KNOWN_TOOLS = _STOCK_TOOLS | {"penny_stock_researcher", "screen_stocks"}
# these return no 'symbol': 'X.NS' field, so they cannot be checked for entity drift
_NO_SYMBOL_TOOLS = {"get_shareholding", "get_forensics", "get_valuation_context",
                    "get_corporate_events", "get_management_commentary",
                    "get_annual_report_notes", "get_broker_consensus", "search_broker_research",
                    "get_trendlyne_view", "search_filings"}

_SYMBOL_RE = re.compile(r"'symbol':\s*'([A-Z0-9&\-]+)\.(?:NS|BO)'")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CURLY_QUOTE_RE = re.compile(r"“([^”\n]+)”")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.+)$", re.MULTILINE)
_WARN_WORDS = ("warning", "caution", "unreliable", "corporate action", "distort",
               "demerger", "inconsist", "treat with care")


def _ok(name: str, passed: bool, detail: str = "", applicable: bool = True) -> dict:
    return {"name": name, "passed": bool(passed), "applicable": applicable, "detail": detail}


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split())


def _tool_blob(trace: dict) -> str:
    """Everything the analyst was given: tool results and the evidence pack's fact sheet."""
    sheet = (trace.get("evidence") or {}).get("fact_sheet") or ""
    return "\n".join([*(c.get("result", "") or "" for c in trace.get("tool_calls", [])), sheet])


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


def _quoted_spans(text: str) -> list[str]:
    """Every span the report presents AS a quotation.

    Straight quotes have no direction, so scanning for '"40+ chars"' also matches the GAP between two
    short quoted words: on a live report, 'maintaining "Buy" ratings ... "super-prime"' was read as a
    quotation and failed provenance. Straight quotes are therefore paired in order, within one line;
    curly quotes and markdown blockquotes are unambiguous."""
    spans = [m.group(1) for m in _CURLY_QUOTE_RE.finditer(text)]
    for line in text.splitlines():
        parts = line.split('"')
        spans += parts[1:len(parts) - 1:2]          # inside each pair; an unpaired quote is ignored
    return spans + [m.group(1) for m in _BLOCKQUOTE_RE.finditer(text)]


def _windows(words: list[str], size: int) -> list[str]:
    return [" ".join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))]


_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _traced(quote: str, corpus: str) -> bool:
    parts = [p for p in _ELLIPSIS.split(quote) if p.strip()]
    if len(parts) > 1:
        # an elided quote ('higher P/E ratios... shrink the margin of safety' leaves out 'and interest
        # rates'): every piece of 3+ words must be verbatim, and the pieces must reach the quote length
        pieces = [_norm(p) for p in parts if len(_norm(p).split()) >= 3]
        return bool(pieces) and all(p in corpus for p in pieces) and \
            sum(len(p.split()) for p in pieces) >= settings.min_quote_words
    return any(w in corpus for w in _windows(_norm(quote).split(), settings.min_quote_words))


def check_citation_provenance(trace: dict) -> dict:
    """Long quoted spans must appear in what the analyst was given (the fabricated-Graham-quote bug): the
    retrieved passages, a tool's result (an earnings-call sentence from get_management_commentary) or the
    evidence pack's fact sheet."""
    chunks = trace.get("retrieved_chunks", [])
    tools = _tool_blob(trace)
    if not chunks and not tools.strip():
        return _ok("citation_provenance", True, "nothing retrieved", applicable=False)
    corpus = _norm(" ".join(c.get("text", "") for c in chunks) + " " + tools)
    quotes = [q.strip() for q in _quoted_spans(trace.get("report", "") or "")
              if len(q.split()) >= settings.min_quote_words]
    if not quotes:
        return _ok("citation_provenance", True, "no long quotes to verify", applicable=False)
    unsupported = [q[:70] for q in quotes if not _traced(q, corpus)]
    return _ok("citation_provenance", not unsupported,
               f"{len(unsupported)}/{len(quotes)} quotes NOT in the evidence: {unsupported[:2]}"
               if unsupported else f"all {len(quotes)} quote(s) traced to the evidence")


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
    """Retrieval must not collapse onto one book (the vanished-Coffee-Can bug). Broker passages are
    not books: two broker notes beside one book is still one book."""
    chunks = [c for c in trace.get("retrieved_chunks", [])
              if c.get("tool", "penny_stock_researcher") == "penny_stock_researcher"]
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
      "news flow", "regulatory", "default", "invoked", "invocation", "insider buying",
      "insider selling", "rating action"), "get_corporate_events"),
    (("management", "guidance", "commentary", "concall", "earnings call", "conference call",
      "what management said", "tone", "credibility", "promised"), "get_management_commentary"),
    (("contingent liabilit", "related party", "related-party", "off balance sheet",
      "off-balance-sheet", "annual report", "guarantees given", "group company",
      "disputed tax", "promoter group dealing"), "get_annual_report_notes"),
    (("screen", "shortlist", "find me stocks", "which stocks", "candidates", "universe",
      "best stocks in"), "screen_stocks"),
    (("broker", "brokers", "brokerage", "target price", "price target", "consensus", "sell-side",
      "sell side", "analyst rating", "analyst ratings", "estimate revision", "earnings revision"),
     "get_broker_consensus"),
    (("broker report", "broker reports", "research report", "research reports",
      "brokerage report", "brokerage reports", "broker thesis", "broker's thesis"),
     "search_broker_research"),
    (("trendlyne", "dvm", "durability score", "momentum score", "valuation score"), "get_trendlyne_view"),
    (("filing", "filings", "disclose", "disclosed", "disclosure", "investor presentation", "rating rationale"),
     "search_filings"),
    (("graham", "philosophy", "principle", "behavioral", "behavioural", "trap", "wisdom", "framework",
      "psychology", "classic error", "old-timer"), "penny_stock_researcher"),
    (("quote", "current price", "pull its", "pull me"), "get_indian_stock_data"),
]


def expected_tools(question: str) -> list[str]:
    """The tools a question clearly needs (one per rule that fires). The analyst graph uses it too."""
    q = (question or "").lower()
    # word-boundary match: plain `in` made "stake" fire on "mistakes" and "fii" on any word
    return [tool for keywords, tool in _TOOL_RULES
            if any(re.search(rf"\b{re.escape(k)}\b", q) for k in keywords)]


def check_tool_coverage(trace: dict) -> dict:
    """If the question clearly needs a tool, that tool must have been called."""
    used = {c.get("name") for c in trace.get("tool_calls", [])}
    wanted = expected_tools(trace.get("query") or "")
    expected, missing = len(wanted), [t for t in wanted if t not in used]
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


def check_context_fits(trace: dict) -> dict:
    """Every model call must fit its window. Ollama drops the oldest messages of an over-long prompt
    WITHOUT an error - which once made a report call data 'not returned' that the tools had returned."""
    calls = trace.get("llm_calls")
    if not calls:
        return _ok("context_fits", True, "no token log in this trace", applicable=False)
    bad = [c for c in calls if c.get("context_full") or c.get("prompt_near_limit")]
    if bad:
        worst = max(bad, key=lambda c: (c.get("input_tokens") or 0) + (c.get("output_tokens") or 0))
        return _ok("context_fits", False,
                   f"{len(bad)} of {len(calls)} model calls filled or nearly filled their window; worst: "
                   f"{worst['role']} {worst.get('input_tokens')}+{worst.get('output_tokens')} "
                   f"of {worst.get('num_ctx')}")
    peak = max(((c.get("input_tokens") or 0) / c["num_ctx"] for c in calls if c.get("num_ctx")), default=0)
    return _ok("context_fits", True, f"{len(calls)} model calls; largest prompt {peak:.0%} of its window")


# A report must not call data missing that a tool returned. The DIXON report of 2026-10-10 said "No
# Piotroski F-Score or Altman Z-Score was returned in the provided data" - get_forensics had returned both;
# the context had dropped them.
_MISSING = re.compile(
    r"\b(not\s+(?:been\s+)?(?:returned|provided|available|included|disclosed|reported|found)|"
    r"unavailable|no\s+data|data\s+gaps?|missing|unverified|unchecked|"
    r"could\s+not\s+be\s+(?:verified|confirmed|assessed|quantified|extracted)|"
    r"no\s+\S[^.]{0,80}?\b(?:was|were|is|are|has\s+been|have\s+been)\s+"
    r"(?:returned|provided|available|included|disclosed|reported|found))\b", re.I)
# topic words in a sentence -> the tool whose result settles it, and how to tell the result really carries it
_HAVE = [
    (("piotroski", "altman", "beneish", "forensic"), "get_forensics", lambda r: "'piotroski'" in r),
    (("promoter holding", "shareholding", "fii", "dii"), "get_shareholding",
     lambda r: "'promoter_pct'" in r and "'promoter_pct': None" not in r),
    (("pledge", "pledging"), "get_shareholding",
     lambda r: "'pledge_pct_of_promoter_holding'" in r and "'pledge_pct_of_promoter_holding': None" not in r),
    (("revenue", "net profit", "net income"), "get_financials", lambda r: "'revenue_cr': {" in r),
    (("historical p/e", "own history", "p/e band", "peer median"), "get_valuation_context",
     lambda r: "'vs_own_history'" in r),
    (("contingent liabilit",), "get_annual_report_notes", lambda r: "'machine_verified': True" in r),
    (("related-party", "related party"), "get_annual_report_notes",
     lambda r: "'source': 'NSE Reg 23(9) XBRL'" in r),
]


def check_no_false_missing(trace: dict) -> dict:
    if not trace.get("report"):
        return _ok("no_false_missing", True, "no report", applicable=False)
    results: dict[str, list[str]] = {}
    for c in trace.get("tool_calls", []):
        results.setdefault(c.get("name"), []).append(c.get("result") or "")
    wrong = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", trace["report"]):
        low = sentence.lower()
        if not _MISSING.search(low):
            continue
        for words, tool, has in _HAVE:
            if any(w in low for w in words) and any(has(r) for r in results.get(tool, [])):
                wrong.append(f"{tool}: '{sentence.strip()[:90]}'")
    return _ok("no_false_missing", not wrong, "; ".join(wrong[:3]) if wrong else "no false 'missing' claims")


# The report must end as a finished text. The DIXON report of 2026-10-10 ended '...a mid-range performer
# (Durability: High' and still scored 4/5. A bullet that ends in a figure ('at Rs 12,097') is finished.
_DANGLING = re.compile(r"\b(and|or|but|the|a|an|of|to|with|for|in|on|at|by|from|as|is|are|was|were|"
                       r"which|that|than|its|their)$", re.I)


def check_report_complete(trace: dict) -> dict:
    report = (trace.get("report") or "").rstrip()
    if not report:
        return _ok("report_complete", False, "no report")
    analyst = [c for c in trace.get("llm_calls") or [] if c.get("role") == "analyst"]
    if analyst and analyst[-1].get("done_reason") == "length":
        return _ok("report_complete", False, "the final draft stopped at its length limit")
    last = report.splitlines()[-1].strip()
    unclosed = last.count("(") > last.count(")") or last.count("[") > last.count("]")
    unfinished = len(last.split()) >= 6 and not re.search(r"[.!?)\]\"'”’*_|%:\d]$", last)
    if unclosed or unfinished or _DANGLING.search(last):
        return _ok("report_complete", False, f"ends mid-sentence: '...{last[-60:]}'")
    return _ok("report_complete", True, f"ends: '...{last[-40:]}'")


ALL_CHECKS = [check_entity_consistency, check_no_unknown_tools, check_task_completion,
              check_citation_provenance, check_numeric_provenance, check_source_diversity,
              check_tool_coverage, check_data_warning_surfaced, check_pledge_disclosed, check_context_fits,
              check_no_false_missing, check_report_complete]


def run_all(trace: dict) -> list[dict]:
    """Run every check; a broken check reports as a failure instead of killing the suite."""
    results = []
    for fn in ALL_CHECKS:
        try:
            results.append(fn(trace))
        except Exception as e:
            results.append(_ok(fn.__name__.replace("check_", ""), False, f"check errored: {e}"))
    return results
