"""Execution tracing - the foundation of the evaluation stack.

Every pipeline run is saved as ONE self-contained JSON file under traces/. A trace IS the
agent's trajectory: the query, every tool call + result, the retrieved passages, the final
report, and the judge's verdict. This is the raw material for (a) error analysis - reading
real runs and categorising failures - and (b) the Layer-0 assertions in pennystock/evals.

Design notes:
  * Field names follow the OpenTelemetry GenAI semantic conventions in spirit (tool name /
    arguments / result, model per role) but stay flat and human-readable: we write local
    files, not OTel spans, because a solo local setup has no collector to ship to. The
    mapping is 1:1 if you later move to an OTel backend.
  * A trajectory is just the message list (the shape LangChain's `agentevals` uses), so no
    extra instrumentation library is required.
  * Tracing NEVER breaks a run: writes are wrapped; failures are logged, not raised.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from pennystock.config import settings

logger = logging.getLogger(__name__)

_CHUNK_RE = re.compile(r"\[source:\s*([^\]]+)\]\s*")
_SCORE_RE = re.compile(r"Score:\s*(\d)", re.IGNORECASE)
JUDGE_PREFIX = "LLM Judge Evaluation"


def _truncate(text: str, limit: int | None = None) -> str:
    limit = limit or settings.trace_max_field_chars
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"...[truncated {len(text) - limit} chars]"


def parse_retrieved_chunks(researcher_output: str) -> list[dict]:
    """Split the researcher tool's '[source: X] text' blob back into structured chunks."""
    parts = _CHUNK_RE.split(str(researcher_output))
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            chunks.append({"source": parts[i].strip(), "text": body})
    return chunks


def _extract_tool_calls(messages) -> list[dict]:
    """Pair each requested tool call with the ToolMessage carrying its result."""
    pending: dict = {}
    calls: list[dict] = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            pending[tc.get("id")] = {"name": tc.get("name", "?"), "args": tc.get("args", {})}
        if isinstance(m, ToolMessage):
            info = pending.pop(getattr(m, "tool_call_id", None), {"name": "?", "args": {}})
            result = str(m.content)
            calls.append({
                "seq": len(calls) + 1,
                "name": info["name"],
                "args": info["args"],
                "result": _truncate(result),
                "ok": not result.lstrip().startswith(("Error", "System Error")),
            })
    return calls


def _extract_report_and_judge(messages) -> tuple[str, dict]:
    """The report is the last substantive AI message; the judge message is tagged by prefix."""
    texts = [str(m.content) for m in messages
             if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and str(m.content).strip()]
    judge_raw = next((t for t in reversed(texts) if t.startswith(JUDGE_PREFIX)), "")
    reports = [t for t in texts if not t.startswith(JUDGE_PREFIX)]
    score = None
    if judge_raw:
        hit = _SCORE_RE.search(judge_raw)
        score = int(hit.group(1)) if hit else None
    return (reports[-1] if reports else ""), {"score": score, "raw": _truncate(judge_raw)}


def build_trace(query: str, final_state: dict, node_path=None, duration_s=None, error=None) -> dict:
    """Turn a finished graph state into a flat, self-contained trace record."""
    messages = (final_state or {}).get("messages", []) or []
    calls = _extract_tool_calls(messages)
    report, judge = _extract_report_and_judge(messages)

    chunks: list[dict] = []
    for c in calls:
        if c["name"] == "penny_stock_researcher":
            chunks.extend(parse_retrieved_chunks(c["result"]))

    return {
        "schema_version": 1,
        "trace_id": f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "query": query,
        "resolved_symbol": (final_state or {}).get("resolved_symbol"),
        "resolved_name": (final_state or {}).get("resolved_name"),
        "node_path": node_path or [],
        "tool_calls": calls,
        "retrieved_chunks": [{"source": c["source"], "text": _truncate(c["text"], 4000)} for c in chunks],
        "report": _truncate(report),
        "judge": judge,
        "revisions": (final_state or {}).get("revisions", 0),
        "grounding_retries": (final_state or {}).get("grounding_retries", 0),
        "models": {"data": settings.data_model, "analyst": settings.analyst_model,
                   "judge": settings.judge_model, "supervisor": settings.supervisor_model,
                   "grounding": settings.grounding_model},
        "retrieval_config": {"collection": settings.collection, "embed_model": settings.embed_model,
                             "reranker_model": settings.reranker_model,
                             "rerank_top_n": settings.rerank_top_n,
                             "research_subqueries": settings.research_subqueries},
        "error": error,
    }


def save_trace(trace: dict) -> Path | None:
    try:
        d = Path(settings.trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{trace['trace_id']}.json"
        p.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
        logger.info(f"Trace saved -> {p}")
        return p
    except Exception as e:
        logger.warning(f"Could not save trace: {e}")     # tracing must never break a run
        return None


def load_traces(directory=None, limit: int | None = None) -> list[dict]:
    d = Path(directory or settings.trace_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("*.json"))
    if limit:
        files = files[-limit:]
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception as e:
            logger.warning(f"Skipping unreadable trace {f.name}: {e}")
    return out
