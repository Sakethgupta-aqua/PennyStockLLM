"""Thin entry point to run the whole pipeline from the package.

    import asyncio
    from pennystock.pipeline import analyze
    result = asyncio.run(analyze("Analyze Suzlon Energy on the NSE"))

Every run is saved as a trace under traces/ (see pennystock.tracing) unless disabled, and the
NSE symbol universe is refreshed if it has gone stale (weekly), so newly listed companies
still resolve.
Requires Ollama running (the models) and the vector DB present.
"""
import asyncio
import logging
import time
from datetime import datetime

from langchain_core.messages import HumanMessage

from pennystock.agents.graph import build_graph
from pennystock.config import settings
from pennystock.data.universe import ensure_fresh
from pennystock.tracing import build_trace, save_trace

logger = logging.getLogger(__name__)


async def analyze(query: str, trace: bool = True) -> dict:
    """Run one analysis end-to-end, save a trace, and return the final graph state."""
    # keep the resolver's symbol list current; cheap when fresh, and never fatal when it is not
    logger.info(f"Universe: {await asyncio.to_thread(ensure_fresh)}")

    app = build_graph()
    inputs = {"messages": [HumanMessage(content=query)]}
    started = time.time()
    node_path: list[str] = []
    final_state: dict | None = None
    error: str | None = None

    try:
        # stream_mode=["updates","values"] gives BOTH the node path and the full final state.
        async for mode, chunk in app.astream(inputs, stream_mode=["updates", "values"]):
            if mode == "updates" and isinstance(chunk, dict):
                node_path.extend(k for k in chunk if k)
            elif mode == "values":
                final_state = chunk
    except (TypeError, ValueError) as e:                 # older LangGraph: no multi-mode streaming
        logger.warning(f"Multi-mode streaming unavailable ({e}); falling back to ainvoke.")
        node_path, final_state = [], await app.ainvoke(inputs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if trace and settings.trace_enabled and final_state is not None:
            try:
                record = build_trace(query=query, final_state=final_state, node_path=node_path,
                                     duration_s=round(time.time() - started, 1), error=error)
                save_trace(record)
                if settings.ledger_enabled:
                    await _record_prediction(record)
            except Exception as e:
                logger.warning(f"Tracing failed (run unaffected): {e}")

    return final_state or {}


async def _record_prediction(trace: dict) -> str | None:
    """Write this run's call into the ledger so it can be graded later.

    THIS IS A PASSENGER, NEVER THE DRIVER. Every failure path returns quietly: the user
    asked for an analysis, and a bookkeeping problem must never cost them one. The whole
    body is wrapped by the caller's try/except as well.

    THREE THINGS IT REFUSES TO DO:
      * Guess. Two failed extractions produce a row marked `extraction_failed`, not an
        invented `buy`. A run that made no gradeable call is itself a fact.
      * Recompute the price later. `entry_price` is the live quote captured NOW, because
        the past changes - the same historical close reads ~10% lower once Yahoo applies
        dividends backwards.
      * Overwrite. The ledger INSERTs; a duplicate id raises rather than silently editing
        what was predicted.
    """
    symbol, report = trace.get("resolved_symbol"), trace.get("report")
    if not symbol:
        logger.info("Ledger: no company resolved in this run; nothing to record.")
        return None

    # The price the call was made at, captured now and frozen. Failing to get it is not
    # fatal - a direction with no entry price is still gradeable against the benchmark.
    entry_price = None
    try:
        from pennystock.data.providers import get_provider
        entry_price = (await asyncio.to_thread(get_provider().get_snapshot, f"{symbol}.NS")).price
    except Exception as e:
        logger.warning(f"Ledger: entry price unavailable for {symbol}: {type(e).__name__}: {e}")

    from pennystock.agents.recommendation import extract_recommendation, to_prediction_fields
    from pennystock.data.ledger import Prediction, save_predictions
    from pennystock.data.sectors import profile_for

    result = await extract_recommendation(str(report or ""), entry_price)
    today = datetime.now().date().isoformat()
    provenance = {"trace_id": trace.get("trace_id"), "models": trace.get("models"),
                  "node_path_len": len(trace.get("node_path") or []),
                  "tool_calls": len(trace.get("tool_calls") or []),
                  "extraction_attempts": result.attempts}

    if result.ok:
        fields = to_prediction_fields(result.recommendation)
        payload = {**fields.pop("payload"), "query": trace.get("query"),
                   "semantic_warnings": result.warnings}
    else:
        # Recorded, not skipped. Silently dropping failed runs would make the hit rate
        # look better than it is by hiding every run the system could not commit to.
        logger.warning(f"Ledger: {result.reason}")
        fields = {"direction": "neutral", "conviction": 1, "horizon": "1m",
                  "invalidation_price": None}
        payload = {"query": trace.get("query"), "extraction_failed": True,
                   "reason": result.reason}

    pred = Prediction(
        symbol=symbol, source="agent", as_of_date=today, entry_price=entry_price,
        entry_price_asof=today, sector=None,
        rulebook=profile_for(None, None).name, payload=payload,
        provenance=provenance, **fields)
    save_predictions([pred])
    logger.info(f"Ledger: recorded {pred.direction} on {symbol} "
                f"at {entry_price} ({pred.prediction_id})")
    return pred.prediction_id
