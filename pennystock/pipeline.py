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
                save_trace(build_trace(query=query, final_state=final_state, node_path=node_path,
                                       duration_s=round(time.time() - started, 1), error=error))
            except Exception as e:
                logger.warning(f"Tracing failed (run unaffected): {e}")

    return final_state or {}
