"""Scheduled data maintenance - one place that knows what must be refreshed and how often.

Three caches feed the analyst, and each changes at a completely different rate:

    universe    NSE listing master   ~weekly    one CSV, cheap
    peers       sector medians       ~monthly   one network call PER company - expensive
    ownership   shareholding+pledge  ~quarterly per symbol; that is the filing cadence

Design principles:
  * STALENESS-DRIVEN, not blind. Every job checks its own cache age first, so running this hourly
    from cron costs three SQLite reads on most invocations. Cheap when fresh is what makes an
    automated job safe to schedule aggressively.
  * REFRESH WHAT YOU ALREADY TRACK. Ownership refreshes only symbols already in the cache and peers
    only symbols already cached (or the seed list when empty), so the job cannot grow unbounded and
    start hammering the sources.
  * IDEMPOTENT + FAIL-SOFT. Every refresh is an upsert; one failing dataset never aborts the others.
  * SINGLE-FLIGHT. A lock file stops a slow run overlapping the next scheduled one - the classic
    way a well-meaning cron job turns into a self-inflicted denial of service.

Per-symbol data ALSO self-heals during a live run (`get_ownership` and `compare_to_peers` fetch on a
cache miss), so this job is about keeping things warm, never about correctness.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from pennystock.config import settings

logger = logging.getLogger(__name__)

LOCK_PATH = Path(settings.project_root) / ".refresh.lock"
LOCK_MAX_AGE_HOURS = 6          # a lock older than this is assumed abandoned (killed process)


@dataclass
class Dataset:
    name: str
    cadence: str
    status: Callable[[], str]
    is_stale: Callable[[], bool]
    refresh: Callable[[], str]


@contextmanager
def single_flight(path: Path = LOCK_PATH):
    """Prevent two refreshes running at once; auto-clears a lock left by a killed process."""
    if path.exists():
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age < timedelta(hours=LOCK_MAX_AGE_HOURS):
            raise RuntimeError(f"another refresh is running (lock held {age}, pid in {path.name})")
        logger.warning(f"clearing stale lock ({age} old)")
        path.unlink(missing_ok=True)
    path.write_text(f"{os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# ------------------------------------------------------------------ jobs
def _universe_refresh() -> str:
    from pennystock.data.universe import ensure_fresh
    return ensure_fresh(force=True)


def _universe_status() -> str:
    from pennystock.data.universe import last_updated
    return f"last updated {last_updated()}"


def _peers_refresh() -> str:
    from pennystock.data.peers import SEED_LIQUID, cached_symbols, fetch_one, save_peers
    symbols = cached_symbols() or SEED_LIQUID
    rows, missed = [], 0
    for sym in symbols:
        row = fetch_one(sym)
        rows.append(row) if row else None
        missed += 0 if row else 1
        if len(rows) >= 25:
            save_peers(rows); rows = []
        time.sleep(settings.refresh_delay_s)
    if rows:
        save_peers(rows)
    return f"refreshed {len(symbols) - missed}/{len(symbols)} peer symbols"


def _peers_status() -> str:
    from pennystock.data.peers import coverage, last_fetched
    c = coverage()
    return f"{c['symbols_cached']} symbols / {c['industries_cached']} industries, last {last_fetched()}"


def _ownership_refresh() -> str:
    from pennystock.data.ownership_providers import get_ownership, tracked_symbols
    symbols = tracked_symbols()
    if not symbols:
        return "no symbols tracked yet (they are added the first time a stock is analysed)"
    ok = 0
    for sym in symbols:
        own = get_ownership(sym, refresh=True)
        ok += 1 if own.quarters else 0
        time.sleep(settings.refresh_delay_s)
    return f"refreshed {ok}/{len(symbols)} tracked symbols"


def _ownership_status() -> str:
    from pennystock.data.ownership_providers import last_fetched, tracked_symbols
    return f"{len(tracked_symbols())} symbols tracked, last {last_fetched()}"


def _metrics_refresh() -> str:
    """Re-fetch the screening row for everything already in the cache. Deliberately does NOT expand
    the universe - widening coverage is an explicit decision, made by running ingest_metrics.py."""
    from pennystock.data.metrics import cached_symbols, fetch_metrics, save_metrics
    symbols = sorted(cached_symbols())
    if not symbols:
        return "no screening metrics cached yet (run scripts/ingest_metrics.py)"
    rows, ok = [], 0
    for sym in symbols:
        row = fetch_metrics(sym)
        if row:
            rows.append(row)
            ok += 1
        if len(rows) >= 25:
            save_metrics(rows); rows = []
        time.sleep(settings.refresh_delay_s)
    save_metrics(rows)
    return f"refreshed {ok}/{len(symbols)} screening rows"


def _metrics_status() -> str:
    from pennystock.data.metrics import cached_symbols, last_fetched
    return f"{len(cached_symbols())} symbols cached, last {last_fetched()}"


def datasets() -> list[Dataset]:
    from pennystock.data import metrics as metrics_mod
    from pennystock.data import peers as peers_mod
    from pennystock.data import universe as universe_mod
    from pennystock.data.ownership_providers import cache_is_stale
    return [
        Dataset("universe", f"~{settings.universe_max_age_days}d",
                _universe_status, universe_mod.is_stale, _universe_refresh),
        Dataset("peers", f"~{settings.peer_max_age_days}d",
                _peers_status, peers_mod.is_stale, _peers_refresh),
        Dataset("ownership", f"~{settings.ownership_max_age_days}d",
                _ownership_status, cache_is_stale, _ownership_refresh),
        Dataset("metrics", f"~{settings.metrics_max_age_days}d",
                _metrics_status, metrics_mod.is_stale, _metrics_refresh),
    ]


def status() -> list[dict]:
    out = []
    for d in datasets():
        try:
            out.append({"dataset": d.name, "cadence": d.cadence,
                        "stale": d.is_stale(), "detail": d.status()})
        except Exception as e:
            out.append({"dataset": d.name, "cadence": d.cadence, "stale": None,
                        "detail": f"status failed: {type(e).__name__}: {e}"})
    return out


def refresh_all(only: list[str] | None = None, force: bool = False) -> list[dict]:
    """Refresh every stale dataset. One failure never stops the others."""
    results = []
    for d in datasets():
        if only and d.name not in only:
            continue
        try:
            if not force and not d.is_stale():
                results.append({"dataset": d.name, "action": "skipped (fresh)", "detail": d.status()})
                continue
            started = time.time()
            detail = d.refresh()
            results.append({"dataset": d.name, "action": "refreshed",
                            "seconds": round(time.time() - started, 1), "detail": detail})
        except Exception as e:                       # fail-soft: keep going
            logger.warning(f"{d.name} refresh failed: {type(e).__name__}: {e}")
            results.append({"dataset": d.name, "action": "FAILED",
                            "detail": f"{type(e).__name__}: {e}"})
    return results
