import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from pennystock.data import maintenance as mt
from pennystock.data.maintenance import Dataset, refresh_all, single_flight, status


def _ds(name, stale, refresh_fn=None):
    return Dataset(name, "~1d", lambda: "detail", lambda: stale,
                   refresh_fn or (lambda: f"{name} refreshed"))


def test_status_covers_every_registered_cache(tmp_path, monkeypatch):
    """Every dataset in the registry must report status. This is what stops a new cache being
    added and then silently never refreshed - the failure mode the registry exists to prevent."""
    # a throwaway database: unpatched, this test created tables in (and read) the REAL market.db
    monkeypatch.setattr(mt.settings, "market_db", tmp_path / "test.db")
    reported = {r["dataset"] for r in status()}
    assert reported == {d.name for d in mt.datasets()}
    assert {"universe", "peers", "ownership", "metrics"} <= reported


def test_every_dataset_declares_a_cadence_and_a_refresh():
    for d in mt.datasets():
        assert d.cadence and callable(d.is_stale) and callable(d.refresh), d.name


def test_refresh_skips_fresh_datasets():
    with patch.object(mt, "datasets", return_value=[_ds("a", stale=False)]):
        assert refresh_all()[0]["action"] == "skipped (fresh)"


def test_force_refreshes_even_when_fresh():
    with patch.object(mt, "datasets", return_value=[_ds("a", stale=False)]):
        assert refresh_all(force=True)[0]["action"] == "refreshed"


def test_only_filters_datasets():
    with patch.object(mt, "datasets", return_value=[_ds("a", True), _ds("b", True)]):
        out = refresh_all(only=["b"])
    assert len(out) == 1 and out[0]["dataset"] == "b"


def test_one_failure_never_stops_the_others():
    def boom():
        raise ConnectionError("source down")
    with patch.object(mt, "datasets", return_value=[_ds("a", True, boom), _ds("b", True)]):
        out = refresh_all()
    assert out[0]["action"] == "FAILED" and out[1]["action"] == "refreshed"   # fail-soft


def test_single_flight_blocks_a_concurrent_run(tmp_path):
    lock = tmp_path / "x.lock"
    with single_flight(lock):
        assert lock.exists()
        with pytest.raises(RuntimeError, match="another refresh is running"):
            with single_flight(lock):
                pass
    assert not lock.exists()          # released on exit


def test_single_flight_clears_an_abandoned_lock(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("999 old")
    old = (datetime.now() - timedelta(hours=mt.LOCK_MAX_AGE_HOURS + 1)).timestamp()
    import os
    os.utime(lock, (old, old))
    with single_flight(lock):         # must not raise - the holder is long gone
        assert lock.exists()


def test_lock_is_released_even_when_the_body_raises(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(ValueError):
        with single_flight(lock):
            raise ValueError("boom")
    assert not lock.exists()


def test_event_signals_are_a_maintained_dataset():
    assert "events" in {d.name for d in mt.datasets()}


def test_the_filings_index_is_a_maintained_dataset():
    assert "filings index" in {d.name for d in mt.datasets()}
