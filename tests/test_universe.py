from datetime import datetime, timedelta
from unittest.mock import patch

from pennystock.data import universe


def test_is_stale_when_never_updated():
    with patch.object(universe, "last_updated", return_value=None):
        assert universe.is_stale()


def test_is_stale_respects_threshold():
    recent = datetime.now() - timedelta(days=2)
    old = datetime.now() - timedelta(days=30)
    with patch.object(universe, "last_updated", return_value=recent):
        assert not universe.is_stale(max_age_days=7)
    with patch.object(universe, "last_updated", return_value=old):
        assert universe.is_stale(max_age_days=7)


def test_ensure_fresh_never_raises_when_the_network_fails():
    universe._checked_this_process = False
    with patch.object(universe, "is_stale", return_value=True), \
         patch.object(universe, "fetch_master", side_effect=ConnectionError("NSE down")):
        status = universe.ensure_fresh(force=True)
    assert "FAILED" in status and "cached universe" in status      # degrades, does not crash


def test_ensure_fresh_skips_work_when_fresh():
    universe._checked_this_process = False
    with patch.object(universe, "is_stale", return_value=False), \
         patch.object(universe, "fetch_master", side_effect=AssertionError("must not fetch")):
        assert "fresh" in universe.ensure_fresh()


def test_ensure_fresh_runs_at_most_once_per_process():
    universe._checked_this_process = True
    with patch.object(universe, "fetch_master", side_effect=AssertionError("must not fetch")):
        assert universe.ensure_fresh() == "already checked this process"


def test_universe_uses_the_resolver_normalizer():
    """The ingest MUST normalise names exactly as the resolver does, or lookups silently fail."""
    from pennystock.data.resolver import normalize
    assert universe.normalize is normalize
