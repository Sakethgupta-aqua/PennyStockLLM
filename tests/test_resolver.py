"""Unit tests for the symbol resolver."""
import pytest

from pennystock.config import settings
from pennystock.data.resolver import normalize, resolve_from_query, resolve_symbol

# The fuzzy/DB tests need the built universe table; skip cleanly if market.db is absent.
needs_db = pytest.mark.skipif(not settings.market_db.exists(), reason="market.db not built")


# ---- pure logic (no DB needed) ----
def test_normalize_strips_suffixes_and_case():
    assert normalize("Kaynes Technology India Limited") == "kaynes technology"
    assert normalize("HDFC Bank Ltd.") == "hdfc bank"
    assert normalize("  Suzlon Energy  ") == "suzlon energy"


# ---- resolution against the real universe DB ----
@needs_db
def test_exact_ticker_resolves():
    m = resolve_symbol("SUZLON")
    assert m and m["symbol"] == "SUZLON" and m["how"] == "exact_ticker"


@needs_db
def test_company_name_resolves():
    m = resolve_symbol("Suzlon Energy")
    assert m and m["symbol"] == "SUZLON"


@needs_db
def test_garbage_is_rejected():
    assert resolve_symbol("asdfghjkl total nonsense") is None


@needs_db
def test_from_query_extracts_the_stock():
    m = resolve_from_query("Analyze Kaynes Technology on the NSE please")
    assert m and m["symbol"] == "KAYNES"


@needs_db
def test_from_query_ignores_non_stock_text():
    assert resolve_from_query("What is the weather in Mumbai today?") is None
