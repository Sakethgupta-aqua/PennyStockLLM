"""Per-stock screening metrics: the cached row the screener ranks on.

Analysing one stock properly costs four or five network calls and several seconds. Doing that live
across a 2,000-name universe is impossible, so the work is split the way every real desk splits it:

    OVERNIGHT (this module)  fetch a compact row of metrics for every stock, cache it in SQLite
    INTERACTIVE (screener)   filter and rank those cached rows in milliseconds

The row deliberately carries the INPUTS a decision needs, not a verdict. Verdicts change when the
thresholds change; inputs do not, so a re-scored screen never needs a re-fetch.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta

from pennystock.config import settings

logger = logging.getLogger(__name__)

# how far the feed's P/E may drift from market cap / filed net profit before we distrust it.
# Wide on purpose: trailing-twelve-months vs last fiscal year legitimately differ by ~30%.
PE_SANITY_LOW, PE_SANITY_HIGH = 0.4, 2.5

_COLUMNS = """symbol TEXT PRIMARY KEY, company TEXT, sector TEXT, industry TEXT,
    market_cap_cr REAL, turnover_cr REAL,
    roe_pct REAL, roce_pct REAL, profit_margin_pct REAL, revenue_cagr_pct REAL,
    debt_to_equity_pct REAL, debt_to_ebitda REAL, interest_coverage REAL, altman_zem REAL,
    piotroski INTEGER, beneish_m REAL, cash_conversion REAL, tax_flag INTEGER,
    pledge_pct_of_promoter REAL, promoter_holding_pct REAL,
    pe REAL, pb REAL, pct_from_peak_5y REAL, above_200dma INTEGER,
    red_flag_count INTEGER, red_flags TEXT, fetched_at TEXT"""
_SCHEMA = f"CREATE TABLE IF NOT EXISTS stock_metrics ({_COLUMNS})"


@dataclass
class StockMetrics:
    symbol: str
    company: str | None = None
    sector: str | None = None
    industry: str | None = None
    # --- can I actually trade it? ---
    market_cap_cr: float | None = None
    turnover_cr: float | None = None          # median daily traded VALUE, the real liquidity test
    # --- is the business any good? ---
    roe_pct: float | None = None
    roce_pct: float | None = None
    profit_margin_pct: float | None = None
    revenue_cagr_pct: float | None = None
    # --- will it survive? ---
    debt_to_equity_pct: float | None = None
    debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    altman_zem: float | None = None           # emerging-market variant, the India-appropriate one
    # --- can I trust the numbers and the people? ---
    piotroski: int | None = None
    beneish_m: float | None = None
    cash_conversion: float | None = None
    tax_flag: int | None = None
    pledge_pct_of_promoter: float | None = None
    promoter_holding_pct: float | None = None
    # --- what am I paying, and where is the price? ---
    pe: float | None = None
    pb: float | None = None
    pct_from_peak_5y: float | None = None
    above_200dma: int | None = None
    # --- carried through so the screen can explain itself ---
    red_flag_count: int | None = None
    red_flags: str | None = None
    fetched_at: str | None = None


_FIELDS = [f.name for f in fields(StockMetrics)]
_NUMERIC = {f.name for f in fields(StockMetrics) if str(f.type).startswith(("float", "int"))}


def _finite_or_none(value):
    """A numeric field's value, or None if it is not a finite number.

    yfinance hands back the STRING 'Infinity' as the P/E of a near-zero-earnings company, and this
    table is not type-strict, so the text was stored as-is. Four such rows crashed the screen the
    moment they reached the percentile sort. Unknown is the honest value for 'not a number'.
    """
    if value is None or isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return value if isinstance(value, (int, float)) else number


def _clean(row: dict) -> dict:
    return {k: (_finite_or_none(v) if k in _NUMERIC else v) for k, v in row.items()}


def _connect():
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def save_metrics(rows: list[StockMetrics]) -> int:
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    cols = ", ".join(_FIELDS)
    placeholders = ", ".join("?" * len(_FIELDS))
    updates = ", ".join(f"{c}=excluded.{c}" for c in _FIELDS if c != "symbol")
    with _connect() as con:
        con.executemany(
            f"INSERT INTO stock_metrics ({cols}) VALUES ({placeholders})"
            f" ON CONFLICT(symbol) DO UPDATE SET {updates}",
            [tuple(_clean(asdict(r) | {"fetched_at": now})[c] for c in _FIELDS) for r in rows])
    return len(rows)


def load_metrics(max_age_days: int | None = None) -> list[StockMetrics]:
    """Every cached row, optionally only those fresh enough to act on."""
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(f"SELECT {', '.join(_FIELDS)} FROM stock_metrics").fetchall()
    out = [StockMetrics(**_clean(dict(r))) for r in rows]
    if max_age_days is None:
        return out
    cutoff = datetime.now() - timedelta(days=max_age_days)
    fresh = [r for r in out if r.fetched_at and datetime.fromisoformat(r.fetched_at) >= cutoff]
    if len(fresh) < len(out):
        logger.info(f"{len(out) - len(fresh)} of {len(out)} cached rows are older than "
                    f"{max_age_days}d and were dropped from the screen")
    return fresh


def last_fetched() -> str | None:
    with _connect() as con:
        r = con.execute("SELECT MAX(fetched_at) FROM stock_metrics").fetchone()
    return r[0] if r else None


def is_stale() -> bool:
    """Screening metrics go stale on a quarterly rhythm - results are filed 4x a year - but prices
    and turnover move constantly, so the cadence is set by the fastest-moving field."""
    last = last_fetched()
    if not last:
        return True
    return datetime.now() - datetime.fromisoformat(last) > timedelta(days=settings.metrics_max_age_days)


def cached_symbols() -> set[str]:
    with _connect() as con:
        return {r[0] for r in con.execute("SELECT symbol FROM stock_metrics")}


def fetch_metrics(symbol: str) -> StockMetrics | None:
    """Build one stock's screening row. Every block fails soft: a stock missing its transcript or
    its shareholding still deserves to be screened on what IS known, and a None simply scores as
    unknown rather than dropping the name."""
    import yfinance as yf

    from pennystock.data.forensics import summarize_forensics
    from pennystock.data.providers import get_provider

    symbol = symbol.upper()
    ticker = f"{symbol}.NS"
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.warning(f"metrics: info failed for {symbol}: {type(e).__name__}: {e}")
        return None
    if not info.get("sector") and not info.get("industry") and not info.get("marketCap"):
        return None                                   # not a live, groupable equity

    # Some Indian listings file in USD (INFY reports financialCurrency=USD while its price and
    # market cap are in INR). Any ratio mixing the two - Altman's market-value term, or a P/E built
    # from filed profit - is then meaningless, so market cap is withheld from those calculations.
    inr_accounts = (info.get("financialCurrency") or "INR").upper() == "INR"
    mc = info.get("marketCap")
    m = StockMetrics(
        symbol=symbol, company=info.get("longName"), sector=info.get("sector"),
        industry=info.get("industry"), market_cap_cr=round(mc / 1e7, 1) if mc else None,
        pe=info.get("trailingPE"), pb=info.get("priceToBook"),
        debt_to_equity_pct=info.get("debtToEquity"),
        roe_pct=round(info["returnOnEquity"] * 100, 2) if info.get("returnOnEquity") else None,
        profit_margin_pct=round(info["profitMargins"] * 100, 2) if info.get("profitMargins") else None)

    prov = get_provider()
    try:
        fin = prov.get_financials(ticker, settings.financials_years)
        f = summarize_forensics(fin, m.market_cap_cr if inr_accounts else None)
        m.piotroski = (f.get("piotroski") or {}).get("f_score")
        m.altman_zem = (f.get("altman_emerging_markets") or {}).get("z_em_score")
        m.beneish_m = (f.get("beneish") or {}).get("m_score")
        m.cash_conversion = (f.get("cash_conversion") or {}).get("cfo_to_pat")
        m.interest_coverage = (f.get("interest_coverage") or {}).get("interest_coverage")
        m.debt_to_ebitda = (f.get("debt_to_ebitda") or {}).get("debt_to_ebitda")
        m.tax_flag = int(bool((f.get("effective_tax_rate") or {}).get("flag")))
        flags = f.get("red_flags") or []
        m.red_flag_count, m.red_flags = len(flags), " | ".join(flags)[:600]
        m.roce_pct, m.revenue_cagr_pct = _roce_and_growth(fin)
        if inr_accounts:
            m.pe = _sanity_check_pe(m.symbol, m.pe, m.market_cap_cr, fin)
        if m.roe_pct is None and fin.net_income_cr and fin.equity_cr and fin.equity_cr[-1]:
            # yfinance simply omits returnOnEquity for many Indian names (HINDALCO among them).
            # ROE is too central to an Indian screen to leave blank when the inputs are right here.
            m.roe_pct = round(fin.net_income_cr[-1] / fin.equity_cr[-1] * 100, 2)
    except Exception as e:
        logger.warning(f"metrics: forensics failed for {symbol}: {type(e).__name__}: {e}")

    try:
        from pennystock.data.ownership_providers import get_ownership
        own = get_ownership(symbol)
        if own and own.quarters:
            latest = own.quarters[-1]                 # quarters run oldest -> newest
            m.pledge_pct_of_promoter = latest.pledge_pct    # % of PROMOTER stake, the convention
            m.promoter_holding_pct = latest.promoter_pct
    except Exception as e:
        logger.warning(f"metrics: ownership failed for {symbol}: {type(e).__name__}: {e}")

    try:
        m.turnover_cr, m.pct_from_peak_5y, m.above_200dma = _price_stats(ticker)
    except Exception as e:
        logger.warning(f"metrics: price stats failed for {symbol}: {type(e).__name__}: {e}")
    return m


def _sanity_check_pe(symbol: str, reported: float | None, mcap_cr: float | None,
                     fin) -> float | None:
    """Cross-check the feed's P/E against market cap / net profit, and prefer our own if they
    disagree wildly.

    Caught live: yfinance reported ZFCVINDIA at a P/E of 2.8 while its own filed accounts imply
    ~56 - a twenty-fold error, most likely a special dividend or demerger corrupting the EPS the
    feed derives. A P/E of 2.8 is a top-decile valuation score, so the bad number walked the stock
    straight into a top-25 shortlist. A screen that ranks on price MUST sanity-check the price.

    Drift of ~30% is normal and expected (the feed is trailing-twelve-months, our figure is the
    last full fiscal year), so the band is deliberately wide - this catches breakage, not noise.

    On a disagreement we DISCARD the P/E rather than substituting our own, because we cannot tell
    which side is broken. ZFCVINDIA's reported 2.8 was wrong and the computed 56 was right; INFY is
    the exact reverse (its accounts arrive in USD, so the computed figure is nonsense while the
    reported 14.8 is correct). Guessing would have made INFY worse. A stock with no P/E is simply
    scored as unknown on valuation - the same treatment any other missing field gets.
    """
    profit = next((x for x in reversed(fin.net_income_cr or []) if x), None)
    if reported is None or not mcap_cr or not profit or profit <= 0:
        return reported
    ratio = (mcap_cr / profit) / reported
    if PE_SANITY_LOW <= ratio <= PE_SANITY_HIGH:
        return reported
    logger.warning(f"{symbol}: reported P/E {reported:.1f} contradicts its filed accounts "
                   f"({mcap_cr / profit:.1f}); dropping P/E rather than trusting either")
    return None


def _roce_and_growth(fin) -> tuple[float | None, float | None]:
    """ROCE and revenue CAGR - the two numbers an Indian analyst quotes before any multiple."""
    roce = cagr = None
    try:
        ebit, assets = fin.operating_income_cr, fin.total_assets_cr
        cl = fin.current_liabilities_cr
        if ebit and assets and cl and assets[-1] and (assets[-1] - cl[-1]):
            roce = round(ebit[-1] / (assets[-1] - cl[-1]) * 100, 2)   # capital employed = TA - CL
    except Exception:
        pass
    try:
        rev = [r for r in (fin.revenue_cr or []) if r and r > 0]
        if len(rev) >= 2:
            yrs = len(rev) - 1
            cagr = round(((rev[-1] / rev[0]) ** (1 / yrs) - 1) * 100, 2)
    except Exception:
        pass
    return roce, cagr


def _price_stats(ticker: str) -> tuple[float | None, float | None, int | None]:
    """Median daily TURNOVER, distance from the 5-year peak, and whether price is above its 200-DMA.

    Turnover (price x volume), not share volume, is the liquidity test that matters: 10 lakh shares
    of a Rs 3 stock is Rs 30 lakh a day, which no real position can enter or exit.
    """
    import yfinance as yf

    h = yf.Ticker(ticker).history(period="5y", auto_adjust=False)
    if h is None or h.empty:
        return None, None, None
    h = h.dropna(subset=["Close"])
    close = h["Close"]
    recent = h.tail(settings.turnover_lookback_days)
    turnover = None
    if "Volume" in recent and not recent.empty:
        val = (recent["Close"] * recent["Volume"]).median()
        turnover = round(float(val) / 1e7, 2) if val == val else None      # paise -> Rs crore
    peak = float(close.max())
    pct_from_peak = round((float(close.iloc[-1]) / peak - 1) * 100, 1) if peak else None
    dma200 = close.tail(200).mean()
    above = int(float(close.iloc[-1]) > float(dma200)) if len(close) >= 200 else None
    return turnover, pct_from_peak, above
