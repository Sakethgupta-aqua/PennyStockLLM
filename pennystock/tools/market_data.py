"""LIVE-DATA tools (the structured half of the data spine).

Tools, one per intent (the doc's 'route by intent' rule):
  get_indian_stock_data  -> live price snapshot          (MarketDataAgent)
  get_stock_fundamentals -> snapshot valuation & health  (FinancialAnalystAgent)
  get_price_history      -> computed technical indicators (FinancialAnalystAgent)
  get_financials         -> 5-year statement trends      (FinancialAnalystAgent)
  get_corporate_actions  -> splits & dividends           (FinancialAnalystAgent)
  get_shareholding       -> promoter/FII/DII + PLEDGE    (FinancialAnalystAgent)
  get_forensics          -> Piotroski/Altman/Beneish     (FinancialAnalystAgent)
  get_valuation_context  -> peer median + own P/E band   (FinancialAnalystAgent)
  get_corporate_events   -> announcements/insider/rating (FinancialAnalystAgent)
  get_management_commentary -> earnings-call language     (FinancialAnalystAgent)

Each resolves a name/ticker via the resolver, then delegates to a swappable provider.
Blocking network calls run in a thread so the LangGraph event loop stays responsive.
"""
import asyncio
import logging
from dataclasses import asdict

from langchain_core.tools import tool

from pennystock.config import settings
from pennystock.data.fundamentals import snapshot_warnings, summarize_financials
from pennystock.data.indicators import detect_structural_break, summarize
from pennystock.data.annual_report import fetch_notes, summarize_notes
from pennystock.data.events import NseEventsProvider, summarize_events
from pennystock.data.forensics import summarize_forensics
from pennystock.data.transcripts import fetch_transcripts, summarize_transcripts
from pennystock.data.ownership import summarize_ownership
from pennystock.data.ownership_providers import get_ownership
from pennystock.data.peers import compare_to_peers
from pennystock.data.providers import get_provider
from pennystock.data.valuation import valuation_bands
from pennystock.data.resolver import resolve_symbol

logger = logging.getLogger(__name__)


def _resolve_ticker(symbol: str, exchange: str) -> tuple[str, str | None]:
    """Resolve a name/ticker to a fully-qualified yfinance ticker + the official name."""
    match = resolve_symbol(symbol)
    if match:
        logger.info(f"Resolved '{symbol}' -> {match['symbol']} "
                    f"({match['company_name']}, {match['how']} {match['score']})")
        symbol, name = match["symbol"], match["company_name"]
    else:
        name = None
        logger.warning(f"Could not confidently resolve '{symbol}' to a listed NSE symbol.")
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol.upper()}{suffix}", name


@tool
async def get_indian_stock_data(symbol: str, exchange: str = "NSE") -> str:
    """Fetches the live PRICE SNAPSHOT for an Indian stock (NSE/BSE).

    `symbol` accepts EITHER an exact ticker (e.g. SUZLON) OR the company name
    (e.g. "Kaynes Technology") - names are resolved automatically."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        snap = await asyncio.to_thread(get_provider().get_snapshot, ticker)
        if snap.price is None:
            return (f"Error: Could not find live data for '{ticker}'. "
                    f"If you guessed the ticker, retry with the full company name instead.")
        data = asdict(snap)
        data["company"] = data.get("company") or name        # visibility guardrail
        return str(data)
    except Exception as e:
        logger.warning(f"snapshot fetch failed for {symbol}: {e}")
        return f"Error fetching data for {symbol}."


@tool
async def get_stock_fundamentals(symbol: str, exchange: str = "NSE") -> str:
    """Fetches valuation & financial-health FUNDAMENTALS: P/E (trailing & forward), EPS,
    book value, price-to-book, debt-to-equity, ROE, profit margin, dividend yield, 52-week
    range, beta, sector/industry.

    Use for VALUE-INVESTING / Graham questions (margin of safety, is it cheap?). Any field
    returned as None is genuinely unavailable - say so, do NOT invent it. `symbol` may be a
    company name or a ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        fund = await asyncio.to_thread(get_provider().get_fundamentals, ticker)
        data = asdict(fund)
        data["company"] = data.get("company") or name
        warnings = snapshot_warnings(fund)
        if warnings:
            data["data_warnings"] = warnings
        return str(data)
    except Exception as e:
        logger.warning(f"fundamentals fetch failed for {symbol}: {e}")
        return f"Error fetching fundamentals for {symbol}."


@tool
async def get_price_history(symbol: str, exchange: str = "NSE", period: str | None = None) -> str:
    """Fetches historical prices and returns COMPUTED technical indicators: 50/200-day moving
    averages, RSI(14), trend, support & resistance, period high/low, % from high, avg volume.

    Use for TECHNICAL-ANALYSIS questions (levels to watch, overbought?). Indicators are
    computed in code, not estimated. `symbol` may be a name or ticker; `period` like
    '6mo', '1y', '5y'."""
    try:
        ticker, _ = _resolve_ticker(symbol, exchange)
        df = await asyncio.to_thread(
            get_provider().get_price_history,
            ticker, period or settings.history_period, settings.history_interval)
        result = summarize(df, settings.sr_lookback, settings.rsi_window, settings.history_recent_days)
        recent = df.tail(settings.history_recent_days) if df is not None and not df.empty else df
        brk = detect_structural_break(recent, settings.structural_break_pct)   # recent window only
        if brk:
            result["data_warning"] = brk
        return str(result)
    except Exception as e:
        logger.warning(f"history fetch failed for {symbol}: {e}")
        return f"Error fetching price history for {symbol}."


@tool
async def get_financials(symbol: str, exchange: str = "NSE") -> str:
    """Fetches 5 YEARS of financial statements and returns a COMPUTED trend summary: revenue +
    CAGR, net income + how many years were profitable, EPS, total-debt trend, latest
    debt-to-equity, free cash flow. Use to judge earnings CONSISTENCY, growth, leverage over
    time and cyclicality (Graham's earnings-stability test). `symbol` may be a name or ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        fin = await asyncio.to_thread(get_provider().get_financials, ticker, settings.financials_years)
        summary = summarize_financials(fin)
        summary["company"] = name
        return str(summary)
    except Exception as e:
        logger.warning(f"financials fetch failed for {symbol}: {e}")
        return f"Error fetching financials for {symbol}."


@tool
async def get_corporate_actions(symbol: str, exchange: str = "NSE") -> str:
    """Fetches recent CORPORATE ACTIONS - stock splits and dividends. NOTE: this source does
    NOT record demergers/spin-offs; for those, a 'data_warning' from get_price_history (a large
    single-day gap) is the better signal. `symbol` may be a name or ticker."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        ca = await asyncio.to_thread(get_provider().get_corporate_actions, ticker)
        return str({"company": name, "recent_splits": ca.splits, "recent_dividends": ca.dividends})
    except Exception as e:
        logger.warning(f"corporate actions fetch failed for {symbol}: {e}")
        return f"Error fetching corporate actions for {symbol}."


@tool
async def get_shareholding(symbol: str, exchange: str = "NSE") -> str:
    """Fetches the OWNERSHIP picture: promoter holding % and its trend, FII/DII (institutional)
    holding and trend, public holding, shareholder count, PROMOTER PLEDGE % with a risk band, and
    explicit governance risk_flags. Includes up to 8 quarters of history.

    Use for governance / red-flag questions, especially on small and micro-caps: promoter
    pledging, promoters selling down, institutions exiting.

    `pledge_pct_of_promoter_holding` is the market-convention figure the risk band uses;
    `pledge_pct_of_total_shares` is NSE's headline number and is much smaller - quote the FORMER.
    If pledge is None it is genuinely UNKNOWN (not filed / source unavailable) - say so explicitly
    and treat it as unverified; NEVER assume it is zero."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]                     # screener uses the bare NSE symbol
        own = await asyncio.to_thread(get_ownership, base)
        summary = summarize_ownership(own, settings.ownership_lookback_quarters)
        summary["company"] = name
        return str(summary)
    except Exception as e:
        logger.warning(f"shareholding fetch failed for {symbol}: {e}")
        return f"Error fetching shareholding for {symbol}."


@tool
async def get_forensics(symbol: str, exchange: str = "NSE") -> str:
    """FORENSIC ACCOUNTING screen - "can I trust these numbers, and can this company survive?".

    Returns the Piotroski F-Score (0-9, fundamental improvement; 8-9 strong, <=3 weak), the Altman
    Z-Score (bankruptcy distress; >2.99 safe, <1.81 distress), the Beneish M-Score (earnings
    manipulation; above -1.78 is a flag), cumulative cash conversion (operating cash flow vs
    reported profit - profit that never becomes cash is paper), receivable/inventory/payable days
    with year-on-year change, and an explicit red_flags list.

    Run this BEFORE judging valuation: if the accounts cannot be trusted or the balance sheet is
    breaking, no multiple is cheap enough. Report the caveats the tool returns - Altman is
    unreliable for banks/NBFCs and Beneish is calibrated on US filers."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        prov = get_provider()
        fin = await asyncio.to_thread(prov.get_financials, ticker, settings.financials_years)
        snap = await asyncio.to_thread(prov.get_snapshot, ticker)
        mcap_cr = (snap.market_cap / 1e7) if snap.market_cap else None
        out = summarize_forensics(fin, mcap_cr)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"forensics failed for {symbol}: {e}")
        return f"Error computing forensics for {symbol}."


@tool
async def get_valuation_context(symbol: str, exchange: str = "NSE") -> str:
    """VALUATION IN CONTEXT - answers "expensive versus WHAT?".

    Two references a bare multiple can never provide:
      peers       - median P/E, P/B, ROE, debt and margin for the company's own industry, plus
                    where this stock ranks (percentile, higher = better)
      own history - its P/E band over the last few fiscal years vs today's multiple

    ALWAYS call this before calling a stock cheap or expensive. A P/E of 33 is high for a metals
    company and low for an FMCG one; a P/E of 13 can still be that stock's own 4-year peak."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        prov = get_provider()
        peers = await asyncio.to_thread(compare_to_peers, base)
        fin = await asyncio.to_thread(prov.get_financials, ticker, settings.financials_years)
        hist = await asyncio.to_thread(prov.get_price_history, ticker,
                                       settings.history_period, settings.history_interval)
        fund = await asyncio.to_thread(prov.get_fundamentals, ticker)
        bands = valuation_bands(fin, hist, fund.pe_trailing, fund.price_to_book, fund.book_value)
        return str({"company": name, "vs_peers": peers, "vs_own_history": bands})
    except Exception as e:
        logger.warning(f"valuation context failed for {symbol}: {e}")
        return f"Error building valuation context for {symbol}."


@tool
async def get_corporate_events(symbol: str, exchange: str = "NSE") -> str:
    """CORPORATE EVENTS & NEWS FLOW from NSE's statutory filings - what moves a stock before the
    numbers show it.

    Returns MATERIAL announcements only (routine compliance filings are screened out): auditor
    resignations, defaults/insolvency, fraud or enforcement action, credit-rating actions, pledge
    events, senior-management exits, litigation, fund raises, M&A. Plus SEBI insider-trading
    disclosures with a buy/sell tally, and the latest credit rating and action.

    An auditor resigning mid-term is never routine and is among the highest-signal warnings in
    Indian markets. If no announcement data comes back, say event risk is UNCHECKED, not clean."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        ev = await asyncio.to_thread(NseEventsProvider().fetch, base)
        out = summarize_events(ev)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"corporate events failed for {symbol}: {e}")
        return f"Error fetching corporate events for {symbol}."


@tool
async def get_management_commentary(symbol: str, exchange: str = "NSE") -> str:
    """EARNINGS-CALL LANGUAGE ANALYSIS - what management SAYS, and what they avoid saying.

    Downloads the last few quarterly earnings-call transcripts from NSE and returns:
      tone_latest / tone_previous - Loughran-McDonald financial sentiment on management's own
        words, including `hedging_ratio` (weak-modal share). "We WILL commission the plant"
        becoming "we MAY commission it" is a downgrade nobody announces.
      evasion - `explicit_non_answers` is the reliable number (management literally declined to
        answer); `possible_deflections` is a noisy heuristic to skim, NOT to score.
      guidance_latest vs guidance_previous - forward-looking sentences from each call. COMPARE
        THEM: a target stated last quarter and absent this quarter is the highest-signal item in
        any transcript. Quote management's own words when you make the point.
      new_risk_language - negative/uncertainty words used this quarter but not last.

    Use for management-quality, guidance and credibility questions. If no transcript is found, say
    management commentary is UNCHECKED - not that it is clean."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        tx = await asyncio.to_thread(fetch_transcripts, base, settings.transcript_max_quarters)
        out = await asyncio.to_thread(summarize_transcripts, tx)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"management commentary failed for {symbol}: {e}")
        return f"Error fetching management commentary for {symbol}."


@tool
async def get_annual_report_notes(symbol: str, exchange: str = "NSE") -> str:
    """OFF-BALANCE-SHEET & PROMOTER-CONFLICT CHECK, read from the annual report itself.

    Two notes at the back of the annual report that no free structured feed carries, and that
    Indian analysts have checked before buying for forty-odd years:

      contingent_liabilities - obligations that are NOT on the balance sheet: guarantees given for
        group companies, disputed tax and regulatory demands. A company can look unlevered on the
        face of the balance sheet while guaranteeing the debts of the wider promoter group. Judge
        the totals AGAINST NET WORTH; a large ratio is real risk sitting off the books.
      related_party_transactions - the classic route by which value leaves a listed company and
        arrives at a promoter-owned one. Judge SIZE against revenue (a long-standing threshold is
        roughly 10%) and, just as importantly, the REASON given for the dealing.

    `figures_mentioned` are numbers scraped from a PDF table that does NOT survive extraction
    cleanly. They are HINTS to verify against `source_url` - never quote them as parsed totals, and
    say they are indicative if you cite them. If nothing is returned, report off-balance-sheet
    exposure and related-party dealings as UNCHECKED - never as clean."""
    try:
        ticker, name = _resolve_ticker(symbol, exchange)
        base = ticker.rsplit(".", 1)[0]
        notes = await asyncio.to_thread(fetch_notes, base)
        # net worth and revenue are the yardsticks the two notes are judged against, so fetch them
        # here rather than hoping the analyst carried them over from another tool call
        net_worth = revenue = None
        try:
            fin = await asyncio.to_thread(get_provider().get_financials, ticker,
                                          settings.financials_years)
            net_worth = fin.equity_cr[0] if fin.equity_cr else None
            revenue = fin.revenue_cr[0] if fin.revenue_cr else None
        except Exception as e:                       # the notes are still worth returning alone
            logger.warning(f"net-worth/revenue context unavailable for {symbol}: {e}")
        out = await asyncio.to_thread(summarize_notes, notes, net_worth, revenue)
        out["company"] = name
        return str(out)
    except Exception as e:
        logger.warning(f"annual report notes failed for {symbol}: {e}")
        return f"Error fetching annual report notes for {symbol}."
