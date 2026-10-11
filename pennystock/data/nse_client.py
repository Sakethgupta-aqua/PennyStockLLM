"""One place that knows how to talk to NSE.

NSE returns 403 to anything that does not look like a real browser tab. The exact recipe took
trial and error and is easy to get subtly wrong, so it lives here and nowhere else:

  1. a FULL Chrome header set - User-Agent alone is not enough; the sec-ch-ua / sec-ch-ua-platform
     / Sec-Fetch-* headers are all checked
  2. a Referer pointing at the SYMBOL'S OWN quote page (not the homepage)
  3. a priming GET to that same page first, so the session picks up NSE's cookies

Endpoints known to work with this (verified): corporate-pledgedata,
corporate-share-holdings-master, corporate-announcements, corporates-pit,
corporate-board-meetings, corporates-corporateActions, corporate-credit-rating.
Known NOT to work: quote-equity?section=corp_info and section=trade_info (still 403).
"""
from __future__ import annotations

import logging

from pennystock.config import settings

logger = logging.getLogger(__name__)

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def nse_session(symbol: str, timeout: int | None = None):
    """A requests.Session primed with cookies for `symbol`'s quote page."""
    import requests
    from urllib.parse import quote

    s = requests.Session()
    s.headers.update(HEADERS)
    # encoded: "symbol=M&MFIN" would reach NSE as symbol=M plus a stray parameter, and time out
    ref = f"{BASE}/get-quotes/equity?symbol={quote(symbol.upper(), safe='')}"
    s.headers["Referer"] = ref
    s.get(ref, timeout=timeout or settings.ownership_timeout_s)
    return s


def nse_api(session, path: str, timeout: int | None = None):
    """GET an /api/ path and return parsed JSON, or None on any failure."""
    try:
        r = session.get(f"{BASE}/api/{path}", timeout=timeout or settings.ownership_timeout_s)
        if r.status_code != 200:
            logger.warning(f"NSE {path.split('?')[0]} returned {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        logger.warning(f"NSE {path.split('?')[0]} failed: {type(e).__name__}: {e}")
        return None


class NseFeed:
    """NSE over one cookie session that is re-primed once when a request fails - NSE drops sessions
    without warning during long runs. `api(path)` -> parsed JSON or None; `get(url)` -> bytes or None."""

    def __init__(self, prime_symbol: str = "RELIANCE"):
        self._prime, self._s = prime_symbol, None

    def _session(self):
        if self._s is None:
            self._s = nse_session(self._prime)
        return self._s

    def api(self, path: str):
        import time
        for attempt in (1, 2):
            try:
                answer = nse_api(self._session(), path)
            except Exception as e:                       # priming itself can time out
                logger.warning(f"NSE session failed: {type(e).__name__}: {e}")
                answer = None
            if answer is not None:
                return answer
            self._s = None
            if attempt == 1:
                time.sleep(10)
        return None

    def get(self, url: str) -> bytes | None:
        import time
        for attempt in (1, 2):
            try:
                r = self._session().get(url, timeout=settings.filings_timeout_s)
                if r.status_code == 200:
                    return r.content
            except Exception as e:
                logger.warning(f"NSE file fetch failed: {type(e).__name__}: {e}")
            self._s = None
            if attempt == 1:
                time.sleep(5)
        return None
