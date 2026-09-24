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

    s = requests.Session()
    s.headers.update(HEADERS)
    ref = f"{BASE}/get-quotes/equity?symbol={symbol.upper()}"
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
