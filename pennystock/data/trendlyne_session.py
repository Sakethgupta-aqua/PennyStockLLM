"""A requests session logged in to Trendlyne through the user's own exported browser cookie.

Trendlyne serves research PDFs and Stock Reports only to a logged-in session. Both fetch scripts
(fetch_midcap_reports.py, fetch_stock_reports.py) build their session here, so the cookie handling and
its help text live in one place. The cookie is the user's login: it is read, never stored or logged.
"""
from __future__ import annotations

from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
COOKIE_HELP = """
How to get cookies.txt:

  1. Log in to https://trendlyne.com in Chrome or Firefox.
  2. Install a cookie exporter extension ("Get cookies.txt LOCALLY" for Chrome,
     "cookies.txt" for Firefox) - both export the standard Netscape format.
  3. With trendlyne.com open, click the extension and export.
  4. Save the file as cookies.txt in the project root.

Alternatively, copy the single `Cookie:` request header from DevTools
(Network tab -> any trendlyne request -> Headers -> Request Headers) and pass it with
--cookie-header "sessionid=...; csrftoken=...".

The cookie is your own login. Keep it out of git - it grants access to your account.
"""


def load_cookies(session, path: Path | None, header: str | None) -> None:
    if header:
        session.headers["Cookie"] = header.strip()
        return
    if not path:
        return
    if not path.exists():
        raise SystemExit(f"cookie file not found: {path}\n{COOKIE_HELP}")
    from http.cookiejar import MozillaCookieJar

    jar = MozillaCookieJar(str(path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:                      # a header paste saved as cookies.txt
        raise SystemExit(f"could not read {path} as Netscape cookies ({e})\n{COOKIE_HELP}")
    session.cookies.update(jar)
