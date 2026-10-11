"""NSE's session recipe: the symbol must reach NSE intact, ampersand and all."""
import requests

from pennystock.data import nse_client


class FakeSession:
    def __init__(self):
        self.headers, self.urls = {}, []

    def get(self, url, timeout=None):
        self.urls.append(url)


def test_an_ampersand_symbol_is_url_encoded_in_the_priming_request(monkeypatch):
    """Unencoded, 'symbol=M&MFIN' reaches NSE as symbol=M plus a stray parameter, and the priming
    request times out - M&M Financial and GE Vernova T&D could never be fetched."""
    fake = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: fake)
    nse_client.nse_session("M&MFIN")
    assert fake.urls == ["https://www.nseindia.com/get-quotes/equity?symbol=M%26MFIN"]
    assert fake.headers["Referer"] == "https://www.nseindia.com/get-quotes/equity?symbol=M%26MFIN"
