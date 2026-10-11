"""The analyst's prompt must describe exactly the tools it is given - and say how many.

It said "FOURTEEN" while listing sixteen (2026-10-07): two tools were added without the count, and a model
told it has fourteen tools has a reason to stop before the last two.
"""
import re

from pennystock.agents.prompts import ANALYST_PROMPT
from pennystock.tools import analyst_tools

_WORDS = {12: "TWELVE", 13: "THIRTEEN", 14: "FOURTEEN", 15: "FIFTEEN", 16: "SIXTEEN", 17: "SEVENTEEN",
          18: "EIGHTEEN", 19: "NINETEEN", 20: "TWENTY"}


def test_the_analyst_is_told_how_many_tools_it_has_and_each_one_is_listed():
    listed = re.findall(r"\b\d+\. (\w+_\w+) - ", ANALYST_PROMPT)
    given = {t.name for t in analyst_tools}
    assert set(listed) == given and len(listed) == len(given)
    assert f"You have {_WORDS[len(given)]} tools" in ANALYST_PROMPT


def test_the_rules_quote_values_the_code_actually_emits():
    # the analyst is told what a statutory 0.0 pledge and the fact sheet's DATA GAPS section mean; renaming
    # either in code would leave the prompt describing something that no longer appears
    import inspect

    from pennystock.agents import fact_sheet
    from pennystock.data import ownership_providers
    assert "EVIDENCE PACK RULE" in ANALYST_PROMPT
    assert "DATA GAPS" in ANALYST_PROMPT and "### DATA GAPS" in inspect.getsource(fact_sheet)
    assert "pledge_source 'nse_shp_xbrl'" in ANALYST_PROMPT
    assert '"nse_shp_xbrl"' in inspect.getsource(ownership_providers)
    assert "FILED figure" in ANALYST_PROMPT and "ALTERNATIVES" in ANALYST_PROMPT
