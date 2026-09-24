"""Run one analysis end-to-end and print the conversation.

  python demo.py "your question here"     # ask about any stock
  python demo.py                          # run the built-in example question

Every run saves a trace under traces/ automatically; score them with scripts/check_traces.py.
"""
import asyncio
import sys

from pennystock.pipeline import analyze

EXAMPLE = """I have followed the Agarwal group since the Sterlite days, and six decades in this market has taught me to hold two thoughts about it at once: the assets are genuinely world class, and the balance sheet has always been somebody else's problem. Vedanta is on my table this week and something about it does not add up, so I want it pulled apart properly rather than admired. Pull me its current quote on the NSE. First, before I look at a single valuation number — can I trust these accounts? Walk me through the forensic picture: earnings quality, cash conversion, working capital, and whether the solvency position is anywhere near the distress zone. I have seen too many resource companies report profits they never collected. Second, the operating record — the last five years of revenue and earnings, what has happened to the debt, and whether the cash generation actually supports the story. Third, the owners — what is the promoter holding, how much of it sits under pledge, and are the institutions adding or leaving? A leveraged promoter in a commodity business is the oldest trap in this market. Fourth, the news flow — any credit rating action, any default, any auditor trouble, and what the insiders themselves have been doing with their own money. Fifth, management — what did they promise on the last earnings call, what are they promising now, and has the guidance quietly changed between the two? I want their own words. Sixth, valuation — where is it trading on earnings and on book value, how does that compare against its peers and against its own history, and is the balance sheet leveraged enough to make a cheap multiple meaningless? And the tape — where does it sit against its 200-day, what is momentum saying, and which support and resistance levels should I respect. Finally, the judgement: reconcile the contradictions for me rather than papering over them, tell me what the accumulated wisdom says about the psychology of buying a high-debt commodity house in the middle of a restructuring, and settle it plainly — accumulate, avoid, or wait for a better level."""

args = sys.argv[1:]
if args and args[0] in ("-h", "--help"):
    print(__doc__)
    raise SystemExit(0)

query = " ".join(args).strip() or EXAMPLE

result = asyncio.run(analyze(query))

for msg in result["messages"]:
    print(f"\n--- {type(msg).__name__} ---\n{msg.content}")
