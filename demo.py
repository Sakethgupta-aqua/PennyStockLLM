"""Run one analysis end-to-end and print the conversation.

  python demo.py "your question here"     # ask about any stock
  python demo.py                          # run the built-in example question

Every run saves a trace under traces/ automatically; score them with scripts/check_traces.py.
"""
import asyncio
import sys

from pennystock.pipeline import analyze

EXAMPLE = """ I'm thinking of investing in Dixon Technologies for the next 3–5 years. What's the current price, and is its P/E too expensive compared with its peers and its own history? How have its revenue and profit grown over the last five years, and can I trust the numbers? Is the promoter reducing their stake or pledging shares, and has there been any insider selling, auditor change or credit rating action? Has it paid a dividend or given a bonus? What did management guide in the last concall, what has the company disclosed in its filings and investor presentation, and is there anything worrying in the annual report, like related-party deals? What target price do brokers give, and what reasons do their research reports give? What is Trendlyne's DVM score, and where does the chart stand against the 200-day moving average? Finally, does it pass Graham's margin of safety, and if not, which stocks would you shortlist instead?"""
args = sys.argv[1:]
if args and args[0] in ("-h", "--help"):
    print(__doc__)
    raise SystemExit(0)

query = " ".join(args).strip() or EXAMPLE

result = asyncio.run(analyze(query))

for msg in result["messages"]:
    print(f"\n--- {type(msg).__name__} ---\n{msg.content}")
