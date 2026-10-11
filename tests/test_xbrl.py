"""Just enough XBRL for NSE's filings: contexts, facts, and the trap that segment rows reuse names."""
from pennystock.data import xbrl

DOC = """<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance">
<xbrli:context id="FY"><xbrli:period><xbrli:startDate>2025-04-01</xbrli:startDate>
<xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="Seg"><xbrli:period><xbrli:startDate>2025-04-01</xbrli:startDate>
<xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period><xbrli:scenario><xbrldi:explicitMember
dimension="a:SegmentsAxis">a:MobilesMember</xbrldi:explicitMember></xbrli:scenario></xbrli:context>
<xbrli:context id="END"><xbrli:period><xbrli:instant>2026-03-31</xbrli:instant></xbrli:period></xbrli:context>
<in-capmkt:RevenueFromOperations contextRef="FY" decimals="-5" unitRef="INR">488728000000
</in-capmkt:RevenueFromOperations>
<in-capmkt:RevenueFromOperations contextRef="Seg" unitRef="INR">99000000000</in-capmkt:RevenueFromOperations>
<in-capmkt:AuditorsFirmName contextRef="FY">S.N. DHAWAN &amp; CO LLP</in-capmkt:AuditorsFirmName>
</xbrli:xbrl>"""


def test_contexts_carry_period_length_and_whether_they_are_a_breakdown():
    ctx = xbrl.contexts(DOC)
    assert ctx["FY"].days == 365 and not ctx["FY"].dimensional
    assert ctx["Seg"].dimensional and ctx["END"].instant == "2026-03-31"


def test_facts_are_read_without_their_prefix_and_unescaped():
    facts = xbrl.facts(DOC)
    revenue = [f for f in facts if f.name == "RevenueFromOperations"]
    assert len(revenue) == 2 and xbrl.number(revenue[0]) == 488728000000.0 and revenue[0].unit == "INR"
    assert next(f.value for f in facts if f.name == "AuditorsFirmName") == "S.N. DHAWAN & CO LLP"
