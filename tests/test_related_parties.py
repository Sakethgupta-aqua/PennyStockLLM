"""Related-party transactions from the XBRL disclosure: classified by relationship, judged against revenue."""
import pytest

from pennystock.data.integrated_filings import RptDisclosure, RptRow
from pennystock.data.related_parties import classify, summarize_rpt


@pytest.mark.parametrize("text, cls", [
    ("Subsidiary", "subsidiary"),
    ("Wholly owned subsidiary", "subsidiary"),
    # a subsidiary OF someone else - here Dixon's joint venture - is outside its group (real DIXON row)
    ("Wholly Owned Subsidiary of Lightanium technologies Private Limited", "jv_associate"),
    ("Joint Venture", "jv_associate"),
    ("Entities over which holding company are able to exercise control/significant influence",
     "jv_associate"),
    ("Entity having significant influence", "group_or_partner"),
    ("Entity under common control", "group_or_partner"),
    ("Fellow subsidiary", "group_or_partner"),
    ("Enterprises over which KMP and their relatives exercise significant influence", "promoter_kmp_entity"),
    ("Promoter group company", "promoter_kmp_entity"),
    ("Executive Director", "people"),
    ("KMP", "people"),
    ("Trust for employees", "other"),
])
def test_relationships_are_classified(text, cls):
    assert classify(text) == cls


def _row(entity, counterparty, relationship, kind, amount, details=""):
    return RptRow(entity, counterparty, relationship, kind, details, amount, None)


DIXON = RptDisclosure("DIXON", "H2 FY26", "2025-10-01", "2026-03-31", [
    _row("Padget Electronics Private Limited", "Ismartu India Private Limited",
         "Entities over which holding company are able to exercise control/significant influence",
         "Sale of goods or services", 3209.9),
    _row("Dixon Electro Appliances Private Limited", "BHARTI AIRTEL LTD",
         "Entity having significant influence", "Sale of goods or services", 1224.67),
    _row("Dixon Technologies (India) Limited", "Padget Electronics Pvt. Ltd", "Subsidiary",
         "Any other transaction", 567.9, "Corporate guarantee given"),
    _row("Dixon Technologies (India) Limited", "Ms X", "Executive Director", "Remuneration", 5.2),
    _row("Dixon Technologies (India) Limited", "Lightanium Technologies Private Limited", "Joint Venture",
         "Any other transaction", 142.8, "Loan given")],
    "12-May-2026 23:38:14", "u/fy26s", 21182.1, "FY26 revenue minus the April-September half")


def test_outside_the_subsidiaries_is_judged_against_the_half_years_revenue():
    s = summarize_rpt(DIXON)
    assert s["period"] == "H2 FY26 (2025-10-01 to 2026-03-31)" and s["filed"] == "12-May-2026"
    assert s["outside_subsidiaries_cr"] == 4582.57                    # 3209.9 + 1224.67 + 5.2 + 142.8
    assert s["outside_subsidiaries_pct_of_revenue"] == 21.6           # 4582.57 / 21182.1
    assert s["promoter_kmp_entities_cr"] == 0.0 and s["by_class_cr"]["subsidiary"] == 567.9
    top = s["largest_outside_subsidiaries"][0]
    assert top["counterparty"] == "Ismartu India Private Limited" and top["class"] == "jv_associate"


def test_loans_and_guarantees_to_non_subsidiaries_are_named():
    loans = summarize_rpt(DIXON)["loans_guarantees_outside_subsidiaries"]
    assert [x["counterparty"] for x in loans] == ["Lightanium Technologies Private Limited"]
    assert loans[0]["type"] == "Loan given"


def test_a_subsidiary_of_the_listed_company_itself_stays_a_subsidiary():
    # SWIGGY H2 FY26: 42 rows say 'Subsidiary of Swiggy Limited' - the listed company, not another one
    assert classify("Subsidiary of Swiggy Limited", listed_word="swiggy") == "subsidiary"
    jv_subsidiary = "Wholly Owned Subsidiary of Lightanium technologies"
    assert classify(jv_subsidiary, listed_word="dixon") == "jv_associate"


@pytest.mark.parametrize("text", ["Controlling Authority",
                                  "Holding more than 10% of the equity shares of the company"])
def test_a_controlling_shareholder_is_its_group(text):
    # PSUs (NTPC, IRCTC) name the government as 'Controlling Authority'
    assert classify(text) == "group_or_partner"


def test_signed_amounts_count_by_size():
    # KALYANKJIL: a released personal guarantee is printed -158.14; a total cannot go negative
    d = RptDisclosure("KALYANKJIL", "H2 FY26", "2025-10-01", "2026-03-31", [
        _row("Kalyan Jewellers India Limited", "T.S.Kalyanaraman", "Promoter and Managing Director",
             "Any other transaction", -158.14, "Personal guarantees released"),
        _row("Kalyan Jewellers India Limited", "M/s Kalyan Textile",
             "Enterprises over which KMP are able to exercise significant influence", "Purchase of goods",
             3.07)],
        None, "u", 1000.0, "test")
    s = summarize_rpt(d)
    assert s["outside_subsidiaries_cr"] == 161.21 and s["promoter_kmp_entities_cr"] == 3.07
