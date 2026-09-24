from pennystock.data.text_quality import junk_reason, looks_like_junk

# real chunks that were retrieved on live runs
INDEX_PAGE = ("90-101, 103-106, 109-111, 112 , 113-123, 124-133, 323-324, 333-351 Graham's concept "
              "of appropriate individual, 17-33 inflation and, 47-56 market fluctuations and, 191-193 "
              "monitoring, 184 negative approach to development of, 134-145, 146-156 overview about, "
              "1-11 positive approach to development of, 157-179, 180-184 rebalancing of, 193 Portnoy, "
              "Dave, 40 positive illusions, 354-355 Posner, Victor, 427n precious metals, 55")

BIBLIOGRAPHY = ("Franklin, Benjamin; The Autobiography of Benjamin Franklin ; Touchstone (December 23, "
                "2003). Galbraith, John Kenneth; A Short History of Financial Euphoria ; Penguin (July 1, "
                "1994). Graham, Benjamin and Dodd, David; Security Analysis: The Classic 1934 Edition ; "
                "McGraw-Hill (October 1, 1996). Greenblatt, Joel; The Little Book That Beats the Market ; "
                "Wiley (November 19, 2005). Hagstrom, Robert; The Warren Buffett Way ; Wiley (2005).")

GOOD_PROSE = ("Graham formally defines the margin of safety as the earnings yield or reciprocal of the "
              "price to earnings ratio minus the interest rate on investment grade bonds. The lower a "
              "stock's ratio, the higher the earnings yield and the greater the likelihood that the "
              "resulting number will exceed the bond yield. The lower interest rates go, the less it "
              "takes for an earnings yield to exceed the bond yield. Conversely, higher ratios and "
              "interest rates shrink the margin of safety for the patient investor.")

GOOD_WITH_NUMBERS = ("When news of regulatory restrictions surfaces, then we need to figure out whether "
                     "the effect will be temporary or permanent and decide to act accordingly. If the "
                     "broader market continues to be in a bad phase in addition to these regulatory "
                     "restrictions, then that would deal a double blow to the stock price. In such a "
                     "situation, the price will remain under pressure and one can see a drawdown of 40 "
                     "to 50 per cent or even more. Exiting from the investment will become difficult.")


def test_flags_index_page():
    assert looks_like_junk(INDEX_PAGE)


def test_flags_bibliography():
    assert looks_like_junk(BIBLIOGRAPHY)


def test_keeps_real_prose():
    assert junk_reason(GOOD_PROSE) is None


def test_keeps_prose_that_contains_numbers():
    assert junk_reason(GOOD_WITH_NUMBERS) is None


def test_flags_short_and_shouty():
    assert looks_like_junk("Chapter 4")
    assert looks_like_junk("BEHAVIORAL OBSTACLES TO VALUE INVESTING " * 8)
