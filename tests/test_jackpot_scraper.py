import unittest
from pathlib import Path
from unittest.mock import patch

from shared.jackpot_scraper import fetch_jackpots, parse_jackpots


class TestParseJackpots(unittest.TestCase):
    def setUp(self) -> None:
        self.html = Path("tests/fixtures/loto_homepage_with_jackpots.html").read_text(
            encoding="utf-8"
        )

    def test_parses_all_three_games(self):
        jackpots = parse_jackpots(self.html)
        self.assertEqual(jackpots["joker"], 65230704.97)
        self.assertEqual(jackpots["loto_649"], 16202003.12)
        self.assertEqual(jackpots["loto_540"], 619353.00)

    def test_missing_labels_yield_none(self):
        jackpots = parse_jackpots("<html><body>no labels here</body></html>")
        self.assertIsNone(jackpots["joker"])
        self.assertIsNone(jackpots["loto_649"])
        self.assertIsNone(jackpots["loto_540"])

    def test_handles_label_without_amount_nearby(self):
        # Label present but no currency-shaped number within window
        html = "<h2>REPORT JOKER</h2>" + "x" * 5000 + "65.230.704,97"
        jackpots = parse_jackpots(html)
        self.assertIsNone(jackpots["joker"])

    def test_european_to_float_via_real_values(self):
        html = "REPORT JOKER</h2> ... 1.234.567,89 ..."
        jackpots = parse_jackpots(html)
        self.assertEqual(jackpots["joker"], 1234567.89)


class TestFetchJackpots(unittest.TestCase):
    def test_network_failure_returns_none_per_game(self):
        with patch("shared.jackpot_scraper.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("no network")
            jackpots = fetch_jackpots()
            self.assertIsNone(jackpots["joker"])
            self.assertIsNone(jackpots["loto_649"])
            self.assertIsNone(jackpots["loto_540"])

    def test_successful_fetch_parses(self):
        html = Path("tests/fixtures/loto_homepage_with_jackpots.html").read_text(
            encoding="utf-8"
        )

        class FakeResp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_args):
                return False

            def read(self_inner):
                return html.encode("utf-8")

        with patch("shared.jackpot_scraper.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeResp()
            jackpots = fetch_jackpots()
            self.assertEqual(jackpots["joker"], 65230704.97)
            self.assertEqual(jackpots["loto_649"], 16202003.12)
            self.assertEqual(jackpots["loto_540"], 619353.00)


if __name__ == "__main__":
    unittest.main()
