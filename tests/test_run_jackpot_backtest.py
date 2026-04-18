import tempfile
import unittest
from pathlib import Path

from scripts.run_jackpot_backtest import run


class TestRunJackpotBacktest(unittest.TestCase):
    def test_emits_report_with_all_builders_and_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            run(
                output_path=out,
                seed=42,
                warmup=50,
                jackpot=1_000_000.0,
            )
            self.assertTrue(out.exists())
            text = out.read_text()
            self.assertIn("# Jackpot Backtest Report", text)
            self.assertIn("Joker", text)
            self.assertIn("Loto 6/49", text)
            self.assertIn("Loto 5/40", text)
            self.assertIn("IndependentBuilder", text)
            self.assertIn("CoreShareBuilder", text)
            self.assertIn("WheelBuilder", text)
            self.assertIn("median_roi", text)
            self.assertIn("p_best_match_ge_4", text)
            self.assertIn("skewness", text)


if __name__ == "__main__":
    unittest.main()
