import os
import tempfile
import unittest
from pathlib import Path

from scripts.run_jackpot_backtest import run


# The full-history backtest walks thousands of iterations and takes 30-60+ min
# on slow CI runners. Skip by default; opt in locally with SLOW_TESTS=1.
SLOW_ENABLED = os.environ.get("SLOW_TESTS") == "1"


@unittest.skipUnless(SLOW_ENABLED, "set SLOW_TESTS=1 to run the full-history backtest")
class TestRunJackpotBacktestFull(unittest.TestCase):
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
            self.assertIn("IndependentBuilder", text)


class TestRunJackpotBacktestQuick(unittest.TestCase):
    """Fast smoke test — uses a high warmup so each game gets few iterations."""

    def test_emits_well_formed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            run(
                output_path=out,
                seed=42,
                warmup=1100,  # leaves <200 test draws per game
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
            self.assertIn("P(>=4)", text)
            self.assertIn("skewness", text)


if __name__ == "__main__":
    unittest.main()
