import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_jackpot_backtest import run


# Full-history backtest against live CSVs takes 30-60 min on slow CI. Skip by
# default; opt in locally with SLOW_TESTS=1.
SLOW_ENABLED = os.environ.get("SLOW_TESTS") == "1"


def _fake_load_draws(csv_path: str, game_key: str):
    """Produce 40 synthetic draws per game so the backtest runs quickly."""
    rng = random.Random(hash(game_key) & 0xFFFFFFFF)
    dates = [f"2025-01-{d:02d}" for d in range(1, 32)] + [f"2025-02-{d:02d}" for d in range(1, 10)]
    if game_key == "joker":
        mains = [sorted(rng.sample(range(1, 46), 5)) for _ in dates]
        bonuses = [rng.randint(1, 20) for _ in dates]
        return mains, dates, bonuses
    pool_max = 49 if game_key == "loto_649" else 40
    k = 6
    mains = [sorted(rng.sample(range(1, pool_max + 1), k)) for _ in dates]
    return mains, dates, None


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


class TestRunJackpotBacktestSynthetic(unittest.TestCase):
    """Fast smoke test — exercises the CLI pipeline on a mocked data source."""

    def test_emits_well_formed_report(self):
        with patch("scripts.run_jackpot_backtest._load_draws", side_effect=_fake_load_draws):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "report.md"
                run(
                    output_path=out,
                    seed=42,
                    warmup=30,
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
