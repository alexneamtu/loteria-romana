"""Tests for cross-game correlation analysis."""

from __future__ import annotations

import random
import unittest
from datetime import date, timedelta

from shared.cross_game_analysis import run_cross_game_analysis


class TestCrossGameAnalysis(unittest.TestCase):
    """Tests for run_cross_game_analysis."""

    def test_independent_games_pass(self) -> None:
        """Two games generated with different seeds should show no correlation."""
        base_date = date(2023, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]

        rng_a = random.Random(42)
        game_a = [(d, sorted(rng_a.sample(range(1, 46), 5))) for d in dates]

        rng_b = random.Random(999)
        game_b = [(d, sorted(rng_b.sample(range(1, 50), 6))) for d in dates]

        game_draws = {
            "joker": game_a,
            "loto649": game_b,
        }

        results = run_cross_game_analysis(game_draws, significance=0.01)
        self.assertTrue(len(results) > 0)
        for result in results:
            if result.passed is not None:
                self.assertTrue(
                    result.passed,
                    f"{result.test_name} unexpectedly failed: {result.summary}",
                )

    def test_correlated_games_detected(self) -> None:
        """Games with correlated sums should be detected by sum_correlation."""
        base_date = date(2023, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]

        rng = random.Random(42)
        base_draws = [sorted(rng.sample(range(1, 46), 5)) for _ in dates]

        game_a = [(d, draw) for d, draw in zip(dates, base_draws)]
        # Shift each number by +1 to create highly correlated sums
        game_b = [
            (d, [min(n + 1, 45) for n in draw])
            for d, draw in zip(dates, base_draws)
        ]

        game_draws = {
            "game_a": game_a,
            "game_b": game_b,
        }

        results = run_cross_game_analysis(game_draws, significance=0.01)
        sum_correlation_results = [
            r for r in results if r.test_name == "sum_correlation"
        ]
        self.assertTrue(len(sum_correlation_results) > 0)
        # At least one sum_correlation test should detect the correlation
        detected = any(r.passed is False for r in sum_correlation_results)
        self.assertTrue(
            detected,
            "Sum correlation between correlated games was not detected",
        )

    def test_no_overlapping_dates(self) -> None:
        """Games with no overlapping dates should be handled gracefully."""
        rng_a = random.Random(42)
        rng_b = random.Random(99)

        dates_a = [date(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        dates_b = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]

        game_a = [(d, sorted(rng_a.sample(range(1, 46), 5))) for d in dates_a]
        game_b = [(d, sorted(rng_b.sample(range(1, 50), 6))) for d in dates_b]

        game_draws = {
            "joker": game_a,
            "loto649": game_b,
        }

        results = run_cross_game_analysis(game_draws, significance=0.01)
        # All results should be inconclusive (passed=None) since no common dates
        for result in results:
            self.assertIsNone(
                result.passed,
                f"{result.test_name} should be inconclusive with no overlap",
            )

    def test_single_game_returns_empty(self) -> None:
        """A single game input should return an empty results list."""
        base_date = date(2023, 1, 1)
        rng = random.Random(42)
        game_a = [
            (base_date + timedelta(days=i), sorted(rng.sample(range(1, 46), 5)))
            for i in range(50)
        ]

        game_draws = {"joker": game_a}

        results = run_cross_game_analysis(game_draws, significance=0.01)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
