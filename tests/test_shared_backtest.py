import unittest
import random

from shared.backtest_base import (
    PrizeTier,
    BacktestResult,
    Backtester,
    CrossValidator,
    wilson_score_interval,
    compute_max_drawdown,
)
from shared.stats import DeltaStrategy, HotColdStrategy


class TestUtilityFunctions(unittest.TestCase):
    def test_wilson_score_interval_basic(self):
        # 50% success rate with 100 trials
        lower, upper = wilson_score_interval(50, 100)
        self.assertGreater(lower, 0.35)
        self.assertLess(upper, 0.65)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)

    def test_wilson_score_interval_zero_trials(self):
        lower, upper = wilson_score_interval(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)

    def test_wilson_score_interval_all_successes(self):
        lower, upper = wilson_score_interval(100, 100)
        self.assertGreater(lower, 0.9)
        self.assertAlmostEqual(upper, 1.0, places=10)

    def test_compute_max_drawdown_no_losses(self):
        history = [True, True, True, True, True]
        self.assertEqual(compute_max_drawdown(history), 0)

    def test_compute_max_drawdown_all_losses(self):
        history = [False, False, False, False, False]
        self.assertEqual(compute_max_drawdown(history), 5)

    def test_compute_max_drawdown_mixed(self):
        history = [True, False, False, True, False, False, False, True]
        self.assertEqual(compute_max_drawdown(history), 3)


class TestPrizeTier(unittest.TestCase):
    def test_prize_tier_creation(self):
        tier = PrizeTier(
            name="3_match",
            matches_required=3,
            bonus_required=False,
            payout=10.0,
        )
        self.assertEqual(tier.name, "3_match")
        self.assertEqual(tier.matches_required, 3)
        self.assertFalse(tier.bonus_required)
        self.assertEqual(tier.payout, 10.0)


class TestBacktestResult(unittest.TestCase):
    def test_backtest_result_roi(self):
        result = BacktestResult(
            strategy_name="test",
            total_draws=100,
            total_tickets=100,
            expected_value=150.0,
        )
        # ROI = (150 - 100) / 100 * 100 = 50%
        self.assertEqual(result.roi, 50.0)

    def test_backtest_result_roi_zero_tickets(self):
        result = BacktestResult(
            strategy_name="test",
            total_draws=0,
            total_tickets=0,
        )
        self.assertEqual(result.roi, 0.0)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.backtester = Backtester(
            number_pool=45,
            numbers_to_pick=5,
        )

    def test_default_prize_tiers(self):
        tiers = self.backtester.prize_tiers
        self.assertTrue(len(tiers) > 0)
        # Should have 3_match, 4_match, 5_match, jackpot
        tier_names = {t.name for t in tiers}
        self.assertIn("3_match", tier_names)
        self.assertIn("jackpot", tier_names)

    def test_evaluate_ticket_no_matches(self):
        ticket = ([1, 2, 3, 4, 5], 10)
        winning = [10, 20, 30, 40, 45]
        results = self.backtester.evaluate_ticket(ticket, winning, 15)

        # No matches, all tiers should be False
        for tier_name, won in results.items():
            self.assertFalse(won, f"Tier {tier_name} should not win")

    def test_evaluate_ticket_three_matches(self):
        ticket = ([1, 2, 3, 40, 45], 10)
        winning = [1, 2, 3, 20, 30]  # 3 matches
        results = self.backtester.evaluate_ticket(ticket, winning, 15)

        self.assertTrue(results["3_match"])
        self.assertFalse(results["4_match"])
        self.assertFalse(results["5_match"])

    def test_evaluate_ticket_all_matches_no_bonus(self):
        ticket = ([1, 2, 3, 4, 5], 10)
        winning = [1, 2, 3, 4, 5]
        results = self.backtester.evaluate_ticket(ticket, winning, 15)  # Wrong bonus

        self.assertTrue(results["3_match"])
        self.assertTrue(results["4_match"])
        self.assertTrue(results["5_match"])
        self.assertFalse(results["jackpot"])  # Bonus doesn't match

    def test_evaluate_ticket_jackpot(self):
        ticket = ([1, 2, 3, 4, 5], 10)
        winning = [1, 2, 3, 4, 5]
        results = self.backtester.evaluate_ticket(ticket, winning, 10)  # Correct bonus

        self.assertTrue(results["jackpot"])

    def test_backtest_basic(self):
        # Create mock draws within valid pool (1-45)
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(100)
        ]

        strategy = DeltaStrategy(45, 5)
        rng = random.Random(42)

        result = self.backtester.backtest(
            strategy=strategy,
            draws=draws,
            tickets_per_draw=1,
            train_window=50,
            rng=rng,
        )

        self.assertEqual(result.strategy_name, "delta")
        self.assertEqual(result.total_draws, 50)  # 100 - 50
        self.assertGreaterEqual(result.total_tickets, 50)
        self.assertIsInstance(result.win_rate, float)
        self.assertIsInstance(result.max_drawdown, int)


class TestCrossValidator(unittest.TestCase):
    def test_split_basic(self):
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(200)
        ]
        cv = CrossValidator(n_splits=5, min_train_size=50)
        splits = cv.split(draws)

        self.assertEqual(len(splits), 5)
        for train, test in splits:
            self.assertGreaterEqual(len(train), 50)
            self.assertGreater(len(test), 0)

    def test_split_insufficient_data(self):
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(30)
        ]
        cv = CrossValidator(n_splits=5, min_train_size=50)
        splits = cv.split(draws)

        # Should return empty when insufficient data
        self.assertEqual(len(splits), 0)

    def test_evaluate_basic(self):
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(200)
        ]
        cv = CrossValidator(n_splits=3, min_train_size=50)
        backtester = Backtester(45, 5)
        strategy = DeltaStrategy(45, 5)
        rng = random.Random(42)

        results = cv.evaluate(
            strategy=strategy,
            draws=draws,
            backtester=backtester,
            rng=rng,
        )

        self.assertEqual(results["n_folds"], 3)
        self.assertIn("avg_win_rate", results)
        self.assertIn("total_tickets", results)


class TestBacktesterEVScoring(unittest.TestCase):
    def setUp(self):
        self.tiers = [
            PrizeTier(name="3_match", matches_required=3, payout=10.0),
            PrizeTier(name="4_match", matches_required=4, payout=100.0),
            PrizeTier(name="5_match", matches_required=5, payout=10000.0),
        ]
        self.backtester = Backtester(
            number_pool=45,
            numbers_to_pick=5,
            prize_tiers=self.tiers,
        )

    def test_ev_per_ticket_computed(self):
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(100)
        ]
        strategy = DeltaStrategy(45, 5)
        result = self.backtester.backtest(
            strategy=strategy,
            draws=draws,
            train_window=50,
            rng=random.Random(42),
        )
        self.assertIsInstance(result.ev_per_ticket, float)
        self.assertGreaterEqual(result.ev_per_ticket, 0.0)

    def test_ev_per_ticket_zero_when_no_wins(self):
        result = BacktestResult(
            strategy_name="test",
            total_draws=100,
            total_tickets=100,
            expected_value=0.0,
        )
        self.assertEqual(result.ev_per_ticket, 0.0)


if __name__ == "__main__":
    unittest.main()
