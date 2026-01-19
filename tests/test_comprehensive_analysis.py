"""Tests for comprehensive lottery analysis modules."""

import unittest
import math


class TestNISTTests(unittest.TestCase):
    """Test NIST randomness test suite."""

    def test_nist_suite_import(self):
        """Test that NIST test suite can be imported."""
        from shared.nist_tests import NISTTestSuite, TestResult
        suite = NISTTestSuite()
        self.assertIsNotNone(suite)

    def test_frequency_test(self):
        """Test basic frequency test."""
        from shared.nist_tests import NISTTestSuite
        suite = NISTTestSuite()

        # Balanced sequence should pass
        bits = [0, 1] * 500
        result = suite.frequency_test(bits)
        self.assertEqual(result.test_name, "Frequency")
        self.assertGreater(result.p_value, 0.01)

        # Highly imbalanced sequence should fail
        bits = [1] * 900 + [0] * 100
        result = suite.frequency_test(bits)
        self.assertLess(result.p_value, 0.01)

    def test_runs_test(self):
        """Test runs test."""
        from shared.nist_tests import NISTTestSuite
        suite = NISTTestSuite()

        # Alternating sequence has maximum runs
        bits = [0, 1] * 500
        result = suite.runs_test(bits)
        self.assertEqual(result.test_name, "Runs")

    def test_lottery_to_bits_presence(self):
        """Test lottery draw to bits conversion (presence encoding)."""
        from shared.nist_tests import lottery_draws_to_bits

        draws = [[1, 2, 3, 4, 5, 6]]
        bits = lottery_draws_to_bits(draws, pool_size=49, encoding="presence")

        # Should have pool_size bits per draw
        self.assertEqual(len(bits), 49)
        # Numbers 1-6 should be 1, rest should be 0
        self.assertEqual(sum(bits), 6)

    def test_lottery_to_bits_binary(self):
        """Test lottery draw to bits conversion (binary encoding)."""
        from shared.nist_tests import lottery_draws_to_bits

        draws = [[1, 2, 3, 4, 5, 6]]
        bits = lottery_draws_to_bits(draws, pool_size=49, encoding="binary")

        # Each number uses ceil(log2(49)) = 6 bits
        expected_bits_per_number = 6
        self.assertEqual(len(bits), 6 * expected_bits_per_number)


class TestBiasDetector(unittest.TestCase):
    """Test statistical bias detection."""

    def test_bias_detector_import(self):
        """Test that BiasDetector can be imported."""
        from shared.bias_detector import BiasDetector, BiasReport
        detector = BiasDetector()
        self.assertIsNotNone(detector)

    def test_frequency_uniformity(self):
        """Test frequency uniformity test."""
        from shared.bias_detector import BiasDetector

        detector = BiasDetector()

        # Create uniform draws
        draws = [[i % 49 + 1 for i in range(j, j + 6)] for j in range(100)]
        result = detector.frequency_uniformity(draws, pool_size=49)

        self.assertEqual(result.test_name, "Frequency Uniformity")
        self.assertGreater(result.p_value, 0)  # Should have valid p-value

    def test_temporal_correlation(self):
        """Test temporal correlation test."""
        from shared.bias_detector import BiasDetector

        detector = BiasDetector()
        draws = [[1, 5, 12, 23, 34, 45]] * 50
        result = detector.temporal_correlation(draws)

        self.assertEqual(result.test_name, "Temporal Correlation")

    def test_full_report(self):
        """Test full bias report generation."""
        from shared.bias_detector import BiasDetector

        detector = BiasDetector()
        draws = [[i % 49 + 1 for i in range(j, j + 6)] for j in range(200)]
        dates = [(2024, 1, d % 28 + 1) for d in range(200)]

        reports = detector.full_report(draws, pool_size=49, dates=dates)

        # Should return multiple reports
        self.assertGreater(len(reports), 5)

        # Summary should work
        summary = detector.summary(reports)
        self.assertIn("total_tests", summary)
        self.assertIn("significant_findings", summary)


class TestEVCalculator(unittest.TestCase):
    """Test expected value calculator."""

    def test_ev_calculator_import(self):
        """Test that EVCalculator can be imported."""
        from shared.ev_calculator import EVCalculator
        calc = EVCalculator()
        self.assertIsNotNone(calc)

    def test_create_loto_649(self):
        """Test Loto 6/49 game creation."""
        from shared.ev_calculator import EVCalculator

        calc = EVCalculator()
        game = calc.create_loto_649()

        self.assertEqual(game.name, "Loto 6/49")
        self.assertEqual(game.pool_size, 49)
        self.assertEqual(game.numbers_drawn, 6)
        self.assertGreater(len(game.prize_tiers), 0)

    def test_calculate_ev_negative(self):
        """Test that EV is negative under normal conditions."""
        from shared.ev_calculator import EVCalculator

        calc = EVCalculator()
        game = calc.create_loto_649()

        # With minimum jackpot, EV should be negative
        result = calc.calculate_ev(game, jackpot=100_000)

        self.assertLess(result.expected_value, 0)
        self.assertFalse(result.is_positive_ev)

    def test_combinations(self):
        """Test combinations calculation."""
        from shared.ev_calculator import EVCalculator

        # C(49, 6) = 13,983,816
        result = EVCalculator._combinations(49, 6)
        self.assertEqual(result, 13_983_816)

        # C(6, 0) = 1
        self.assertEqual(EVCalculator._combinations(6, 0), 1)

        # C(6, 6) = 1
        self.assertEqual(EVCalculator._combinations(6, 6), 1)


class TestPaperTrader(unittest.TestCase):
    """Test paper trading system."""

    def test_paper_trader_import(self):
        """Test that PaperTrader can be imported."""
        from shared.paper_trader import PaperTrader, Trade
        trader = PaperTrader()
        self.assertIsNotNone(trader)

    def test_execute_trade(self):
        """Test executing a paper trade."""
        from shared.paper_trader import PaperTrader

        trader = PaperTrader(initial_balance=1000, ticket_cost=6.0)

        # Execute a trade
        trade = trader.execute_trade(
            draw_id=1,
            strategy_name="test",
            tickets=[[1, 2, 3, 4, 5, 6]],
            actual_draw=[7, 8, 9, 10, 11, 12]  # No matches
        )

        self.assertEqual(trade.prize_won, 0)
        self.assertEqual(trader.balance, 1000 - 6)

    def test_winning_trade(self):
        """Test a winning paper trade."""
        from shared.paper_trader import PaperTrader

        def prize_calc(picks, actual):
            matches = len(set(picks) & set(actual))
            return {3: 20, 4: 80, 5: 5000, 6: 1000000}.get(matches, 0)

        trader = PaperTrader(
            initial_balance=1000,
            ticket_cost=6.0,
            prize_calculator=prize_calc
        )

        # Match 3 numbers
        trade = trader.execute_trade(
            draw_id=1,
            strategy_name="test",
            tickets=[[1, 2, 3, 4, 5, 6]],
            actual_draw=[1, 2, 3, 40, 41, 42]
        )

        self.assertEqual(trade.prize_won, 20)  # 3 matches
        self.assertEqual(trade.matches_per_ticket[0], 3)

    def test_get_stats(self):
        """Test statistics calculation."""
        from shared.paper_trader import PaperTrader

        trader = PaperTrader(initial_balance=1000, ticket_cost=6.0)

        # Execute several trades
        for i in range(10):
            trader.execute_trade(
                draw_id=i,
                strategy_name="test",
                tickets=[[1, 2, 3, 4, 5, 6]],
                actual_draw=[7, 8, 9, 10, 11, 12]
            )

        stats = trader.get_stats("test")

        self.assertEqual(stats.total_trades, 10)
        self.assertEqual(stats.total_tickets, 10)
        self.assertEqual(stats.total_spent, 60)
        self.assertEqual(stats.winning_trades, 0)


class TestTransformerModel(unittest.TestCase):
    """Test transformer model (stub tests when PyTorch unavailable)."""

    def test_transformer_config(self):
        """Test transformer config creation."""
        from shared.transformer_model import TransformerConfig, create_default_config

        config = create_default_config("loto_649")

        self.assertEqual(config.pool_size, 49)
        self.assertEqual(config.draw_size, 6)

    def test_torch_availability_check(self):
        """Test that PyTorch availability is properly detected."""
        from shared.transformer_model import TORCH_AVAILABLE
        # Just check it's a boolean
        self.assertIsInstance(TORCH_AVAILABLE, bool)


class TestVAEModel(unittest.TestCase):
    """Test VAE model (stub tests when PyTorch unavailable)."""

    def test_vae_config(self):
        """Test VAE config creation."""
        from shared.vae_model import VAEConfig, create_default_config

        config = create_default_config("loto_649")

        self.assertEqual(config.pool_size, 49)
        self.assertEqual(config.draw_size, 6)
        self.assertGreater(config.latent_dim, 0)

    def test_torch_availability_check(self):
        """Test that PyTorch availability is properly detected."""
        from shared.vae_model import TORCH_AVAILABLE
        # Just check it's a boolean
        self.assertIsInstance(TORCH_AVAILABLE, bool)


if __name__ == "__main__":
    unittest.main()
