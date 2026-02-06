import random
import unittest

from shared.ensemble_blend import generate_blended_picks, _apply_significance_gate
from shared.game_config import JOKER_CONFIG


class TestSignificanceGateFunction(unittest.TestCase):
    def test_gate_removes_low_scoring_strategies(self):
        scores = {"random": 5, "frequency": 5, "bayesian": 5, "weak": 1}
        baseline_score = 5
        total_draws = 100
        gated = _apply_significance_gate(scores, baseline_score, total_draws)
        # "weak" should be removed since it doesn't outperform baseline
        self.assertNotIn("weak", gated)
        # random is always kept as the baseline
        self.assertIn("random", gated)

    def test_gate_keeps_strong_strategies(self):
        scores = {"random": 5, "frequency": 20, "bayesian": 15}
        baseline_score = 5
        total_draws = 100
        gated = _apply_significance_gate(scores, baseline_score, total_draws)
        self.assertIn("frequency", gated)
        self.assertIn("bayesian", gated)

    def test_gate_always_keeps_random(self):
        scores = {"random": 10}
        gated = _apply_significance_gate(scores, 10, 100)
        self.assertIn("random", gated)

    def test_gate_with_insufficient_data_keeps_all(self):
        scores = {"random": 1, "frequency": 1, "bayesian": 1}
        gated = _apply_significance_gate(scores, 1, 5)
        self.assertEqual(len(gated), 3)


class TestSignificanceGateEndToEnd(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_blend_with_significance_gating(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))


if __name__ == "__main__":
    unittest.main()
