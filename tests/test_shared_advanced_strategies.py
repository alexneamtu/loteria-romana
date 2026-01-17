import unittest

from shared.advanced_strategies import compute_composite_scores
from shared.features import compute_weighted_gap_averages
from shared.game_config import GameConfig


class TestAdvancedStrategies(unittest.TestCase):
    def test_composite_scores_weighted_frequency(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=4,
            numbers_drawn=2,
            numbers_to_pick=2,
        )
        draws = [
            [1, 2],  # oldest
            [3, 4],  # newest
        ]
        weights = {
            "frequency": 1.0,
            "recency": 0.0,
            "gap": 0.0,
            "position": 0.0,
            "trend": 0.0,
            "balance": 0.0,
        }
        scores = compute_composite_scores(config, draws, weights=weights, half_life=1.0)
        self.assertGreater(scores[3], scores[1])

    def test_weighted_gap_averages_use_recent_gaps(self):
        draws = [[1], [1], [2], [2], [1]]
        weights = [0.1, 0.2, 0.3, 0.6, 1.0]
        gaps = compute_weighted_gap_averages(draws, pool_size=2, weights=weights)
        expected = (1 * 0.2 + 3 * 1.0) / (0.2 + 1.0)
        self.assertAlmostEqual(gaps[1], expected, places=6)

    def test_trend_weighting_favors_recent_draw(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=2,
            numbers_drawn=1,
            numbers_to_pick=1,
        )
        draws = [[1]] * 10 + [[1]] * 9 + [[2]]
        weights = {
            "frequency": 0.0,
            "recency": 0.0,
            "gap": 0.0,
            "position": 0.0,
            "trend": 1.0,
            "balance": 0.0,
        }
        scores = compute_composite_scores(config, draws, weights=weights, half_life=0.5)
        self.assertGreater(scores[2], scores[1])


if __name__ == "__main__":
    unittest.main()
