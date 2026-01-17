import unittest

from shared.advanced_strategies import compute_composite_scores
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


if __name__ == "__main__":
    unittest.main()
