import unittest

from shared.game_config import GameConfig
from shared.game_strategies import build_frequency


class TestGameStrategies(unittest.TestCase):
    def test_build_frequency_weights_recent(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=3,
            numbers_drawn=2,
            numbers_to_pick=2,
        )
        draws = [
            [1, 2],  # oldest
            [2, 3],  # newest
        ]

        freq = build_frequency(config, draws, half_life=1.0)

        self.assertAlmostEqual(freq[1], 0.5, places=6)
        self.assertAlmostEqual(freq[2], 1.5, places=6)
        self.assertAlmostEqual(freq[3], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
