import unittest

from shared.recency import draw_weights, resolve_half_life, DEFAULT_HALF_LIFE


class TestRecencyWeights(unittest.TestCase):
    def test_draw_weights_half_life(self):
        weights = draw_weights(11, 10.0)
        self.assertAlmostEqual(weights[-1], 1.0, places=6)
        self.assertAlmostEqual(weights[0], 0.5, places=6)

    def test_draw_weights_monotonic(self):
        weights = draw_weights(5, 10.0)
        self.assertGreater(weights[-1], weights[0])

    def test_resolve_half_life_default_and_env(self):
        self.assertEqual(resolve_half_life(None, None), DEFAULT_HALF_LIFE)
        self.assertEqual(resolve_half_life(None, "25"), 25.0)

    def test_resolve_half_life_invalid(self):
        with self.assertRaises(ValueError):
            resolve_half_life(None, "0")


if __name__ == "__main__":
    unittest.main()
