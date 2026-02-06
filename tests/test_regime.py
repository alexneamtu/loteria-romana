import unittest
import random

from shared.regime import (
    detect_regimes,
    segment_draws_by_regime,
    RegimeBoundary,
)


class TestRegimeBoundary(unittest.TestCase):
    def test_fields(self):
        b = RegimeBoundary(index=50, confidence=0.95, pre_mean=100.0, post_mean=130.0)
        self.assertEqual(b.index, 50)
        self.assertAlmostEqual(b.confidence, 0.95)


class TestDetectRegimes(unittest.TestCase):
    def test_no_regime_change(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        boundaries = detect_regimes(values)
        self.assertEqual(len(boundaries), 0)

    def test_single_regime_change(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        boundaries = detect_regimes(values)
        self.assertGreaterEqual(len(boundaries), 1)
        self.assertTrue(any(150 < b.index < 250 for b in boundaries))

    def test_multiple_regime_changes(self):
        rng = random.Random(42)
        seg1 = [rng.gauss(100, 10) for _ in range(150)]
        seg2 = [rng.gauss(150, 10) for _ in range(150)]
        seg3 = [rng.gauss(80, 10) for _ in range(150)]
        values = seg1 + seg2 + seg3
        boundaries = detect_regimes(values)
        self.assertGreaterEqual(len(boundaries), 2)

    def test_empty_input(self):
        boundaries = detect_regimes([])
        self.assertEqual(len(boundaries), 0)

    def test_short_input(self):
        boundaries = detect_regimes([1.0, 2.0])
        self.assertEqual(len(boundaries), 0)

    def test_custom_min_segment(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        boundaries = detect_regimes(values, min_segment_size=100)
        self.assertEqual(len(boundaries), 0)


class TestSegmentDrawsByRegime(unittest.TestCase):
    def test_no_boundaries(self):
        draws = [[1, 2, 3]] * 100
        segments = segment_draws_by_regime(draws, [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 100)

    def test_single_boundary(self):
        draws = [[1, 2, 3]] * 100
        boundaries = [RegimeBoundary(index=50, confidence=0.95, pre_mean=10.0, post_mean=20.0)]
        segments = segment_draws_by_regime(draws, boundaries)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 50)
        self.assertEqual(len(segments[1]), 50)

    def test_multiple_boundaries(self):
        draws = [[1, 2, 3]] * 150
        boundaries = [
            RegimeBoundary(index=50, confidence=0.9, pre_mean=10.0, post_mean=20.0),
            RegimeBoundary(index=100, confidence=0.9, pre_mean=20.0, post_mean=30.0),
        ]
        segments = segment_draws_by_regime(draws, boundaries)
        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[0]), 50)
        self.assertEqual(len(segments[1]), 50)
        self.assertEqual(len(segments[2]), 50)

    def test_boundary_at_edges(self):
        draws = [[1, 2, 3]] * 100
        boundaries = [RegimeBoundary(index=0, confidence=0.9, pre_mean=10.0, post_mean=20.0)]
        segments = segment_draws_by_regime(draws, boundaries)
        self.assertTrue(all(len(s) > 0 for s in segments if s))


if __name__ == "__main__":
    unittest.main()
