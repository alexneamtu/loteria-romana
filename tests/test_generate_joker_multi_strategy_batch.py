import random
import unittest

import scripts.generate_joker_multi_strategy_batch as batch_script


class TestMultiStrategyBatchHelpers(unittest.TestCase):
    def test_parse_joker_lines(self):
        text = "\n".join([
            "1. 1, 2, 3, 4, 5 + J7",
            "2. 6, 7, 8, 9, 10 + J14",
        ])
        parsed = batch_script.parse_joker_lines(text)
        self.assertEqual(parsed[0], ([1, 2, 3, 4, 5], 7))
        self.assertEqual(parsed[1], ([6, 7, 8, 9, 10], 14))

    def test_build_final_lines_deduplicates_and_covers_jokers(self):
        candidates = [[1, 2, 3, 4, 5]] * 6 + [
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
        lines = batch_script.build_final_lines(
            candidates,
            count=5,
            rng=random.Random(123),
        )
        mains = [tuple(main) for main, _ in lines]
        jokers = [joker for _, joker in lines]
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(set(mains)), 5)
        self.assertEqual(len(set(jokers)), 5)


if __name__ == "__main__":
    unittest.main()
