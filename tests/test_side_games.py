import random
import unittest

from shared.side_games import (
    NOROC_DIGITS,
    SUPER_NOROC_DIGITS,
    NOROC_PLUS_MIN,
    NOROC_PLUS_MAX,
    generate_noroc,
    generate_super_noroc,
    generate_noroc_plus,
    validate_side_game_number,
)


class TestSideGames(unittest.TestCase):
    def test_noroc_constants(self):
        self.assertEqual(NOROC_DIGITS, 7)
        self.assertEqual(SUPER_NOROC_DIGITS, 6)
        self.assertEqual((NOROC_PLUS_MIN, NOROC_PLUS_MAX), (1, 20))

    def test_generate_noroc_is_7_digit_string(self):
        rng = random.Random(0)
        n = generate_noroc(rng)
        self.assertEqual(len(n), 7)
        self.assertTrue(n.isdigit())

    def test_generate_noroc_preserves_leading_zeros(self):
        rng = random.Random(1)  # seed picked so first draw has leading zero
        samples = {generate_noroc(rng) for _ in range(200)}
        self.assertTrue(any(s.startswith("0") for s in samples))

    def test_generate_super_noroc_is_6_digit_string(self):
        rng = random.Random(0)
        n = generate_super_noroc(rng)
        self.assertEqual(len(n), 6)
        self.assertTrue(n.isdigit())

    def test_generate_noroc_plus_is_np_prefix(self):
        rng = random.Random(0)
        n = generate_noroc_plus(rng)
        self.assertTrue(n.startswith("NP"))
        inner = int(n[2:])
        self.assertTrue(NOROC_PLUS_MIN <= inner <= NOROC_PLUS_MAX)

    def test_validate_noroc(self):
        self.assertTrue(validate_side_game_number("loto_649", "1234567"))
        self.assertFalse(validate_side_game_number("loto_649", "123456"))
        self.assertFalse(validate_side_game_number("loto_649", "12345678"))
        self.assertFalse(validate_side_game_number("loto_649", "12a4567"))

    def test_validate_super_noroc(self):
        self.assertTrue(validate_side_game_number("loto_540", "012345"))
        self.assertFalse(validate_side_game_number("loto_540", "12345"))

    def test_validate_noroc_plus(self):
        self.assertTrue(validate_side_game_number("joker", "NP14"))
        self.assertTrue(validate_side_game_number("joker", "NP01"))
        self.assertFalse(validate_side_game_number("joker", "14"))
        self.assertFalse(validate_side_game_number("joker", "NP21"))

    def test_seed_reproduces(self):
        r1 = random.Random(42)
        r2 = random.Random(42)
        self.assertEqual(generate_noroc(r1), generate_noroc(r2))


if __name__ == "__main__":
    unittest.main()
