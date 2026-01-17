import unittest
from math import comb

from shared.wheeling import (
    WheelConfig,
    WheelGenerator,
    KeyNumberWheel,
    generate_full_wheel,
    generate_balanced_wheel,
    verify_wheel_coverage,
    estimate_full_wheel_size,
)


class TestWheelConfig(unittest.TestCase):
    def test_valid_config(self):
        config = WheelConfig(total_numbers=10, numbers_per_line=5, guarantee=3)
        self.assertEqual(config.total_numbers, 10)
        self.assertEqual(config.numbers_per_line, 5)
        self.assertEqual(config.guarantee, 3)

    def test_invalid_guarantee_exceeds_line(self):
        with self.assertRaises(ValueError):
            WheelConfig(total_numbers=10, numbers_per_line=5, guarantee=6)

    def test_invalid_line_exceeds_total(self):
        with self.assertRaises(ValueError):
            WheelConfig(total_numbers=5, numbers_per_line=6, guarantee=3)


class TestFullWheel(unittest.TestCase):
    def test_generate_full_wheel_basic(self):
        numbers = [1, 2, 3, 4, 5]
        tickets = generate_full_wheel(numbers, 3)

        # C(5,3) = 10 combinations
        self.assertEqual(len(tickets), 10)

        # All tickets should have 3 numbers
        for ticket in tickets:
            self.assertEqual(len(ticket), 3)

    def test_generate_full_wheel_all_combinations(self):
        numbers = [1, 2, 3, 4]
        tickets = generate_full_wheel(numbers, 2)

        # C(4,2) = 6
        self.assertEqual(len(tickets), 6)

        # Verify all pairs are present
        pairs = {tuple(t) for t in tickets}
        expected = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
        self.assertEqual(pairs, expected)

    def test_estimate_full_wheel_size(self):
        size = estimate_full_wheel_size(10, 5)
        self.assertEqual(size, comb(10, 5))  # 252


class TestWheelGenerator(unittest.TestCase):
    def test_abbreviated_wheel_coverage(self):
        generator = WheelGenerator(numbers_per_line=5, guarantee=3)
        numbers = list(range(1, 11))  # 10 numbers

        tickets = generator.generate_abbreviated_wheel(numbers)

        # Verify coverage
        is_valid, uncovered = verify_wheel_coverage(tickets, numbers, 3)
        self.assertTrue(is_valid, f"Uncovered: {uncovered}")

    def test_abbreviated_wheel_smaller_than_full(self):
        generator = WheelGenerator(numbers_per_line=5, guarantee=3)
        numbers = list(range(1, 11))  # 10 numbers

        abbreviated = generator.generate_abbreviated_wheel(numbers)
        full = generator.generate_full_wheel(numbers)

        # Abbreviated should be smaller than full wheel
        self.assertLess(len(abbreviated), len(full))

    def test_abbreviated_wheel_max_tickets(self):
        generator = WheelGenerator(numbers_per_line=5, guarantee=3)
        numbers = list(range(1, 11))

        tickets = generator.generate_abbreviated_wheel(numbers, max_tickets=5)

        self.assertLessEqual(len(tickets), 5)

    def test_estimate_reduction(self):
        generator = WheelGenerator(numbers_per_line=5, guarantee=3)
        stats = generator.estimate_reduction(total_numbers=10)

        self.assertEqual(stats["total_numbers"], 10)
        self.assertEqual(stats["numbers_per_line"], 5)
        self.assertEqual(stats["guarantee"], 3)
        self.assertEqual(stats["full_wheel_size"], comb(10, 5))
        self.assertGreater(stats["reduction_ratio"], 1.0)


class TestVerifyWheelCoverage(unittest.TestCase):
    def test_full_coverage(self):
        # Full wheel should have complete coverage
        numbers = [1, 2, 3, 4, 5]
        tickets = generate_full_wheel(numbers, 3)

        is_valid, uncovered = verify_wheel_coverage(tickets, numbers, 2)
        self.assertTrue(is_valid)
        self.assertEqual(len(uncovered), 0)

    def test_partial_coverage(self):
        # Single ticket can't cover all 2-combinations of 5 numbers
        numbers = [1, 2, 3, 4, 5]
        tickets = [[1, 2, 3]]  # Only covers pairs from 1,2,3

        is_valid, uncovered = verify_wheel_coverage(tickets, numbers, 2)
        self.assertFalse(is_valid)
        self.assertGreater(len(uncovered), 0)

    def test_empty_tickets(self):
        numbers = [1, 2, 3, 4, 5]
        is_valid, uncovered = verify_wheel_coverage([], numbers, 2)
        self.assertFalse(is_valid)


class TestBalancedWheel(unittest.TestCase):
    def test_balanced_wheel_basic(self):
        numbers = list(range(1, 11))  # 10 numbers
        tickets = generate_balanced_wheel(numbers, numbers_per_line=5, target_tickets=10)

        self.assertEqual(len(tickets), 10)
        for ticket in tickets:
            self.assertEqual(len(ticket), 5)
            self.assertTrue(all(n in numbers for n in ticket))

    def test_balanced_wheel_unique_tickets(self):
        numbers = list(range(1, 11))
        tickets = generate_balanced_wheel(numbers, numbers_per_line=5, target_tickets=10)

        # All tickets should be unique
        ticket_tuples = [tuple(t) for t in tickets]
        self.assertEqual(len(ticket_tuples), len(set(ticket_tuples)))


class TestKeyNumberWheel(unittest.TestCase):
    def test_key_wheel_basic(self):
        wheel = KeyNumberWheel(numbers_per_line=5, key_numbers=[1, 2])
        other_numbers = [3, 4, 5, 6, 7]

        tickets = wheel.generate(other_numbers)

        # C(5,3) = 10 combinations of other numbers
        self.assertEqual(len(tickets), 10)

        # Every ticket should contain key numbers
        for ticket in tickets:
            self.assertIn(1, ticket)
            self.assertIn(2, ticket)
            self.assertEqual(len(ticket), 5)

    def test_key_wheel_estimate_size(self):
        wheel = KeyNumberWheel(numbers_per_line=5, key_numbers=[1, 2])
        size = wheel.estimate_size(other_numbers_count=5)

        # Need 3 more numbers from 5 available: C(5,3) = 10
        self.assertEqual(size, 10)

    def test_key_wheel_invalid_too_many_keys(self):
        with self.assertRaises(ValueError):
            KeyNumberWheel(numbers_per_line=5, key_numbers=[1, 2, 3, 4, 5])

    def test_key_wheel_insufficient_other_numbers(self):
        wheel = KeyNumberWheel(numbers_per_line=5, key_numbers=[1, 2])
        # Need 3 other numbers but only have 2
        tickets = wheel.generate([3, 4])
        self.assertEqual(len(tickets), 0)


class TestWheelIntegration(unittest.TestCase):
    def test_joker_wheel_5_from_10(self):
        """Test wheeling 10 numbers for Joker (5 numbers per ticket)."""
        generator = WheelGenerator(numbers_per_line=5, guarantee=3)
        numbers = list(range(1, 11))

        tickets = generator.generate_abbreviated_wheel(numbers)

        # Verify 3-if-10 coverage
        is_valid, uncovered = verify_wheel_coverage(tickets, numbers, 3)
        self.assertTrue(is_valid)

        # Should be significantly smaller than full wheel
        full_size = comb(10, 5)  # 252
        self.assertLess(len(tickets), full_size // 2)

    def test_loto_649_wheel_6_from_12(self):
        """Test wheeling 12 numbers for Loto 6/49 (6 numbers per ticket)."""
        generator = WheelGenerator(numbers_per_line=6, guarantee=4)
        numbers = list(range(1, 13))

        tickets = generator.generate_abbreviated_wheel(numbers)

        # Verify 4-if-12 coverage
        is_valid, uncovered = verify_wheel_coverage(tickets, numbers, 4)
        self.assertTrue(is_valid)

        # Should be smaller than full wheel
        full_size = comb(12, 6)  # 924
        self.assertLess(len(tickets), full_size)


if __name__ == "__main__":
    unittest.main()
