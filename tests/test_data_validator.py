"""Tests for data_validator module."""

import unittest

from shared.data_validator import ValidationIssue, validate_draws
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG


def make_draw(date, main_numbers):
    return type("Draw", (), {"date": date, "main_numbers": main_numbers})()


class TestValidateDraws(unittest.TestCase):
    def test_valid_draws_no_issues(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3, 4, 5]),
            make_draw("2024-01-08", [6, 7, 8, 9, 10]),
            make_draw("2024-01-15", [11, 12, 13, 14, 15]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        self.assertEqual(issues, [])

    def test_duplicate_dates_detected(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3, 4, 5]),
            make_draw("2024-01-01", [6, 7, 8, 9, 10]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        errors = [i for i in issues if i.severity == "error"]
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(
            any("duplicate" in i.description.lower() for i in errors)
        )

    def test_number_out_of_range(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3, 4, 99]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        errors = [i for i in issues if i.severity == "error"]
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("range" in i.description.lower() for i in errors))

    def test_wrong_count_of_numbers(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        errors = [i for i in issues if i.severity == "error"]
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("count" in i.description.lower() for i in errors))

    def test_non_chronological_order(self):
        draws = [
            make_draw("2024-01-15", [1, 2, 3, 4, 5]),
            make_draw("2024-01-01", [6, 7, 8, 9, 10]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        errors = [i for i in issues if i.severity == "error"]
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(
            any("chronolog" in i.description.lower() for i in errors)
        )

    def test_duplicate_numbers_in_draw(self):
        draws = [
            make_draw("2024-01-01", [1, 1, 3, 4, 5]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        errors = [i for i in issues if i.severity == "error"]
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(
            any("duplicate number" in i.description.lower() for i in errors)
        )

    def test_schedule_gap_warning(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3, 4, 5]),
            make_draw("2024-01-08", [6, 7, 8, 9, 10]),
            make_draw("2024-01-15", [11, 12, 13, 14, 15]),
            make_draw("2024-03-15", [16, 17, 18, 19, 20]),
        ]
        issues = validate_draws(draws, JOKER_CONFIG)
        warnings = [i for i in issues if i.severity == "warning"]
        self.assertTrue(len(warnings) >= 1)
        self.assertTrue(
            any("gap" in i.description.lower() for i in warnings)
        )

    def test_loto649_config(self):
        draws = [
            make_draw("2024-01-01", [1, 2, 3, 4, 5, 6]),
            make_draw("2024-01-08", [7, 8, 9, 10, 11, 12]),
            make_draw("2024-01-15", [13, 14, 15, 16, 17, 18]),
        ]
        issues = validate_draws(draws, LOTO_649_CONFIG)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
