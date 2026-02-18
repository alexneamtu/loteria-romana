import unittest
import re
from types import SimpleNamespace
from unittest import mock

import scripts.generate_joker_picks as joker_script
import scripts.generate_loto_649_picks as loto_649_script
import scripts.generate_loto_540_picks as loto_540_script


class TestCliParsers(unittest.TestCase):
    def test_joker_parser_accepts_half_life_mode(self):
        parser = joker_script.build_parser()
        args = parser.parse_args(["--half-life-mode", "days"])
        self.assertEqual(args.half_life_mode, "days")

    def test_loto_649_parser_accepts_half_life_mode(self):
        parser = loto_649_script.build_parser()
        args = parser.parse_args(["--half-life-mode", "days"])
        self.assertEqual(args.half_life_mode, "days")

    def test_loto_540_parser_accepts_half_life_mode(self):
        parser = loto_540_script.build_parser()
        args = parser.parse_args(["--half-life-mode", "days"])
        self.assertEqual(args.half_life_mode, "days")


class TestCliDrawDates(unittest.TestCase):
    def test_joker_smart_passes_draw_dates(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5], joker=7),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6], joker=8),
        ]
        with mock.patch.object(joker_script, "update_dataset"), \
            mock.patch.object(joker_script, "load_draws", return_value=draws), \
            mock.patch.object(joker_script, "resolve_seed", return_value=123), \
            mock.patch.object(joker_script, "resolve_recency_settings", return_value=(10.0, "days")), \
            mock.patch.object(joker_script, "generate_smart_picks", return_value=[[1, 2, 3, 4, 5]]) as smart, \
            mock.patch("sys.argv", ["generate_joker_picks.py", "--strategy", "smart", "-n", "1"]):
            joker_script.main()

        _, kwargs = smart.call_args
        self.assertEqual(kwargs["draw_dates"], ["2024-01-01", "2024-01-05"])
        self.assertEqual(kwargs["half_life_mode"], "days")

    def test_loto_649_smart_passes_draw_dates(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5, 6]),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6, 7]),
        ]
        with mock.patch.object(loto_649_script, "update_dataset"), \
            mock.patch.object(loto_649_script, "load_draws", return_value=draws), \
            mock.patch.object(loto_649_script, "resolve_seed", return_value=123), \
            mock.patch.object(loto_649_script, "resolve_recency_settings", return_value=(10.0, "days")), \
            mock.patch.object(loto_649_script, "generate_smart_picks", return_value=[[1, 2, 3, 4, 5, 6]]) as smart, \
            mock.patch("sys.argv", ["generate_loto_649_picks.py", "--strategy", "smart", "-n", "1"]):
            loto_649_script.main()

        _, kwargs = smart.call_args
        self.assertEqual(kwargs["draw_dates"], ["2024-01-01", "2024-01-05"])
        self.assertEqual(kwargs["half_life_mode"], "days")

    def test_loto_540_smart_passes_draw_dates(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5]),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6]),
        ]
        with mock.patch.object(loto_540_script, "update_dataset"), \
            mock.patch.object(loto_540_script, "load_draws", return_value=draws), \
            mock.patch.object(loto_540_script, "resolve_seed", return_value=123), \
            mock.patch.object(loto_540_script, "resolve_recency_settings", return_value=(10.0, "days")), \
            mock.patch.object(loto_540_script, "generate_smart_picks", return_value=[[1, 2, 3, 4, 5]]) as smart, \
            mock.patch("sys.argv", ["generate_loto_540_picks.py", "--strategy", "smart", "-n", "1"]):
            loto_540_script.main()

        _, kwargs = smart.call_args
        self.assertEqual(kwargs["draw_dates"], ["2024-01-01", "2024-01-05"])
        self.assertEqual(kwargs["half_life_mode"], "days")


class TestJokerCliSetOptimization(unittest.TestCase):
    @staticmethod
    def _extract_output_lines(mock_print):
        rendered = []
        for call in mock_print.call_args_list:
            if not call.args:
                continue
            line = str(call.args[0])
            if re.match(r"^\d+\.\s", line):
                rendered.append(line)
        return rendered

    @staticmethod
    def _parse_main_and_joker(line: str):
        match = re.match(r"^\d+\.\s*([\d,\s]+)\s\+\sJ(\d+)$", line)
        if not match:
            return [], -1
        main = [int(value.strip()) for value in match.group(1).split(",")]
        joker = int(match.group(2))
        return main, joker

    def test_assigns_unique_joker_numbers_for_small_count(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5], joker=7),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6], joker=8),
        ]
        duplicate_main = [[1, 2, 3, 4, 5]] * 5

        with mock.patch.object(joker_script, "update_dataset"), \
            mock.patch.object(joker_script, "load_draws", return_value=draws), \
            mock.patch.object(joker_script, "resolve_seed", return_value=123), \
            mock.patch.object(joker_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(joker_script, "generate_smart_picks", return_value=duplicate_main), \
            mock.patch("builtins.print") as mock_print, \
            mock.patch("sys.argv", ["generate_joker_picks.py", "--strategy", "smart", "-n", "5"]):
            joker_script.main()

        lines = self._extract_output_lines(mock_print)
        jokers = [self._parse_main_and_joker(line)[1] for line in lines]
        self.assertEqual(len(jokers), 5)
        self.assertEqual(len(set(jokers)), 5)

    def test_deduplicates_duplicate_main_candidates(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5], joker=7),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6], joker=8),
        ]
        duplicate_main = [[1, 2, 3, 4, 5]] * 5

        with mock.patch.object(joker_script, "update_dataset"), \
            mock.patch.object(joker_script, "load_draws", return_value=draws), \
            mock.patch.object(joker_script, "resolve_seed", return_value=123), \
            mock.patch.object(joker_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(joker_script, "generate_smart_picks", return_value=duplicate_main), \
            mock.patch("builtins.print") as mock_print, \
            mock.patch("sys.argv", ["generate_joker_picks.py", "--strategy", "smart", "-n", "5"]):
            joker_script.main()

        lines = self._extract_output_lines(mock_print)
        mains = [tuple(self._parse_main_and_joker(line)[0]) for line in lines]
        self.assertEqual(len(mains), 5)
        self.assertEqual(len(set(mains)), 5)
