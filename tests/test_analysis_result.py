import json
import unittest

from shared.analysis_result import AnalysisResult


class TestAnalysisResult(unittest.TestCase):
    def test_create_passing_result(self):
        r = AnalysisResult(
            test_name="frequency_monobit",
            game="joker",
            passed=True,
            p_value=0.45,
            statistic=1.2,
            threshold=0.01,
            sample_size=1000,
            details={"per_number": [0.45, 0.51]},
            summary="Frequency test passed (p=0.45)",
        )
        self.assertEqual(r.test_name, "frequency_monobit")
        self.assertTrue(r.passed)

    def test_create_inconclusive_result(self):
        r = AnalysisResult(
            test_name="serial",
            game="loto_649",
            passed=None,
            p_value=None,
            statistic=0.0,
            threshold=0.01,
            sample_size=50,
            details={},
            summary="Insufficient data for serial test",
        )
        self.assertIsNone(r.passed)
        self.assertIsNone(r.p_value)

    def test_to_json_roundtrip(self):
        r = AnalysisResult(
            test_name="runs",
            game="joker",
            passed=False,
            p_value=0.003,
            statistic=3.1,
            threshold=0.01,
            sample_size=500,
            details={"failed_numbers": [7, 22]},
            summary="Runs test failed (p=0.003)",
        )
        json_str = r.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["test_name"], "runs")
        self.assertFalse(parsed["passed"])
        self.assertAlmostEqual(parsed["p_value"], 0.003)

    def test_results_to_json_list(self):
        results = [
            AnalysisResult("a", "joker", True, 0.5, 1.0, 0.01, 100, {}, "ok"),
            AnalysisResult("b", "joker", False, 0.001, 5.0, 0.01, 100, {}, "fail"),
        ]
        json_str = AnalysisResult.results_to_json(results)
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["test_name"], "a")

    def test_summary_line(self):
        r = AnalysisResult("freq", "joker", True, 0.5, 1.0, 0.01, 100, {}, "Passed")
        line = r.summary_line()
        self.assertIn("PASS", line)
        self.assertIn("freq", line)

    def test_summary_line_fail(self):
        r = AnalysisResult("freq", "joker", False, 0.001, 5.0, 0.01, 100, {}, "Failed")
        line = r.summary_line()
        self.assertIn("FAIL", line)

    def test_summary_line_inconclusive(self):
        r = AnalysisResult("freq", "joker", None, None, 0.0, 0.01, 10, {}, "N/A")
        line = r.summary_line()
        self.assertIn("???", line)
