"""Shared data structure for all analysis test results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AnalysisResult:
    """Result of a single statistical analysis test."""

    test_name: str
    game: str
    passed: bool | None  # True=pass, False=fail, None=inconclusive
    p_value: float | None
    statistic: float
    threshold: float
    sample_size: int
    details: dict[str, Any]
    summary: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @staticmethod
    def results_to_json(results: list[AnalysisResult]) -> str:
        return json.dumps(
            [asdict(r) for r in results], indent=2, default=str,
        )

    def summary_line(self) -> str:
        if self.passed is True:
            verdict = "PASS"
        elif self.passed is False:
            verdict = "FAIL"
        else:
            verdict = "???"
        p_str = f"p={self.p_value:.4f}" if self.p_value is not None else "p=N/A"
        return f"[{verdict}] {self.test_name:.<30s} {p_str}  ({self.game}, n={self.sample_size})"
