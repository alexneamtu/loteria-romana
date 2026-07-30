import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import scripts.backfill_loto_ro as backfill
from joker_model.models import JokerDraw
from joker_model.storage import append_draws, load_draws


class TestBackfillLotoRo(unittest.TestCase):
    def _run(self, csv_path, since, today):
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        url, parse, storage, _ = backfill.GAMES["joker"]
        requested = []

        def fetcher(_url, year, month):
            requested.append((year, month))
            return html if (year, month) == (2026, 1) else ""

        with mock.patch.dict(backfill.GAMES, {"joker": (url, parse, storage, csv_path)}):
            added, months = backfill.backfill("joker", since, today, fetcher=fetcher)
        return requested, added, months

    def test_fetches_from_newest_draw_month_and_appends_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "joker_draws.csv"
            append_draws(csv_path, [JokerDraw("2025-12-28", [2, 4, 6, 8, 10], 3, None)])

            requested, added, months = self._run(csv_path, None, date(2026, 2, 10))

            self.assertEqual(requested, [(2025, 12), (2026, 1), (2026, 2)])
            self.assertEqual((added, months), (2, 3))
            self.assertEqual(
                [d.date for d in load_draws(csv_path)],
                ["2025-12-28", "2026-01-11", "2026-01-15"],
            )

            # Second pass over the same months must not duplicate anything.
            _, added, _ = self._run(csv_path, None, date(2026, 2, 10))
            self.assertEqual(added, 0)
            self.assertEqual(len(load_draws(csv_path)), 3)

    def test_since_overrides_newest_draw_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "joker_draws.csv"
            append_draws(csv_path, [JokerDraw("2026-02-01", [2, 4, 6, 8, 10], 3, None)])

            requested, added, _ = self._run(csv_path, "2025-11", date(2026, 2, 10))

            self.assertEqual(requested[0], (2025, 11))
            self.assertEqual(added, 2)


if __name__ == "__main__":
    unittest.main()
