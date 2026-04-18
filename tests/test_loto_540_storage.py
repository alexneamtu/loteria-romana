import csv
import tempfile
import unittest
from pathlib import Path

from loto_540_model.models import Loto540Draw
from loto_540_model.storage import append_draws, load_draws


class TestLoto540StorageSuperNoroc(unittest.TestCase):
    def test_round_trip_with_super_noroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                Loto540Draw(
                    date="2026-01-15",
                    main_numbers=[2, 9, 15, 27, 34, 38],
                    super_noroc="012345",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(loaded[0].super_noroc, "012345")

    def test_load_legacy_csv_without_super_noroc_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,main_6\n"
                "2026-01-15,2,9,15,27,34,38\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].super_noroc)


if __name__ == "__main__":
    unittest.main()
