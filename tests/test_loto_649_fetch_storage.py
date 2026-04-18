import csv
import tempfile
import unittest
from pathlib import Path

from loto_649_model.fetch import update_dataset
from loto_649_model.models import Loto649Draw
from loto_649_model.storage import append_draws, load_draws


class TestLoto649FetchStorage(unittest.TestCase):
    def test_update_dataset_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            data_dir = Path(tmpdir) / "data"
            cache_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            html_path = cache_dir / "loto_649_results.html"
            html_path.write_text("<p>Detalii castiguri la loto 6/49 din <span>15.01.2026</span></p>", encoding="utf-8")

            updated = update_dataset(
                url="https://example.invalid",
                cache_path=html_path,
                csv_path=data_dir / "loto_649_draws.csv",
                fetcher=lambda _: html_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(updated, 0)


class TestLoto649StorageNoroc(unittest.TestCase):
    def test_round_trip_with_noroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                Loto649Draw(
                    date="2026-01-15",
                    main_numbers=[3, 12, 27, 31, 38, 44],
                    noroc="0123456",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(loaded[0].noroc, "0123456")

    def test_load_legacy_csv_without_noroc_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,main_6\n"
                "2026-01-15,3,12,27,31,38,44\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].noroc)


if __name__ == "__main__":
    unittest.main()
