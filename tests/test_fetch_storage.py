import csv
import tempfile
import unittest
from pathlib import Path

from joker_model.fetch import update_dataset
from joker_model.models import JokerDraw
from joker_model.storage import append_draws, load_draws


class TestFetchStorage(unittest.TestCase):
    def test_update_dataset_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            data_dir = Path(tmpdir) / "data"
            cache_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            html_path = cache_dir / "joker_results.html"
            html_path.write_text("<p>Detalii castiguri  la joker din <span>15.01.2026</span></p>", encoding="utf-8")

            updated = update_dataset(
                url="https://example.invalid",
                cache_path=html_path,
                csv_path=data_dir / "joker_draws.csv",
                fetcher=lambda _: html_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(updated, 0)


class TestJokerStorageNorocPlus(unittest.TestCase):
    def test_round_trip_with_noroc_plus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                JokerDraw(
                    date="2026-01-15",
                    main_numbers=[7, 11, 44, 45, 46],
                    joker=13,
                    noroc_plus="NP07",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].noroc_plus, "NP07")

    def test_load_legacy_csv_without_noroc_plus_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,joker\n"
                "2026-01-15,7,11,44,45,46,13\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertEqual(len(loaded), 1)
            self.assertIsNone(loaded[0].noroc_plus)

    def test_round_trip_with_noroc_plus_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].noroc_plus)
            with path.open() as f:
                header = next(csv.reader(f))
            self.assertIn("noroc_plus", header)


if __name__ == "__main__":
    unittest.main()
