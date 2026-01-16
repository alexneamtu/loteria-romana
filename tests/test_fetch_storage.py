import tempfile
import unittest
from pathlib import Path

from joker_model.fetch import update_dataset


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


if __name__ == "__main__":
    unittest.main()
