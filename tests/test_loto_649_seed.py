import unittest

from loto_649_model.seed import resolve_seed


class TestLoto649Seed(unittest.TestCase):
    def test_env_seed_parsing(self):
        self.assertEqual(resolve_seed(None, "123"), 123)

    def test_env_seed_invalid(self):
        with self.assertRaises(ValueError):
            resolve_seed(None, "nope")


if __name__ == "__main__":
    unittest.main()
