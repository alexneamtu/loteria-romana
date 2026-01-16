import unittest

from joker_model.seed import resolve_seed


class TestSeed(unittest.TestCase):
    def test_cli_seed_overrides_env(self):
        self.assertEqual(resolve_seed(cli_seed=7, env_seed="11"), 7)

    def test_env_seed_used_when_cli_missing(self):
        self.assertEqual(resolve_seed(cli_seed=None, env_seed="11"), 11)

    def test_empty_env_returns_none(self):
        self.assertIsNone(resolve_seed(cli_seed=None, env_seed=""))

    def test_invalid_env_raises(self):
        with self.assertRaises(ValueError):
            resolve_seed(cli_seed=None, env_seed="abc")


if __name__ == "__main__":
    unittest.main()
