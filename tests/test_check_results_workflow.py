from pathlib import Path
import unittest


class TestCheckResultsWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.workflow = root / ".github/workflows/check-results.yml"
        self.text = self.workflow.read_text()

    def test_uses_check_results_script(self):
        self.assertIn("python scripts/check_results.py", self.text)
        self.assertIn("MAX_RETRIES", self.text)
        self.assertIn("RETRY_DELAY_MINUTES", self.text)

    def test_has_database_url_env(self):
        self.assertIn("DATABASE_URL", self.text)
        self.assertIn("secrets.DATABASE_URL", self.text)

    def test_installs_db_driver(self):
        self.assertIn("Install DB driver", self.text)
        self.assertIn("psycopg[binary]", self.text)


if __name__ == "__main__":
    unittest.main()
