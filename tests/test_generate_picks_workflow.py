from pathlib import Path
import unittest


class TestGeneratePicksWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.workflow = root / ".github/workflows/generate-picks.yml"
        self.text = self.workflow.read_text()

    def test_budget_input_present(self):
        self.assertIn("budget", self.text)
        self.assertIn("Budget in RON", self.text)
        self.assertIn("ev_gate", self.text)
        self.assertIn("Enable EV/jackpot gate", self.text)

    def test_default_budget_present(self):
        self.assertIn("default: '53'", self.text)
        self.assertIn('budget="53"', self.text)
        # ev_gate defaults OFF: the per-game filter forced 5/40, the worst
        # game for P(any prize).
        self.assertIn('ev_gate="false"', self.text)
        self.assertNotIn('ev_gate="true"\n          fi', self.text)

    def test_database_env_and_driver_present(self):
        self.assertIn("DATABASE_URL", self.text)
        self.assertIn("secrets.DATABASE_URL", self.text)
        self.assertIn("Install DB driver", self.text)
        self.assertIn("psycopg[binary]", self.text)

    def test_recommended_picks_script_used(self):
        self.assertIn("generate_recommended_picks.py", self.text)
        self.assertIn("--budget", self.text)
        self.assertIn("--output-dir picks", self.text)
        self.assertIn("--ev-gate", self.text)
        self.assertIn("--ev-min-ratio", self.text)

    def test_telegram_step_uses_file_existence_check(self):
        self.assertIn('if [ ! -f picks/tickets.json ]', self.text)
        # Skip path now prefers picks/skip_notice.txt written by the orchestrator.
        self.assertIn('skip_notice.txt', self.text)
        self.assertIn('no tickets emitted', self.text)

    def test_telegram_step_sends_messages(self):
        self.assertIn("workflow_messages.py", self.text)
        self.assertIn("TELEGRAM_BOT_TOKEN", self.text)
        self.assertIn("TELEGRAM_CHAT_ID", self.text)
        self.assertIn("curl", self.text)

    def test_ledger_is_committed_to_repo(self):
        # Ledger must persist across scheduled runs via a git commit.
        self.assertIn("Commit budget ledger", self.text)
        self.assertIn("data/budget_bank.json", self.text)
        self.assertIn("contents: write", self.text)
