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
        self.assertIn("default: '40'", self.text)
        self.assertIn("default: 'false'", self.text)

    def test_recommended_picks_script_used(self):
        self.assertIn("generate_recommended_picks.py", self.text)
        self.assertIn("--budget", self.text)
        self.assertIn("--output-dir picks", self.text)
        self.assertIn("--ev-gate", self.text)
        self.assertIn("--ev-min-ratio", self.text)

    def test_telegram_step_uses_file_existence_check(self):
        self.assertIn('if [ -f "picks/joker.txt" ]', self.text)
        self.assertIn('if [ -f "picks/loto649.txt" ]', self.text)
        self.assertIn('if [ -f "picks/loto540.txt" ]', self.text)

    def test_telegram_step_sends_messages(self):
        self.assertIn("send_message", self.text)
        self.assertIn("TELEGRAM_BOT_TOKEN", self.text)
        self.assertIn("TELEGRAM_CHAT_ID", self.text)
