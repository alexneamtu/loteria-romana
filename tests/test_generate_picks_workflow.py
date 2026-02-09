from pathlib import Path
import unittest


class TestGeneratePicksWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.workflow = root / ".github/workflows/generate-picks.yml"
        self.text = self.workflow.read_text()

    def test_inputs_present(self):
        for name in ("joker_lines", "loto649_lines", "loto540_lines"):
            self.assertIn(name, self.text)

    def test_default_values_present(self):
        self.assertIn("default: '5'", self.text)  # Joker
        self.assertIn("default: '5'", self.text)  # Loto 6/49
        self.assertIn("default: '5'", self.text)  # Loto 5/40

    def test_resolved_counts_used(self):
        self.assertIn("steps.counts.outputs.joker_lines", self.text)
        self.assertIn("steps.counts.outputs.loto649_lines", self.text)
        self.assertIn("steps.counts.outputs.loto540_lines", self.text)

    def test_outputs_not_used_in_shell_conditionals(self):
        self.assertNotIn('if [ -n "${{ steps.joker.outputs.PICKS }}"', self.text)
        self.assertNotIn('if [ -n "${{ steps.loto649.outputs.PICKS }}"', self.text)
        self.assertNotIn('if [ -n "${{ steps.loto540.outputs.PICKS }}"', self.text)

    def test_telegram_step_avoids_inline_cat_for_missing_files(self):
        self.assertNotIn("cat picks/joker.txt", self.text)
        self.assertNotIn("cat picks/loto649.txt", self.text)
        self.assertNotIn("cat picks/loto540.txt", self.text)
