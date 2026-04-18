import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGeneratePicksWorkflowMessages(unittest.TestCase):
    def test_emit_messages_writes_one_per_line_null_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tickets.json"
            p.write_text(json.dumps({
                "generated_at": "x",
                "budget_ron": 70.0,
                "total_cost_ron": 40.0,
                "allocation": {"joker": 1, "loto_649": 0, "loto_540": 1},
                "tickets": [
                    {
                        "game": "joker",
                        "variants": [
                            {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                            {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                        ],
                        "side_game_number": "NP07",
                        "strategy": "core_share",
                        "cost_ron": 17.5,
                    }
                ],
            }))
            out = subprocess.check_output(
                [
                    sys.executable, "scripts/workflow_messages.py",
                    "--tickets-json", str(p),
                    "--date", "2026-04-21",
                    "--format", "picks",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
                text=True,
            )
            # Summary + 1 ticket = 2 messages, NUL-separated with trailing NUL
            self.assertEqual(out.count("\x00"), 2)
            self.assertIn("Lottery Picks", out)
            self.assertIn("core_share", out)


if __name__ == "__main__":
    unittest.main()
