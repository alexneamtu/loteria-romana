import json
import tempfile
import unittest
from pathlib import Path

from shared.telegram_formatter import (
    format_summary,
    format_tickets,
    format_check_results,
    emit_messages_for_workflow,
)


def _write_fixture(path: Path, tickets_list: list[dict]) -> Path:
    path.write_text(
        json.dumps({
            "generated_at": "2026-04-21T10:00:00Z",
            "budget_ron": 70.0,
            "total_cost_ron": 70.0,
            "allocation": {"joker": 1, "loto_649": 0, "loto_540": 1},
            "tickets": tickets_list,
        })
    )
    return path


class TestTelegramFormatter(unittest.TestCase):
    def test_joker_ticket_message_includes_variants_and_side_game(self):
        msg = format_tickets([{
            "game": "joker",
            "variants": [
                {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
            ],
            "side_game_number": "NP07",
            "strategy": "core_share",
            "cost_ron": 17.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("JOKER", msg)
        self.assertIn("core_share", msg)
        self.assertIn("17.5", msg)
        self.assertIn("3, 7, 12, 19, 28", msg)
        self.assertIn("+ J11", msg)
        self.assertIn("NP07", msg)

    def test_loto_649_message_shows_noroc(self):
        msg = format_tickets([{
            "game": "loto_649",
            "variants": [
                {"main_numbers": [1, 5, 17, 23, 34, 49], "bonus_number": None},
                {"main_numbers": [2, 6, 18, 24, 35, 47], "bonus_number": None},
                {"main_numbers": [3, 7, 19, 25, 36, 48], "bonus_number": None},
            ],
            "side_game_number": "1234567",
            "strategy": "wheel",
            "cost_ron": 28.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("LOTO 6/49", msg)
        self.assertIn("1234567", msg)
        self.assertIn("Noroc", msg)

    def test_loto_540_message_shows_super_noroc(self):
        msg = format_tickets([{
            "game": "loto_540",
            "variants": [
                {"main_numbers": [2, 9, 15, 27, 34], "bonus_number": None},
            ] * 4,
            "side_game_number": "012345",
            "strategy": "independent",
            "cost_ron": 22.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("LOTO 5/40", msg)
        self.assertIn("012345", msg)
        self.assertIn("Super Noroc", msg)

    def test_summary_aggregates_all_tickets(self):
        tickets = [
            {"game": "joker", "variants": [], "side_game_number": "NP07", "strategy": "core_share", "cost_ron": 17.5},
            {"game": "loto_540", "variants": [], "side_game_number": "012345", "strategy": "independent", "cost_ron": 22.5},
        ]
        msg = format_summary(tickets, budget_ron=70.0, total_cost_ron=40.0, draw_date="2026-04-21")
        self.assertIn("Lottery Picks", msg)
        self.assertIn("70.0", msg)
        self.assertIn("Joker", msg)
        self.assertIn("Loto 5/40", msg)

    def test_emit_messages_for_workflow_returns_ordered_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(Path(tmp) / "tickets.json", [{
                "game": "joker",
                "variants": [
                    {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                    {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                ],
                "side_game_number": "NP07",
                "strategy": "core_share",
                "cost_ron": 17.5,
            }])
            msgs = emit_messages_for_workflow(path, draw_date="2026-04-21")
            self.assertEqual(len(msgs), 2)  # summary + 1 ticket
            self.assertIn("Lottery Picks", msgs[0])
            self.assertIn("core_share", msgs[1])

    def test_check_results_message_shows_match_and_side_game_digits(self):
        msg = format_check_results(
            game="loto_649",
            draw_date="2026-04-21",
            winning_main=[1, 5, 17, 23, 34, 49],
            winning_side="1234567",
            ticket_outcomes=[
                {
                    "ticket_id": "t-01",
                    "strategy": "wheel",
                    "best_main_match": 4,
                    "side_game_match": 0,
                    "side_game_digits_matched": 4,
                    "variants_rendered": [
                        "1, 5, 17, 23, 34, 40 [4 hit]",
                        "2, 5, 17, 23, 34, 49 [5 hit]",
                        "3, 5, 17, 23, 34, 48 [4 hit]",
                    ],
                    "ticket_side": "9994567",
                }
            ],
        )
        self.assertIn("LOTO 6/49", msg)
        self.assertIn("Winning:", msg)
        self.assertIn("1234567", msg)
        self.assertIn("4 digits", msg)
        self.assertIn("[5 hit]", msg)


if __name__ == "__main__":
    unittest.main()


class TestBuyList(unittest.TestCase):
    def _ticket(self, n_variants=4, cost=22.5):
        return {
            "game": "loto_540",
            "variants": [{"main_numbers": [2, 9, 15, 27, 34], "bonus_number": None}] * n_variants,
            "side_game_number": "012345",
            "strategy": "independent",
            "cost_ron": cost,
        }

    def test_one_message_per_game_with_bulletin_shape(self):
        msgs = format_tickets([self._ticket(), self._ticket()], draw_date="2026-08-17")
        self.assertEqual(len(msgs), 1)
        self.assertIn("buy 2 ticket(s)", msgs[0])
        self.assertIn("Each bulletin: 4 variants + Super Noroc", msgs[0])
        self.assertIn("Total: 45.00 RON", msgs[0])
        self.assertIn("#1", msgs[0])
        self.assertIn("#2", msgs[0])

    def test_flags_ticket_that_cannot_fill_a_bulletin(self):
        msg = format_tickets([self._ticket(n_variants=2)], draw_date="2026-08-17")[0]
        self.assertIn("⚠️", msg)
        self.assertIn("do not have 4 variants", msg)

    def test_splits_when_over_telegram_limit(self):
        msgs = format_tickets([self._ticket() for _ in range(80)], draw_date="2026-08-17")
        self.assertGreater(len(msgs), 1)
        self.assertTrue(all(len(m) < 4096 for m in msgs))
        self.assertEqual(sum(m.count("#") for m in msgs), 80)
