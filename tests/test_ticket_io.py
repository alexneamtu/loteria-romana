import json
import tempfile
import unittest
from pathlib import Path

from shared.ticket import Ticket, Variant
from shared.ticket_io import dump_tickets, load_tickets


def _joker_ticket() -> Ticket:
    return Ticket(
        game="joker",
        variants=(
            Variant((3, 7, 12, 19, 28), 11, "joker"),
            Variant((3, 7, 12, 19, 33), 11, "joker"),
        ),
        side_game_number="NP14",
        strategy="core_share",
        cost_ron=17.5,
    )


class TestTicketIO(unittest.TestCase):
    def test_round_trip_single_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            dump_tickets(
                path,
                tickets=[_joker_ticket()],
                budget_ron=70.0,
                total_cost_ron=17.5,
                allocation={"joker": 1, "loto_649": 0, "loto_540": 0},
                generated_at="2026-04-19T10:00:00Z",
            )
            loaded = load_tickets(path)
            self.assertEqual(len(loaded["tickets"]), 1)
            t = loaded["tickets"][0]
            self.assertEqual(t.game, "joker")
            self.assertEqual(t.side_game_number, "NP14")
            self.assertEqual(t.variants[0].main_numbers, (3, 7, 12, 19, 28))
            self.assertEqual(t.variants[0].bonus_number, 11)
            self.assertEqual(loaded["budget_ron"], 70.0)
            self.assertEqual(loaded["allocation"]["joker"], 1)

    def test_json_preserves_leading_zero_side_game(self):
        t = Ticket(
            game="joker",
            variants=(
                Variant((1, 2, 3, 4, 5), 1, "joker"),
                Variant((6, 7, 8, 9, 10), 2, "joker"),
            ),
            side_game_number="NP07",
            strategy="independent",
            cost_ron=17.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            dump_tickets(path, [t], budget_ron=70.0, total_cost_ron=17.5, allocation={"joker": 1}, generated_at="x")
            raw = json.loads(path.read_text())
            self.assertEqual(raw["tickets"][0]["side_game_number"], "NP07")

    def test_load_raises_on_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            path.write_text("{ not valid json")
            with self.assertRaises(ValueError):
                load_tickets(path)


if __name__ == "__main__":
    unittest.main()
