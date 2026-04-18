import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.generate_recommended_picks as recommended_script
from shared.ticket_allocator import TicketAllocation


class TestEVGate(unittest.TestCase):
    def test_gate_disabled_keeps_allocation(self):
        allocation = TicketAllocation(
            tickets={"joker": 2, "loto_649": 1, "loto_540": 0},
            total_cost=22.0,
            p_any_win=0.10,
        )
        gated, details = recommended_script.apply_ev_gate(
            allocation=allocation,
            enabled=False,
            min_ratio=0.8,
            jackpots={},
        )
        self.assertEqual(gated.tickets, allocation.tickets)
        self.assertEqual(gated.total_cost, allocation.total_cost)
        self.assertEqual(details, {})

    def test_gate_filters_below_threshold_games(self):
        allocation = TicketAllocation(
            tickets={"joker": 2, "loto_649": 2, "loto_540": 1},
            total_cost=32.0,
            p_any_win=0.15,
        )
        jackpots = {
            "joker": 1.0,
            "loto_649": 1_000_000_000.0,
            "loto_540": 1.0,
        }
        gated, details = recommended_script.apply_ev_gate(
            allocation=allocation,
            enabled=True,
            min_ratio=0.8,
            jackpots=jackpots,
        )
        self.assertEqual(gated.tickets["joker"], 0)
        self.assertGreaterEqual(gated.tickets["loto_649"], 1)
        self.assertEqual(gated.tickets["loto_540"], 0)
        self.assertIn("joker", details)
        self.assertIn("loto_649", details)


class TestGenerationDBPersistenceHook(unittest.TestCase):
    def test_main_persists_even_when_allocation_has_zero_win_probability(self):
        zero_allocation = TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
        )
        with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "best_allocation", return_value=zero_allocation), \
            mock.patch.object(recommended_script, "apply_ev_gate", return_value=(zero_allocation, {})), \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True) as persist, \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", ["generate_recommended_picks.py", "--budget", "40"]):
            recommended_script.main()

        self.assertEqual(persist.call_count, 1)
        _, kwargs = persist.call_args
        self.assertEqual(kwargs["tickets"], [])
        self.assertEqual(kwargs["allocation"]["p_any_win"], 0.0)

    def test_main_persists_generated_ticket_rows(self):
        from shared.ticket import Ticket, Variant
        allocation = TicketAllocation(
            tickets={"joker": 1, "loto_649": 0, "loto_540": 0},
            total_cost=17.5,
            p_any_win=0.1,
        )
        fake_ticket = Ticket(
            game="joker",
            variants=(
                Variant(main_numbers=(1, 2, 3, 4, 5), bonus_number=7, game="joker"),
                Variant(main_numbers=(2, 3, 4, 5, 6), bonus_number=8, game="joker"),
            ),
            side_game_number="123456789",
            strategy="independent",
            cost_ron=17.5,
        )
        draws = [SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5], joker=7)]
        with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "best_allocation", return_value=allocation), \
            mock.patch.object(recommended_script, "apply_ev_gate", return_value=(allocation, {})), \
            mock.patch.object(recommended_script, "load_joker_draws", return_value=draws), \
            mock.patch.object(recommended_script, "_build_tickets_for_allocation", return_value=[fake_ticket]), \
            mock.patch.object(recommended_script, "dump_tickets"), \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True) as persist, \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", ["generate_recommended_picks.py", "--budget", "20"]):
            recommended_script.main()

        self.assertEqual(persist.call_count, 1)
        _, kwargs = persist.call_args
        # One dict per variant (legacy DB schema counts lines, not tickets)
        self.assertGreater(len(kwargs["tickets"]), 0)
        self.assertEqual(kwargs["tickets"][0]["game"], "joker")
        self.assertEqual(kwargs["tickets"][0]["main_numbers"], [1, 2, 3, 4, 5])
        self.assertEqual(kwargs["tickets"][0]["joker_number"], 7)


class TestGenerateTicketsJSON(unittest.TestCase):
    def test_emits_tickets_json_with_expected_allocation_at_70_ron(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "70",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "independent",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            tickets_path = out / "tickets.json"
            self.assertTrue(tickets_path.exists())
            doc = json.loads(tickets_path.read_text())
            self.assertEqual(doc["budget_ron"], 70.0)
            self.assertGreater(len(doc["tickets"]), 0)
            games = [t["game"] for t in doc["tickets"]]
            # At 70 RON with best_allocation, expect all three games
            self.assertIn("joker", games)

    def test_each_ticket_has_correct_variant_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "70",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "core_share",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            doc = json.loads((out / "tickets.json").read_text())
            expected_variants = {"joker": 2, "loto_649": 3, "loto_540": 4}
            for t in doc["tickets"]:
                self.assertEqual(len(t["variants"]), expected_variants[t["game"]])

    def test_legacy_txt_files_no_longer_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "70",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "independent",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            self.assertEqual(list(out.glob("*.txt")), [])


class TestEVSkipBoost(unittest.TestCase):
    def test_skip_when_all_ratios_below_skip_ratio(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            # All jackpots zero → ratio 0 → skip
            subprocess.check_call([
                sys.executable, "scripts/generate_recommended_picks.py",
                "--budget", "70", "--seed", "42", "--output-dir", str(out),
                "--ev-gate", "--ev-skip-ratio", "0.5",
                "--joker-jackpot", "0",
                "--loto649-jackpot", "0",
                "--loto540-jackpot", "0",
                "--ledger-path", str(out / "ledger.json"),
            ], env={"PYTHONPATH": "src", "PATH": ""})
            # Expect: no tickets.json emitted, ledger has +70 credit
            self.assertFalse((out / "tickets.json").exists())
            led = json.loads((out / "ledger.json").read_text())
            self.assertEqual(led["balance"], 70.0)

    def test_boost_when_ratio_above_boost_ratio(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            ledger_path = out / "ledger.json"
            out.mkdir(parents=True)
            # Pre-credit the ledger
            ledger_path.write_text(json.dumps({"balance": 40.0, "entries": []}))
            subprocess.check_call([
                sys.executable, "scripts/generate_recommended_picks.py",
                "--budget", "70", "--seed", "42", "--output-dir", str(out),
                "--ev-gate", "--ev-boost-ratio", "1.2",
                "--joker-jackpot", "600000000",  # huge jackpot → ratio > 1.2 → boost
                "--ledger-path", str(ledger_path),
            ], env={"PYTHONPATH": "src", "PATH": ""})
            # tickets.json emitted with budget > 70 (boosted)
            doc = json.loads((out / "tickets.json").read_text())
            self.assertGreater(doc["budget_ron"], 70.0)
            # Ledger debited
            led = json.loads(ledger_path.read_text())
            self.assertLess(led["balance"], 40.0)


if __name__ == "__main__":
    unittest.main()
