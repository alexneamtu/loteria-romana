import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.generate_recommended_picks as recommended_script
from shared.ev_calculator import EVCalculator
from shared.ticket_allocator import TicketAllocation


def _breakeven(factory) -> float:
    calc = EVCalculator()
    return calc._calculate_positive_ev_jackpot(factory(), 1.0, None)  # noqa: SLF001


def joker_jackpot_at(ratio: float) -> float:
    """Joker jackpot giving this jackpot/breakeven ratio."""
    return round(ratio * _breakeven(EVCalculator.create_joker), 2)


def loto540_jackpot_at(ratio: float) -> float:
    """5/40 jackpot giving this jackpot/breakeven ratio.

    Derived, not hard-coded: these tests exercise the gate's skip/play/boost
    bands, and a frozen jackpot silently changes band whenever the breakeven
    is corrected (as it was when Category I dropped from 6/C(40,5) to 1).
    """
    return round(ratio * _breakeven(EVCalculator.create_loto_540), 2)


# Subprocess-based tests spawn the real orchestrator end-to-end (loading
# ~2500 draws, training sklearn strategies, auto-scraping live jackpots).
# Each call is 1-2 min on CI. Gate behind SLOW_TESTS=1 so normal CI stays fast.
SLOW_ENABLED = os.environ.get("SLOW_TESTS") == "1"


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


class TestApplyEVGateV2(unittest.TestCase):
    # Per-line breakevens from EVCalculator (main-ticket cost / variants):
    # joker ≈ 169M, loto_649 ≈ 108M, loto_540 ≈ 3.36M.
    SKIP_RATIO = 0.5
    BOOST_RATIO = 1.2

    def test_skip_when_all_ratios_below_threshold(self):
        decision = recommended_script.apply_ev_gate_v2(
            budget=40.0,
            jackpots={"joker": 50_000_000.0, "loto_649": 50_000_000.0, "loto_540": loto540_jackpot_at(0.36)},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
        )
        self.assertEqual(decision.action, "skip")
        self.assertIn("joker=", decision.reason)
        self.assertIn("loto_649=", decision.reason)
        self.assertIn("loto_540=", decision.reason)

    def test_play_when_some_ratio_in_band(self):
        # ratio ~0.91: above skip_ratio, below boost_ratio -> play
        decision = recommended_script.apply_ev_gate_v2(
            budget=40.0,
            jackpots={"joker": 1.0, "loto_649": 1.0, "loto_540": loto540_jackpot_at(0.91)},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
        )
        self.assertEqual(decision.action, "play")
        self.assertIn("loto_540=", decision.reason)

    def test_boost_when_max_ratio_above_threshold(self):
        # ratio ~6.4: well above boost_ratio -> boost
        decision = recommended_script.apply_ev_gate_v2(
            budget=40.0,
            jackpots={"joker": 1.0, "loto_649": 1.0, "loto_540": loto540_jackpot_at(6.38)},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
        )
        self.assertEqual(decision.action, "boost")
        # No bank yet, so nothing to release.
        self.assertEqual(decision.extra_budget, 0.0)
        self.assertIn("loto_540=", decision.reason)

    def test_boost_releases_a_share_of_the_bank_not_a_flat_budget(self):
        # A flat one-budget release could never drain a bank fed by far
        # more skips than boosts; it grew monotonically to 2240 RON.
        decision = recommended_script.apply_ev_gate_v2(
            budget=70.0,
            jackpots={"joker": 1.0, "loto_649": 1.0, "loto_540": loto540_jackpot_at(6.38)},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
            ledger_balance=2240.0,
            boost_fraction=0.25,
        )
        self.assertEqual(decision.action, "boost")
        self.assertEqual(decision.extra_budget, 560.0)

    def test_boost_drains_geometrically_and_never_goes_negative(self):
        balance = 2240.0
        releases = []
        for _ in range(6):
            decision = recommended_script.apply_ev_gate_v2(
                budget=70.0,
                jackpots={"joker": 1.0, "loto_649": 1.0, "loto_540": loto540_jackpot_at(6.38)},
                skip_ratio=self.SKIP_RATIO,
                boost_ratio=self.BOOST_RATIO,
                ledger_balance=balance,
                boost_fraction=0.25,
            )
            releases.append(decision.extra_budget)
            balance -= decision.extra_budget

        self.assertEqual(releases, sorted(releases, reverse=True))
        self.assertLess(balance, 2240.0 * 0.2)
        self.assertGreater(balance, 0.0)

    def test_joker_uses_main_ticket_cost_not_bundled(self):
        # Ratio 0.56 on main-ticket-only cost (14.5) -> play. Had the gate
        # used the bundled cost (17.5 incl. Noroc Plus), breakeven would be
        # ~21% higher and the same jackpot would read 0.46 -> skip.
        decision = recommended_script.apply_ev_gate_v2(
            budget=40.0,
            jackpots={"joker": joker_jackpot_at(0.56), "loto_649": 1.0, "loto_540": 1.0},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
        )
        self.assertEqual(decision.action, "play")

    def test_missing_jackpot_treated_as_zero_ratio(self):
        decision = recommended_script.apply_ev_gate_v2(
            budget=40.0,
            jackpots={"joker": None, "loto_649": None, "loto_540": None},
            skip_ratio=self.SKIP_RATIO,
            boost_ratio=self.BOOST_RATIO,
        )
        self.assertEqual(decision.action, "skip")
        self.assertIn("joker=0.00", decision.reason)


class TestGateThresholdsStayCoherent(unittest.TestCase):
    """The global skip gate and the per-game filter must agree by default.

    They used to default to 0.5 and 0.8 independently, so a draw whose max
    ratio landed between them was "played" with every game filtered out:
    no tickets, no ledger credit, budget silently lost.
    """

    def test_skip_ratio_defaults_to_min_ratio(self):
        args = recommended_script.build_parser().parse_args(["--budget", "70", "--ev-gate"])
        self.assertIsNone(args.ev_skip_ratio)

        with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "apply_ev_gate_v2") as gate_v2, \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
            mock.patch.object(recommended_script.BudgetLedger, "credit_skip"), \
            mock.patch.object(recommended_script.BudgetLedger, "balance", return_value=0.0), \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", [
                "generate_recommended_picks.py", "--budget", "70", "--ev-gate",
                "--ev-min-ratio", "0.10",
                "--joker-jackpot", "1", "--loto649-jackpot", "1", "--loto540-jackpot", "1",
            ]):
            gate_v2.return_value = recommended_script.EVDecision(action="skip", reason="test")
            recommended_script.main()

        self.assertEqual(gate_v2.call_args.kwargs["skip_ratio"], 0.10)

    def test_dead_band_credits_ledger_and_writes_skip_notice(self):
        # ratio ≈ 0.22: above skip_ratio (plays) but below min_ratio
        # (every game blocked) — the dead band.
        nonzero = TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 1},
            total_cost=20.5,
            p_any_win=0.05,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "bank.json"
            out_dir = Path(tmp) / "picks"
            with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
                mock.patch.object(recommended_script, "best_allocation", return_value=nonzero), \
                mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
                mock.patch("builtins.print"), \
                mock.patch("sys.argv", [
                    "generate_recommended_picks.py", "--budget", "70", "--ev-gate",
                    "--ev-skip-ratio", "0.01", "--ev-min-ratio", "0.80",
                    "--joker-jackpot", "1", "--loto649-jackpot", "1",
                    "--loto540-jackpot", str(loto540_jackpot_at(0.22)),
                    "--ledger-path", str(ledger_path),
                    "--output-dir", str(out_dir),
                ]):
                recommended_script.main()

            self.assertEqual(json.loads(ledger_path.read_text())["balance"], 70.0)
            notice = (out_dir / "skip_notice.txt").read_text()
            self.assertIn("no game reached min ratio", notice)
            self.assertIn("loto_540=0.22", notice)

    def test_dead_band_credits_ledger_in_multi_mix_path_too(self):
        # --mixes > 1 filters allocations in its own loop; an empty result
        # made the render loop a no-op and skipped the ledger credit.
        nonzero = TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 1},
            total_cost=20.5,
            p_any_win=0.05,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "bank.json"
            out_dir = Path(tmp) / "picks"
            with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
                mock.patch.object(recommended_script, "_top_n_diverse_allocations", return_value=[nonzero, nonzero]), \
                mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
                mock.patch("builtins.print"), \
                mock.patch("sys.argv", [
                    "generate_recommended_picks.py", "--budget", "70", "--ev-gate",
                    "--mixes", "3",
                    "--ev-skip-ratio", "0.01", "--ev-min-ratio", "0.80",
                    "--joker-jackpot", "1", "--loto649-jackpot", "1",
                    "--loto540-jackpot", str(loto540_jackpot_at(0.22)),
                    "--ledger-path", str(ledger_path),
                    "--output-dir", str(out_dir),
                ]):
                recommended_script.main()

            self.assertEqual(json.loads(ledger_path.read_text())["balance"], 70.0)
            self.assertIn(
                "no game reached min ratio",
                (out_dir / "skip_notice.txt").read_text(),
            )

    def test_unspendable_boost_is_returned_to_the_ledger(self):
        # The allocator caps tickets per game, so a release can exceed what
        # is buyable. The surplus must go back to the bank, not vanish.
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "bank.json"
            ledger_path.write_text(json.dumps({"balance": 20000.0, "entries": []}))
            small = TicketAllocation(
                tickets={"joker": 0, "loto_649": 0, "loto_540": 20},
                total_cost=450.0,
                p_any_win=0.01,
            )
            with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
                mock.patch.object(recommended_script, "best_allocation", return_value=small), \
                mock.patch.object(recommended_script, "apply_ev_gate", return_value=(small, {})), \
                mock.patch.object(recommended_script, "_build_tickets_for_allocation", return_value=[]), \
                mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
                mock.patch("builtins.print"), \
                mock.patch("sys.argv", [
                    "generate_recommended_picks.py", "--budget", "70", "--ev-gate",
                    "--ev-min-ratio", "0.10", "--ev-boost-fraction", "0.25",
                    "--joker-jackpot", "1", "--loto649-jackpot", "1",
                    "--loto540-jackpot", str(loto540_jackpot_at(1.82)),
                    "--ledger-path", str(ledger_path),
                ]):
                recommended_script.main()

            data = json.loads(ledger_path.read_text())
            kinds = [e["kind"] for e in data["entries"]]
            self.assertEqual(kinds, ["debit", "credit"])
            # Released 5000, spent 450 of the 5070 effective budget. The
            # first 70 comes from the base budget, so only 380 of the boost
            # was consumed and the rest goes back.
            self.assertEqual(data["entries"][0]["amount"], 5000.0)
            self.assertEqual(data["entries"][1]["amount"], 5070.0 - 450.0)
            self.assertEqual(data["balance"], 20000.0 - 380.0)

    def test_no_double_credit_when_global_gate_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "bank.json"
            with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
                mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
                mock.patch("builtins.print"), \
                mock.patch("sys.argv", [
                    "generate_recommended_picks.py", "--budget", "70", "--ev-gate",
                    "--ev-min-ratio", "0.10",
                    "--joker-jackpot", "1", "--loto649-jackpot", "1", "--loto540-jackpot", "1",
                    "--ledger-path", str(ledger_path),
                ]):
                recommended_script.main()

            self.assertEqual(json.loads(ledger_path.read_text())["balance"], 70.0)


class TestBuilderInvocation(unittest.TestCase):
    def test_independent_allocation_builds_in_one_call_per_game(self):
        # IndependentBuilder runs one ensemble backtest per build() call, so
        # a game's n tickets must be built in a single call — not n calls,
        # which re-ran the backtest n times (~18 min for a boosted run on
        # the ARM runner).
        alloc = TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 5},
            total_cost=112.5,
            p_any_win=0.1,
        )
        calls: list[int] = []

        class _FakeBuilder:
            def __init__(self, n: int):
                self.n = n

            def build(self, ctx):
                return [object() for _ in range(self.n)]

        def _fake_spec(spec, n_tickets=1):
            calls.append(n_tickets)
            return _FakeBuilder(n_tickets)

        with mock.patch.object(recommended_script, "_builder_for_spec", side_effect=_fake_spec):
            tickets = recommended_script._build_tickets_for_allocation(
                alloc, random.Random(1), 50.0, "draws", "independent"
            )

        self.assertEqual(calls, [5])
        self.assertEqual(len(tickets), 5)


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


@unittest.skipUnless(SLOW_ENABLED, "set SLOW_TESTS=1 to run orchestrator subprocess tests")
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


@unittest.skipUnless(SLOW_ENABLED, "set SLOW_TESTS=1 to run orchestrator subprocess tests")
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
            # Pre-credit enough that 25% of the balance funds more than one
            # extra Joker ticket, so the released boost is actually deployed
            # and the ledger is net-debited. A smaller release would be spent
            # only up to ticket granularity and the remainder credited back
            # by the unused-boost refund, leaving the balance unchanged.
            ledger_path.write_text(json.dumps({"balance": 1000.0, "entries": []}))
            subprocess.check_call([
                sys.executable, "scripts/generate_recommended_picks.py",
                "--budget", "70", "--seed", "42", "--output-dir", str(out),
                "--ev-gate", "--ev-boost-ratio", "1.2",
                "--joker-jackpot", "600000000",  # ratio > 1.2 → boost
                # Provide the other two so the test never hits the network.
                "--loto649-jackpot", "0",
                "--loto540-jackpot", "0",
                "--ledger-path", str(ledger_path),
            ], env={"PYTHONPATH": "src", "PATH": ""})
            # tickets.json emitted with budget > 70 (boosted)
            doc = json.loads((out / "tickets.json").read_text())
            self.assertGreater(doc["budget_ron"], 70.0)
            # Ledger net-debited after the boost was deployed
            led = json.loads(ledger_path.read_text())
            self.assertLess(led["balance"], 1000.0)


if __name__ == "__main__":
    unittest.main()


class TestBudgetIsNotMultiplied(unittest.TestCase):
    def test_bucket_budget_over_total_is_rejected(self):
        with self.assertRaises(SystemExit):
            recommended_script._parse_bucket_budget(  # noqa: SLF001
                "joker=400,loto_649=400,loto_540=400", 400.0
            )

    def test_bucket_budget_within_total_is_accepted(self):
        buckets = recommended_script._parse_bucket_budget(  # noqa: SLF001
            "joker=100,loto_649=200,loto_540=100", 400.0
        )
        self.assertEqual(sum(buckets.values()), 400.0)

    def test_mixes_share_the_budget_instead_of_each_taking_it(self):
        seen = []
        with mock.patch.object(
            recommended_script, "_top_n_diverse_allocations",
            side_effect=lambda budget, n: seen.append(budget) or [],
        ), mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True), \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", [
                "generate_recommended_picks.py", "--budget", "400", "--mixes", "4",
            ]):
            recommended_script.main()
        self.assertEqual(seen, [100.0])  # 400 / 4, not 400 four times
