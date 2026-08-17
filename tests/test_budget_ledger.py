import tempfile
import unittest
from pathlib import Path

from shared.budget_ledger import BudgetLedger


class TestBudgetLedger(unittest.TestCase):
    def test_new_ledger_has_zero_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            self.assertEqual(led.balance(), 0.0)
            self.assertEqual(led.entries(), [])

    def test_credit_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            self.assertEqual(led.balance(), 40.0)
            es = led.entries()
            self.assertEqual(len(es), 1)
            self.assertEqual(es[0]["kind"], "credit")
            self.assertEqual(es[0]["reason"], "low-ev")

    def test_debit_boost_cannot_exceed_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            actual = led.debit_boost(draw_date="2026-05-04", amount=100.0, reason="jackpot")
            self.assertEqual(actual, 40.0)  # clipped to balance
            self.assertEqual(led.balance(), 0.0)

    def test_round_trip_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            led = BudgetLedger(path)
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            led2 = BudgetLedger(path)
            self.assertEqual(led2.balance(), 40.0)

    def test_debit_without_balance_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            actual = led.debit_boost(draw_date="d", amount=10.0, reason="r")
            self.assertEqual(actual, 0.0)
            self.assertEqual(led.balance(), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestLedgerIdempotency(unittest.TestCase):
    """One draw gets one credit and one debit, no matter how often it runs."""

    def _ledger(self, tmp):
        return BudgetLedger(Path(tmp) / "bank.json")

    def test_rerunning_the_same_draw_does_not_credit_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            led.credit_skip("2026-08-17", 70.0, "skip")
            led.credit_skip("2026-08-17", 70.0, "skip")
            self.assertEqual(led.balance(), 70.0)
            self.assertEqual(len(led.entries()), 1)

    def test_rerunning_the_same_draw_does_not_debit_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            led.credit_skip("2026-08-16", 2240.0, "bank")
            first = led.debit_boost("2026-08-17", 560.0, "boost")
            second = led.debit_boost("2026-08-17", 420.0, "boost")
            self.assertEqual(first, 560.0)
            self.assertEqual(second, 0.0)  # was 420.0: the live 3x double-dip
            self.assertEqual(led.balance(), 1680.0)

    def test_a_later_draw_still_debits(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = self._ledger(tmp)
            led.credit_skip("2026-08-16", 1000.0, "bank")
            led.debit_boost("2026-08-17", 100.0, "boost")
            self.assertEqual(led.debit_boost("2026-08-20", 100.0, "boost"), 100.0)
