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
