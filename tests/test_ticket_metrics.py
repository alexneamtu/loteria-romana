import statistics
import unittest

from shared.ticket import Ticket, Variant
from shared.ticket_metrics import (
    TicketOutcome,
    estimate_ticket_payout,
    median_roi,
    p_best_match_at_least,
    sample_skewness,
    summarize,
)


def _make_joker_ticket(mains_a, mains_b, bonus=11) -> Ticket:
    return Ticket(
        game="joker",
        variants=(
            Variant(tuple(sorted(mains_a)), bonus, "joker"),
            Variant(tuple(sorted(mains_b)), bonus, "joker"),
        ),
        side_game_number="NP14",
        strategy="test",
        cost_ron=17.5,
    )


class TestTicketMetrics(unittest.TestCase):
    def test_payout_jackpot_tier(self):
        # 5 mains + joker = Category I (jackpot)
        t = _make_joker_ticket(
            mains_a=[3, 7, 12, 19, 28],
            mains_b=[5, 9, 13, 20, 30],
            bonus=11,
        )
        winning = ([3, 7, 12, 19, 28], 11)
        payout = estimate_ticket_payout(t, winning_main=winning[0], winning_joker=winning[1], winning_side=None, jackpot=1_000_000)
        self.assertGreater(payout, 500_000)  # At least half the jackpot

    def test_payout_small_fixed_tier(self):
        # 3 mains, no joker = Category VI (14 RON fixed)
        t = _make_joker_ticket(
            mains_a=[3, 7, 12, 40, 41],
            mains_b=[3, 7, 12, 42, 43],
            bonus=19,  # not the winning joker
        )
        winning_main = [3, 7, 12, 19, 28]
        payout = estimate_ticket_payout(t, winning_main=winning_main, winning_joker=11, winning_side=None, jackpot=1_000_000)
        # Best variant has 3 matches, no joker → Category VI = 14 RON
        self.assertGreaterEqual(payout, 14.0)
        self.assertLess(payout, 1000.0)

    def test_payout_zero_when_no_tier_hit(self):
        t = _make_joker_ticket(
            mains_a=[1, 2, 40, 41, 42],
            mains_b=[3, 4, 43, 44, 45],
            bonus=19,
        )
        payout = estimate_ticket_payout(t, winning_main=[30, 31, 32, 33, 34], winning_joker=11, winning_side=None, jackpot=1_000_000)
        self.assertEqual(payout, 0.0)

    def test_median_roi_handles_losses(self):
        outcomes = [
            TicketOutcome(payout=0.0, cost=14.5, best_match=1),
            TicketOutcome(payout=14.0, cost=14.5, best_match=3),
            TicketOutcome(payout=0.0, cost=14.5, best_match=2),
            TicketOutcome(payout=100.0, cost=14.5, best_match=4),
        ]
        # ROIs: -14.5, -0.5, -14.5, 85.5 → median ≈ -7.5
        self.assertAlmostEqual(median_roi(outcomes), -7.5, places=1)

    def test_p_best_match_at_least(self):
        outcomes = [
            TicketOutcome(payout=0, cost=14.5, best_match=1),
            TicketOutcome(payout=14, cost=14.5, best_match=3),
            TicketOutcome(payout=100, cost=14.5, best_match=4),
            TicketOutcome(payout=0, cost=14.5, best_match=2),
        ]
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 3), 0.5)
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 4), 0.25)
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 5), 0.0)

    def test_sample_skewness_positive_for_heavy_right_tail(self):
        # Heavy right tail (jackpot-like): many losses, one huge win
        values = [-14.5] * 99 + [1_000_000]
        skew = sample_skewness(values)
        self.assertGreater(skew, 5.0)

    def test_sample_skewness_near_zero_for_symmetric_distribution(self):
        values = [i for i in range(-50, 51)]
        self.assertAlmostEqual(sample_skewness(values), 0.0, places=1)

    def test_summarize_returns_all_three_metrics(self):
        outcomes = [
            TicketOutcome(payout=0, cost=14.5, best_match=1),
            TicketOutcome(payout=100, cost=14.5, best_match=4),
        ]
        s = summarize(outcomes)
        self.assertIn("median_roi", s)
        self.assertIn("p_best_match_ge_4", s)
        self.assertIn("skewness", s)
        self.assertIn("n", s)
        self.assertEqual(s["n"], 2)


if __name__ == "__main__":
    unittest.main()
