import random
import unittest

from shared.game_config import LOTO_649_CONFIG
from shared.ticket_backtester import TicketBacktester
from shared.ticket_builders import IndependentBuilder


class TestTicketBacktester(unittest.TestCase):
    def test_walk_forward_produces_outcome_per_draw_after_warmup(self):
        # 100 fake draws
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 50), 6)) for _ in range(100)]
        bt = TicketBacktester(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=draws,
            draw_dates=None,
            warmup=50,
            rng=random.Random(0),
        )
        outcomes = bt.run(IndependentBuilder(n_tickets=1))
        # 100 draws − 50 warmup = 50 outcomes
        self.assertEqual(len(outcomes), 50)
        for o in outcomes:
            self.assertGreaterEqual(o.best_match, 0)
            self.assertLessEqual(o.best_match, 6)

    def test_summary_keys(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 50), 6)) for _ in range(60)]
        bt = TicketBacktester(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=draws,
            draw_dates=None,
            warmup=50,
            rng=random.Random(0),
        )
        outcomes = bt.run(IndependentBuilder(n_tickets=1))
        summary = bt.summarize(outcomes)
        self.assertIn("median_roi", summary)
        self.assertIn("skewness", summary)
        self.assertIn("p_best_match_ge_4", summary)
        self.assertEqual(summary["n"], 10)

    def test_reproducible_with_seeded_rng(self):
        rng_draws = random.Random(1)
        draws = [sorted(rng_draws.sample(range(1, 50), 6)) for _ in range(60)]
        bt1 = TicketBacktester(
            game="loto_649", config=LOTO_649_CONFIG, draws=draws,
            draw_dates=None, warmup=50, rng=random.Random(0),
        )
        bt2 = TicketBacktester(
            game="loto_649", config=LOTO_649_CONFIG, draws=draws,
            draw_dates=None, warmup=50, rng=random.Random(0),
        )
        o1 = bt1.run(IndependentBuilder(n_tickets=1))
        o2 = bt2.run(IndependentBuilder(n_tickets=1))
        self.assertEqual(
            [o.best_match for o in o1],
            [o.best_match for o in o2],
        )


if __name__ == "__main__":
    unittest.main()
