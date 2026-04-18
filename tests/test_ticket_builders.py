import random
import unittest

from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ticket import Ticket
from shared.ticket_builders import BuilderContext, IndependentBuilder


def _joker_draws():
    return [
        [3, 7, 12, 19, 28],
        [5, 17, 18, 24, 31],
        [6, 9, 18, 22, 27],
    ] * 20  # 60 draws — enough for softmax to fit


def _649_draws():
    return [[1, 5, 17, 23, 34, 49], [3, 12, 27, 31, 38, 44]] * 20


def _540_draws():
    return [[2, 9, 15, 27, 34, 38], [1, 5, 19, 22, 28, 40]] * 20


class TestIndependentBuilder(unittest.TestCase):
    def test_joker_builds_one_ticket_with_2_variants(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        tickets = IndependentBuilder(n_tickets=1).build(ctx)
        self.assertEqual(len(tickets), 1)
        t = tickets[0]
        self.assertIsInstance(t, Ticket)
        self.assertEqual(t.game, "joker")
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.strategy, "independent")
        for v in t.variants:
            self.assertIsNotNone(v.bonus_number)

    def test_joker_variants_are_independent_not_shared_core(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertNotEqual(t.variants[0].main_numbers, t.variants[1].main_numbers)

    def test_loto_649_builds_3_variants(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(len(t.variants), 3)
        self.assertEqual(t.game, "loto_649")
        for v in t.variants:
            self.assertIsNone(v.bonus_number)
            self.assertEqual(len(v.main_numbers), 6)

    def test_loto_540_builds_4_variants(self):
        ctx = BuilderContext(
            game="loto_540",
            config=LOTO_540_CONFIG,
            draws=_540_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(len(t.variants), 4)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(7),
        )
        ctx2 = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(7),
        )
        t1 = IndependentBuilder(n_tickets=1).build(ctx1)[0]
        t2 = IndependentBuilder(n_tickets=1).build(ctx2)[0]
        self.assertEqual(t1.variants[0].main_numbers, t2.variants[0].main_numbers)

    def test_ticket_cost_matches_pricing_module(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(t.cost_ron, 17.5)  # 2 * 7.0 + 0.5 fee + 3.0 Noroc Plus

    def test_multi_ticket_output(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        tickets = IndependentBuilder(n_tickets=3).build(ctx)
        self.assertEqual(len(tickets), 3)


if __name__ == "__main__":
    unittest.main()
