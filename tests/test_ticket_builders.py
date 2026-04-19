import random
import unittest

from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ticket import Ticket
from shared.ticket_builders import BuilderContext, CoreShareBuilder, IndependentBuilder, WheelBuilder


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


class TestCoreShareBuilder(unittest.TestCase):
    def test_joker_all_variants_share_3_core_numbers(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared_main = set(t.variants[0].main_numbers) & set(t.variants[1].main_numbers)
        self.assertGreaterEqual(len(shared_main), 3)
        self.assertEqual(t.strategy, "core_share")

    def test_loto_649_all_variants_share_4_core_numbers(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared = (
            set(t.variants[0].main_numbers)
            & set(t.variants[1].main_numbers)
            & set(t.variants[2].main_numbers)
        )
        self.assertGreaterEqual(len(shared), 4)

    def test_loto_540_all_variants_share_3_core_numbers(self):
        ctx = BuilderContext(
            game="loto_540",
            config=LOTO_540_CONFIG,
            draws=_540_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared = (
            set(t.variants[0].main_numbers)
            & set(t.variants[1].main_numbers)
            & set(t.variants[2].main_numbers)
            & set(t.variants[3].main_numbers)
        )
        self.assertGreaterEqual(len(shared), 3)

    def test_variants_are_distinct(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        self.assertNotEqual(t.variants[0].main_numbers, t.variants[1].main_numbers)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="joker", config=JOKER_CONFIG, draws=_joker_draws(),
            draw_dates=None, rng=random.Random(99),
        )
        ctx2 = BuilderContext(
            game="joker", config=JOKER_CONFIG, draws=_joker_draws(),
            draw_dates=None, rng=random.Random(99),
        )
        t1 = CoreShareBuilder().build(ctx1)[0]
        t2 = CoreShareBuilder().build(ctx2)[0]
        self.assertEqual(t1.variants[0].main_numbers, t2.variants[0].main_numbers)


class TestCoreShareAntiCrowding(unittest.TestCase):
    def _make_ctx(self, game="loto_649", seed=42):
        draws_by_game = {"joker": _joker_draws, "loto_649": _649_draws, "loto_540": _540_draws}
        config_by_game = {"joker": JOKER_CONFIG, "loto_649": LOTO_649_CONFIG, "loto_540": LOTO_540_CONFIG}
        return BuilderContext(
            game=game,
            config=config_by_game[game],
            draws=draws_by_game[game](),
            draw_dates=None,
            rng=random.Random(seed),
        )

    def test_anti_crowding_off_by_default(self):
        builder = CoreShareBuilder()
        self.assertFalse(builder.anti_crowding)

    def test_anti_crowding_on_raises_average_score(self):
        # Over enough samples, anti_crowding should yield a higher mean score
        # than the default. Single-ticket variance is high; average 20 tickets.
        from shared.crowding import anti_crowding_score

        base_scores: list[float] = []
        acr_scores: list[float] = []
        for seed in range(20):
            ctx_base = self._make_ctx(seed=seed)
            ctx_acr = self._make_ctx(seed=seed)
            t_base = CoreShareBuilder().build(ctx_base)[0]
            t_acr = CoreShareBuilder(anti_crowding=True, anti_crowding_candidates=8).build(ctx_acr)[0]
            for v in t_base.variants:
                base_scores.append(anti_crowding_score(list(v.main_numbers), ctx_base.config.pool_size))
            for v in t_acr.variants:
                acr_scores.append(anti_crowding_score(list(v.main_numbers), ctx_acr.config.pool_size))
        base_mean = sum(base_scores) / len(base_scores)
        acr_mean = sum(acr_scores) / len(acr_scores)
        self.assertGreater(acr_mean, base_mean)

    def test_anti_crowding_preserves_core_share_invariant(self):
        # Even with anti-crowding on, all variants must still share the core.
        ctx = self._make_ctx(game="loto_649", seed=7)
        t = CoreShareBuilder(anti_crowding=True).build(ctx)[0]
        shared = set(t.variants[0].main_numbers)
        for v in t.variants[1:]:
            shared &= set(v.main_numbers)
        self.assertGreaterEqual(len(shared), 4)  # _CORE_K["loto_649"] = 4

    def test_anti_crowding_seeded_is_deterministic(self):
        ctx1 = self._make_ctx(game="joker", seed=99)
        ctx2 = self._make_ctx(game="joker", seed=99)
        t1 = CoreShareBuilder(anti_crowding=True).build(ctx1)[0]
        t2 = CoreShareBuilder(anti_crowding=True).build(ctx2)[0]
        self.assertEqual(
            [v.main_numbers for v in t1.variants],
            [v.main_numbers for v in t2.variants],
        )


class TestWheelBuilder(unittest.TestCase):
    def test_joker_wheel_produces_2_variants_covering_pool(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = WheelBuilder(pool_size=7).build(ctx)[0]
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.strategy, "wheel")
        all_mains = set()
        for v in t.variants:
            all_mains.update(v.main_numbers)
        # 2 joker variants (5 each) from a 7-pool should cover all 7
        self.assertGreaterEqual(len(all_mains), 7)

    def test_loto_649_wheel_covers_full_pool(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = WheelBuilder(pool_size=8).build(ctx)[0]
        self.assertEqual(len(t.variants), 3)
        all_mains: set[int] = set()
        for v in t.variants:
            all_mains.update(v.main_numbers)
        # 3 variants of 6 each, from 8-pool: should cover all 8
        self.assertEqual(len(all_mains), 8)

    def test_pool_size_must_exceed_per_variant(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        with self.assertRaises(ValueError):
            WheelBuilder(pool_size=5).build(ctx)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="loto_649", config=LOTO_649_CONFIG, draws=_649_draws(),
            draw_dates=None, rng=random.Random(7),
        )
        ctx2 = BuilderContext(
            game="loto_649", config=LOTO_649_CONFIG, draws=_649_draws(),
            draw_dates=None, rng=random.Random(7),
        )
        t1 = WheelBuilder(pool_size=8).build(ctx1)[0]
        t2 = WheelBuilder(pool_size=8).build(ctx2)[0]
        for v1, v2 in zip(t1.variants, t2.variants):
            self.assertEqual(v1.main_numbers, v2.main_numbers)


if __name__ == "__main__":
    unittest.main()
