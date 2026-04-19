import unittest

from shared.ticket_allocator import TicketAllocation, enumerate_allocations, best_allocation


class TestTicketAllocator(unittest.TestCase):
    def test_40_ron_fits_joker_plus_loto_540_full(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        # Expect: 1 joker + 1 loto_540 = 17.5 + 22.5 = 40.0
        match = [
            a for a in allocs
            if a.tickets == {"joker": 1, "loto_649": 0, "loto_540": 1}
        ]
        self.assertEqual(len(match), 1)
        self.assertAlmostEqual(match[0].total_cost, 40.0, places=2)

    def test_40_ron_excludes_joker_plus_loto_649(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        joker_plus_649 = [
            a for a in allocs
            if a.tickets.get("joker", 0) == 1
            and a.tickets.get("loto_649", 0) == 1
            and a.tickets.get("loto_540", 0) == 0
        ]
        # 17.5 + 28.5 = 46 — over budget
        self.assertEqual(joker_plus_649, [])

    def test_70_ron_fits_all_three_games(self):
        allocs = enumerate_allocations(budget_ron=70.0)
        # Expect: 1 joker + 1 loto_649 + 1 loto_540 = 17.5 + 28.5 + 22.5 = 68.5
        all_three = [
            a for a in allocs
            if a.tickets == {"joker": 1, "loto_649": 1, "loto_540": 1}
        ]
        self.assertEqual(len(all_three), 1)
        self.assertAlmostEqual(all_three[0].total_cost, 68.5, places=2)

    def test_zero_allocation_excluded(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        for a in allocs:
            self.assertGreater(sum(a.tickets.values()), 0)

    def test_under_cheapest_ticket_returns_empty(self):
        allocs = enumerate_allocations(budget_ron=17.0)  # cheapest = 17.5 Joker
        self.assertEqual(allocs, [])

    def test_best_allocation_at_70_ron_picks_nonzero(self):
        best = best_allocation(budget_ron=70.0)
        self.assertGreater(sum(best.tickets.values()), 0)
        self.assertGreater(best.p_any_win, 0)

    def test_allowed_games_filter(self):
        allocs = enumerate_allocations(budget_ron=40.0, allowed_games={"joker"})
        for a in allocs:
            self.assertEqual(a.tickets.get("loto_649", 0), 0)
            self.assertEqual(a.tickets.get("loto_540", 0), 0)

    def test_total_cost_recompute(self):
        alloc = TicketAllocation(tickets={"joker": 2, "loto_649": 0, "loto_540": 0}, total_cost=0.0)
        # 2 × 17.5 = 35.0
        self.assertEqual(alloc.recompute_total_cost(), 35.0)


class TestBestPerGameAllocation(unittest.TestCase):
    def test_buckets_allocate_one_ticket_per_game(self):
        from shared.ticket_allocator import best_per_game_allocation

        buckets = {"joker": 20.0, "loto_649": 30.0, "loto_540": 25.0}
        alloc = best_per_game_allocation(buckets)
        self.assertEqual(alloc.tickets, {"joker": 1, "loto_649": 1, "loto_540": 1})
        # Costs: 17.5 + 28.5 + 22.5 = 68.5
        self.assertAlmostEqual(alloc.total_cost, 68.5, places=2)

    def test_bucket_below_ticket_price_spends_zero(self):
        from shared.ticket_allocator import best_per_game_allocation

        buckets = {"joker": 10.0, "loto_649": 30.0, "loto_540": 25.0}
        alloc = best_per_game_allocation(buckets)
        # Joker bucket can't afford a 17.5 ticket → 0 joker.
        self.assertEqual(alloc.tickets["joker"], 0)
        self.assertEqual(alloc.tickets["loto_649"], 1)
        self.assertEqual(alloc.tickets["loto_540"], 1)

    def test_empty_or_zero_buckets_yield_empty_allocation(self):
        from shared.ticket_allocator import best_per_game_allocation

        alloc = best_per_game_allocation({})
        self.assertEqual(sum(alloc.tickets.values()), 0)
        self.assertEqual(alloc.total_cost, 0.0)

    def test_unknown_game_in_buckets_is_ignored(self):
        from shared.ticket_allocator import best_per_game_allocation

        alloc = best_per_game_allocation({"joker": 20.0, "powerball": 999.0})
        self.assertEqual(alloc.tickets["joker"], 1)
        self.assertEqual(alloc.total_cost, 17.5)

    def test_large_bucket_fits_multiple_tickets_of_same_game(self):
        from shared.ticket_allocator import best_per_game_allocation

        # 55 RON on 6/49 → two tickets (28.5 × 2 = 57 > 55, so one; we want 55 → 1 ticket)
        # Try 60 RON → two tickets (28.5 × 2 = 57 ≤ 60) ✓
        alloc = best_per_game_allocation({"loto_649": 60.0})
        self.assertEqual(alloc.tickets["loto_649"], 2)
        self.assertAlmostEqual(alloc.total_cost, 57.0, places=2)


if __name__ == "__main__":
    unittest.main()
