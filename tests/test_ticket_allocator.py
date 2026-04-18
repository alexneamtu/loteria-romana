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


if __name__ == "__main__":
    unittest.main()
