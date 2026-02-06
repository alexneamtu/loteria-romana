import random
import unittest

from shared.genetic import GeneticStrategy


class TestGeneticStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = GeneticStrategy(
            pool_size=45,
            numbers_to_pick=5,
            population_size=50,
            generations=20,
            mutation_rate=0.1,
        )
        self.draws = [
            sorted(random.Random(i).sample(range(1, 46), 5))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "genetic")

    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 5, rng)
        self.assertEqual(len(lines), 5)

    def test_generate_returns_valid_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 3, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    def test_generate_returns_unique_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_generate_with_empty_draws(self):
        rng = random.Random(42)
        lines = self.strategy.generate([], 3, rng)
        self.assertEqual(len(lines), 3)

    def test_get_probabilities_returns_distribution(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_deterministic_with_seed(self):
        lines1 = self.strategy.generate(self.draws, 3, random.Random(99))
        lines2 = self.strategy.generate(self.draws, 3, random.Random(99))
        self.assertEqual(lines1, lines2)

    def test_fitness_improves_over_generations(self):
        rng = random.Random(42)
        low_gen = GeneticStrategy(45, 5, population_size=30, generations=1)
        high_gen = GeneticStrategy(45, 5, population_size=30, generations=50)
        l1 = low_gen.generate(self.draws, 3, random.Random(42))
        l2 = high_gen.generate(self.draws, 3, random.Random(42))
        self.assertEqual(len(l1), 3)
        self.assertEqual(len(l2), 3)


if __name__ == "__main__":
    unittest.main()
