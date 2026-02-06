"""Genetic algorithm strategy for lottery number optimization.

Evolves a population of ticket candidates using selection, crossover,
and mutation. Fitness is measured by historical prize matches against
recent draws. This strategy optimizes ticket sets directly rather
than predicting probability distributions.

Uses only the Python standard library.
"""

import random
from collections import Counter


class GeneticStrategy:
    """Genetic algorithm that evolves lottery ticket populations."""

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elite_count: int = 2,
        half_life: float = 100.0,
        half_life_mode: str = "draws",
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "genetic"

    def _random_individual(self, rng: random.Random) -> list[int]:
        """Create a random ticket."""
        return sorted(rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick))

    def _fitness(self, individual: list[int], draws: list[list[int]]) -> float:
        """Score an individual against historical draws."""
        if not draws:
            return 0.0

        ind_set = set(individual)
        score = 0.0
        n = len(draws)

        for i, draw in enumerate(draws):
            weight = 0.5 ** ((n - 1 - i) / max(self.half_life, 1.0))
            matches = len(ind_set & set(draw))
            if matches >= 3:
                score += weight * (matches ** 2)
            elif matches >= 2:
                score += weight * matches * 0.5

        return score

    def _tournament_select(
        self,
        population: list[list[int]],
        fitnesses: list[float],
        rng: random.Random,
    ) -> list[int]:
        """Select an individual via tournament selection."""
        indices = rng.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return list(population[best_idx])

    def _crossover(
        self,
        parent1: list[int],
        parent2: list[int],
        rng: random.Random,
    ) -> list[int]:
        """Uniform crossover: combine numbers from two parents."""
        combined = list(set(parent1) | set(parent2))
        if len(combined) < self.numbers_to_pick:
            extra = [n for n in range(1, self.pool_size + 1) if n not in combined]
            combined.extend(rng.sample(extra, self.numbers_to_pick - len(combined)))

        child = sorted(rng.sample(combined, self.numbers_to_pick))
        return child

    def _mutate(self, individual: list[int], rng: random.Random) -> list[int]:
        """Mutate by replacing one random number with another from the pool."""
        result = list(individual)
        available = [n for n in range(1, self.pool_size + 1) if n not in result]
        if not available:
            return result

        idx = rng.randrange(len(result))
        result[idx] = rng.choice(available)
        return sorted(result)

    def _evolve(
        self,
        draws: list[list[int]],
        rng: random.Random,
    ) -> list[list[int]]:
        """Run the genetic algorithm and return the final population."""
        population = [self._random_individual(rng) for _ in range(self.population_size)]

        for _ in range(self.generations):
            fitnesses = [self._fitness(ind, draws) for ind in population]

            sorted_indices = sorted(
                range(len(population)),
                key=lambda i: fitnesses[i],
                reverse=True,
            )
            elites = [list(population[i]) for i in sorted_indices[:self.elite_count]]

            new_population = list(elites)

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses, rng)
                parent2 = self._tournament_select(population, fitnesses, rng)

                if rng.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, rng)
                else:
                    child = list(parent1)

                if rng.random() < self.mutation_rate:
                    child = self._mutate(child, rng)

                new_population.append(child)

            population = new_population[:self.population_size]

        fitnesses = [self._fitness(ind, draws) for ind in population]
        sorted_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )
        return [population[i] for i in sorted_indices]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate picks by evolving and selecting top individuals."""
        evolved = self._evolve(draws, rng)

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for individual in evolved:
            if len(lines) >= count:
                break
            key = tuple(individual)
            if key not in seen:
                seen.add(key)
                lines.append(individual)

        while len(lines) < count:
            extra = sorted(rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick))
            key = tuple(extra)
            if key not in seen:
                seen.add(key)
                lines.append(extra)

        return lines[:count]

    def get_probabilities(
        self,
        draws: list[list[int]],
        **kwargs,
    ) -> list[float]:
        """Get probability distribution from evolved population."""
        rng = random.Random(0)
        evolved = self._evolve(draws, rng)

        counts = Counter()
        for individual in evolved:
            for num in individual:
                counts[num] += 1

        total = sum(counts.values()) or 1
        return [counts.get(n, 0) / total for n in range(1, self.pool_size + 1)]
