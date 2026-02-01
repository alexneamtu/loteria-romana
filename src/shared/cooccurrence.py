"""Co-occurrence graph strategy for lottery number selection.

Builds an NxN co-occurrence matrix from historical draws and
generates picks using adjacency-weighted sequential sampling:
each subsequent number is chosen based on how frequently it
has appeared alongside the already-selected numbers.
"""

import random

from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE, draw_weights


class CooccurrenceStrategy:
    """Generate picks using number co-occurrence patterns."""

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 0,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "cooccurrence"

    def build_matrix(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[float]]:
        """Build recency-weighted co-occurrence matrix.

        Returns an NxN matrix where entry [i][j] is the weighted count
        of draws where numbers i+1 and j+1 appeared together.
        """
        n = self.number_pool
        matrix = [[0.0] * n for _ in range(n)]

        if not draws:
            return matrix

        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )

        for draw, weight in zip(draws, weights):
            nums = [num for num in draw if 1 <= num <= n]
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    a, b = nums[i] - 1, nums[j] - 1
                    matrix[a][b] += weight
                    matrix[b][a] += weight

        return matrix

    def _marginal_scores(self, matrix: list[list[float]]) -> list[float]:
        """Compute marginal score for each number (row sum)."""
        return [sum(row) + 1.0 for row in matrix]

    def _adjacency_scores(
        self,
        matrix: list[list[float]],
        selected: list[int],
    ) -> list[float]:
        """Score candidates by average co-occurrence with selected numbers.

        Args:
            matrix: Co-occurrence matrix (0-indexed).
            selected: Already-selected numbers (1-indexed).
        """
        n = self.number_pool
        scores = [0.0] * n
        if not selected:
            return self._marginal_scores(matrix)

        for candidate in range(n):
            total = 0.0
            for s in selected:
                total += matrix[candidate][s - 1]
            scores[candidate] = total / len(selected) + 0.1

        return scores

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Return marginal probability distribution over numbers."""
        matrix = self.build_matrix(draws, draw_dates, half_life_mode)
        scores = self._marginal_scores(matrix)
        total = sum(scores)
        return [s / total for s in scores]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate picks using adjacency-weighted sequential sampling."""
        matrix = self.build_matrix(draws, draw_dates, half_life_mode)
        marginals = self._marginal_scores(matrix)

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for _ in range(count * 20):
            if len(lines) >= count:
                break

            pick = self._sample_line(matrix, marginals, rng)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)

        return lines[:count]

    def _sample_line(
        self,
        matrix: list[list[float]],
        marginals: list[float],
        rng: random.Random,
    ) -> list[int]:
        """Sample a single line using sequential adjacency weighting."""
        selected: list[int] = []
        available = set(range(1, self.number_pool + 1))

        for step in range(self.numbers_to_pick):
            if not available:
                break

            if step == 0:
                scores = list(marginals)
            else:
                scores = self._adjacency_scores(matrix, selected)

            candidates = sorted(available)
            weights = [max(scores[c - 1], 0.001) for c in candidates]

            pick = rng.choices(candidates, weights=weights, k=1)[0]
            selected.append(pick)
            available.discard(pick)

        return sorted(selected)
