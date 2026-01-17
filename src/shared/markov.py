"""Markov chain analysis for lottery draws.

IMPORTANT: This module is provided for educational purposes only.
Markov chains assume that future states depend on the current state.
Lottery draws are INDEPENDENT events - they have no memory of past draws.
Therefore, Markov chain methods are fundamentally inapplicable to
lottery prediction.

As stated in research:
"Not every process has the Markov Property, such as the Lottery -
this week's winning numbers have no dependence to the previous
week's winning numbers."
"""

import random
from dataclasses import dataclass


@dataclass
class MarkovResult:
    """Result from Markov chain analysis."""

    transition_matrix: dict[int, dict[int, float]]
    stationary_distribution: dict[int, float]
    order: int
    sample_size: int

    def get_transition_prob(self, from_num: int, to_num: int) -> float:
        """Get transition probability from one number to another."""
        return self.transition_matrix.get(from_num, {}).get(to_num, 0.0)


def build_first_order_transitions(
    draws: list[list[int]], pool_size: int
) -> dict[int, dict[int, int]]:
    """Build first-order transition counts between draws.

    For each number n that appeared in draw t, count how often each
    number m appeared in draw t+1.

    Args:
        draws: List of draws (oldest first)
        pool_size: Total numbers in pool

    Returns:
        Nested dict: transitions[from_num][to_num] = count
    """
    transitions: dict[int, dict[int, int]] = {
        n: {m: 0 for m in range(1, pool_size + 1)} for n in range(1, pool_size + 1)
    }

    for i in range(len(draws) - 1):
        current = set(draws[i])
        next_draw = set(draws[i + 1])

        for n in current:
            for m in next_draw:
                transitions[n][m] += 1

    return transitions


def normalize_transitions(
    counts: dict[int, dict[int, int]]
) -> dict[int, dict[int, float]]:
    """Convert transition counts to probabilities.

    Args:
        counts: Transition count matrix

    Returns:
        Transition probability matrix
    """
    probs: dict[int, dict[int, float]] = {}

    for n, row in counts.items():
        total = sum(row.values())
        if total > 0:
            probs[n] = {m: c / total for m, c in row.items()}
        else:
            # Uniform if no data
            pool_size = len(row)
            probs[n] = {m: 1.0 / pool_size for m in row.keys()}

    return probs


def first_order_markov(draws: list[list[int]], pool_size: int) -> MarkovResult:
    """Build first-order Markov chain from historical draws.

    First-order means: P(X_{t+1} | X_t) - next state depends only on current.

    Args:
        draws: List of draws (oldest first)
        pool_size: Total numbers in pool

    Returns:
        MarkovResult with transition matrix and stationary distribution
    """
    counts = build_first_order_transitions(draws, pool_size)
    transitions = normalize_transitions(counts)

    # Estimate stationary distribution (average of rows converges to this)
    stationary = compute_stationary_distribution(transitions, pool_size)

    return MarkovResult(
        transition_matrix=transitions,
        stationary_distribution=stationary,
        order=1,
        sample_size=len(draws),
    )


def build_second_order_transitions(
    draws: list[list[int]], pool_size: int
) -> dict[tuple[int, int], dict[int, int]]:
    """Build second-order transition counts.

    Second-order: P(X_{t+1} | X_t, X_{t-1})

    Args:
        draws: List of draws (oldest first)
        pool_size: Total numbers in pool

    Returns:
        Dict: transitions[(prev_num, curr_num)][next_num] = count
    """
    transitions: dict[tuple[int, int], dict[int, int]] = {}

    for i in range(len(draws) - 2):
        prev_draw = set(draws[i])
        curr_draw = set(draws[i + 1])
        next_draw = set(draws[i + 2])

        for p in prev_draw:
            for c in curr_draw:
                key = (p, c)
                if key not in transitions:
                    transitions[key] = {m: 0 for m in range(1, pool_size + 1)}
                for n in next_draw:
                    transitions[key][n] += 1

    return transitions


def second_order_markov(draws: list[list[int]], pool_size: int) -> MarkovResult:
    """Build second-order Markov chain from historical draws.

    Args:
        draws: List of draws (oldest first)
        pool_size: Total numbers in pool

    Returns:
        MarkovResult with second-order transition probabilities
    """
    counts = build_second_order_transitions(draws, pool_size)

    # Normalize to probabilities
    transitions: dict[tuple[int, int], dict[int, float]] = {}
    for key, row in counts.items():
        total = sum(row.values())
        if total > 0:
            transitions[key] = {m: c / total for m, c in row.items()}
        else:
            transitions[key] = {m: 1.0 / pool_size for m in row.keys()}

    # For compatibility, flatten to single-key dict (loses some info)
    flattened: dict[int, dict[int, float]] = {}
    for (p, c), probs in transitions.items():
        if c not in flattened:
            flattened[c] = {m: 0.0 for m in range(1, pool_size + 1)}
        for m, prob in probs.items():
            flattened[c][m] += prob

    # Re-normalize flattened
    for c in flattened:
        total = sum(flattened[c].values())
        if total > 0:
            flattened[c] = {m: p / total for m, p in flattened[c].items()}

    stationary = compute_stationary_distribution(flattened, pool_size)

    return MarkovResult(
        transition_matrix=flattened,
        stationary_distribution=stationary,
        order=2,
        sample_size=len(draws),
    )


def compute_stationary_distribution(
    transitions: dict[int, dict[int, float]], pool_size: int, iterations: int = 100
) -> dict[int, float]:
    """Compute stationary distribution via power iteration.

    The stationary distribution is the long-run probability of being
    in each state. For a fair lottery, this should be uniform.

    Args:
        transitions: Transition probability matrix
        pool_size: Total numbers in pool
        iterations: Number of power iterations

    Returns:
        Dict mapping each number to its stationary probability
    """
    # Start with uniform distribution
    dist = {n: 1.0 / pool_size for n in range(1, pool_size + 1)}

    for _ in range(iterations):
        new_dist = {n: 0.0 for n in range(1, pool_size + 1)}
        for from_n, from_prob in dist.items():
            for to_n, trans_prob in transitions.get(from_n, {}).items():
                new_dist[to_n] += from_prob * trans_prob
        dist = new_dist

    # Normalize
    total = sum(dist.values())
    if total > 0:
        dist = {n: p / total for n, p in dist.items()}

    return dist


def position_based_transitions(
    draws: list[list[int]], pool_size: int, numbers_per_draw: int
) -> list[dict[int, dict[int, float]]]:
    """Build position-based transition matrices.

    For each sorted position, track which numbers follow which.
    Position 0 = smallest number in draw, position k-1 = largest.

    Args:
        draws: List of draws (oldest first)
        pool_size: Total numbers in pool
        numbers_per_draw: Numbers per draw

    Returns:
        List of transition matrices, one per position
    """
    # Initialize counts for each position
    position_counts: list[dict[int, dict[int, int]]] = [
        {n: {m: 0 for m in range(1, pool_size + 1)} for n in range(1, pool_size + 1)}
        for _ in range(numbers_per_draw)
    ]

    for i in range(len(draws) - 1):
        curr_sorted = sorted(draws[i])
        next_sorted = sorted(draws[i + 1])

        for pos in range(min(len(curr_sorted), len(next_sorted))):
            curr_num = curr_sorted[pos]
            next_num = next_sorted[pos]
            position_counts[pos][curr_num][next_num] += 1

    # Normalize
    position_probs: list[dict[int, dict[int, float]]] = []
    for pos_counts in position_counts:
        pos_probs: dict[int, dict[int, float]] = {}
        for n, row in pos_counts.items():
            total = sum(row.values())
            if total > 0:
                pos_probs[n] = {m: c / total for m, c in row.items()}
            else:
                pos_probs[n] = {m: 1.0 / pool_size for m in row.keys()}
        position_probs.append(pos_probs)

    return position_probs


class MarkovGenerator:
    """Generate numbers using Markov chain transitions.

    WARNING: This cannot improve prediction because lottery draws
    are independent. This class exists for educational purposes only.
    """

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        order: int = 1,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.order = order
        self.name = f"markov_order_{order}"
        self.markov_result: MarkovResult | None = None

    def fit(self, draws: list[list[int]]) -> None:
        """Fit Markov chain to historical draws."""
        if self.order == 1:
            self.markov_result = first_order_markov(draws, self.pool_size)
        else:
            self.markov_result = second_order_markov(draws, self.pool_size)

    def get_probabilities(self, draws: list[list[int]]) -> list[float]:
        """Get probability distribution based on Markov analysis."""
        if self.markov_result is None:
            self.fit(draws)

        # Use stationary distribution as probabilities
        probs = [0.0] * self.pool_size
        if self.markov_result:
            for n, p in self.markov_result.stationary_distribution.items():
                if 1 <= n <= self.pool_size:
                    probs[n - 1] = p

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / self.pool_size] * self.pool_size

        return probs

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
    ) -> list[list[int]]:
        """Generate lines using Markov chain sampling.

        Note: This will NOT improve prediction because lottery draws
        are independent events.
        """
        if self.markov_result is None:
            self.fit(draws)

        if not self.markov_result or not draws:
            # Fallback to random
            return [
                sorted(rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick))
                for _ in range(count)
            ]

        lines = []
        seen = set()
        last_draw = draws[-1] if draws else []

        for _ in range(count * 10):
            if len(lines) >= count:
                break

            line = self._sample_line(last_draw, rng)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        return lines[:count]

    def _sample_line(
        self, last_draw: list[int], rng: random.Random
    ) -> list[int]:
        """Sample a single line using transition probabilities."""
        if not self.markov_result:
            return sorted(
                rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick)
            )

        line = []
        transitions = self.markov_result.transition_matrix

        # Start from a number in the last draw (or random)
        if last_draw:
            current = rng.choice(last_draw)
        else:
            current = rng.randint(1, self.pool_size)

        for _ in range(self.numbers_to_pick):
            # Get transition probabilities from current number
            trans_probs = transitions.get(current, {})
            if not trans_probs:
                # Fallback to uniform
                next_num = rng.randint(1, self.pool_size)
            else:
                numbers = list(trans_probs.keys())
                weights = [trans_probs[n] for n in numbers]
                next_num = rng.choices(numbers, weights=weights, k=1)[0]

            # Avoid duplicates
            while next_num in line:
                next_num = rng.randint(1, self.pool_size)

            line.append(next_num)
            current = next_num

        return sorted(line)


def analyze_markov_properties(draws: list[list[int]], pool_size: int) -> dict:
    """Analyze whether lottery draws exhibit Markov properties.

    Spoiler: They don't (and shouldn't).

    Args:
        draws: Historical draws
        pool_size: Total numbers in pool

    Returns:
        Analysis results showing lack of Markov structure
    """
    if len(draws) < 10:
        return {"error": "Insufficient data for Markov analysis"}

    # Build transition matrix
    result = first_order_markov(draws, pool_size)

    # Check if transitions are uniform (they should be for random lottery)
    uniformity_score = 0.0
    expected_prob = 1.0 / pool_size

    for from_n, row in result.transition_matrix.items():
        for to_n, prob in row.items():
            uniformity_score += (prob - expected_prob) ** 2

    uniformity_score /= pool_size * pool_size
    uniformity_score = 1.0 - min(1.0, uniformity_score * 1000)

    # Check stationarity (should be uniform for random lottery)
    stationary_uniformity = 0.0
    for n, prob in result.stationary_distribution.items():
        stationary_uniformity += (prob - expected_prob) ** 2
    stationary_uniformity = 1.0 - min(1.0, stationary_uniformity * 100)

    return {
        "transition_uniformity": uniformity_score,
        "stationary_uniformity": stationary_uniformity,
        "sample_size": len(draws),
        "conclusion": (
            "Transitions are approximately uniform, confirming lottery draws "
            "do NOT have Markov structure. Each draw is independent."
            if uniformity_score > 0.9
            else "Some non-uniformity detected (likely sampling noise)."
        ),
        "predictive_value": "None - lottery draws are independent events.",
    }
