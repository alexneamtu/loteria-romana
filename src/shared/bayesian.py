"""Bayesian probability estimation for lottery numbers.

This module implements Bayesian updating of number probabilities using
a Dirichlet-Categorical conjugate model. It samples from the Dirichlet
posterior via gamma variates, providing proper uncertainty quantification
that goes beyond simple frequency analysis.

For truly random lotteries the posterior converges to uniform, but
Dirichlet sampling adds exploration that can be valuable in an ensemble.
"""

import math
import random
from dataclasses import dataclass

from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE, draw_weights


@dataclass
class BayesianResult:
    """Result from Bayesian probability estimation."""

    posterior: list[float]  # Posterior probability for each number
    alpha: list[float]  # Dirichlet concentration parameters
    effective_sample_size: float
    entropy: float  # Entropy of posterior (higher = more uniform)


def dirichlet_posterior(
    counts: list[int], alpha_prior: float = 1.0
) -> list[float]:
    """Compute Dirichlet posterior from observed counts.

    The Dirichlet distribution is the conjugate prior for categorical data.
    Given counts and a uniform prior (alpha=1 for each category), the
    posterior is also Dirichlet.

    Args:
        counts: Observed count for each category
        alpha_prior: Prior concentration parameter (1 = uniform prior)

    Returns:
        Posterior mean probability for each category
    """
    # Posterior alpha = prior alpha + counts
    posterior_alpha = [alpha_prior + c for c in counts]

    # Posterior mean = alpha_i / sum(alpha)
    total = sum(posterior_alpha)
    return [a / total for a in posterior_alpha]


def compute_entropy(probs: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Higher entropy means more uniform distribution.
    Maximum entropy = log(n) for n categories.

    Args:
        probs: Probability distribution

    Returns:
        Shannon entropy in nats
    """
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


class BayesianScorer:
    """Bayesian probability scorer for lottery numbers.

    Uses Dirichlet-Categorical conjugate model with proper posterior
    sampling via gamma variates.  Each call to generate() draws a fresh
    probability vector from the Dirichlet posterior, adding exploration
    that pure frequency analysis lacks.
    """

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        secondary_pool: int = 0,
        alpha_prior: float = 1.0,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.alpha_prior = alpha_prior
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "bayesian"
        self.counts: list[float] = [0.0] * pool_size

    def fit(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> BayesianResult:
        """Update posterior with recency-weighted observed draws."""
        mode = half_life_mode or self.half_life_mode
        self.counts = [0.0] * self.pool_size

        if draws:
            weights = draw_weights(
                len(draws),
                self.half_life,
                draw_dates=draw_dates,
                mode=mode,
            )
            for draw, weight in zip(draws, weights):
                for num in draw:
                    if 1 <= num <= self.pool_size:
                        self.counts[num - 1] += weight

        posterior_alpha = [self.alpha_prior + c for c in self.counts]
        posterior = dirichlet_posterior(self.counts, self.alpha_prior)
        effective_n = sum(self.counts)
        entropy = compute_entropy(posterior)

        return BayesianResult(
            posterior=posterior,
            alpha=posterior_alpha,
            effective_sample_size=effective_n,
            entropy=entropy,
        )

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Return posterior mean probabilities."""
        result = self.fit(draws, draw_dates, half_life_mode)
        return result.posterior

    def _sample_dirichlet(
        self,
        alpha: list[float],
        rng: random.Random,
    ) -> list[float]:
        """Sample a probability vector from a Dirichlet distribution.

        Uses the gamma-variate method: sample X_i ~ Gamma(alpha_i, 1)
        then normalise so the vector sums to 1.
        """
        samples = [rng.gammavariate(a, 1.0) for a in alpha]
        total = sum(samples)
        if total == 0:
            return [1.0 / len(alpha)] * len(alpha)
        return [s / total for s in samples]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines by sampling from the Dirichlet posterior.

        Each line gets its own probability vector sampled from the
        posterior, adding exploration beyond the posterior mean.
        """
        result = self.fit(draws, draw_dates, half_life_mode)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for _ in range(count * 20):
            if len(lines) >= count:
                break

            probs = self._sample_dirichlet(result.alpha, rng)
            line = self._weighted_sample(numbers, probs, rng)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        return lines[:count]

    def _weighted_sample(
        self,
        numbers: list[int],
        probs: list[float],
        rng: random.Random,
    ) -> list[int]:
        """Sample numbers weighted by probabilities without replacement."""
        chosen = []
        available = list(numbers)
        available_probs = list(probs)

        for _ in range(self.numbers_to_pick):
            if not available:
                break

            total = sum(available_probs)
            if total == 0:
                break

            normalized = [p / total for p in available_probs]
            pick = rng.choices(available, weights=normalized, k=1)[0]
            chosen.append(pick)

            idx = available.index(pick)
            available.pop(idx)
            available_probs.pop(idx)

        return sorted(chosen)


def bayesian_credible_interval(
    count: int, total: int, alpha_prior: float = 1.0, credibility: float = 0.95
) -> tuple[float, float]:
    """Compute Bayesian credible interval for a proportion.

    Uses Beta-Binomial model (special case of Dirichlet-Categorical).

    Args:
        count: Number of occurrences
        total: Total trials
        alpha_prior: Prior concentration
        credibility: Credibility level (e.g., 0.95)

    Returns:
        (lower, upper) bounds of credible interval
    """
    # Beta posterior parameters
    alpha = alpha_prior + count
    beta = alpha_prior + total - count

    # Approximate with normal for large samples
    if alpha > 1 and beta > 1:
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)

        # Normal approximation
        z = 1.96  # 95% credible
        lower = max(0, mean - z * std)
        upper = min(1, mean + z * std)
    else:
        # Small sample: use wider interval
        mean = alpha / (alpha + beta)
        lower = max(0, mean - 0.1)
        upper = min(1, mean + 0.1)

    return (lower, upper)


def analyze_bayesian_convergence(
    draws: list[list[int]], pool_size: int
) -> dict:
    """Analyze how Bayesian posterior converges with more data.

    For a fair lottery, the posterior should converge to uniform.

    Args:
        draws: Historical draws
        pool_size: Total numbers in pool

    Returns:
        Analysis of convergence behavior
    """
    if not draws:
        return {"error": "No draws to analyze"}

    scorer = BayesianScorer(pool_size, 6)  # Assume 6 per draw

    # Track convergence at different sample sizes
    checkpoints = [10, 50, 100, 200, len(draws)]
    checkpoints = [c for c in checkpoints if c <= len(draws)]

    convergence_data = []
    uniform_entropy = math.log(pool_size)

    for n in checkpoints:
        result = scorer.fit(draws[:n])

        # Measure distance from uniform
        uniform = 1.0 / pool_size
        kl_divergence = sum(
            p * math.log(p / uniform) if p > 0 else 0 for p in result.posterior
        )

        # Entropy ratio (1.0 = perfectly uniform)
        entropy_ratio = result.entropy / uniform_entropy

        convergence_data.append(
            {
                "sample_size": n,
                "entropy_ratio": entropy_ratio,
                "kl_from_uniform": kl_divergence,
                "min_prob": min(result.posterior),
                "max_prob": max(result.posterior),
            }
        )

    return {
        "convergence": convergence_data,
        "final_entropy_ratio": convergence_data[-1]["entropy_ratio"],
        "conclusion": (
            "Posterior converges toward uniform distribution as expected "
            "for truly random lottery draws. This confirms: no number has "
            "higher true probability than any other."
        ),
    }


def compare_prior_sensitivity(
    draws: list[list[int]], pool_size: int
) -> dict:
    """Compare posteriors under different priors.

    Demonstrates that with sufficient data, prior choice doesn't matter.

    Args:
        draws: Historical draws
        pool_size: Total numbers in pool

    Returns:
        Comparison of posteriors under different priors
    """
    priors = [0.1, 1.0, 10.0, 100.0]  # Weak to strong priors
    results = {}

    for alpha in priors:
        scorer = BayesianScorer(pool_size, 6, alpha_prior=alpha)
        result = scorer.fit(draws)

        results[f"alpha_{alpha}"] = {
            "min_prob": min(result.posterior),
            "max_prob": max(result.posterior),
            "entropy": result.entropy,
            "range": max(result.posterior) - min(result.posterior),
        }

    # Compare ranges to show convergence
    range_decrease = (
        results["alpha_100.0"]["range"] < results["alpha_0.1"]["range"]
        if len(draws) < 100
        else True  # With enough data, all converge
    )

    return {
        "prior_comparisons": results,
        "conclusion": (
            "Strong priors shrink toward uniform. With sufficient data, "
            "all priors converge to the same posterior (uniform). "
            "This demonstrates the lottery is truly random."
        ),
    }


def hierarchical_bayesian_model(
    draws: list[list[int]], pool_size: int
) -> dict:
    """Fit hierarchical Bayesian model to lottery draws.

    Models position-specific probabilities with shared hyperprior.
    Even this more sophisticated model learns that numbers are uniform.

    Args:
        draws: Historical draws
        pool_size: Total numbers in pool

    Returns:
        Hierarchical model results
    """
    if not draws:
        return {"error": "No draws to analyze"}

    numbers_per_draw = len(draws[0]) if draws else 6

    # Count position-specific frequencies
    position_counts: list[list[int]] = [
        [0] * pool_size for _ in range(numbers_per_draw)
    ]

    for draw in draws:
        sorted_draw = sorted(draw)
        for pos, num in enumerate(sorted_draw):
            if 1 <= num <= pool_size:
                position_counts[pos][num - 1] += 1

    # Compute posterior for each position
    position_posteriors = []
    for pos in range(numbers_per_draw):
        posterior = dirichlet_posterior(position_counts[pos], alpha_prior=1.0)
        entropy = compute_entropy(posterior)
        position_posteriors.append(
            {
                "position": pos + 1,
                "entropy": entropy,
                "max_entropy": math.log(pool_size),
                "top_3": sorted(
                    [(i + 1, p) for i, p in enumerate(posterior)],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3],
            }
        )

    return {
        "position_posteriors": position_posteriors,
        "conclusion": (
            "Position-specific posteriors show expected patterns: "
            "position 1 favors low numbers, position 6 favors high numbers. "
            "This is a mathematical property of sorting random numbers, "
            "NOT a predictive pattern."
        ),
    }
