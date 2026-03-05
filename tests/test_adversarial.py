"""Adversarial CI gate: no strategy should beat random on synthetic uniform data.

When draws are generated from a perfectly uniform distribution, frequency-based
and other pattern-detecting strategies have no real signal to exploit.  Any
apparent edge must be indistinguishable from noise.  These tests verify that
the significance gate correctly filters out spurious advantages.
"""

import random
import unittest

from shared.ensemble_blend import (
    _score_random,
    _score_frequency,
    _apply_significance_gate,
)
from shared.game_config import JOKER_CONFIG


def _generate_uniform_draws(
    config,
    count: int,
    seed: int,
) -> list[list[int]]:
    """Generate synthetic draws from a uniform distribution."""
    rng = random.Random(seed)
    pool = list(config.pool_range)
    return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]


class TestAdversarialUniformData(unittest.TestCase):
    """Verify no strategy beats random on purely uniform synthetic draws."""

    SEED = 20260306
    DRAW_COUNT = 200

    def setUp(self):
        self.config = JOKER_CONFIG
        self.draws = _generate_uniform_draws(
            self.config, self.DRAW_COUNT, seed=self.SEED,
        )

    def test_frequency_does_not_beat_random(self):
        """Frequency strategy should not dramatically outscore random on uniform data.

        We aggregate over multiple independent seeds to smooth out single-run
        variance.  The combined frequency win-rate must not exceed the combined
        random win-rate by more than a generous z > 3.0 threshold (p < 0.001).
        """
        import math

        total_random_wins = 0
        total_freq_wins = 0
        total_draws = 0
        num_trials = 5

        for trial in range(num_trials):
            trial_seed = self.SEED + trial * 1000
            draws = _generate_uniform_draws(
                self.config, self.DRAW_COUNT, seed=trial_seed,
            )
            random_score = _score_random(
                self.config, draws, random.Random(trial_seed + 1),
            )
            frequency_score = _score_frequency(
                self.config,
                draws,
                random.Random(trial_seed + 2),
                half_life=20.0,
                half_life_mode="draws",
                draw_dates=None,
            )
            total_random_wins += random_score
            total_freq_wins += frequency_score
            total_draws += len(draws)

        baseline_rate = total_random_wins / total_draws
        freq_rate = total_freq_wins / total_draws

        # If frequency does not exceed random, nothing to check.
        if freq_rate <= baseline_rate:
            return

        # Two-proportion z-test with a very generous threshold (z < 3.0).
        se = math.sqrt(
            baseline_rate * (1 - baseline_rate) / total_draws,
        ) if 0 < baseline_rate < 1 else 0.0

        if se > 0:
            z = (freq_rate - baseline_rate) / se
            self.assertLessEqual(
                z,
                3.0,
                f"Frequency aggregated win-rate ({freq_rate:.4f}) dramatically "
                f"exceeds random ({baseline_rate:.4f}) with z={z:.2f} over "
                f"{total_draws} draws — suggests a bug or data leakage.",
            )

    def test_significance_gate_filters_on_uniform_data(self):
        """The significance gate should not promote non-random strategies on uniform draws."""
        scoring_rng_random = random.Random(self.SEED + 10)
        scoring_rng_freq = random.Random(self.SEED + 11)

        random_score = _score_random(self.config, self.draws, scoring_rng_random)
        frequency_score = _score_frequency(
            self.config,
            self.draws,
            scoring_rng_freq,
            half_life=20.0,
            half_life_mode="draws",
            draw_dates=None,
        )

        scores = {
            "random": max(random_score, 1),
            "frequency": max(frequency_score, 1),
        }

        gated = _apply_significance_gate(
            scores,
            baseline_score=scores["random"],
            total_draws=len(self.draws),
        )

        # "random" must always survive the gate.
        self.assertIn("random", gated)

        # If frequency survived, its score must not be statistically
        # significantly higher than random (z <= 1.645 at p < 0.05).
        # We re-check the z-score here to make the assertion transparent.
        if "frequency" in gated:
            baseline_rate = scores["random"] / len(self.draws)
            freq_rate = scores["frequency"] / len(self.draws)
            if baseline_rate > 0 and baseline_rate < 1:
                import math
                se = math.sqrt(
                    baseline_rate * (1 - baseline_rate) / len(self.draws),
                )
                if se > 0 and freq_rate > baseline_rate:
                    z = (freq_rate - baseline_rate) / se
                    self.assertLessEqual(
                        z,
                        1.645,
                        f"Frequency passed the gate but z={z:.3f} > 1.645 — "
                        f"significance gate appears broken.",
                    )


if __name__ == "__main__":
    unittest.main()
