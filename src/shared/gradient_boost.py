"""Gradient Boosted Trees strategy for lottery number prediction.

Trains per-number binary classifiers using features extracted from
historical draws. Uses scikit-learn's GradientBoostingClassifier
with optional import -- falls back to frequency-based selection if
scikit-learn is not available.
"""

import random
from collections import Counter

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class GradientBoostStrategy:
    """Gradient boosting strategy using per-number classifiers."""

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        numbers_drawn: int,
        window_size: int = 10,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_train_draws: int = 30,
        half_life: float = 100.0,
        half_life_mode: str = "draws",
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.numbers_drawn = numbers_drawn
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_train_draws = min_train_draws
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "gradient_boost"
        self._sklearn_available = SKLEARN_AVAILABLE

    def _build_features(self, draws: list[list[int]], idx: int) -> list[float]:
        """Build feature vector for predicting draw at idx."""
        window = draws[max(0, idx - self.window_size):idx]
        if not window:
            return [0.0] * (self.pool_size + 6)

        freq = Counter()
        for draw in window:
            for n in draw:
                freq[n] += 1

        total_numbers = sum(len(d) for d in window)
        freq_features = [
            freq.get(n, 0) / max(total_numbers, 1)
            for n in range(1, self.pool_size + 1)
        ]

        last_draw = window[-1]
        draw_sum = sum(last_draw)
        odd_count = sum(1 for n in last_draw if n % 2 == 1)
        high_count = sum(1 for n in last_draw if n > self.pool_size // 2)

        last_seen = {}
        for i, draw in enumerate(window):
            for n in draw:
                last_seen[n] = i
        avg_gap = sum(
            len(window) - last_seen.get(n, 0)
            for n in range(1, self.pool_size + 1)
        ) / self.pool_size

        return freq_features + [
            draw_sum / max(self.pool_size * self.numbers_drawn, 1),
            odd_count / max(self.numbers_drawn, 1),
            high_count / max(self.numbers_drawn, 1),
            len(window) / max(self.window_size, 1),
            avg_gap / max(len(window), 1),
            len(freq) / self.pool_size,
        ]

    def _train_and_predict(self, draws: list[list[int]]) -> list[float]:
        """Train per-number classifiers and return prediction scores."""
        if not self._sklearn_available or len(draws) < self.min_train_draws:
            return self._fallback_probabilities(draws)

        feature_start = self.window_size
        if len(draws) - feature_start < 10:
            return self._fallback_probabilities(draws)

        X = []
        y_per_number = {n: [] for n in range(1, self.pool_size + 1)}

        for idx in range(feature_start, len(draws)):
            features = self._build_features(draws, idx)
            X.append(features)
            draw_set = set(draws[idx])
            for n in range(1, self.pool_size + 1):
                y_per_number[n].append(1 if n in draw_set else 0)

        scores = []
        for n in range(1, self.pool_size + 1):
            y = y_per_number[n]
            if sum(y) == 0 or sum(y) == len(y):
                scores.append(sum(y) / max(len(y), 1))
                continue

            clf = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=0,
            )
            clf.fit(X, y)

            last_features = self._build_features(draws, len(draws))
            prob = clf.predict_proba([last_features])[0]
            positive_idx = (
                list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            )
            scores.append(prob[positive_idx])

        return scores

    def _fallback_probabilities(self, draws: list[list[int]]) -> list[float]:
        """Simple frequency-based fallback when sklearn unavailable."""
        freq = Counter()
        for draw in draws:
            for n in draw:
                freq[n] += 1
        total = sum(freq.values()) or 1
        probs = [
            (freq.get(n, 0) + 1) / (total + self.pool_size)
            for n in range(1, self.pool_size + 1)
        ]
        s = sum(probs)
        return [p / s for p in probs]

    def get_probabilities(self, draws: list[list[int]], **kwargs) -> list[float]:
        """Return normalized prediction scores as probabilities."""
        scores = self._train_and_predict(draws)
        total = sum(scores) or 1.0
        return [s / total for s in scores]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate picks weighted by classifier predictions."""
        probs = self.get_probabilities(draws)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for _ in range(count * 20):
            if len(lines) >= count:
                break

            chosen = []
            available = list(numbers)
            available_probs = list(probs)

            for _ in range(self.numbers_to_pick):
                if not available:
                    break
                total = sum(available_probs)
                if total == 0:
                    pick = rng.choice(available)
                else:
                    normalized = [p / total for p in available_probs]
                    pick = rng.choices(available, weights=normalized, k=1)[0]
                chosen.append(pick)
                idx = available.index(pick)
                available.pop(idx)
                available_probs.pop(idx)

            line = sorted(chosen)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        while len(lines) < count:
            extra = sorted(rng.sample(numbers, self.numbers_to_pick))
            key = tuple(extra)
            if key not in seen:
                seen.add(key)
                lines.append(extra)

        return lines[:count]
