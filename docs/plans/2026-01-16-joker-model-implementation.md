# Joker Model Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Joker-only pipeline that ingests loto.ro results, trains statistical and neural samplers, backtests them, and outputs 7 weekly Joker lines.

**Architecture:** A small Python package handles HTML parsing, dataset updates, strategy models, and backtesting. A CLI script fetches data, selects the best-performing strategy from backtests, and prints weekly picks.

**Tech Stack:** Python 3.x (stdlib only), `unittest`, CSV, JSON.

### Task 1: Draw model + HTML parser

**Files:**
- Create: `src/joker_model/__init__.py`
- Create: `src/joker_model/models.py`
- Create: `src/joker_model/parser.py`
- Create: `tests/fixtures/joker_results_snippet.html`
- Test: `tests/test_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_parser.py
import unittest
from pathlib import Path

from joker_model.parser import parse_joker_results


class TestParser(unittest.TestCase):
    def test_parse_snippet(self):
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].date, "2026-01-15")
        self.assertEqual(draws[0].main_numbers, [7, 11, 44, 45, 46])
        self.assertEqual(draws[0].joker, 13)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joker_model'` or missing `parse_joker_results`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/models.py
from dataclasses import dataclass


@dataclass(frozen=True)
class JokerDraw:
    date: str
    main_numbers: list[int]
    joker: int
```

```python
# src/joker_model/parser.py
import re
from datetime import datetime

from .models import JokerDraw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_joker_results(html: str) -> list[JokerDraw]:
    draws = []
    date_pattern = re.compile(r"Detalii castiguri\s+la joker\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>")

    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 800)
        window = html[window_start:match.start()]

        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]
        joker_nums = [int(n) for n in re.findall(r"/bile/joker/(\d{1,2})\.png", window)]

        if len(main_nums) < 5 or not joker_nums:
            continue

        main = sorted(main_nums[-5:])
        joker = joker_nums[-1]
        draws.append(JokerDraw(_normalize_date(match.group(1)), main, joker))

    return draws
```

```python
# src/joker_model/__init__.py
from .models import JokerDraw
from .parser import parse_joker_results

__all__ = ["JokerDraw", "parse_joker_results"]
```

```html
<!-- tests/fixtures/joker_results_snippet.html -->
<div class="content-rezultate">
  <img src="/bile/45.png" />
  <img src="/bile/11.png" />
  <img src="/bile/46.png" />
  <img src="/bile/7.png" />
  <img src="/bile/44.png" />
  <img src="/bile/joker/13.png" />
  <p>Detalii castiguri  la joker din <span>15.01.2026</span></p>
</div>
<div class="content-rezultate">
  <img src="/bile/1.png" />
  <img src="/bile/22.png" />
  <img src="/bile/13.png" />
  <img src="/bile/30.png" />
  <img src="/bile/33.png" />
  <img src="/bile/joker/6.png" />
  <p>Detalii castiguri  la joker din <span>11.01.2026</span></p>
</div>
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/__init__.py src/joker_model/models.py src/joker_model/parser.py tests/fixtures/joker_results_snippet.html tests/test_parser.py
git commit -m "feat: add Joker draw parser"
```

### Task 2: Data fetch + storage

**Files:**
- Create: `src/joker_model/fetch.py`
- Create: `src/joker_model/storage.py`
- Test: `tests/test_fetch_storage.py`

**Step 1: Write the failing test**

```python
# tests/test_fetch_storage.py
import tempfile
import unittest
from pathlib import Path

from joker_model.fetch import update_dataset


class TestFetchStorage(unittest.TestCase):
    def test_update_dataset_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            data_dir = Path(tmpdir) / "data"
            cache_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)

            html_path = cache_dir / "joker_results.html"
            html_path.write_text("<p>Detalii castiguri  la joker din <span>15.01.2026</span></p>", encoding="utf-8")

            updated = update_dataset(
                url="https://example.invalid",
                cache_path=html_path,
                csv_path=data_dir / "joker_draws.csv",
                fetcher=lambda _: html_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(updated, 0)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_fetch_storage.py -v`
Expected: FAIL with `ImportError` or missing `update_dataset`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/storage.py
import csv
from pathlib import Path

from .models import JokerDraw


def load_draws(path: Path) -> list[JokerDraw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 6)]
            rows.append(JokerDraw(row["date"], sorted(main), int(row["joker"])))
    return rows


def append_draws(path: Path, draws: list[JokerDraw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = ["date"] + [f"main_{i}" for i in range(1, 6)] + ["joker"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for draw in draws:
            writer.writerow({
                "date": draw.date,
                "main_1": draw.main_numbers[0],
                "main_2": draw.main_numbers[1],
                "main_3": draw.main_numbers[2],
                "main_4": draw.main_numbers[3],
                "main_5": draw.main_numbers[4],
                "joker": draw.joker,
            })
    return len(draws)
```

```python
# src/joker_model/fetch.py
from pathlib import Path
import urllib.request

from .parser import parse_joker_results
from .storage import load_draws, append_draws


def update_dataset(url: str, cache_path: Path, csv_path: Path, fetcher=None) -> int:
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        fetcher = fetcher or (lambda u: urllib.request.urlopen(u).read().decode("utf-8", "ignore"))
        html = fetcher(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")

    draws = parse_joker_results(html)
    existing = {d.date for d in load_draws(csv_path)}
    new_draws = [d for d in draws if d.date not in existing]
    return append_draws(csv_path, new_draws)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_fetch_storage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/fetch.py src/joker_model/storage.py tests/test_fetch_storage.py
git commit -m "feat: add dataset update flow"
```

### Task 3: Metrics + statistical strategies

**Files:**
- Create: `src/joker_model/metrics.py`
- Create: `src/joker_model/strategies.py`
- Test: `tests/test_strategies.py`

**Step 1: Write the failing test**

```python
# tests/test_strategies.py
import unittest
import random

from joker_model.metrics import is_joker_prize
from joker_model.strategies import generate_random_lines, generate_frequency_lines


class TestStrategies(unittest.TestCase):
    def test_joker_prize_rules(self):
        self.assertTrue(is_joker_prize(main_matches=5, joker_match=True))
        self.assertTrue(is_joker_prize(main_matches=3, joker_match=False))
        self.assertTrue(is_joker_prize(main_matches=1, joker_match=True))
        self.assertFalse(is_joker_prize(main_matches=2, joker_match=False))

    def test_generate_random_lines_unique(self):
        rng = random.Random(1234)
        lines = generate_random_lines(3, rng=rng)
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({tuple(l[0]) + (l[1],) for l in lines}), 3)

    def test_generate_frequency_lines_unique(self):
        freq = {n: 1 for n in range(1, 46)}
        rng = random.Random(42)
        lines = generate_frequency_lines(2, freq, rng=rng)
        self.assertEqual(len(lines), 2)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py -v`
Expected: FAIL with `ImportError` or missing functions.

**Step 3: Write minimal implementation**

```python
# src/joker_model/metrics.py

def is_joker_prize(main_matches: int, joker_match: bool) -> bool:
    if main_matches >= 3:
        return True
    if joker_match and main_matches >= 1:
        return True
    return False
```

```python
# src/joker_model/strategies.py
import random


def _sample_weighted(numbers, weights, count, rng):
    chosen = []
    pool = list(numbers)
    pool_weights = list(weights)
    for _ in range(count):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return sorted(chosen)


def generate_random_lines(count: int, rng=None):
    rng = rng or random.SystemRandom()
    lines = []
    seen = set()
    while len(lines) < count:
        main = sorted(rng.sample(range(1, 46), 5))
        joker = rng.randint(1, 20)
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))
    return lines


def generate_frequency_lines(count: int, freq: dict[int, int], rng=None):
    rng = rng or random.SystemRandom()
    numbers = list(range(1, 46))
    weights = [freq.get(n, 1) for n in numbers]
    lines = []
    seen = set()
    while len(lines) < count:
        main = _sample_weighted(numbers, weights, 5, rng)
        joker = rng.randint(1, 20)
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))
    return lines
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/metrics.py src/joker_model/strategies.py tests/test_strategies.py
git commit -m "feat: add prize rules and basic strategies"
```

### Task 4: Neural sampler

**Files:**
- Create: `src/joker_model/neural.py`
- Test: `tests/test_neural.py`

**Step 1: Write the failing test**

```python
# tests/test_neural.py
import unittest
import random

from joker_model.neural import SoftmaxModel, generate_neural_lines


class TestNeural(unittest.TestCase):
    def test_softmax_probs_sum_to_one(self):
        model = SoftmaxModel(input_size=4, output_size=3, rng=random.Random(0))
        probs = model.predict_probs([1, 0, 0, 0])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertEqual(len(probs), 3)

    def test_training_reduces_loss(self):
        model = SoftmaxModel(input_size=2, output_size=2, rng=random.Random(0))
        inputs = [[1, 0], [0, 1]]
        targets = [[1, 0], [0, 1]]
        loss_before = model.loss(inputs, targets)
        model.train(inputs, targets, epochs=50, lr=0.5)
        loss_after = model.loss(inputs, targets)
        self.assertLess(loss_after, loss_before)

    def test_generate_neural_lines(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        rng = random.Random(0)
        lines = generate_neural_lines(draws, count=2, rng=rng, epochs=10, lr=0.1)
        self.assertEqual(len(lines), 2)
        for main, joker in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))
            self.assertTrue(1 <= joker <= 20)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_neural.py -v`
Expected: FAIL with `ImportError` or missing `SoftmaxModel`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/neural.py
import math
import random


def _softmax(logits):
    max_logit = max(logits)
    exp_vals = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]


class SoftmaxModel:
    def __init__(self, input_size: int, output_size: int, rng=None):
        self.input_size = input_size
        self.output_size = output_size
        self.rng = rng or random.Random()
        self.weights = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(input_size)]
            for _ in range(output_size)
        ]

    def predict_probs(self, inputs):
        logits = []
        for row in self.weights:
            logits.append(sum(w * x for w, x in zip(row, inputs)))
        return _softmax(logits)

    def loss(self, inputs_list, targets_list):
        total = 0.0
        for inputs, target in zip(inputs_list, targets_list):
            probs = self.predict_probs(inputs)
            for p, t in zip(probs, target):
                if t:
                    total -= math.log(max(p, 1e-12))
        return total / max(1, len(inputs_list))

    def train(self, inputs_list, targets_list, epochs=10, lr=0.1):
        for _ in range(epochs):
            for inputs, target in zip(inputs_list, targets_list):
                probs = self.predict_probs(inputs)
                for i in range(self.output_size):
                    error = probs[i] - target[i]
                    for j in range(self.input_size):
                        self.weights[i][j] -= lr * error * inputs[j]


def _one_hot(indices, size, value=1.0):
    vec = [0.0] * size
    for idx in indices:
        vec[idx] = value
    return vec


def _sample_without_replacement(weights, count, rng):
    pool = list(range(len(weights)))
    chosen = []
    local_weights = list(weights)
    for _ in range(count):
        pick = rng.choices(pool, weights=local_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        local_weights.pop(idx)
    return chosen


def generate_neural_lines(draws, count, rng=None, epochs=10, lr=0.1):
    rng = rng or random.SystemRandom()
    if len(draws) < 2:
        return []

    input_size = 65
    main_model = SoftmaxModel(input_size=input_size, output_size=45, rng=random.Random(0))
    joker_model = SoftmaxModel(input_size=input_size, output_size=20, rng=random.Random(1))

    inputs = []
    main_targets = []
    joker_targets = []

    for prev, nxt in zip(draws[:-1], draws[1:]):
        prev_main, prev_joker = prev
        x = _one_hot([n - 1 for n in prev_main], 45) + _one_hot([prev_joker - 1], 20)
        inputs.append(x)
        main_targets.append(_one_hot([n - 1 for n in nxt[0]], 45, value=1.0 / 5.0))
        joker_targets.append(_one_hot([nxt[1] - 1], 20))

    main_model.train(inputs, main_targets, epochs=epochs, lr=lr)
    joker_model.train(inputs, joker_targets, epochs=epochs, lr=lr)

    last_main, last_joker = draws[-1]
    last_x = _one_hot([n - 1 for n in last_main], 45) + _one_hot([last_joker - 1], 20)
    main_probs = main_model.predict_probs(last_x)
    joker_probs = joker_model.predict_probs(last_x)

    lines = []
    seen = set()
    while len(lines) < count:
        main_idxs = _sample_without_replacement(main_probs, 5, rng)
        main = sorted([i + 1 for i in main_idxs])
        joker = rng.choices(range(1, 21), weights=joker_probs, k=1)[0]
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))

    return lines
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_neural.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/neural.py tests/test_neural.py
git commit -m "feat: add softmax neural baseline"
```

### Task 5: Backtest + weekly picks script

**Files:**
- Create: `src/joker_model/backtest.py`
- Create: `scripts/generate_joker_picks.py`
- Test: `tests/test_backtest.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest.py
import unittest
import random

from joker_model.backtest import pick_best_strategy


class TestBacktest(unittest.TestCase):
    def test_pick_best_strategy(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        best = pick_best_strategy(draws, rng=random.Random(0))
        self.assertIn(best, {"random", "frequency", "neural"})
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_backtest.py -v`
Expected: FAIL with `ImportError` or missing `pick_best_strategy`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/backtest.py
import random

from .metrics import is_joker_prize
from .neural import generate_neural_lines
from .strategies import generate_random_lines, generate_frequency_lines


def _score_strategy(draws, generator, rng):
    wins = 0
    for main, joker in draws:
        lines = generator(1, rng=rng)
        line_main, line_joker = lines[0]
        main_matches = len(set(line_main) & set(main))
        joker_match = line_joker == joker
        if is_joker_prize(main_matches, joker_match):
            wins += 1
    return wins


def pick_best_strategy(draws, rng=None):
    rng = rng or random.Random()
    freq = {n: 1 for n in range(1, 46)}
    scores = {
        "random": _score_strategy(draws, generate_random_lines, rng),
        "frequency": _score_strategy(draws, lambda c, rng=None: generate_frequency_lines(c, freq, rng=rng), rng),
    }

    if len(draws) >= 2:
        scores["neural"] = _score_strategy(draws, lambda c, rng=None: generate_neural_lines(draws, c, rng=rng), rng)
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)
```

```python
# scripts/generate_joker_picks.py
import random
from pathlib import Path

from joker_model.fetch import update_dataset
from joker_model.storage import load_draws
from joker_model.backtest import pick_best_strategy
from joker_model.strategies import generate_random_lines, generate_frequency_lines
from joker_model.neural import generate_neural_lines


def main():
    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
    cache_path = Path("data/raw/joker_results.html")
    csv_path = Path("data/clean/joker_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    rng = random.SystemRandom()
    best = pick_best_strategy([(d.main_numbers, d.joker) for d in draws])

    if best == "neural":
        lines = generate_neural_lines([(d.main_numbers, d.joker) for d in draws], 7, rng=rng)
    elif best == "frequency":
        freq = {n: 1 for n in range(1, 46)}
        lines = generate_frequency_lines(7, freq, rng=rng)
    else:
        lines = generate_random_lines(7, rng=rng)

    for idx, (main, joker) in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)} + J{joker}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_backtest.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/backtest.py scripts/generate_joker_picks.py tests/test_backtest.py
git commit -m "feat: add backtest and weekly picks script"
```
