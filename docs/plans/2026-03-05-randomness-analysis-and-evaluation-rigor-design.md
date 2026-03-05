# Randomness Analysis & Evaluation Rigor — Design

**Date:** 2026-03-05
**Status:** Approved
**Approach:** Modular library + thin script layer (Approach 2)

## Goals

Establish whether the lottery data contains any detectable non-randomness, then tighten the evaluation framework so the system can honestly measure whether its strategies have any edge.

Two phases executed in order:
- **Phase D:** Randomness analysis (items 26-29 from improvement prompt)
- **Phase C:** Data validation + evaluation rigor (items 1-3 partial, 15-19)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Output format | Terminal + JSON | Lean, machine-readable, CI-friendly |
| NIST tests | Applicable subset only (6 tests) | Honest results at our sample size |
| Holdout size | 100 draws (~10%) | Solid statistical power, 950+ draws still available for training |
| Hypothesis correction | Tiered (BH gate + effect size scaling) | Bonferroni too aggressive, soft-only lets noise through |

## Shared Foundation

```python
@dataclass
class AnalysisResult:
    test_name: str           # e.g. "frequency_monobit"
    game: str                # e.g. "joker", "loto_649", "all"
    passed: bool | None      # True=pass, False=fail, None=inconclusive
    p_value: float | None
    statistic: float
    threshold: float         # significance threshold (0.01)
    sample_size: int
    details: dict            # test-specific extras
    summary: str             # human-readable one-liner
```

All modules return `list[AnalysisResult]`. Significance threshold: 0.01 throughout.

## Phase D: Randomness Analysis

### Module 1: `src/shared/randomness_tests.py`

NIST SP 800-22 applicable subset:

| Test | Detects | Min draws |
|------|---------|-----------|
| Frequency (monobit) | Overall frequency imbalance per number | ~100 |
| Runs test | Too many/few consecutive appearance streaks | ~100 |
| Longest run | Unusually long streaks | ~200 |
| Serial test | Pairwise sequential dependencies | ~500 |
| Approximate entropy | Regularity/predictability | ~500 |
| Cumulative sums | Persistent frequency drift over time | ~100 |

Encoding: Each draw becomes a multi-hot binary vector (length = pool_size). Tests operate on per-number binary sequences and aggregate properties.

```python
def run_randomness_tests(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]
```

### Module 2: `src/shared/benford.py`

First-digit distribution analysis. Chi-square goodness-of-fit against both Benford's distribution and expected uniform. Reports which fits better.

```python
def run_benford_analysis(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]
```

### Module 3: `src/shared/hurst.py`

Rescaled Range (R/S) analysis for Hurst exponent per number's inter-appearance gap time series. Pure standard library. Flags numbers with H significantly different from 0.5 via bootstrap confidence interval.

```python
def run_hurst_analysis(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]
```

### Module 4: `src/shared/cross_game_analysis.py`

Same-date analysis across Joker, Loto 6/49, and Loto 5/40:
- Sum correlation (Pearson)
- Number overlap vs chance
- Cross-game entropy reduction

```python
def run_cross_game_analysis(
    game_draws: dict[str, list[tuple[date, list[int]]]],
    significance: float = 0.01,
) -> list[AnalysisResult]
```

### Script: `scripts/analyze_randomness.py`

CLI wrapper: loads all games, runs all four modules, prints terminal summary, writes `data/analysis/randomness_report.json`.

## Phase C: Data Quality & Evaluation Rigor

### Data Validator: `src/shared/data_validator.py`

```python
@dataclass
class ValidationIssue:
    severity: str        # "error" | "warning"
    game: str
    description: str
    draw_date: str | None
    details: dict

def validate_draws(
    draws: list,
    game_config: GameConfig,
) -> list[ValidationIssue]
```

Checks: duplicate dates, numbers in range, correct count per draw, no duplicate numbers within a draw, chronological order, schedule gap detection.

### Holdout Split: `src/shared/holdout.py`

```python
@dataclass
class TemporalSplit:
    train: list
    holdout: list
    holdout_size: int
    split_date: date

def temporal_holdout_split(
    draws: list,
    holdout_size: int = 100,
) -> TemporalSplit
```

Last 100 draws chronologically. No shuffling. Used once for final evaluation.

### Corrected Significance: update `src/shared/backtest_base.py`

```python
@dataclass
class CorrectedResult:
    strategy: str
    raw_p_value: float
    adjusted_p_value: float    # BH-corrected
    effect_size: float         # Cohen's h
    verdict: str               # "excluded" | "included"
    weight_scale: float        # 0.0 if excluded, effect-size-scaled if included

def correct_significance(
    strategy_results: list[BacktestResult],
    fdr_threshold: float = 0.10,
    min_effect_size: float = 0.01,
) -> list[CorrectedResult]
```

Tiered logic:
1. Raw p-values per strategy (existing z-test)
2. Benjamini-Hochberg at FDR=0.10 — failures excluded (weight=0)
3. Survivors weighted by Cohen's h effect size
4. Below min_effect_size also excluded

### Calibration: `src/shared/calibration.py`

```python
@dataclass
class CalibrationResult:
    strategy: str
    brier_score: float
    expected_calibration_error: float
    bins: list[dict]

def compute_calibration(
    predicted_probs: list[float],
    observed: list[bool],
    n_bins: int = 10,
) -> CalibrationResult
```

### Adversarial Test: `tests/test_adversarial.py`

Generates uniform random draws, runs full ensemble pipeline, asserts NO strategy passes significance gate. CI gate — failure means the pipeline has a bug.

## File Layout

### New files

```
src/shared/
    analysis_result.py
    randomness_tests.py
    benford.py
    hurst.py
    cross_game_analysis.py
    data_validator.py
    holdout.py
    calibration.py

scripts/
    analyze_randomness.py

tests/
    test_randomness_tests.py
    test_benford.py
    test_hurst.py
    test_cross_game.py
    test_data_validator.py
    test_holdout.py
    test_calibration.py
    test_adversarial.py
    test_corrected_significance.py
```

### Modified files

```
src/shared/backtest_base.py    # Add correct_significance(), CorrectedResult
src/shared/ensemble_blend.py   # Wire in corrected significance gating
.gitignore                     # Add data/analysis/
```

### Dependencies

All standard library. No numpy, scipy, or matplotlib required.

### Branch strategy

1. `feature/randomness-analysis` — Phase D
2. `feature/evaluation-rigor` — Phase C
