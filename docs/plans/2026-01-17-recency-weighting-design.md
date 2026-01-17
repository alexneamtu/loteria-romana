# Recency Weighting Design

## Goal
Make newer lottery draws influence all heuristic strategies more than older draws using a shared exponential recency weighting, while keeping defaults simple and configurable. Ensure weekly pick generation on Thursday and Sunday uses the latest results by backfilling before picks are generated.

## Configuration
- CLI: add `--half-life` (float) to `scripts/generate_joker_picks.py`, `scripts/generate_loto_649_picks.py`, and `scripts/generate_loto_540_picks.py`.
- Env: add `RECENCY_HALF_LIFE` as a global default.
- Default: 50 draws if neither is set.
- Validation: reject non-positive values with a clear `ValueError`.

## Recency Weights
Introduce a shared helper (e.g., `shared/recency.py`) that builds per-draw weights from oldest to newest:

```
weight(age) = 0.5 ** (age / half_life)
```

Where `age = (len(draws) - 1 - idx)` so the newest draw has weight 1.0. The helper should expose:
- `resolve_half_life(cli_value, env_value, default=50.0)`
- `draw_weights(draw_count, half_life)` returning a list aligned with draws order
- Optional utility for weighted mean/std if needed by sum distributions

## Strategy Integration
Apply the weights anywhere historical draws are aggregated so all heuristic strategies inherit the same decay:

- `shared/game_strategies.build_frequency` and the per-game `build_frequency` helpers use weighted counts.
- `shared/stats` updates:
  - `build_delta_distribution`, `compute_odd_even_distribution`, `compute_high_low_distribution`, `compute_consecutive_distribution`, `build_position_frequency` use weighted counts.
  - `SumConstraintStrategy.compute_sum_distribution` uses weighted mean/std.
  - `HotColdStrategy` switches from a fixed decay rate to using the shared half-life weights.
  - `PairStrategy` uses weighted co-occurrence counts.
  - `SkipGapStrategy` keeps current gap computation, but expected gap averages use weighted history so newer gaps matter more.
- `shared/advanced_strategies.compute_composite_scores` replaces the hardcoded `decay_rate=0.95` with the shared half-life and uses weighted distributions for recency/frequency/position where applicable.
- CLI scripts pass the resolved half-life into all strategy constructors and into advanced strategy helpers.

Neural strategies remain unweighted in this change.

## Workflow Update
Update `.github/workflows/generate-picks.yml` to run:
```
PYTHONPATH=src python scripts/backfill_noroc_chior.py
```
Before generating picks, ensuring the Thu/Sun runs include the latest results.

## Error Handling
- Empty draws fall back to uniform distributions as today.
- If weights sum to zero (should not happen with valid half-life), fall back to uniform.

## Testing
- Add unit tests for recency weights (monotonic decay, newest weight 1.0, half-life behavior).
- Update stats tests to allow weighted behavior; verify relative ordering or approximate ratios instead of exact integer counts.
- Add at least one test validating weighted frequency (newer draw outweighs older draw) to cover the shared path.

## Non-goals
- Neural training with sample weights.
- Changing draw ingestion or dataset structure.
