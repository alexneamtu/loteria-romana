# Odds and Recency Enhancements Design

## Goal

- Document official prize tiers and odds in README for Loto 6/49, Joker, and Loto 5/40.
- Add consistent recency weighting across heuristics and neural training.
- Support both draw-count and time-based half-life weighting modes.

## Scope

- Base games only: Loto 6/49, Joker, Loto 5/40.
- No add-ons (Noroc, Noroc Plus, Super Noroc) in README or calculations.
- No changes to data sources or claims about improving odds.

## References

- `docs/05-loto-ro-specific.md` for official tier rules and baseline odds.
- Official loto.ro odds pages (linked in `docs/loto-ro-any-prize-checklist.md`).
- Existing recency helpers in `src/shared/recency.py`.

## Design

### README Odds Tables

For each game, add two tables and summary lines:

- Official Prize Tiers table: tier rule, category label, odds as "1 in X", and percent.
- Simplified Win Rules table: plain-language conditions for each tier.
- Summary lines for any-prize (sum of tier probabilities) and jackpot odds.
- A short note that odds are fixed by the rules and independent of historical data.

Odds will be computed from combinatorics to keep results reproducible and aligned with
the documented rules. A short formula footnote per game will indicate the base
combination used (for example, C(49,6) for Loto 6/49).

### Recency Weighting Modes

Add a config setting to choose the half-life mode:

- `draws`: current behavior, half-life measured in number of draws.
- `days`: half-life measured in calendar days between draw dates.

In `days` mode, weights require draw dates. If dates are missing, raise a clear error.
Most recent draw weight stays 1.0 and reaches 0.5 at the configured half-life.

### Heuristic Strategy Weighting

Existing weighted frequency helpers remain the default for frequency, delta, pair,
and related strategies. Update advanced gap and trend scoring to compute weighted
averages based on the same recency weights, so newer draws affect these components
more than older draws.

### Neural Sample Weighting

Apply per-sample weights in training loops for MLP, LSTM, and softmax models. The
weight for each training sample aligns with the target draw date (the "next draw"
being predicted). If dates are unavailable, weights default to 1.0 with a warning.

### Data Flow Summary

1. Load draws and draw dates from CSV.
2. Build recency weights by draw index or draw date.
3. Use weights in heuristic stats builders and advanced gap/trend scoring.
4. Build sample weights for neural training aligned to target draw dates.

## Error Handling

- If `days` mode is enabled but draw dates are missing, raise `ValueError` with a
  clear message.
- If weights or dates are mismatched in length, raise a validation error.

## Testing

- Recency weights: draw-count half-life and day-based half-life correctness.
- Gap/trend: weighted averages shift toward recent draws in a deterministic test.
- Neural: weighted loss matches unweighted when weights are all 1.0 and changes
  when weights differ.
- README: verify odds and percent values match combinatorics and documented rules.

## Documentation

- Update `README.md` with tier tables and any-prize/jackpot summaries.
- Keep references to official odds pages and the existing honesty note.
