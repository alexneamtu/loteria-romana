# Jackpot Backtest Report

seed=42 warmup=1000 jackpot=1000000 RON

## Joker

| builder | n | median_roi | mean_roi | P(>=3) | P(>=4) | P(>=5) | skewness |
|---|---|---|---|---|---|---|---|
| IndependentBuilder | 54 | -17.50 | -17.20 | 0.000% | 0.000% | 0.000% | +4.903 |
| CoreShareBuilder | 54 | -17.50 | -16.94 | 0.000% | 0.000% | 0.000% | +4.428 |
| WheelBuilder | 54 | -17.50 | -17.35 | 0.000% | 0.000% | 0.000% | +7.143 |

## Loto 6/49

| builder | n | median_roi | mean_roi | P(>=3) | P(>=4) | P(>=5) | skewness |
|---|---|---|---|---|---|---|---|
| IndependentBuilder | 258 | -28.50 | -26.95 | 6.589% | 0.388% | 0.000% | +6.959 |
| CoreShareBuilder | 258 | -28.50 | -26.99 | 5.039% | 0.388% | 0.388% | +12.323 |
| WheelBuilder | 258 | -28.50 | -27.57 | 4.651% | 0.000% | 0.000% | +4.307 |

## Loto 5/40

| builder | n | median_roi | mean_roi | P(>=3) | P(>=4) | P(>=5) | skewness |
|---|---|---|---|---|---|---|---|
| IndependentBuilder | 200 | -22.50 | -22.08 | 7.000% | 0.000% | 0.000% | +3.371 |
| CoreShareBuilder | 200 | -22.50 | -22.11 | 6.500% | 0.000% | 0.000% | +3.529 |
| WheelBuilder | 200 | -22.50 | -21.96 | 5.000% | 1.000% | 0.000% | +8.105 |

## Interpretation

Directional findings from this run (seed=42, warmup=1000, jackpot=1M RON):

- **CoreShare on Loto 6/49 produced the only `best_match >= 5` event in the sample** (1/258, skewness +12.3 vs +7.0 for Independent). The correlated-variant tilt shifted variance into the upper tail as intended. `p_best_match_ge_4` tied with Independent (both 1/258) — sample too small to separate on that metric.
- **Wheel on Loto 5/40** hit `best_match >= 4` twice in 200 draws (1%) vs 0% for Independent. Wheel's guaranteed coverage pays off when ≥ 3 of the 8-pool are drawn. Skewness +8.1 vs +3.4 Independent.
- **Wheel on Loto 6/49 underperforms** both other builders on `p_best_match_ge_3` (4.65% vs Independent's 6.59%). A pool_size=8 wheel where the pool doesn't match the draw kills all three variants together — the downside of the coverage approach when the pool is poorly chosen.
- **Joker data is under-powered at warmup=1000** — only 54 usable draws and all three builders scored zero `best_match >= 3`. Drop to warmup=500 or lower when iterating on Joker strategies.
- All `median_roi` values are the full ticket cost negated (`-17.5` / `-28.5` / `-22.5`) because the median outcome is a loss on every builder — expected, lotteries have house edge. Comparing builders on `mean_roi` reveals small differences driven by rare payouts; those tails are precisely what we're trying to thicken.

**Next-action takeaway:** CoreShare is the front-runner for 6/49, Wheel for 5/40. Joker needs either a longer backtest window or a different signal source before CoreShare can be validated there. None of these results are statistically significant at this sample size — treat them as hypothesis-generating.
