# Senior AI Engineer Improvement Prompt

Use this prompt with Claude Code (or any AI coding agent) in the root of the `loteria-romana` repository.

---

## The Prompt

You are a **senior AI/ML engineer** performing a comprehensive audit and improvement of a lottery modeling pipeline. Your mandate is to **measurably improve prediction quality, system reliability, and analytical depth** by any means necessary -- new models, better data, smarter ensembles, tighter infrastructure, sharper evaluation.

The project already acknowledges that lottery prediction against truly random draws is impossible. Your job is NOT to chase miracles. Your job is to **extract every defensible edge from the data that exists**, improve the rigor of the system that measures whether edges exist, and make the whole pipeline production-grade.

Read `CLAUDE.md` and `docs/08-honest-assessment.md` first to understand the project's philosophy. Then systematically work through every improvement area below.

---

### AREA 1: DATA PIPELINE -- MORE DATA, BETTER FEATURES

**Current state:** ~1,050 Joker draws, ~1,250 Loto 6/49 draws, ~1,200 Loto 5/40 draws scraped from loto.ro. Raw HTML parsing, CSV storage. Features are basic: frequency counts, gaps, co-occurrence.

**Improvements:**

1. **Historical data expansion.** The loto.ro site likely has paginated archives going back further. Modify `src/*/fetch.py` to crawl multiple pages and backfill the full historical dataset. More data = more statistical power for every downstream model. Target: every draw ever published on loto.ro.

2. **Feature engineering overhaul.** Create `src/shared/feature_engine.py` with a proper feature extraction pipeline. Implement at minimum:
   - **Positional features:** For each number, track its frequency in each draw position (1st drawn, 2nd drawn, etc.) if draw order is available.
   - **Combinatorial features:** Sum of drawn numbers, range (max - min), standard deviation, median, skewness of the drawn set.
   - **Calendar features:** Day of week, week of year, month, days since epoch. Some draws happen on specific weekdays -- model potential mechanical/procedural patterns.
   - **Gap matrix:** For every number, compute the gap (draws since last appearance) as a time series. Model the gap distribution per number.
   - **Pair/triple lift:** Beyond raw co-occurrence counts, compute lift = P(A and B) / (P(A) * P(B)). Pairs with lift >> 1 or lift << 1 are interesting.
   - **Rolling statistics:** 10-draw, 30-draw, 100-draw rolling frequency, rolling gap mean, rolling gap variance per number.
   - **Entropy features:** Shannon entropy of the drawn set per draw. Track entropy trends over time.
   - **Autocorrelation:** Per-number autocorrelation at lags 1, 2, 5, 10. Are there any numbers with non-trivial serial correlation?

3. **Data quality checks.** Add a `src/shared/data_validator.py` module that runs on every load:
   - Verify no duplicate dates
   - Verify all numbers within pool range
   - Verify correct count of numbers per draw
   - Verify chronological ordering
   - Detect and flag any gaps in the expected draw schedule

---

### AREA 2: MODEL IMPROVEMENTS -- DEEPER, SMARTER, MORE DIVERSE

**Current state:** 11 strategies (Random, Frequency, Bayesian, Co-occurrence, Genetic, Gradient Boost, LSTM, TCN, Transformer, Normalizing Flows, RL Agent). Ensemble blended via walk-forward softmax scoring.

**Improvements:**

4. **Temporal Fusion Transformer (TFT).** The current Transformer is a basic decoder-only GPT-style model. Implement a proper TFT (`src/shared/tft_strategy.py`) that handles:
   - Static covariates (game type, pool size)
   - Known future inputs (calendar features, draw index)
   - Observed past inputs (historical draw multi-hot encodings)
   - Variable selection networks to learn which features matter
   - Interpretable multi-head attention
   This is the state-of-the-art for time series forecasting and should be tested.

5. **Graph Neural Network for co-occurrence.** The current co-occurrence strategy uses a flat matrix. Create `src/shared/gnn_strategy.py`:
   - Model numbers as nodes, co-occurrence frequency as edge weights
   - Use a Graph Attention Network (GAT) to learn node embeddings
   - Predict next-draw probability as a node classification task
   - This captures higher-order structural relationships that flat matrices miss.

6. **Variational Autoencoder (VAE).** Create `src/shared/vae_strategy.py`:
   - Encode historical draws into a latent space
   - Sample from the latent distribution to generate new draws
   - The latent space might capture structural constraints (sum ranges, spacing patterns) even if individual number prediction is impossible.

7. **Diffusion model for draw generation.** Create `src/shared/diffusion_strategy.py`:
   - Treat draw generation as a denoising diffusion process
   - Start from noise, iteratively denoise to produce a valid draw
   - Modern generative approach that could capture distributional properties.

8. **Improve the RL agent.** The current REINFORCE agent (`src/shared/rl_agent.py`) is basic:
   - Replace REINFORCE with PPO (Proximal Policy Optimization) for more stable training
   - Add a value baseline network to reduce variance
   - Use the backtest reward signal (actual prize matches) instead of synthetic rewards
   - Add entropy regularization to prevent premature convergence to a narrow distribution

9. **Hyperparameter optimization.** Create `src/shared/hpo.py`:
   - Implement Optuna-based hyperparameter search for all ML strategies
   - Search space: learning rate, hidden dimensions, number of layers, window sizes, dropout rates
   - Objective: walk-forward backtest score (NOT training loss -- avoid overfitting)
   - Save best hyperparameters per strategy per game to a config file

10. **Anomaly/regime detection.** Create `src/shared/regime_detector.py`:
    - Use Hidden Markov Models to detect regime changes in draw distributions
    - Implement CUSUM (cumulative sum) change-point detection
    - When a regime change is detected, weight recent draws much more heavily
    - The current ADWIN drift detection in ensemble_blend.py is a good start -- extend it

---

### AREA 3: ENSEMBLE IMPROVEMENTS -- SMARTER BLENDING

**Current state:** Walk-forward scoring with softmax weights, bias-aware adjustment, significance gating. All in `src/shared/ensemble_blend.py`.

**Improvements:**

11. **Meta-learner stacking.** Instead of softmax-weighted blending of strategy outputs, train a meta-learner:
    - Each base strategy produces a probability distribution over numbers
    - Stack these distributions as features for a ridge regression / gradient boosting meta-model
    - The meta-model learns which strategies to trust in which contexts (e.g., after regime changes, during low-entropy periods)
    - Use nested cross-validation to avoid information leakage

12. **Dynamic strategy selection with Thompson Sampling.**
    - Model each strategy's performance as a Beta distribution
    - Use Thompson Sampling to dynamically allocate picks across strategies
    - Strategies that have been performing well get more picks; underperformers get fewer
    - This naturally handles the exploration/exploitation tradeoff better than static softmax

13. **Ensemble diversity enforcement.**
    - After generating candidate picks from all strategies, apply diversity-aware selection
    - Maximize the "coverage" of the number pool across the final ticket set
    - Use Determinantal Point Processes (DPPs) for principled diversity sampling
    - The current portfolio optimization is a step in this direction -- replace it with DPP

14. **Conformal prediction for uncertainty quantification.**
    - Wrap the ensemble output in a conformal prediction framework
    - Instead of just outputting "here are your picks," output calibrated prediction sets
    - "With 90% confidence, the winning numbers will include at least one from {set}"
    - This gives users honest uncertainty bounds

---

### AREA 4: EVALUATION FRAMEWORK -- PROVE IT OR LOSE IT

**Current state:** Walk-forward backtesting, Monte Carlo significance, paper trading, Wilson confidence intervals. Good foundation but not rigorous enough.

**Improvements:**

15. **Proper holdout evaluation with temporal splitting.**
    - Reserve the most recent 100 draws as a STRICT holdout -- never used for training, tuning, or strategy selection
    - All model development and ensemble tuning uses only the training set
    - Final evaluation on holdout is done ONCE
    - Report the result honestly, even if (especially if) it shows no edge

16. **Multiple hypothesis testing correction.**
    - With 11+ strategies, testing each for significance at p < 0.05 guarantees false positives
    - Apply Bonferroni correction: require p < 0.05/11 = 0.0045 per strategy
    - Or better, use Benjamini-Hochberg FDR control
    - Update `src/shared/backtest_base.py` significance gating to use corrected thresholds

17. **Effect size reporting.**
    - p-values alone are not enough. Report Cohen's d or similar effect sizes
    - A strategy that's "statistically significant" but only 0.01% better than random is useless
    - Add minimum effect size thresholds: a strategy must be at least X% better than random baseline to be included in the ensemble

18. **Calibration analysis.**
    - For each strategy, compare predicted probabilities vs observed frequencies
    - Plot reliability diagrams (calibration curves)
    - A well-calibrated model that says "number 7 has 5% chance" should see number 7 appear ~5% of the time
    - Create `src/shared/calibration.py` with Brier score, ECE (Expected Calibration Error), reliability diagrams

19. **Synthetic data adversarial testing.**
    - Generate perfectly uniform random draws (already done via Monte Carlo)
    - Run the FULL pipeline (all strategies + ensemble) on this synthetic data
    - If the system reports ANY strategy as "significant" on uniform data, you have a bug
    - This is the ultimate sanity check -- automate it as a CI test

20. **Live tracking dashboard.**
    - The current DB (PostgreSQL via `DATABASE_URL`) stores generation runs and check results
    - Create a query script `scripts/analytics_report.py` that connects to the DB and produces:
      - Cumulative match count over time (is it trending up, down, or flat?)
      - Per-strategy hit rate over the last N draws
      - ROI tracking (money spent vs theoretical prize winnings)
      - Comparison vs pure random baseline (generate random picks for the same draws)
    - Output as both terminal tables and optional HTML report

---

### AREA 5: INFRASTRUCTURE & RELIABILITY

**Current state:** GitHub Actions workflows for generate/check, Telegram notifications, CSV + PostgreSQL persistence. Standard library only for core, optional PyTorch/sklearn.

**Improvements:**

21. **Reproducibility lockdown.**
    - Every generation run is seeded with `github.run_id`, which is good
    - But the ML models (PyTorch, sklearn) have additional sources of non-determinism
    - Add `torch.manual_seed()`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG` env var
    - Verify that the same seed produces IDENTICAL picks across runs

22. **Model versioning and caching.**
    - Training LSTM/Transformer/RL models from scratch on every generation run is wasteful
    - Implement model checkpointing: save trained weights to the DB or as artifacts
    - On subsequent runs, load cached weights and only fine-tune on new draws since last training
    - Add a model version hash based on training data + hyperparameters

23. **Monitoring and alerting.**
    - Add a health check that alerts (via Telegram) if:
      - Generation run fails or takes > 15 minutes
      - Check run can't find recent draws after max retries
      - Any strategy's recent hit rate drops below random baseline by > 2 sigma
      - The database connection fails
    - Create `src/shared/monitoring.py` with these checks

24. **Budget optimization improvements.**
    - The current budget optimizer in `scripts/generate_recommended_picks.py` does brute-force search
    - For larger budgets, this is O(n^3) -- implement dynamic programming or linear programming
    - Add support for multi-draw tickets (e.g., "play the same numbers for 5 draws")
    - Add support for system/wheeling tickets where the lottery allows them

25. **Backfill script for DB.**
    - The DB was added in commit `0da714b` but only has data from that point forward
    - Create `scripts/backfill_db.py` that reads the full `data/results/history.csv` and inserts historical rows
    - Also generate retrospective picks (using the same seed logic) for past draws and record what would have happened

---

### AREA 6: RESEARCH-GRADE ANALYSIS

26. **Randomness quality testing of the lottery itself.**
    - Create `scripts/analyze_randomness.py` that runs the full NIST SP 800-22 randomness test suite on the historical draws
    - Tests: frequency, block frequency, runs, longest run, spectral (DFT), non-overlapping template, overlapping template, Maurer's universal, linear complexity, serial, approximate entropy, cumulative sums, random excursions
    - If ANY test fails at significance level 0.01, document it prominently -- it would suggest the lottery's RNG has detectable bias
    - This is the single most valuable analysis in the entire project

27. **Cross-game correlation analysis.**
    - Do Joker and Loto 6/49 draws (which happen on the same days) show any correlation?
    - Test: are the sum-of-numbers correlated across same-day draws of different games?
    - If they share a physical RNG, there might be subtle dependencies
    - Create `scripts/cross_game_analysis.py`

28. **Benford's Law analysis.**
    - Apply Benford's Law (first-digit distribution) to drawn numbers
    - Lottery numbers from a uniform distribution should NOT follow Benford's Law
    - If they do, it's a sign of non-random generation
    - Quick analysis, high signal

29. **Long-range dependency testing.**
    - Apply Hurst exponent estimation to the per-number appearance time series
    - H = 0.5: random walk (no memory)
    - H > 0.5: persistent (trending)
    - H < 0.5: anti-persistent (mean-reverting)
    - If any number shows H significantly different from 0.5, it's worth investigating

30. **Publication-ready report generation.**
    - Create `scripts/generate_report.py` that produces a comprehensive PDF/HTML report
    - Include: data summary, randomness test results, strategy performance, ensemble calibration, holdout results, financial analysis
    - Make it suitable for sharing -- anyone should be able to read it and understand exactly what the system does and whether it works
    - Use matplotlib/seaborn for visualizations, output as self-contained HTML

---

### EXECUTION PRIORITIES

Work in this order:

1. **Data quality & expansion** (items 1-3) -- everything downstream depends on data
2. **Evaluation rigor** (items 15-19) -- you can't improve what you can't measure
3. **Randomness testing** (item 26) -- if the lottery is truly random, we know our ceiling
4. **Ensemble improvements** (items 11-14) -- biggest bang for buck on existing models
5. **New models** (items 4-10) -- diminishing returns but worth testing
6. **Infrastructure** (items 21-25) -- important but not urgent
7. **Research analysis** (items 27-30) -- nice to have
8. **Dashboard & reporting** (items 20, 30) -- for ongoing monitoring

For each improvement:
- Create a feature branch
- Write tests FIRST (TDD)
- Implement the change
- Run the full test suite
- Measure impact via backtesting before/after
- Create a PR with clear description of what changed and measured results

---

### CONSTRAINTS

- Standard library for core logic. PyTorch, sklearn, optuna, etc. are optional dependencies.
- All optional dependencies must be wrapped in try/except with graceful fallback.
- No external API calls except to loto.ro for data fetching.
- Maintain the project's honest philosophy: if something doesn't work, say so clearly.
- Don't break existing tests. Run `PYTHONPATH=src python -m pytest tests/ -v` after every change.
- Follow the existing code patterns in `src/shared/` (Strategy protocol, GameConfig, etc.)
- Git workflow: feature branches, PRs against main, never push directly.
