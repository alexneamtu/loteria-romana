# Phase 2: Anti-Crowding, EV Gate, and Multi-Strategy Batch Aggregator

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three practical improvements:
1) anti-crowding scoring for line selection,
2) optional EV/jackpot gate in automated generation flow,
3) a script that aggregates picks across strategies and deduplicates them into a final set.

**Architecture:** Keep existing strategy generators unchanged. Add a selection-quality layer in shared utilities, add optional gating logic in `generate_recommended_picks.py`, and add a new batch script that orchestrates multiple strategy runs then optimizes the merged candidate pool.

**Tech Stack:** Python 3.12, standard library, existing shared modules (`portfolio`, `ev_calculator`, `joker_set_optimizer`), GitHub Actions workflow + unittest/pytest.

---

### Task 1: Add failing tests for anti-crowding scoring

**Files:**
- Create: `tests/test_crowding.py`
- Modify: `tests/test_joker_set_optimizer.py`

**Checks:**
- Anti-crowding score should penalize birthday-heavy lines.
- Anti-crowding score should penalize obvious consecutive/arithmetic patterns.
- `optimize_main_ticket_set(...)` with anti-crowding weight should prefer less-crowded lines when diversity is similar.

### Task 2: Implement anti-crowding module and integrate optimizer

**Files:**
- Create: `src/shared/crowding.py`
- Modify: `src/shared/joker_set_optimizer.py`

**Implementation:**
- Add `anti_crowding_score(line, pool_size) -> float`.
- Add `average_anti_crowding_score(lines, pool_size) -> float`.
- Extend `optimize_main_ticket_set(...)` with `anti_crowding_weight` and local-improvement pass using objective:
  - `diversity_score + anti_crowding_weight * average_anti_crowding_score`.

### Task 3: Add failing tests for EV/jackpot gate

**Files:**
- Modify: `tests/test_generate_recommended_picks.py`
- Modify: `tests/test_generate_picks_workflow.py`

**Checks:**
- Gate blocks ticket allocation when jackpot inputs are below threshold.
- Gate allows only passing games.
- Workflow exposes gate/jackpot inputs and forwards them to script.

### Task 4: Implement EV/jackpot gate

**Files:**
- Modify: `scripts/generate_recommended_picks.py`
- Modify: `.github/workflows/generate-picks.yml`

**Implementation:**
- Add CLI options:
  - `--ev-gate`, `--ev-min-ratio`,
  - `--joker-jackpot`, `--loto649-jackpot`, `--loto540-jackpot`.
- Compute per-game breakeven jackpot via `EVCalculator`.
- Filter budget allocation to games whose `jackpot / breakeven >= ev_min_ratio`.
- Keep default behavior unchanged when `--ev-gate` is not enabled.
- Wire workflow dispatch inputs + command args.

### Task 5: Add failing tests for batch aggregator script

**Files:**
- Create: `tests/test_generate_joker_multi_strategy_batch.py`

**Checks:**
- Parse strategy output lines correctly.
- Merge candidates from multiple strategies and dedupe.
- Final output count and unique Joker coverage for small counts.

### Task 6: Implement batch aggregator script

**Files:**
- Create: `scripts/generate_joker_multi_strategy_batch.py`

**Implementation:**
- Accept strategies list, per-strategy count, final count, seed, recency args.
- Execute `generate_joker_picks.py` per strategy, parse outputs.
- Aggregate all main candidates, apply `optimize_main_ticket_set(...)`.
- Assign Joker values via `assign_max_coverage_jokers(...)`.
- Print and optionally write outputs.

### Task 7: Verification sweep

**Run:**
- `PYTHONPATH=src python -m pytest tests/test_crowding.py tests/test_joker_set_optimizer.py tests/test_generate_recommended_picks.py tests/test_generate_picks_workflow.py tests/test_generate_joker_multi_strategy_batch.py -v`
- `PYTHONPATH=src python -m pytest tests/test_cli_scripts.py tests/test_portfolio.py -v`
