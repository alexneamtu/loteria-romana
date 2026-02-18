# Joker Set Optimizer Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve practical Joker ticket-set quality by reducing duplicated/overlapping main lines and maximizing Joker-number coverage across generated tickets.

**Architecture:** Add a small shared optimizer module for Joker ticket sets, then apply it at the CLI integration point (`scripts/generate_joker_picks.py`) as a post-processing layer over generated main-number candidates. Keep strategy generation unchanged and only optimize the final set.

**Tech Stack:** Python 3.12, standard library, existing shared portfolio optimizer (`shared.portfolio`), unittest/pytest test suite.

---

### Task 1: Add failing tests for Joker set optimization helpers

**Files:**
- Create: `tests/test_joker_set_optimizer.py`
- Reference: `src/shared/portfolio.py`

**Step 1: Write failing tests**
- Test Joker assignment uses unique Joker values for first 20 picks.
- Test Joker assignment cycles deterministically with seeded RNG for counts > 20.
- Test main-line optimizer removes duplicates and returns requested count.
- Test main-line optimizer improves diversity when duplicates dominate candidates.

**Step 2: Run tests to verify failure**
- Run: `PYTHONPATH=src python -m pytest tests/test_joker_set_optimizer.py -v`
- Expected: FAIL because helper module does not exist yet.

### Task 2: Implement shared Joker set optimizer

**Files:**
- Create: `src/shared/joker_set_optimizer.py`

**Step 1: Add Joker assignment helper**
- `assign_max_coverage_jokers(count, rng, joker_pool=20)`:
  - Generate shuffled full pool cycles.
  - Ensure maximal unique coverage before repeats.

**Step 2: Add main-line set optimizer**
- `optimize_main_ticket_set(candidates, select_count, pool_size, numbers_to_pick, rng)`:
  - De-duplicate candidates.
  - Add random valid candidates if pool is too small.
  - Delegate final subset selection to `optimize_ticket_portfolio`.

**Step 3: Run tests**
- Run: `PYTHONPATH=src python -m pytest tests/test_joker_set_optimizer.py -v`
- Expected: PASS.

### Task 3: Integrate optimizer into Joker CLI flow

**Files:**
- Modify: `scripts/generate_joker_picks.py`
- Test: `tests/test_cli_scripts.py`

**Step 1: Integrate post-processing**
- For all non-wheel strategy branches that produce main picks, route through `optimize_main_ticket_set`.
- Replace per-line random Joker assignment with `assign_max_coverage_jokers`.
- In wheel mode, assign Joker values using the same max-coverage helper.

**Step 2: Add focused integration tests**
- Add tests verifying:
  - Joker numbers are unique when `count <= 20`.
  - Duplicate main picks are reduced by optimizer output path.

**Step 3: Run targeted tests**
- Run: `PYTHONPATH=src python -m pytest tests/test_joker_set_optimizer.py tests/test_cli_scripts.py -v`
- Expected: PASS.

### Task 4: Verification sweep

**Files:**
- No code changes expected.

**Step 1: Run additional related tests**
- Run:
  - `PYTHONPATH=src python -m pytest tests/test_portfolio.py -v`
  - `PYTHONPATH=src python -m pytest tests/test_generate_picks_workflow.py -v`

**Step 2: Validate outcome**
- Confirm no regressions.
- Confirm new optimization helpers are covered.
