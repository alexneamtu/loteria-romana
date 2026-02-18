# Results DB Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist generate and check workflow outcomes to a database without breaking the existing CSV/text outputs.

**Architecture:** Add a shared persistence module that initializes schema and writes normalized records for workflow runs, generation runs/tickets, and check results. Keep CSV history append in place (dual-write) and make DB writes optional via environment configuration so current workflows remain stable.

**Tech Stack:** Python standard library (`sqlite3`, `json`, `uuid`) with optional PostgreSQL driver (`psycopg`) when `DATABASE_URL` is set to Postgres.

---

### Task 1: Add failing DB persistence tests

**Files:**
- Create: `tests/test_results_db.py`

**Steps:**
1. Write a test that persists a generation run and asserts rows exist in `workflow_runs`, `generation_runs`, `generated_tickets`.
2. Write a test that persists check results and asserts strategy rows are expanded into `check_results`.
3. Run: `PYTHONPATH=src python -m pytest tests/test_results_db.py -v` and confirm failure before implementation.

### Task 2: Implement shared DB module

**Files:**
- Create: `src/shared/results_db.py`

**Steps:**
1. Implement DSN resolution from `DATABASE_URL`/`RESULTS_DB_PATH`.
2. Implement SQLite and optional PostgreSQL connection handling.
3. Implement schema bootstrap and insert helpers.
4. Implement high-level `persist_generation_run` and `persist_check_results`.
5. Run tests from Task 1 and confirm green.

### Task 3: Wire generation workflow script

**Files:**
- Modify: `scripts/generate_recommended_picks.py`

**Steps:**
1. Collect emitted ticket lines into a normalized list for DB persistence.
2. Call DB persistence helper near script end (and for zero-ticket allocation path).
3. Ensure failures in DB writes do not fail pick generation.

### Task 4: Wire check workflow script

**Files:**
- Modify: `scripts/check_results.py`

**Steps:**
1. Call DB persistence helper after building `history_rows`.
2. Keep existing `append_history(...)` CSV behavior unchanged.
3. Ensure DB write errors are logged but non-fatal.

### Task 5: Configure workflows and verify

**Files:**
- Modify: `.github/workflows/generate-picks.yml`
- Modify: `.github/workflows/check-results.yml`

**Steps:**
1. Pass `DATABASE_URL` into the run steps (from secrets).
2. Add optional `psycopg` install step in workflows to support Postgres runtime.
3. Run:
   - `PYTHONPATH=src python -m pytest tests/test_results_db.py tests/test_generate_recommended_picks.py tests/test_check_results.py -v`
4. Commit with focused message after green results.
