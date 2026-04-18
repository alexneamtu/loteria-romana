# Jackpot Redesign — Plan D: Workflows & Telegram

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GitHub Actions workflows and Telegram messages ticket-aware. Users see one Telegram message per physical ticket (header + variants + side-game + cost). The check-results workflow scrapes side-game winning numbers, emits `picks_detail.jsonl` history, and reports side-game outcomes alongside main-game results.

**Architecture:** Telegram formatting moves out of the bash block in `generate-picks.yml` into a Python module `src/shared/telegram_formatter.py` that consumes `picks/tickets.json`. The yaml calls it via a one-liner and pipes the output through `curl`. Same refactor for `check-results.yml`. Side-game fetch reuses the existing loto.ro HTML cache via the parsers Plan A extended. `picks_detail.jsonl` is appended from `check_results.py` per draw.

**Tech Stack:** Plans A–C plus bash + curl. No new dependencies.

**Prerequisites:** Plans A, B, C merged. Feature branch: `feature/jackpot-d-workflow-telegram`.

**Scope boundary:**
- DO: telegram_formatter, workflow yaml updates, jsonl appender, workflow integration tests, delete legacy `.txt` shim after confirming.
- DO NOT: change pricing, builder logic, or allocator — those are locked by Plans A/B/C.

---

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW | `src/shared/telegram_formatter.py` | Build Telegram-Markdown messages per ticket + summary |
| NEW | `src/shared/picks_detail_history.py` | Append-only `data/results/picks_detail.jsonl` writer |
| MODIFY | `.github/workflows/generate-picks.yml` | Python formatter step, per-ticket message loop |
| MODIFY | `.github/workflows/check-results.yml` | Per-ticket results + side-game outcomes |
| MODIFY | `scripts/check_results.py` | Append to `picks_detail.jsonl`, emit per-ticket result files |
| MODIFY | `scripts/generate_recommended_picks.py` | Stop emitting legacy `.txt` shim (flag-gated first, then removed) |
| NEW | `tests/test_telegram_formatter.py` | unittest |
| NEW | `tests/test_picks_detail_history.py` | unittest |
| MODIFY | `tests/test_generate_picks_workflow.py` | coverage for new message-generation path |
| MODIFY | `tests/test_check_results_workflow.py` | coverage for per-ticket output + jsonl append |

---

## Task 1: Telegram formatter module

**Files:**
- Create: `src/shared/telegram_formatter.py`
- Create: `tests/test_telegram_formatter.py`

Telegram wants one message per physical ticket plus a summary. Markdown `parse_mode=Markdown` (not V2). Each message:
```
🃏 *JOKER Ticket (core_share) - 2026-04-21*
Cost: 17.5 RON
```
V1: 3, 7, 12, 19, 28 + J11
V2: 3, 7, 12, 19, 33 + J11
Noroc Plus: NP07
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_telegram_formatter.py`:
```python
import json
import tempfile
import unittest
from pathlib import Path

from shared.telegram_formatter import (
    format_summary,
    format_tickets,
    format_check_results,
    emit_messages_for_workflow,
)


def _write_fixture(path: Path, tickets_list: list[dict]) -> Path:
    path.write_text(
        json.dumps({
            "generated_at": "2026-04-21T10:00:00Z",
            "budget_ron": 40.0,
            "total_cost_ron": 40.0,
            "allocation": {"joker": 1, "loto_649": 0, "loto_540": 1},
            "tickets": tickets_list,
        })
    )
    return path


class TestTelegramFormatter(unittest.TestCase):
    def test_joker_ticket_message_includes_variants_and_side_game(self):
        msg = format_tickets([{
            "game": "joker",
            "variants": [
                {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
            ],
            "side_game_number": "NP07",
            "strategy": "core_share",
            "cost_ron": 17.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("JOKER", msg)
        self.assertIn("core_share", msg)
        self.assertIn("17.5", msg)
        self.assertIn("3, 7, 12, 19, 28", msg)
        self.assertIn("+ J11", msg)
        self.assertIn("NP07", msg)

    def test_loto_649_message_shows_noroc(self):
        msg = format_tickets([{
            "game": "loto_649",
            "variants": [
                {"main_numbers": [1, 5, 17, 23, 34, 49], "bonus_number": None},
                {"main_numbers": [2, 6, 18, 24, 35, 47], "bonus_number": None},
                {"main_numbers": [3, 7, 19, 25, 36, 48], "bonus_number": None},
            ],
            "side_game_number": "1234567",
            "strategy": "wheel",
            "cost_ron": 28.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("LOTO 6/49", msg)
        self.assertIn("1234567", msg)
        self.assertIn("Noroc", msg)

    def test_loto_540_message_shows_super_noroc(self):
        msg = format_tickets([{
            "game": "loto_540",
            "variants": [
                {"main_numbers": [2, 9, 15, 27, 34], "bonus_number": None},
            ] * 4,
            "side_game_number": "012345",
            "strategy": "independent",
            "cost_ron": 22.5,
        }], draw_date="2026-04-21")[0]
        self.assertIn("LOTO 5/40", msg)
        self.assertIn("012345", msg)
        self.assertIn("Super Noroc", msg)

    def test_summary_aggregates_all_tickets(self):
        tickets = [
            {"game": "joker", "variants": [], "side_game_number": "NP07", "strategy": "core_share", "cost_ron": 17.5},
            {"game": "loto_540", "variants": [], "side_game_number": "012345", "strategy": "independent", "cost_ron": 22.5},
        ]
        msg = format_summary(tickets, budget_ron=40.0, total_cost_ron=40.0, draw_date="2026-04-21")
        self.assertIn("Lottery Picks", msg)
        self.assertIn("40.0", msg)
        self.assertIn("Joker", msg)
        self.assertIn("Loto 5/40", msg)

    def test_emit_messages_for_workflow_returns_ordered_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(Path(tmp) / "tickets.json", [{
                "game": "joker",
                "variants": [
                    {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                    {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                ],
                "side_game_number": "NP07",
                "strategy": "core_share",
                "cost_ron": 17.5,
            }])
            msgs = emit_messages_for_workflow(path, draw_date="2026-04-21")
            self.assertEqual(len(msgs), 2)  # summary + 1 ticket
            self.assertIn("Lottery Picks", msgs[0])
            self.assertIn("core_share", msgs[1])

    def test_check_results_message_shows_match_and_side_game_digits(self):
        msg = format_check_results(
            game="loto_649",
            draw_date="2026-04-21",
            winning_main=[1, 5, 17, 23, 34, 49],
            winning_side="1234567",
            ticket_outcomes=[
                {
                    "ticket_id": "t-01",
                    "strategy": "wheel",
                    "best_main_match": 4,
                    "side_game_match": 0,
                    "side_game_digits_matched": 4,
                    "variants_rendered": [
                        "1, 5, 17, 23, 34, 40 [4 hit]",
                        "2, 5, 17, 23, 34, 49 [5 hit]",
                        "3, 5, 17, 23, 34, 48 [4 hit]",
                    ],
                    "ticket_side": "9994567",
                }
            ],
        )
        self.assertIn("LOTO 6/49", msg)
        self.assertIn("Winning:", msg)
        self.assertIn("1234567", msg)
        self.assertIn("4 digits", msg)
        self.assertIn("[5 hit]", msg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_telegram_formatter -v
```
Expected: `ModuleNotFoundError: No module named 'shared.telegram_formatter'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/telegram_formatter.py`:
```python
"""Telegram message builders for ticket-aware workflow output.

Consumers (GitHub Actions) call emit_messages_for_workflow with the
path to tickets.json and the draw date; they receive an ordered list
of Telegram-Markdown messages to send. Each message should fit under
Telegram's 4096-char limit (tickets max out around ~300 chars).
"""
from __future__ import annotations

import json
from pathlib import Path

_GAME_LABEL = {"joker": "JOKER", "loto_649": "LOTO 6/49", "loto_540": "LOTO 5/40"}
_GAME_EMOJI = {"joker": "🃏", "loto_649": "🎱", "loto_540": "🎯"}
_SIDE_NAME = {"joker": "Noroc Plus", "loto_649": "Noroc", "loto_540": "Super Noroc"}
_DISPLAY_GAME = {"joker": "Joker", "loto_649": "Loto 6/49", "loto_540": "Loto 5/40"}


def _format_variant(game: str, v: dict) -> str:
    nums = ", ".join(str(n) for n in v["main_numbers"])
    if game == "joker":
        return f"{nums} + J{v['bonus_number']}"
    return nums


def format_tickets(tickets: list[dict], draw_date: str) -> list[str]:
    messages: list[str] = []
    for t in tickets:
        game = t["game"]
        header = f"{_GAME_EMOJI[game]} *{_GAME_LABEL[game]} Ticket ({t['strategy']}) - {draw_date}*"
        cost_line = f"Cost: {t['cost_ron']} RON"
        body_lines = [
            f"V{i}: {_format_variant(game, v)}"
            for i, v in enumerate(t["variants"], start=1)
        ]
        side_line = f"{_SIDE_NAME[game]}: {t['side_game_number']}"
        text = "\n".join([header, cost_line, "```", *body_lines, side_line, "```"])
        messages.append(text)
    return messages


def format_summary(
    tickets: list[dict],
    *,
    budget_ron: float,
    total_cost_ron: float,
    draw_date: str,
) -> str:
    games_count: dict[str, int] = {}
    for t in tickets:
        games_count[t["game"]] = games_count.get(t["game"], 0) + 1
    if not games_count:
        body = "_no tickets (EV gate skipped this draw)_"
    else:
        body = "\n".join(
            f"- {_DISPLAY_GAME[g]}: {n} ticket(s)"
            for g, n in games_count.items()
        )
    header = f"🎰 *Lottery Picks - {draw_date}*"
    totals = f"Budget: {budget_ron} RON\nSpent:  {total_cost_ron} RON"
    return "\n".join([header, "", totals, "", body])


def emit_messages_for_workflow(tickets_json_path: Path, draw_date: str) -> list[str]:
    if not tickets_json_path.exists():
        return [
            f"🎰 *Lottery Picks - {draw_date}*\n\n_No tickets emitted (skip or error)._"
        ]
    doc = json.loads(tickets_json_path.read_text(encoding="utf-8"))
    summary = format_summary(
        doc["tickets"],
        budget_ron=doc["budget_ron"],
        total_cost_ron=doc["total_cost_ron"],
        draw_date=draw_date,
    )
    ticket_msgs = format_tickets(doc["tickets"], draw_date)
    return [summary, *ticket_msgs]


def format_check_results(
    *,
    game: str,
    draw_date: str,
    winning_main: list[int] | tuple[int, ...],
    winning_side: str | None,
    ticket_outcomes: list[dict],
) -> str:
    header = f"{_GAME_EMOJI[game]} *{_GAME_LABEL[game]} Results - {draw_date}*"
    win_main = ", ".join(str(n) for n in winning_main)
    winning_line = f"Winning: `{win_main}`"
    side_line = (
        f"{_SIDE_NAME[game]}: `{winning_side or '—'}`"
    )
    if not ticket_outcomes:
        return "\n".join([header, winning_line, side_line, "\n_No picks to compare_"])

    parts = [header, winning_line, side_line, ""]
    for out in ticket_outcomes:
        parts.append(f"*{out['strategy']}* (best {out['best_main_match']} main)")
        parts.append("```")
        parts.extend(out.get("variants_rendered", []))
        side_note = (
            f"side: {out['ticket_side']} → {out['side_game_digits_matched']} digits"
            + (" ✅" if out["side_game_match"] else "")
        )
        parts.append(side_note)
        parts.append("```")
    return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_telegram_formatter -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/telegram_formatter.py tests/test_telegram_formatter.py
git commit -m "feat(telegram): add ticket-aware formatter for picks + check results"
```

---

## Task 2: `picks_detail.jsonl` writer

**Files:**
- Create: `src/shared/picks_detail_history.py`
- Create: `tests/test_picks_detail_history.py`

Append-only JSONL history. Each draw's check writes one line per ticket containing the full ticket dict plus match results. Lets Plan B's backtester replay on real live data rather than synthetic draws.

- [ ] **Step 1: Write the failing test**

Create `tests/test_picks_detail_history.py`:
```python
import json
import tempfile
import unittest
from pathlib import Path

from shared.picks_detail_history import append_detail_rows


class TestPicksDetailHistory(unittest.TestCase):
    def test_appends_one_json_line_per_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            rows = [
                {
                    "draw_date": "2026-04-21",
                    "game": "joker",
                    "ticket_id": "t-01",
                    "best_main_match": 3,
                    "side_game_match": 0,
                    "side_game_digits_matched": 0,
                    "cost_ron": 17.5,
                },
                {
                    "draw_date": "2026-04-21",
                    "game": "loto_540",
                    "ticket_id": "t-02",
                    "best_main_match": 2,
                    "side_game_match": 0,
                    "side_game_digits_matched": 1,
                    "cost_ron": 22.5,
                },
            ]
            append_detail_rows(path, rows)
            with path.open() as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(ln) for ln in lines]
            self.assertEqual(parsed[0]["ticket_id"], "t-01")
            self.assertEqual(parsed[1]["best_main_match"], 2)

    def test_appends_preserve_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            append_detail_rows(path, [{"draw_date": "2026-04-21", "ticket_id": "a"}])
            append_detail_rows(path, [{"draw_date": "2026-04-24", "ticket_id": "b"}])
            with path.open() as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["ticket_id"], "a")
            self.assertEqual(json.loads(lines[1])["ticket_id"], "b")

    def test_empty_rows_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            append_detail_rows(path, [])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_picks_detail_history -v
```
Expected: `ModuleNotFoundError: No module named 'shared.picks_detail_history'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/picks_detail_history.py`:
```python
"""Append-only picks_detail.jsonl — full per-ticket check results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_detail_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_picks_detail_history -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/picks_detail_history.py tests/test_picks_detail_history.py
git commit -m "feat(shared): add picks_detail.jsonl append writer for replayable history"
```

---

## Task 3: `check_results.py` writes detail JSONL and uses formatter

**Files:**
- Modify: `scripts/check_results.py`
- Modify: `tests/test_check_results.py`

Wire the formatter and detail writer in. Existing behavior (per-game `.txt` output to `results/`, `history.csv`, `persist_check_results`) all stays; we ADD jsonl output and REPLACE the text-formatter inside `build_game_message`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_results.py`:
```python
class TestCheckResultsWritesJSONL(unittest.TestCase):
    def test_check_run_appends_picks_detail_jsonl(self):
        # This is a thin test: we call the inner writer used by main().
        import tempfile
        from pathlib import Path
        from scripts.check_results import write_picks_detail

        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "picks_detail.jsonl"
            write_picks_detail(
                jsonl_path=jsonl,
                rows=[{
                    "draw_date": "2026-04-21",
                    "game": "joker",
                    "ticket_id": "tkt-1",
                    "strategy": "core_share",
                    "best_main_match": 3,
                    "variants": [{"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11}],
                    "side_game_number": "NP07",
                    "winning_side_game": "NP14",
                    "side_game_match": 0,
                    "side_game_digits_matched": 0,
                    "cost_ron": 17.5,
                }],
            )
            self.assertTrue(jsonl.exists())
            text = jsonl.read_text().strip()
            self.assertIn("tkt-1", text)
            self.assertIn("core_share", text)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_check_results.TestCheckResultsWritesJSONL -v
```
Expected: `ImportError: cannot import name 'write_picks_detail'`.

- [ ] **Step 3: Implement**

Add to `scripts/check_results.py`:
```python
from shared.picks_detail_history import append_detail_rows
from shared.telegram_formatter import format_check_results as _fmt_check_results


def write_picks_detail(jsonl_path: Path, rows: list[dict]) -> None:
    append_detail_rows(jsonl_path, rows)
```

In `main()`, after the per-ticket scoring loop, collect `detail_rows` (one dict per ticket per game with the fields from the test) and call `write_picks_detail(Path(os.environ.get("PICKS_DETAIL_PATH", "data/results/picks_detail.jsonl")), detail_rows)`.

Replace the content of `build_game_message` with a call to `_fmt_check_results` passing the new per-ticket outcomes. The helper already exists in Plan D Task 1.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_check_results -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/check_results.py tests/test_check_results.py
git commit -m "feat(checker): append picks_detail.jsonl and use ticket-aware formatter"
```

---

## Task 4: Generate-picks workflow yaml

**Files:**
- Modify: `.github/workflows/generate-picks.yml`

Replace the manual bash loop that reads `picks/joker.txt`, `picks/joker_mix1.txt`, etc. with a single Python call that emits a messages.txt (one message per line, separated by `\x00`).

- [ ] **Step 1: Read the current yaml**

Check the block between `# Send picks to Telegram` and end of file. That's the logic being replaced.

- [ ] **Step 2: Write the failing test**

Create `tests/test_generate_picks_workflow_v2.py`:
```python
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGeneratePicksWorkflowMessages(unittest.TestCase):
    def test_emit_messages_writes_one_per_line_null_separated(self):
        # Build a tickets.json fixture and run the workflow helper.
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tickets.json"
            p.write_text(json.dumps({
                "generated_at": "x",
                "budget_ron": 40.0,
                "total_cost_ron": 40.0,
                "allocation": {"joker": 1, "loto_649": 0, "loto_540": 1},
                "tickets": [
                    {
                        "game": "joker",
                        "variants": [
                            {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                            {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                        ],
                        "side_game_number": "NP07",
                        "strategy": "core_share",
                        "cost_ron": 17.5,
                    }
                ],
            }))
            # workflow_messages.py is invoked by the yaml; run it directly
            out = subprocess.check_output(
                [
                    sys.executable, "scripts/workflow_messages.py",
                    "--tickets-json", str(p),
                    "--date", "2026-04-21",
                    "--format", "picks",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
                text=True,
            )
            # Summary + 1 ticket = 2 messages, NUL-separated
            self.assertEqual(out.count("\x00"), 2)
            self.assertIn("Lottery Picks", out)
            self.assertIn("core_share", out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Create `scripts/workflow_messages.py`**

Create `scripts/workflow_messages.py`:
```python
"""Emit Telegram-Markdown messages separated by NUL bytes.

Used by .github/workflows/*.yml; consumed by a bash while-read loop
that splits on NUL and sends each to Telegram.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared.telegram_formatter import emit_messages_for_workflow


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickets-json", type=Path, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--format", choices=["picks"], default="picks")
    args = p.parse_args()

    msgs = emit_messages_for_workflow(args.tickets_json, args.date)
    sys.stdout.write("\x00".join(msgs))
    sys.stdout.write("\x00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run the test:
```bash
PYTHONPATH=src python -m unittest tests.test_generate_picks_workflow_v2 -v
```
Expected: passes.

- [ ] **Step 4: Edit `.github/workflows/generate-picks.yml`**

Replace the entire `Send picks to Telegram` step with:

```yaml
      - name: Send picks to Telegram
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          DATE=$(date +"%Y-%m-%d")
          if [ ! -f picks/tickets.json ]; then
            echo "No tickets.json emitted (EV gate skip or allocator empty). Posting skip notice."
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
              -d chat_id="${TELEGRAM_CHAT_ID}" \
              -d text="🎰 Lottery Picks - ${DATE}: skipped (no tickets emitted)" \
              -d parse_mode=Markdown > /dev/null
            exit 0
          fi

          PYTHONPATH=src python scripts/workflow_messages.py \
            --tickets-json picks/tickets.json \
            --date "${DATE}" \
            --format picks \
            | while IFS= read -r -d '' msg; do
              curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d chat_id="${TELEGRAM_CHAT_ID}" \
                --data-urlencode "text=${msg}" \
                -d parse_mode=Markdown > /dev/null
            done
```

The `--data-urlencode` is important — prior yaml used `-d text=...` which mangles special characters in ticket numbers and asterisks. Switch every curl that posts user content to `--data-urlencode`.

Validate yaml syntax locally:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/generate-picks.yml'))"
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/generate-picks.yml scripts/workflow_messages.py tests/test_generate_picks_workflow_v2.py
git commit -m "feat(workflow): ticket-aware Telegram messages via workflow_messages.py"
```

---

## Task 5: Check-results workflow yaml

**Files:**
- Modify: `.github/workflows/check-results.yml`

Replace the bash that reads `results/loto540.txt`, `results/loto649.txt`, `results/joker.txt` with a single call to `scripts/workflow_messages.py --format check-results` that emits per-game + per-ticket messages.

- [ ] **Step 1: Extend `scripts/workflow_messages.py`**

Add a new `--format check-results` mode. It reads `results/*.txt` — wait, those are now produced by `check_results.py` already using `format_check_results`. Keep the bash loop simple: just cat those files and curl each one.

Actually simpler: `scripts/check_results.py` already writes `results/{game}.txt` via the new formatter (Plan D Task 3). So the yaml only needs to loop those. Change the existing loop to use `--data-urlencode`:

```yaml
      - name: Send results to Telegram
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          for file in results/loto540.txt results/loto649.txt results/joker.txt; do
            if [ -f "$file" ]; then
              msg="$(cat "$file")"
              curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d chat_id="${TELEGRAM_CHAT_ID}" \
                --data-urlencode "text=${msg}" \
                -d parse_mode=Markdown > /dev/null
            fi
          done

          if [ -f "results/comparison.txt" ]; then
            msg="$(cat results/comparison.txt)"
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
              -d chat_id="${TELEGRAM_CHAT_ID}" \
              --data-urlencode "text=${msg}" \
              -d parse_mode=Markdown > /dev/null
          fi
```

- [ ] **Step 2: Ensure history commits include jsonl**

In the `Commit results history` step of the same yaml, stage both files:
```yaml
      - name: Commit results history
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/results/history.csv data/results/picks_detail.jsonl
          git diff --staged --quiet || git commit -m "chore: update results history"
          git push
```

- [ ] **Step 3: Validate yaml**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/check-results.yml'))"
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/check-results.yml
git commit -m "feat(workflow): commit picks_detail.jsonl and use url-encoded Telegram posts"
```

---

## Task 6: Delete legacy `.txt` shim from generator

**Files:**
- Modify: `scripts/generate_recommended_picks.py`
- Modify: `tests/test_generate_recommended_picks.py`

Plan C's orchestrator kept the per-game `.txt` files for backward compat. Plan D no longer needs them (Telegram reads `tickets.json` directly via `workflow_messages.py`; `check_results.py` reads `tickets.json` via `parse_tickets_json`). Delete.

- [ ] **Step 1: Adjust the test that required legacy .txt**

In `tests/test_generate_recommended_picks.py` find `test_legacy_txt_files_still_emitted` from Plan C and invert it:
```python
    def test_legacy_txt_files_no_longer_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "40", "--seed", "42",
                    "--output-dir", str(out), "--strategy", "independent",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            self.assertEqual(list(out.glob("*.txt")), [])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks.TestGenerateTicketsJSON.test_legacy_txt_files_no_longer_emitted -v
```
Expected: fails (txt files still emitted).

- [ ] **Step 3: Delete the shim**

In `scripts/generate_recommended_picks.py`, remove the loop that writes the per-game `.txt` files. The file should only emit `tickets.json` (and still call `persist_generation_run`).

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks -v
PYTHONPATH=src python -m unittest -v 2>&1 | tail -5
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_recommended_picks.py tests/test_generate_recommended_picks.py
git commit -m "refactor(orchestrator): drop legacy .txt shim — tickets.json is the only artifact"
```

---

## Task 7: Side-game fetch retry in check-results

**Files:**
- Modify: `scripts/check_results.py`

`fetch_results_with_retry` currently considers the fetch successful if any game's main draws parsed. With Plan A's parsers, the same HTML pages carry side-game numbers — but only sometimes (loto.ro may publish mains first and side games a few minutes later). Add a *secondary* retry that waits for `noroc_plus`/`noroc`/`super_noroc` to appear, but don't block indefinitely.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_results.py`:
```python
class TestSideGameRetry(unittest.TestCase):
    def test_fetch_retry_recognizes_side_game_as_success_condition(self):
        # Unit test: is_side_game_ready detects presence of non-None side field.
        from scripts.check_results import is_side_game_ready

        class FakeDraw:
            def __init__(self, side=None):
                self.noroc = side
                self.noroc_plus = side
                self.super_noroc = side

        self.assertTrue(is_side_game_ready(FakeDraw(side="NP07"), attr="noroc_plus"))
        self.assertFalse(is_side_game_ready(FakeDraw(side=None), attr="noroc_plus"))
```

- [ ] **Step 2: Implement**

Add to `scripts/check_results.py`:
```python
def is_side_game_ready(draw, attr: str) -> bool:
    return getattr(draw, attr, None) is not None
```

Wire it into `fetch_results_with_retry` as an *informational* check: after main draws are confirmed fresh, if any of `joker_draws[-1].noroc_plus`, `loto_draws[-1].noroc`, `loto540_draws[-1].super_noroc` is None, wait one more retry cycle (but don't fail the workflow). Log clearly:
```python
missing = []
if joker_draws and not is_side_game_ready(joker_draws[-1], "noroc_plus"):
    missing.append("joker/noroc_plus")
# ... same for other two
if missing and attempt < max_retries - 1:
    log(f"Side games not yet published: {missing}. Waiting {delay_minutes} more minute(s)...")
    time.sleep(delay_minutes * 60)
    continue
```

- [ ] **Step 3: Commit**

```bash
git add scripts/check_results.py tests/test_check_results.py
git commit -m "feat(checker): retry fetch while side-game numbers are missing"
```

---

## Task 8: End-to-end dry run

- [ ] **Step 1: Regenerate picks + check against synthetic winning set**

```bash
rm -rf /tmp/pd-e2e && mkdir -p /tmp/pd-e2e
PYTHONPATH=src python scripts/generate_recommended_picks.py \
  --budget 40 --seed 42 --strategy core_share --output-dir /tmp/pd-e2e
echo "--- tickets.json ---"
cat /tmp/pd-e2e/tickets.json | python -m json.tool
echo ""
echo "--- simulate telegram ---"
PYTHONPATH=src python scripts/workflow_messages.py \
  --tickets-json /tmp/pd-e2e/tickets.json --date 2026-04-21 --format picks \
  | tr '\x00' '\n' | head -40
```

Expected: JSON is well-formed; the "simulate telegram" output shows a summary message followed by one ticket-block per physical ticket, each with proper markdown.

- [ ] **Step 2: Full suite**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -10
```
Expected: OK.

- [ ] **Step 3: Yaml validation**

```bash
python -c "
import yaml, glob
for f in glob.glob('.github/workflows/*.yml'):
    yaml.safe_load(open(f))
    print(f'{f}: ok')
"
```

- [ ] **Step 4: Push + PR**

```bash
git push -u origin feature/jackpot-d-workflow-telegram
gh pr create --fill --base main --title "Jackpot redesign D: workflows + Telegram"
```

PR description must include:
- Screenshot or captured text of the new Telegram messages (run workflow_dispatch against a fixture if needed).
- Confirmation that `picks/budget_bank.json` persists across runs via artifact upload (add to `upload-artifact` in generate-picks.yml if not already there).
- Note any visible changes a user would see at draw time: more messages per Telegram run, ticket-grouped layout with Noroc numbers.

---

## Follow-up (outside Plan D scope)

1. **Live ticket pricing monitor.** If loto.ro changes variant prices or side-game stakes, downstream math breaks silently. Add a follow-up task to scrape loto.ro's pricing page monthly and open an issue on mismatch.
2. **Budget-bank artifact upload.** Verify that `picks/budget_bank.json` is included in the `upload-artifact` path in generate-picks.yml (it should be if the path is `picks/`). If not, explicitly add.
3. **Separate the "Commit results history" step failure mode.** If the bot has nothing new to commit (because the workflow was skipped), the `git commit` will return 1 → but the `|| true` pattern via `git diff --staged --quiet` handles that; verify it holds with the new `picks_detail.jsonl` path.
4. **Metrics dashboard.** `picks_detail.jsonl` enables a cheap Grafana panel showing rolling P(best_main_match ≥ 4) — worth building once there's 30+ draws of data.

---

## Risks & open questions

1. **Telegram Markdown escaping.** User-controlled content (the side-game numbers or ticket numbers) is numeric and safe. Strategy names can contain colons (`wheel:8`) which Telegram Markdown treats as normal text — no escape needed. If future strategies use `*`, `_`, or `` ` `` in their names, the formatter must escape them.
2. **4096-char message limit.** Each ticket message is roughly 200 chars; safely under. If Plan C ever allows >10 variants per ticket or multiple mixes per ticket the limit could bite — split into multiple messages then.
3. **Workflow secrets.** Same `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` as today. No new secrets introduced.
4. **Side-game scraping accuracy.** If Plan A's parser for one game is wrong, the checker will silently report "side_game_digits_matched: 0" forever. The `is_side_game_ready` retry is a canary but not foolproof — monitor the first two weeks of live runs manually.
5. **Removing the legacy `.txt` shim breaks any out-of-tree consumer** that may be reading those files. Check for external consumers before merging; the repo has only the workflow yaml and `check_results.py` reading them, both updated in this plan.
6. **`workflow_messages.py` tests invoke a subprocess.** These are slower than pure unit tests but catch real shell integration issues. They run under `unittest` without pytest-specific plugins.
