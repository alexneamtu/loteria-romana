# Generate Picks Workflow Inputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the `generate-picks.yml` workflow to accept per-game line counts with defaults (Joker=7, 6/49=0, 5/40=0) while keeping `check-results.yml` working for any input combination.

**Architecture:** Add `workflow_dispatch` inputs and a line-count resolution step in `.github/workflows/generate-picks.yml`. Gate each generation step on resolved counts and only write picks files when a game is enabled. Build the Telegram message conditionally so skipped games do not add empty sections. Add a small `unittest` that asserts the workflow contains the new inputs, defaults, and uses resolved counts.

**Tech Stack:** GitHub Actions YAML, Python 3 stdlib `unittest`.

### Task 1: Add failing workflow tests

**Files:**
- Create: `tests/test_generate_picks_workflow.py`

**Step 1: Write the failing tests**

```python
from pathlib import Path
import unittest


class TestGeneratePicksWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.workflow = root / ".github/workflows/generate-picks.yml"
        self.text = self.workflow.read_text()

    def test_inputs_present(self):
        for name in ("joker_lines", "loto649_lines", "loto540_lines"):
            self.assertIn(name, self.text)

    def test_default_values_present(self):
        self.assertIn("default: '7'", self.text)
        self.assertGreaterEqual(self.text.count("default: '0'"), 2)

    def test_resolved_counts_used(self):
        self.assertIn("steps.counts.outputs.joker_lines", self.text)
        self.assertIn("steps.counts.outputs.loto649_lines", self.text)
        self.assertIn("steps.counts.outputs.loto540_lines", self.text)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_generate_picks_workflow.py -v`  
Expected: FAIL because the workflow lacks the new inputs and counts step.

### Task 2: Update generate-picks workflow to use inputs and defaults

**Files:**
- Modify: `.github/workflows/generate-picks.yml`

**Step 1: Add workflow inputs with defaults**

Add under `workflow_dispatch`:

```yaml
  workflow_dispatch:
    inputs:
      joker_lines:
        description: "Number of Joker lines to generate"
        required: false
        default: "7"
        type: string
      loto649_lines:
        description: "Number of Loto 6/49 lines to generate"
        required: false
        default: "0"
        type: string
      loto540_lines:
        description: "Number of Loto 5/40 lines to generate"
        required: false
        default: "0"
        type: string
```

**Step 2: Add a line-count resolution step**

Insert before generation steps:

```yaml
      - name: Resolve line counts
        id: counts
        run: |
          normalize() {
            local input="$1"
            local fallback="$2"
            if [ -z "$input" ]; then
              echo "$fallback"
              return
            fi
            if ! [[ "$input" =~ ^[0-9]+$ ]]; then
              echo "$fallback"
              return
            fi
            echo "$input"
          }

          joker=$(normalize "${{ github.event.inputs.joker_lines }}" "7")
          loto649=$(normalize "${{ github.event.inputs.loto649_lines }}" "0")
          loto540=$(normalize "${{ github.event.inputs.loto540_lines }}" "0")

          echo "joker_lines=$joker" >> "$GITHUB_OUTPUT"
          echo "loto649_lines=$loto649" >> "$GITHUB_OUTPUT"
          echo "loto540_lines=$loto540" >> "$GITHUB_OUTPUT"
```

**Step 3: Gate generation steps on resolved counts**

Update each game step to:

```yaml
      - name: Generate Joker picks
        if: ${{ steps.counts.outputs.joker_lines != '0' }}
        id: joker
        run: |
          {
            echo 'PICKS<<EOF'
            PYTHONPATH=src python scripts/generate_joker_picks.py -s smart -n ${{ steps.counts.outputs.joker_lines }} --seed ${{ github.run_id }}
            echo 'EOF'
          } >> $GITHUB_OUTPUT
```

Repeat for Loto 6/49 and 5/40 using their respective counts.

**Step 4: Save picks only for enabled games**

Update the save step:

```bash
mkdir -p picks
if [ -n "${{ steps.joker.outputs.PICKS }}" ]; then
  echo "${{ steps.joker.outputs.PICKS }}" > picks/joker.txt
fi
if [ -n "${{ steps.loto649.outputs.PICKS }}" ]; then
  echo "${{ steps.loto649.outputs.PICKS }}" > picks/loto649.txt
fi
if [ -n "${{ steps.loto540.outputs.PICKS }}" ]; then
  echo "${{ steps.loto540.outputs.PICKS }}" > picks/loto540.txt
fi
date +"%Y-%m-%d" > picks/date.txt
```

**Step 5: Build Telegram message conditionally**

Replace the static message with a conditional builder, e.g.:

```bash
DATE=$(date +"%Y-%m-%d")
MESSAGE="🎰 *Lottery Picks - ${DATE}*"

if [ -n "${{ steps.joker.outputs.PICKS }}" ]; then
  MESSAGE="${MESSAGE}\n\n🃏 *JOKER*\n\`\`\`\n${{ steps.joker.outputs.PICKS }}\n\`\`\`"
fi
if [ -n "${{ steps.loto649.outputs.PICKS }}" ]; then
  MESSAGE="${MESSAGE}\n\n🎱 *LOTO 6/49*\n\`\`\`\n${{ steps.loto649.outputs.PICKS }}\n\`\`\`"
fi
if [ -n "${{ steps.loto540.outputs.PICKS }}" ]; then
  MESSAGE="${MESSAGE}\n\n🎯 *LOTO 5/40*\n\`\`\`\n${{ steps.loto540.outputs.PICKS }}\n\`\`\`"
fi

if [ -z "${{ steps.joker.outputs.PICKS }}${{ steps.loto649.outputs.PICKS }}${{ steps.loto540.outputs.PICKS }}" ]; then
  MESSAGE="${MESSAGE}\n\n_No picks generated (all line counts set to 0)._"
fi

MESSAGE="${MESSAGE}\n\n_Generated with smart strategy_\n_Results will be checked tomorrow morning_"
```

**Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests/test_generate_picks_workflow.py -v`  
Expected: PASS

**Step 7: Commit**

```bash
git add .github/workflows/generate-picks.yml tests/test_generate_picks_workflow.py
git commit -m "feat: add line-count inputs to generate-picks workflow"
```

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-17-generate-picks-inputs.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration  
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
