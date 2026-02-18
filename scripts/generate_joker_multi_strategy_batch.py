"""Generate a deduplicated Joker batch from multiple strategies.

This script runs several Joker strategies, merges all generated
main-number candidates, then applies set optimization:
- dedupe + diversity optimization
- anti-crowding selection
- Joker-number max coverage assignment
"""

import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from shared.joker_set_optimizer import (
    assign_max_coverage_jokers,
    optimize_main_ticket_set,
)


DEFAULT_STRATEGIES = [
    "blend",
    "bayesian",
    "cooccurrence",
    "smart",
    "genetic",
]


def parse_joker_lines(text: str) -> list[tuple[list[int], int]]:
    """Parse `generate_joker_picks.py` output lines."""
    parsed: list[tuple[list[int], int]] = []
    pattern = re.compile(r"^\d+\.\s*([\d,\s]+)\s+\+\s*J(\d+)\s*$")
    for raw in text.splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        main = [int(value.strip()) for value in match.group(1).split(",")]
        joker = int(match.group(2))
        parsed.append((main, joker))
    return parsed


def build_final_lines(
    main_candidates: list[list[int]],
    count: int,
    rng: random.Random,
) -> list[tuple[list[int], int]]:
    """Build final Joker lines from candidate main-number sets."""
    selected_main = optimize_main_ticket_set(
        candidates=main_candidates,
        select_count=count,
        pool_size=45,
        numbers_to_pick=5,
        rng=rng,
        anti_crowding_weight=0.35,
    )
    jokers = assign_max_coverage_jokers(
        count=len(selected_main),
        rng=rng,
        joker_pool=20,
    )
    return list(zip(selected_main, jokers))


def _run_strategy(
    strategy: str,
    count: int,
    seed: int | None,
    half_life: float | None,
    half_life_mode: str | None,
) -> list[tuple[list[int], int]]:
    cmd = [
        sys.executable,
        "scripts/generate_joker_picks.py",
        "--strategy",
        strategy,
        "--count",
        str(count),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if half_life is not None:
        cmd.extend(["--half-life", str(half_life)])
    if half_life_mode is not None:
        cmd.extend(["--half-life-mode", half_life_mode])

    env = os.environ.copy()
    if not env.get("PYTHONPATH"):
        env["PYTHONPATH"] = "src"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Strategy '{strategy}' failed ({result.returncode}): {result.stderr.strip()}"
        )
    return parse_joker_lines(result.stdout)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate merged Joker picks across multiple strategies",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Final number of deduplicated lines to output",
    )
    parser.add_argument(
        "--per-strategy",
        type=int,
        default=5,
        help="Lines to request from each strategy",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategy list",
    )
    parser.add_argument("--seed", type=int, help="Base seed for reproducibility")
    parser.add_argument("--half-life", type=float, help="Recency half-life value")
    parser.add_argument(
        "--half-life-mode",
        choices=["draws", "days"],
        help="Recency half-life mode",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Optional output file path",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategies:
        raise SystemExit("At least one strategy is required")

    base_rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    main_candidates: list[list[int]] = []

    for idx, strategy in enumerate(strategies):
        strategy_seed = None
        if args.seed is not None:
            strategy_seed = args.seed + (idx + 1) * 1009
        lines = _run_strategy(
            strategy=strategy,
            count=args.per_strategy,
            seed=strategy_seed,
            half_life=args.half_life,
            half_life_mode=args.half_life_mode,
        )
        main_candidates.extend(main for main, _ in lines)

    if not main_candidates:
        raise SystemExit("No candidates produced by selected strategies")

    final_lines = build_final_lines(
        main_candidates=main_candidates,
        count=args.count,
        rng=base_rng,
    )

    rendered = []
    for idx, (main, joker) in enumerate(final_lines, 1):
        rendered.append(f"{idx}. {', '.join(str(n) for n in main)} + J{joker}")

    output = "\n".join(rendered)
    print(output)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
