import argparse
import os
import random
from pathlib import Path

from joker_model.fetch import update_dataset
from joker_model.storage import load_draws
from joker_model.picks import generate_picks
from joker_model.seed import resolve_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    args = parser.parse_args()

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
    cache_path = Path("data/raw/joker_results.html")
    csv_path = Path("data/clean/joker_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    try:
        seed = resolve_seed(args.seed, os.getenv("JOKER_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    draw_tuples = [(d.main_numbers, d.joker) for d in draws]
    lines = generate_picks(draw_tuples, count=2, rng=rng)

    for idx, (main, joker) in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)} + J{joker}")


if __name__ == "__main__":
    main()
