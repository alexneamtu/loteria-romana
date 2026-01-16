import argparse
import os
import random
from pathlib import Path

from loto_649_model.fetch import update_dataset
from loto_649_model.storage import load_draws
from loto_649_model.picks import generate_picks
from loto_649_model.seed import resolve_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    args = parser.parse_args()

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html"
    cache_path = Path("data/raw/loto_649_results.html")
    csv_path = Path("data/clean/loto_649_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    try:
        seed = resolve_seed(args.seed, os.getenv("LOTO_649_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    draw_tuples = [(d.main_numbers, d.noroc) for d in draws]
    lines = generate_picks(draw_tuples, count=2, rng=rng)

    for idx, (main, noroc) in enumerate(lines, 1):
        noroc_str = f"{noroc:07d}"
        print(f"{idx}. {', '.join(str(n) for n in main)} + N{noroc_str}")


if __name__ == "__main__":
    main()
