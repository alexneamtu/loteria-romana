import random
from pathlib import Path

from joker_model.fetch import update_dataset
from joker_model.storage import load_draws
from joker_model.backtest import pick_best_strategy
from joker_model.strategies import build_frequency, generate_random_lines, generate_frequency_lines
from joker_model.neural import generate_neural_lines


def main():
    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
    cache_path = Path("data/raw/joker_results.html")
    csv_path = Path("data/clean/joker_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    rng = random.SystemRandom()
    draw_tuples = [(d.main_numbers, d.joker) for d in draws]
    best = pick_best_strategy(draw_tuples)

    if best == "neural":
        lines = generate_neural_lines(draw_tuples, 7, rng=rng)
    elif best == "frequency":
        freq = build_frequency(draw_tuples)
        lines = generate_frequency_lines(7, freq, rng=rng)
    else:
        lines = generate_random_lines(7, rng=rng)

    for idx, (main, joker) in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)} + J{joker}")


if __name__ == "__main__":
    main()
