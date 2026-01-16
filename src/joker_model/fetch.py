from pathlib import Path
import urllib.request

from .parser import parse_joker_results
from .storage import load_draws, append_draws


def update_dataset(url: str, cache_path: Path, csv_path: Path, fetcher=None) -> int:
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        fetcher = fetcher or (lambda u: urllib.request.urlopen(u).read().decode("utf-8", "ignore"))
        html = fetcher(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")

    draws = parse_joker_results(html)
    existing = {d.date for d in load_draws(csv_path)}
    new_draws = [d for d in draws if d.date not in existing]
    return append_draws(csv_path, new_draws)
