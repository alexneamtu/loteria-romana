"""Current-jackpot scraper for loto.ro.

The homepage renders three "REPORT" headings (Joker, Loto 6/49, Loto 5/40)
followed by the current jackpot amount in European notation, e.g.
"65.230.704,97" for 65,230,704.97 RON. We scan for the label and take the
first currency-shaped number within ~3500 chars downstream.

The scraper is best-effort: any failure (network, HTML changed, label not
found) yields None for the affected game. The orchestrator treats None as
"no jackpot signal" — the EV gate skips that game rather than crashing.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

LOTO_HOMEPAGE_URL = "https://www.loto.ro"
FETCH_TIMEOUT_SECONDS = 30

_LABEL_PATTERNS: dict[str, str] = {
    "joker": r"REPORT\s+JOKER",
    "loto_649": r"REPORT\s+LOTO\s+6/49",
    "loto_540": r"REPORT\s+LOTO\s+5/40",
}

# Matches "65.230.704,97" or "619.353,00" — 1-3 digits, one or more thousands
# groups, mandatory decimal group with 1-2 digits.
_AMOUNT_PATTERN = r"([0-9]{1,3}(?:\.[0-9]{3})+,[0-9]{1,2})"


def parse_jackpots(html: str) -> dict[str, float | None]:
    """Extract current jackpots from loto.ro homepage HTML."""
    results: dict[str, float | None] = {}
    for game, label in _LABEL_PATTERNS.items():
        pattern = re.compile(label + r".{0,3500}?" + _AMOUNT_PATTERN, re.IGNORECASE | re.DOTALL)
        match = pattern.search(html)
        if match:
            raw = match.group(1)
            results[game] = _european_to_float(raw)
        else:
            results[game] = None
    return results


def _european_to_float(value: str) -> float:
    return float(value.replace(".", "").replace(",", "."))


def fetch_jackpots(
    url: str = LOTO_HOMEPAGE_URL,
    timeout: float = FETCH_TIMEOUT_SECONDS,
) -> dict[str, float | None]:
    """Fetch current jackpots from loto.ro. Returns None per game on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; loteria-romana-bot)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"joker": None, "loto_649": None, "loto_540": None}
    return parse_jackpots(html)
