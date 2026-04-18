"""Test-time configuration.

We want tests to verify *our code* (orchestrator, allocator, builders,
formatters), not the external ML training cost. Real calls to
`generate_blended_picks` run walk-forward scoring of 10+ strategies
(including LSTM/TCN/Transformer/NormalizingFlow/RL when torch is installed).
On CI a single call is minutes; tests that use `TicketBuilder` subclasses
trigger dozens of calls.

So:
1. Disable torch-backed strategies in `shared.ensemble_blend`. Torch-specific
   tests (`test_lstm_strategy.py`, etc.) exercise the models directly and
   don't go through `ensemble_blend` — they remain unaffected.
2. Replace `generate_blended_picks` with a lightweight random-sample stub for
   every test except `test_ensemble_blend.py`, which is the one file that
   explicitly tests the blending logic end-to-end.
"""
import importlib
import random

import pytest

import shared.ensemble_blend as _eb

_eb._TORCH_AVAILABLE = False


def _fast_blended_picks(
    config,
    draws,
    count,
    rng=None,
    half_life=None,
    half_life_mode=None,
    draw_dates=None,
):
    """Return `count` unique random picks matching the game's pool + size."""
    rng = rng or random.Random()
    pool = list(config.pool_range)
    k = config.numbers_to_pick
    picks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count * 50):
        if len(picks) >= count:
            break
        pick = tuple(sorted(rng.sample(pool, k)))
        if pick in seen:
            continue
        seen.add(pick)
        picks.append(list(pick))
    while len(picks) < count:
        picks.append(sorted(rng.sample(pool, k)))
    return picks


_PATCH_TARGETS = ("shared.ensemble_blend", "shared.ticket_builders")


@pytest.fixture(autouse=True)
def _stub_generate_blended_picks(request, monkeypatch):
    """Patch generate_blended_picks to a fast stub for every test except the
    one that exercises the blending logic directly.

    Callers that import the function via `from shared.ensemble_blend import
    generate_blended_picks` capture the original reference at module load
    time, so the stub must also be installed on every known import site.
    """
    if "test_ensemble_blend" in request.node.nodeid:
        return
    for mod_name in _PATCH_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "generate_blended_picks"):
            monkeypatch.setattr(
                f"{mod_name}.generate_blended_picks",
                _fast_blended_picks,
            )
