from datetime import date


DEFAULT_HALF_LIFE = 50.0
DEFAULT_HALF_LIFE_MODE = "draws"
VALID_HALF_LIFE_MODES = {"draws", "days"}


def resolve_half_life(cli_value, env_value, default: float = DEFAULT_HALF_LIFE) -> float:
    if cli_value is not None:
        raw = cli_value
    elif env_value is not None:
        raw = env_value
    else:
        raw = default

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("RECENCY_HALF_LIFE must be a positive number") from exc

    if value <= 0:
        raise ValueError("RECENCY_HALF_LIFE must be a positive number")

    return value


def resolve_half_life_mode(
    cli_value, env_value, default: str = DEFAULT_HALF_LIFE_MODE
) -> str:
    if cli_value is not None:
        mode = cli_value
    elif env_value is not None:
        mode = env_value
    else:
        mode = default

    if mode not in VALID_HALF_LIFE_MODES:
        raise ValueError("RECENCY_HALF_LIFE_MODE must be 'draws' or 'days'")

    return mode


def _parse_draw_dates(draw_dates: list[str]) -> list[date]:
    try:
        return [date.fromisoformat(value) for value in draw_dates]
    except (TypeError, ValueError) as exc:
        raise ValueError("draw_dates must be ISO YYYY-MM-DD strings") from exc


def draw_weights(
    draw_count: int,
    half_life: float,
    draw_dates: list[str] | None = None,
    mode: str = DEFAULT_HALF_LIFE_MODE,
) -> list[float]:
    if draw_count <= 0:
        return []
    if mode == "draws":
        return [
            0.5 ** ((draw_count - 1 - idx) / half_life)
            for idx in range(draw_count)
        ]

    if not draw_dates or len(draw_dates) != draw_count:
        raise ValueError("draw_dates must be provided for RECENCY_HALF_LIFE_MODE=days")

    parsed = _parse_draw_dates(draw_dates)
    latest = parsed[-1]
    return [0.5 ** (((latest - value).days) / half_life) for value in parsed]
