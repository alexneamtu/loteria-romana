DEFAULT_HALF_LIFE = 50.0


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


def draw_weights(draw_count: int, half_life: float) -> list[float]:
    if draw_count <= 0:
        return []
    return [0.5 ** ((draw_count - 1 - idx) / half_life) for idx in range(draw_count)]
