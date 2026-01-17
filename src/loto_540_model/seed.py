def resolve_seed(cli_seed: int | None, env_seed: str | None = None) -> int | None:
    """Return seed from CLI arg, env var LOTO_540_SEED, or None."""
    if cli_seed is not None:
        return cli_seed
    if env_seed:
        try:
            return int(env_seed)
        except ValueError:
            raise ValueError(f"Invalid LOTO_540_SEED value: {env_seed}")
    return None
