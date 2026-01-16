def resolve_seed(cli_seed, env_seed=None):
    if cli_seed is not None:
        return cli_seed
    if not env_seed:
        return None
    try:
        return int(env_seed)
    except ValueError as exc:
        raise ValueError("JOKER_SEED must be an integer") from exc
