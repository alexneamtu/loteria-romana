def is_loto_649_prize(main_matches: int, noroc_match: bool, include_noroc: bool = True) -> bool:
    if main_matches >= 3:
        return True
    if include_noroc and noroc_match:
        return True
    return False
