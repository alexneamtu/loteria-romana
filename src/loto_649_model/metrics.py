def is_loto_649_prize(main_matches: int, noroc_match: bool) -> bool:
    if main_matches >= 3:
        return True
    if noroc_match:
        return True
    return False
