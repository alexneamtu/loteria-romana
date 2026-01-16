def is_joker_prize(main_matches: int, joker_match: bool) -> bool:
    if main_matches >= 3:
        return True
    if joker_match and main_matches >= 1:
        return True
    return False
