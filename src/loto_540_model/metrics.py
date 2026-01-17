def is_loto_540_prize(main_matches: int) -> bool:
    """Check if the pick wins a prize.

    In Loto 5/40:
    - 6 numbers are drawn from 1-40
    - Players pick 5 numbers
    - Win by matching 4 or 5 of the 6 drawn numbers
    """
    return main_matches >= 4
