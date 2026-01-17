def is_loto_540_prize(
    main_matches: int,
    super_noroc_match: bool,
    include_super_noroc: bool = True
) -> bool:
    """Check if the pick wins a prize.

    In Loto 5/40:
    - 6 numbers are drawn from 1-40
    - Players pick 5 numbers
    - Win by matching 4 or 5 of the 6 drawn numbers
    - Super Noroc is a separate bonus game
    """
    # Main game: match 4+ of 6 drawn numbers with your 5 picks
    if main_matches >= 4:
        return True
    # Super Noroc bonus (if playing)
    if include_super_noroc and super_noroc_match:
        return True
    return False
