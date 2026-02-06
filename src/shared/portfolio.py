"""Portfolio optimization for lottery ticket selection.

Treats ticket selection as a mean-variance portfolio problem.
Each ticket is characterized by its number coverage. The goal
is to select a subset of tickets that maximizes coverage diversity
and minimizes redundancy (overlapping numbers).

Uses a greedy algorithm that iteratively selects the ticket with
the lowest covariance to the already-selected set.
"""


def compute_ticket_covariance(
    tickets: list[list[int]],
    pool_size: int,
) -> list[list[float]]:
    """Compute covariance matrix between tickets based on number overlap.

    Each ticket is represented as a binary vector over the number pool.
    Covariance measures how much two tickets share the same numbers.

    Args:
        tickets: List of tickets (each a sorted list of numbers).
        pool_size: Total numbers in the pool.

    Returns:
        n x n covariance matrix where n = len(tickets).
    """
    n = len(tickets)
    if n == 0:
        return []

    # Convert tickets to binary vectors
    vectors = []
    for ticket in tickets:
        vec = [0.0] * pool_size
        for num in ticket:
            if 1 <= num <= pool_size:
                vec[num - 1] = 1.0
        vectors.append(vec)

    # Compute mean vector
    mean_vec = [0.0] * pool_size
    for vec in vectors:
        for j in range(pool_size):
            mean_vec[j] += vec[j]
    for j in range(pool_size):
        mean_vec[j] /= n

    # Compute covariance matrix
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(
                (vectors[i][k] - mean_vec[k]) * (vectors[j][k] - mean_vec[k])
                for k in range(pool_size)
            ) / pool_size
            cov[i][j] = val
            cov[j][i] = val

    return cov


def _overlap_fraction(ticket_a: list[int], ticket_b: list[int], pool_size: int) -> float:
    """Compute overlap fraction between two tickets."""
    set_a = set(ticket_a)
    set_b = set(ticket_b)
    overlap = len(set_a & set_b)
    max_possible = min(len(ticket_a), len(ticket_b))
    return overlap / max_possible if max_possible > 0 else 0.0


def optimize_ticket_portfolio(
    candidates: list[list[int]],
    select_count: int,
    pool_size: int,
) -> list[list[int]]:
    """Select a diverse subset of tickets minimizing redundancy.

    Uses greedy selection: at each step, add the candidate with
    the lowest average overlap to the already-selected tickets.

    Args:
        candidates: Pool of candidate tickets to choose from.
        select_count: Number of tickets to select.
        pool_size: Total numbers in the pool.

    Returns:
        Selected tickets optimized for diversity.
    """
    if not candidates:
        return []

    if len(candidates) <= select_count:
        return list(candidates)

    # Start with the ticket covering the most unique numbers
    selected_indices: list[int] = []
    selected_sets: list[set[int]] = []

    # Pick first ticket (widest spread)
    best_first = 0
    best_score = -1.0
    for i, ticket in enumerate(candidates):
        if len(ticket) >= 2:
            spread = ticket[-1] - ticket[0]
        else:
            spread = 0
        if spread > best_score:
            best_score = spread
            best_first = i

    selected_indices.append(best_first)
    selected_sets.append(set(candidates[best_first]))

    # Greedily add tickets with minimum overlap
    while len(selected_indices) < select_count:
        best_idx = -1
        best_avg_overlap = float("inf")

        for i, ticket in enumerate(candidates):
            if i in selected_indices:
                continue

            ticket_set = set(ticket)
            total_overlap = 0.0
            for s_set in selected_sets:
                overlap = len(ticket_set & s_set)
                total_overlap += overlap

            avg_overlap = total_overlap / len(selected_sets)

            if avg_overlap < best_avg_overlap:
                best_avg_overlap = avg_overlap
                best_idx = i

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        selected_sets.append(set(candidates[best_idx]))

    return [candidates[i] for i in selected_indices]


def diversity_score(
    tickets: list[list[int]],
    pool_size: int,
) -> float:
    """Compute a diversity score for a set of tickets.

    Score ranges from 0 (all identical) to 1 (no overlap).
    Based on the ratio of unique numbers covered to theoretical
    maximum, and penalized by pairwise overlap.

    Args:
        tickets: List of tickets.
        pool_size: Total numbers in the pool.

    Returns:
        Diversity score between 0 and 1.
    """
    if not tickets:
        return 0.0

    if len(tickets) == 1:
        return 1.0

    # Coverage component: what fraction of the pool is covered
    all_numbers = set()
    for ticket in tickets:
        all_numbers.update(ticket)
    total_numbers = sum(len(t) for t in tickets)
    max_unique = min(total_numbers, pool_size)
    coverage = len(all_numbers) / max_unique if max_unique > 0 else 0.0

    # Overlap penalty: average pairwise overlap
    n = len(tickets)
    total_overlap = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = _overlap_fraction(tickets[i], tickets[j], pool_size)
            total_overlap += overlap
            pair_count += 1

    avg_overlap = total_overlap / pair_count if pair_count > 0 else 0.0

    # Diversity = coverage * (1 - avg_overlap)
    return coverage * (1.0 - avg_overlap)
