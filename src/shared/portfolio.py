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

    # Precompute candidate sets once and track each candidate's cumulative
    # overlap against the selected set incrementally. The naive version
    # rebuilt set(ticket) and re-summed overlap against every selected line
    # on every round — O(select_count² · candidates) with a set-intersection
    # inside, which blew up to ~10 min on a boosted 112-from-448 selection.
    # Dividing by the (constant-per-round) selected count doesn't change the
    # argmin, so we compare raw cumulative overlap directly. Same greedy
    # result, ~O(select_count · candidates).
    cand_sets = [frozenset(c) for c in candidates]
    selected_indices: list[int] = [best_first]
    selected_flags = [False] * len(candidates)
    selected_flags[best_first] = True
    cum_overlap = [len(cand_sets[i] & cand_sets[best_first]) for i in range(len(candidates))]

    while len(selected_indices) < select_count:
        best_idx = -1
        best_overlap = float("inf")
        for i in range(len(candidates)):
            if selected_flags[i]:
                continue
            if cum_overlap[i] < best_overlap:
                best_overlap = cum_overlap[i]
                best_idx = i

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        selected_flags[best_idx] = True
        new_set = cand_sets[best_idx]
        for i in range(len(candidates)):
            if not selected_flags[i]:
                cum_overlap[i] += len(cand_sets[i] & new_set)

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
