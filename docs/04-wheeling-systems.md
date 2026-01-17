# Wheeling Systems for Lottery Coverage

Wheeling systems are the **only mathematically guaranteed approach** in lottery playing. Unlike prediction methods (which don't work), wheels provide provable coverage guarantees for smaller prizes.

## What is a Wheeling System?

A wheeling system is a set of tickets that guarantees a minimum match level if certain numbers are drawn.

### Example

If you believe 10 numbers will include the 6 winners, a wheel can guarantee:
- At least one ticket with 4 matches if all 6 winners are in your 10
- Fewer tickets than playing all C(10,6) = 210 combinations

## Types of Wheels

| Type | Description | Tickets | Coverage |
|------|-------------|---------|----------|
| Full Wheel | All combinations | C(n,k) | Complete |
| Abbreviated Wheel | Subset with guarantee | Reduced | Guaranteed minimum |
| Key Number Wheel | Fixed numbers in all tickets | Reduced | Key-dependent |
| Balanced Wheel | Equal number coverage | Target count | Balanced |

## Mathematical Foundation

### Covering Designs

A (v, k, t)-covering design is a collection of k-subsets (blocks) from a v-set such that every t-subset is contained in at least one block.

For lottery wheels:
- v = total numbers in wheel
- k = numbers per ticket
- t = guaranteed match level

### Covering Number

The covering number C(v, k, t) is the minimum number of blocks needed:
```
C(v, k, t) ≥ C(v, t) / C(k, t)
```

This is a theoretical lower bound. Actual wheel sizes are typically 20-40% larger.

## Full Wheel

### Definition

Play every possible combination of your selected numbers.

### Calculation

For n numbers with k per ticket:
```
Tickets = C(n, k) = n! / (k! × (n-k)!)
```

### Examples

| Numbers | Per Ticket | Tickets |
|---------|------------|---------|
| 7 | 6 | 7 |
| 8 | 6 | 28 |
| 9 | 6 | 84 |
| 10 | 6 | 210 |
| 12 | 6 | 924 |
| 15 | 6 | 5,005 |

### Implementation

```python
from itertools import combinations

def generate_full_wheel(numbers: list[int], per_ticket: int) -> list[list[int]]:
    return [list(combo) for combo in combinations(numbers, per_ticket)]
```

### When to Use

- Small number sets (≤ 8 numbers)
- When you want absolute maximum coverage
- When budget allows all combinations

## Abbreviated Wheel

### Definition

A reduced set of tickets that guarantees a minimum match level.

### Guarantee Notation

"3-if-5" means: If 5 of your wheeled numbers are among the winners, at least one ticket will have 3 matches.

### Algorithm

Greedy covering algorithm:

```python
def generate_abbreviated_wheel(numbers: list[int],
                               per_ticket: int,
                               guarantee: int) -> list[list[int]]:
    # Sub-combinations that need coverage
    required = set(combinations(numbers, guarantee))
    covered = set()
    tickets = []

    while covered != required:
        # Find ticket covering most uncovered subcombos
        best_ticket = max(
            combinations(numbers, per_ticket),
            key=lambda t: len(set(combinations(t, guarantee)) - covered)
        )

        tickets.append(list(best_ticket))
        covered |= set(combinations(best_ticket, guarantee))

    return tickets
```

### Example: 10 Numbers, 6 per Ticket, 4-if-6 Guarantee

Full wheel: 210 tickets
Abbreviated wheel: ~15-20 tickets

**Reduction: 90%+ fewer tickets with guaranteed 4-match coverage**

### Trade-off

| Guarantee | Typical Reduction | What You Lose |
|-----------|-------------------|---------------|
| 6-if-6 | 0% (full wheel) | Nothing |
| 5-if-6 | 40-60% | Some 6-match scenarios |
| 4-if-6 | 80-90% | Some 5+ match scenarios |
| 3-if-6 | 95%+ | Most higher match scenarios |

## Key Number Wheel

### Definition

One or more "key" numbers appear in every ticket. Remaining slots are filled with combinations of other numbers.

### When to Use

- Strong confidence in specific numbers
- Reducing ticket count dramatically
- Focusing coverage on likely numbers

### Implementation

```python
def generate_key_wheel(key_numbers: list[int],
                       other_numbers: list[int],
                       per_ticket: int) -> list[list[int]]:
    remaining_slots = per_ticket - len(key_numbers)
    tickets = []

    for combo in combinations(other_numbers, remaining_slots):
        ticket = sorted(list(key_numbers) + list(combo))
        tickets.append(ticket)

    return tickets
```

### Example

Key numbers: [7, 13]
Other numbers: [1, 2, 3, 4, 5, 6]
Per ticket: 6

Tickets needed: C(6, 4) = 15

Each ticket contains 7 and 13 plus 4 others.

### Risk

If a key number is NOT among the winners, all tickets lose that potential match.

## Balanced Wheel

### Definition

Each number appears in approximately the same number of tickets.

### Purpose

- Avoid over-concentration on any number
- Spread risk evenly
- Fair coverage for all selected numbers

### Implementation

```python
def generate_balanced_wheel(numbers: list[int],
                            per_ticket: int,
                            target_tickets: int) -> list[list[int]]:
    counts = {n: 0 for n in numbers}
    tickets = []
    seen = set()

    for _ in range(target_tickets * 10):
        if len(tickets) >= target_tickets:
            break

        # Prefer numbers with lower counts
        sorted_nums = sorted(numbers, key=lambda x: counts[x])
        ticket = tuple(sorted(sorted_nums[:per_ticket]))

        if ticket not in seen:
            seen.add(ticket)
            tickets.append(list(ticket))
            for n in ticket:
                counts[n] += 1

    return tickets
```

## Coverage Verification

### Purpose

Confirm a wheel provides claimed guarantees.

### Implementation

```python
def verify_wheel(tickets: list[list[int]],
                 numbers: list[int],
                 guarantee: int) -> tuple[bool, list]:
    required = set(tuple(sorted(c)) for c in combinations(numbers, guarantee))
    covered = set()

    for ticket in tickets:
        for subcombo in combinations(ticket, guarantee):
            covered.add(tuple(sorted(subcombo)))

    uncovered = required - covered
    return len(uncovered) == 0, list(uncovered)
```

### Testing Your Wheel

```python
# Example: Verify 4-if-6 wheel
numbers = list(range(1, 11))  # 10 numbers
tickets = generate_abbreviated_wheel(numbers, 6, 4)

is_valid, uncovered = verify_wheel(tickets, numbers, 4)
print(f"Valid: {is_valid}, Uncovered: {len(uncovered)}")
```

## Wheel Optimization

### Professor Bluskov's Research

Professor Iliya Bluskov (University of Northern British Columbia) has published optimal and near-optimal covering designs for lottery applications.

Key contributions:
- Mathematically proven minimum-size wheels
- Tables for common lottery configurations
- Algorithms for wheel generation

### Optimization Goals

1. **Minimize tickets** - Fewer tickets for same guarantee
2. **Maximize coverage** - More sub-combinations covered
3. **Balance numbers** - Equal appearances

### Greedy vs Optimal

| Approach | Tickets | Computation | Quality |
|----------|---------|-------------|---------|
| Greedy | ~30% above optimal | Fast | Good |
| Optimal | Minimum possible | Slow (NP-hard) | Best |
| Heuristic | ~10-20% above optimal | Medium | Very Good |

## Practical Wheel Examples

### Joker (5 from 45)

**8 numbers, 3-if-5 guarantee:**
- Full wheel: C(8,5) = 56 tickets
- Abbreviated: ~8 tickets

**10 numbers, 4-if-5 guarantee:**
- Full wheel: C(10,5) = 252 tickets
- Abbreviated: ~15-20 tickets

### Loto 6/49 (6 from 49)

**10 numbers, 4-if-6 guarantee:**
- Full wheel: C(10,6) = 210 tickets
- Abbreviated: ~15 tickets

**12 numbers, 4-if-6 guarantee:**
- Full wheel: C(12,6) = 924 tickets
- Abbreviated: ~40-50 tickets

### Loto 5/40 (5 from 40)

**8 numbers, 3-if-5 guarantee:**
- Full wheel: C(8,5) = 56 tickets
- Abbreviated: ~8 tickets

## CLI Usage

```bash
# Generate wheel with 10 numbers and 3-match guarantee
PYTHONPATH=src python scripts/generate_joker_picks.py --wheel 10 --wheel-guarantee 3 -v

# Generate Loto 6/49 wheel with 12 numbers and 4-match guarantee
PYTHONPATH=src python scripts/generate_loto_649_picks.py --wheel 12 --wheel-guarantee 4 -v
```

### Output Format

```
=== Wheeling System ===
Numbers in wheel: 10
Numbers per ticket: 6
Guarantee: 4-if-6

Tickets: 18
Coverage: 100% of 4-number combinations

1. 1, 2, 3, 4, 5, 6
2. 1, 2, 3, 7, 8, 9
...
```

## Cost-Benefit Analysis

### Jackpot Odds

Wheeling does NOT improve jackpot odds proportionally.

| Without Wheel | With 10-number Wheel |
|---------------|---------------------|
| 1 in 13,983,816 | ~210 in 13,983,816 |
| 1 ticket | 210 tickets |
| Same odds per ticket | Same odds per ticket |

**You're buying more tickets, not getting better odds per ticket.**

### Small Prize Improvement

Where wheels DO help:

| Scenario | Without Wheel | With 4-if-6 Wheel |
|----------|---------------|-------------------|
| 6 of your 10 win | Maybe 6 matches | Guaranteed ≥4 |
| 5 of your 10 win | Maybe 5 matches | Likely ≥3 |
| 4 of your 10 win | Maybe 4 matches | Possible ≥3 |

### Break-Even Analysis

For a 4-if-6 wheel to be "profitable":
```
Expected small prizes × probability > Ticket cost × number of tickets
```

This rarely works out due to:
- Small prize payouts being low
- Probability still being against you
- Jackpot odds not improving proportionally

## When Wheeling Makes Sense

### Good Use Cases

1. **Lottery pools** - Group plays where more tickets are affordable
2. **Entertainment** - Systematic approach adds structure
3. **Guaranteed coverage** - Psychological benefit of "knowing you're covered"
4. **Small prize focus** - When small wins matter to you

### Bad Use Cases

1. **"Investment"** - Wheels don't make lottery profitable
2. **Solo play with limited budget** - Better to buy fewer random tickets
3. **Chasing jackpot** - Wheels don't improve jackpot odds meaningfully

## Summary

| Wheel Type | Tickets | Guarantee | Best For |
|------------|---------|-----------|----------|
| Full | All C(n,k) | Maximum | Small n |
| Abbreviated | Reduced | Specified minimum | Medium n |
| Key Number | Reduced | Key-dependent | High confidence numbers |
| Balanced | Target count | Even coverage | Fair distribution |

### The Honest Truth About Wheels

1. **Mathematically sound** - Guarantees are provable
2. **Cost effective** - Reduce tickets while maintaining coverage
3. **Still negative EV** - Expected return is still negative
4. **Best lottery "strategy"** - If you must play, wheels are optimal

## Implementation in This Codebase

See `src/shared/wheeling.py` for:
- `WheelGenerator` - Main wheel generation class
- `generate_full_wheel()` - All combinations
- `generate_abbreviated_wheel()` - Reduced with guarantee
- `KeyNumberWheel` - Fixed key numbers
- `generate_balanced_wheel()` - Equal coverage
- `verify_wheel_coverage()` - Guarantee verification

## Next Steps

- [Loto.ro Specific](05-loto-ro-specific.md) - Romanian lottery details
- [Implementation Guide](06-implementation-guide.md) - Using wheels in practice
- [Honest Assessment](08-honest-assessment.md) - Realistic expectations
