# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Check block-cell decoding (Section 4) and the asymmetric example (Section 5)."""
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from math import prod

from check_decoder import decode_cell
from verify_rollback import bits


def block_cells(n, edges, support):
    """Compare tagged executions with componentwise decoding of every cell."""
    blocks = list(product(support, repeat=2))
    successors = [0] * n
    for a, b in edges:
        successors[a] |= 1 << b
    # These small fixture orders are topologically numbered.
    for a in reversed(range(n)):
        for b in bits(successors[a]):
            successors[a] |= successors[b]
    tagged = successors + [row << n for row in successors]

    @cache
    def words(state):
        result = {()}
        for e in bits(state):
            if not tagged[e] & state:
                result.update((e,) + rest for rest in words(state ^ (1 << e)))
        return result

    @cache
    def region(states):
        events, _ = decode_cell(n, edges, (bits(x) for x in states))
        return sum(1 << e for e in events)

    checked = correlated = 0
    for size in range(1, len(blocks) + 1):
        for cell in combinations(blocks, size):
            left = frozenset(x for x, y in cell)
            right = frozenset(y for x, y in cell)
            common = set.intersection(*(words(x | (y << n)) for x, y in cell))
            predicted = words(region(left) | (region(right) << n))
            if common != predicted:
                raise AssertionError((cell, len(common), len(predicted)))
            checked += 1
            correlated += size != len(left) * len(right)
    return checked, correlated


def asymmetric_counting_bound():
    # A dual witness at mean depth one. Validate every posterior-support
    # constraint and compare the resulting bound without rounding logs.
    support = [frozenset(s) for s in ("ab", "ac", "de")]
    outputs = {frozenset(f) for x in support for j in range(3)
               for f in combinations(x, j)}
    tilt = Fraction(7, 4)
    prices = [3 / (2 * tilt), 3 / (2 * tilt), 3 / tilt**2]
    for f in outputs:
        mass = sum(u for u, x in zip(prices, support) if f <= x) / 3
        if mass * tilt**len(f) > 1:
            raise AssertionError(f)
    # The information lower bound is log2(t) + mean(log2(u_x)).
    # Its exponential cubed must exceed the cubed counting value (3/2)^3.
    if tilt**3 * prod(prices) <= Fraction(3, 2)**3:
        raise AssertionError("expected a strict gap above the counting point")


if __name__ == "__main__":
    # Antichain with a common event z and one worker from three.
    first = block_cells(4, [], [0b0011, 0b0101, 0b1001])
    # r<a, r<b with a common independent event z. The shared r may be blocked.
    second = block_cells(4, [(0, 1), (0, 2)], [0b1011, 0b1101])
    print(f"{first[0] + second[0]} block cells passed, including "
          f"{first[1] + second[1]} nonrectangular cells.")
    asymmetric_counting_bound()
    print("Strict gap for the uniform nontransitive source verified exactly.")
