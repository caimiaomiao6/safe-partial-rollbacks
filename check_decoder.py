# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Sparse-list decoder in Section 3. Event IDs are integers 0,...,n-1."""
from collections import deque
from itertools import combinations

from verify_rollback import bits, orders


def decode_cell(n, edges, checkpoints):
    """Return the region and one longest safe sequence for a nonempty cell.

    The caller supplies a causal DAG and ideal lists without repeated IDs.
    """
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for cause, effect in edges:
        parents[effect].append(cause)
        children[cause].append(effect)
    count = [0] * n
    m = 0
    for checkpoint in checkpoints:
        m += 1
        for event in checkpoint:
            count[event] += 1
    if not m:
        raise ValueError("the message cell must contain at least one checkpoint")
    marked = [0 < t < m for t in count]
    pending = deque(e for e in range(n) if marked[e])
    while pending:
        effect = pending.popleft()
        for cause in parents[effect]:
            if not marked[cause]:
                marked[cause] = True
                pending.append(cause)
    region = {e for e in range(n) if count[e] == m and not marked[e]}
    remaining = [len(row) for row in children]
    pending = deque(e for e in range(n) if remaining[e] == 0)
    sequence = []
    while pending:
        effect = pending.popleft()
        if effect in region:
            sequence.append(effect)
        for cause in parents[effect]:
            remaining[cause] -= 1
            if remaining[cause] == 0:
                pending.append(cause)
    return region, sequence


if __name__ == "__main__":
    count = 0
    for n in range(1, 5):
        for successors in orders(n):
            edges = [(a, b) for a in range(n) for b in bits(successors[a])
                     if not any(successors[c] & (1 << b) for c in bits(successors[a]))]
            ideals = [x for x in range(1 << n)
                      if all(x & (1 << a) or not (x & successors[a]) for a in range(n))]
            for m in range(1, len(ideals) + 1):
                for cell in combinations(ideals, m):
                    union, meet = 0, (1 << n) - 1
                    for x in cell:
                        union |= x
                        meet &= x
                    expected = {a for a in bits(meet) if successors[a] & (union ^ meet) == 0}
                    region, sequence = decode_cell(n, edges, (bits(x) for x in cell))
                    if region != expected or set(sequence) != expected or len(sequence) != len(expected):
                        raise AssertionError((successors, cell, sequence))
                    for x in cell:
                        for event in sequence:
                            if not x & (1 << event) or x & successors[event]:
                                raise AssertionError((x, event))
                            x ^= 1 << event
                    count += 1
    print(f"Regions and longest sequences verified on {count} nonempty cells.")
