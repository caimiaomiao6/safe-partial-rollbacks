# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Reproduce the finite checks in Safe Partial Rollback from Compressed Logs.

Run with Python 3.10+ (standard library): python verify_rollback.py
Bit i denotes occurrence i. All orders use a fixed topological labelling.
"""
from functools import cache
from itertools import combinations
from math import ceil, comb, log2


def submasks(mask):
    part = mask
    while part:
        yield part
        part = (part - 1) & mask
    yield 0


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def orders(n):
    edges = list(combinations(range(n), 2))
    seen = set()
    for graph in range(1 << len(edges)):
        succ = [0] * n
        for i in bits(graph):
            a, b = edges[i]
            succ[a] |= 1 << b
        for a in reversed(range(n)):
            for b in bits(succ[a]):
                succ[a] |= succ[b]
        key = tuple(succ)
        if key not in seen:
            seen.add(key)
            yield key


def language_checks():
    order_count = cell_count = 0
    for n in range(1, 5):
        full = (1 << n) - 1
        for succ in orders(n):
            order_count += 1
            ideals = [x for x in range(full + 1)
                      if all(not (succ[a] & x) for a in bits(full ^ x))]

            @cache
            def words(x):
                ans = {()}
                for a in bits(x):
                    if not succ[a] & x:
                        ans.update((a,) + tail for tail in words(x ^ (1 << a)))
                return ans

            def visit(i, union, meet, shared):
                nonlocal cell_count
                for j in range(i, len(ideals)):
                    x = ideals[j]
                    u, v = union | x, meet & x
                    common = words(x) if shared is None else shared & words(x)
                    region = sum(1 << a for a in bits(v) if not succ[a] & (u ^ v))
                    assert common == words(region), (succ, u, v)
                    cell_count += 1
                    visit(j + 1, u, v, common)

            visit(0, 0, full, None)
            words.cache_clear()
    return order_count, cell_count


def hull(points):
    result = []
    for g, r in sorted(points):
        while len(result) >= 2:
            a, s = result[-2]
            b, t = result[-1]
            if (t-s)*(g-b) < (r-t)*(b-a) - 1e-10:
                break
            result.pop()
        result.append((g, r))
    return result


def fork_checks(b, w=8, k=4):
    trunk = (1 << b) - 1
    states = [trunk | sum(1 << (b+i) for i in x)
              for x in combinations(range(w), k)]
    succ = [sum(1 << j for j in range(i+1, b+w)) if i < b else 0
            for i in range(b+w)]
    # Build every operationally sound output, without using the binomial formula.
    degree = {}
    for x in states:
        for f in submasks(x):
            if all(not (succ[a] & (x ^ f)) for a in bits(f)):
                degree[f] = degree.get(f, 0) + 1
    best = {}
    for f, d in degree.items():
        size = f.bit_count()
        best[size] = max(d, best.get(size, 0))
    entropy = log2(comb(w, k))
    observed = hull([(g, log2(len(states)/d)) for g, d in best.items()])
    exact = hull([(j, log2(comb(w,k)/comb(w-j,k-j))) for j in range(k)]
                 + [(b+k, entropy)])
    # Extra singleton-support outputs can lie above the lower boundary:
    # the upper useful endpoint dominates these at the same information cost.
    assert len(observed) == len(exact)
    assert all(a == c and abs(s-t) < 1e-10
               for (a,s), (c,t) in zip(observed,exact))
    return observed


def hard_depth_check():
    edges = list(combinations(range(4),2))
    blocks = list(combinations(range(4),2))
    # K4 source: certify at least one vertex in EACH of two coordinates.
    pairs = [(x,y) for x in edges for y in blocks]
    rectangles = [{i for i,(x,y) in enumerate(pairs) if a in x and b in y}
                  for a in range(4) for b in range(4)]
    for s in range(1, len(rectangles)+1):
        for choice in combinations(rectangles, s):
            if len(set.union(*choice)) == len(pairs):
                return s
    raise AssertionError('uncovered state')


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('Validation requires assertions. Run Python without -O.')
    n_orders, n_cells = language_checks()
    print(f'Language identity: {n_orders} naturally labelled orders, {n_cells} cells.')
    for b in (0, 1, 2, 3, 6):
        vertices = fork_checks(b)
        print(f'b={b}: ' + ', '.join(f'({g},{r:.6f})' for g,r in vertices))
    print('Critical trunk length (8,4):', ceil(log2(70)/log2(2)-4))
    print('K4, two checkpoints, minimum hard-depth-one codebook:', hard_depth_check())
