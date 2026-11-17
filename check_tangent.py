# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Find the last partial vertex of the uniform-fork rate curve (Section 6).

Input: integers w > k >= 2 and b >= 0. A return value of 0 gives the
endpoint chord. Otherwise the partial vertices are 1 through the result.
All orientation tests compare integers; no logarithm is rounded.
"""
from math import comb


def last_partial_vertex(w, k, b):
    if not (w > k >= 2 and b >= 0):
        raise ValueError("expected w > k >= 2 and b >= 0")
    low, high = 0, k - 1
    while low < high:
        j = (low + high) // 2
        span = b + k - j
        completions = comb(w - j, k - j)
        denominator = k - j
        # Each group of denominator factors raises the ratio by at least two.
        # Avoid constructing powers proportional to a binary-encoded chain length.
        large_span = span >= denominator * completions.bit_length()
        if large_span or completions * denominator ** span <= (w - j) ** span:
            high = j
        else:
            low = j + 1
    return low


def reference_hull(w, k, b):
    # Each point is (depth, completion count). Its height is log2(N/count).
    points = [(j, comb(w - j, k - j)) for j in range(k)]
    points.append((b + k, 1))
    hull = []
    for c, dc in points:
        while len(hull) >= 2:
            (a, da), (m, dm) = hull[-2:]
            if da ** (c - m) * dc ** (m - a) < dm ** (c - a):
                break
            hull.pop()
        hull.append((c, dc))
    return [depth for depth, _ in hull]


if __name__ == "__main__":
    count = 0
    for w in range(3, 25):
        for k in range(2, w):
            for b in range(25):
                last = last_partial_vertex(w, k, b)
                expected = list(range(last + 1)) + [b + k]
                if reference_hull(w, k, b) != expected:
                    raise AssertionError((w, k, b))
                count += 1
    print(f"{count} exact comparisons with the full lower-hull scan passed.")

    # At w=8 and k=4 the proved threshold is three. This input exercises
    # the binary-length cutoff, independently of the general hull oracle.
    if last_partial_vertex(8, 4, 10**100) != 0:
        raise AssertionError("large chain should have the endpoint chord")
    print("Binary-encoded chain of length 10^100: endpoint chord.")
