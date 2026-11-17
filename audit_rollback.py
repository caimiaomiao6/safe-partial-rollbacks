# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Independent checks added during the submission audit (Python 3.10+).

Run beside verify_rollback.py:
    python audit_rollback.py
Uses only the Python standard library.
"""
from collections import deque
from itertools import combinations
from math import comb, log2
import json
import verify_rollback as original


def certificate(order, x, f):
    return f & x == f and all(not (order[a] & (x ^ f)) for a in original.bits(f))


def structural_checks():
    counts = dict(orders=0, interval_pairs=0, net_markings=0, net_steps=0)
    for n in range(1, 6):
        full = (1 << n) - 1
        for order in original.orders(n):
            counts["orders"] += 1
            predecessors = [sum(1 << a for a in range(n) if order[a] & (1 << e))
                            for e in range(n)]
            ideals = [x for x in range(full + 1)
                      if all(predecessors[a] & x == predecessors[a] for a in original.bits(x))]
            for f in range(full + 1):
                convex = all((order[a] & predecessors[c]) & f == order[a] & predecessors[c]
                             for a in original.bits(f) for c in original.bits(f)
                             if order[a] & (1 << c))
                down = f
                up = f
                for a in original.bits(f):
                    down |= predecessors[a]
                    up |= order[a]
                for x in ideals:
                    interval = convex and down & x == down and not (x & (up ^ f))
                    assert certificate(order, x, f) == interval
                    counts["interval_pairs"] += 1
            covers = [(a,b) for a in range(n) for b in original.bits(order[a])
                      if not (order[a] & predecessors[b])]
            forward = []
            for e in range(n):
                inputs = {("ready",e)} | {("edge",a,b) for a,b in covers if b == e}
                outputs = {("done",e)} | {("edge",a,b) for a,b in covers if a == e}
                forward.append((inputs,outputs))
            start = frozenset(("ready",e) for e in range(n))
            seen, queue = {start}, deque([start])
            while queue:
                marking = queue.popleft()
                x = sum(1 << e for e in range(n) if ("done",e) in marking)
                expected = ({("done",e) if x & (1<<e) else ("ready",e) for e in range(n)}
                            | {("edge",a,b) for a,b in covers if x & (1<<a) and not x & (1<<b)})
                assert marking == expected and x in ideals
                counts["net_markings"] += 1
                for e,(inputs,outputs) in enumerate(forward):
                    for reverse,ins,outs in [(False,inputs,outputs),(True,outputs,inputs)]:
                        enabled = ins <= marking
                        required = (bool(x & (1<<e)) and not order[e] & x) if reverse else (
                            not x & (1<<e) and predecessors[e] & x == predecessors[e])
                        assert enabled == required
                        counts["net_steps"] += 1
                        if enabled:
                            assert not (outs & (marking - ins)), "one-safety"
                            next_marking = frozenset((marking - ins) | outs)
                            if next_marking not in seen:
                                seen.add(next_marking)
                                queue.append(next_marking)
            assert len(seen) == len(ideals)
    return counts


def numerical_checks():
    cases, max_error, max_gap = 0, 0.0, 0.0
    for b in (0,1,3):
        w,k = 5,2
        states = [((1<<b)-1) | sum(1 << (b+i) for i in s) for s in combinations(range(w),k)]
        order = [sum(1 << j for j in range(a+1,b+w)) if a < b else 0 for a in range(b+w)]
        outputs = [f for f in range(1 << (b+w)) if any(certificate(order,x,f) for x in states)]
        support = [[j for j,f in enumerate(outputs) if certificate(order,x,f)] for x in states]
        gain = [f.bit_count() for f in outputs]
        p = 1 / len(states)
        for beta in (0.1,0.3,0.7,1.1,1.5,2.4,4.0):
            q = [1 / len(outputs)] * len(outputs)
            powers = [2**(beta*g) for g in gain]
            for iteration in range(100000):
                z = [sum(q[j]*powers[j] for j in row) for row in support]
                channel = [[q[j]*powers[j]/z[i] if j in support[i] else 0.0
                            for j in range(len(q))] for i in range(len(states))]
                marginal = [sum(p*row[j] for row in channel) for j in range(len(q))]
                objective = sum(p*val*(log2(val/marginal[j])-beta*gain[j])
                                for row in channel for j,val in enumerate(row) if val)
                # A dual certificate, computed over ALL feasible output symbols.
                max_gradient = max(sum(p*powers[j]/z[i] for i,row in enumerate(support) if j in row)
                                   for j in range(len(q)))
                lower = -sum(p*log2(v) for v in z)-log2(max_gradient)
                gap = objective-lower
                if gap < 1e-9:
                    break
                q = marginal
            else:
                raise AssertionError("BA did not obtain the requested certified gap")
            exact = min([log2(comb(w,k)/comb(w-j,k-j))-beta*j for j in range(k)]
                        + [log2(comb(w,k))-beta*(b+k)])
            assert lower-1e-10 <= exact <= objective+1e-10
            max_error = max(max_error, abs(objective-exact))
            max_gap = max(max_gap,gap)
            cases += 1
    return dict(cases=cases,max_objective_error=max_error,max_duality_gap=max_gap)


def entropy(p):
    return -p*log2(p)-(1-p)*log2(1-p)


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit('Validation requires assertions. Run Python without -O.')
    structural = structural_checks()
    numerical = numerical_checks()
    nonuniform = dict(selective_disclosure=entropy(.1)-.9*entropy(1/9),
                      independent_timesharing=.1*entropy(.1))
    assert nonuniform["selective_disclosure"] < nonuniform["independent_timesharing"]
    print(json.dumps(dict(structural=structural,numerical=numerical,
                         nonuniform_counterexample=nonuniform),indent=2))
