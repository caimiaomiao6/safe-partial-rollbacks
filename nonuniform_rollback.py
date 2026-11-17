# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Fixed-depth rate curves for the three-worker fork. Python 3.11+.

Search with NumPy: python nonuniform_rollback.py --output reproduced.json
Check saved rational witnesses without running the optimizer:
     python nonuniform_rollback.py --check nonuniform-results.json
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from functools import lru_cache
import json
import math
from pathlib import Path


# Bit 0 is the common prerequisite. Workers have bits 1, 2 and 3.
STATES = (7, 11, 13)  # rAB, rAC, rBC
def sound(x, f):
    return f & x == f and (not f & 1 or f == x)


OUTPUTS = [f for f in range(16) if any(sound(x, f) for x in STATES)]
ALLOWED = [[sound(x, f) for f in OUTPUTS] for x in STATES]
GAIN = [f.bit_count() for f in OUTPUTS]
DEN = 10**14
GRID = [Q(i, 100) for i in (0, 1, 3, 10, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 297, 300)]


@lru_cache(maxsize=8192)
def ln_bounds(n):
    with localcontext() as ctx:
        ctx.prec = 55
        val = Decimal(n).ln()
        if n == 1:
            return Q(0), Q(0)
        return Q(val.next_minus()), Q(val.next_plus())


def log_bounds(value):
    # Decimal.ln is correctly rounded. All subsequent interval arithmetic is rational.
    a, b = ln_bounds(value.numerator)
    c, d = ln_bounds(value.denominator)
    lo2, hi2 = ln_bounds(2)
    candidates = [v / w for v in (a-d, b-c) for w in (lo2, hi2)]
    return min(candidates), max(candidates)


def exact_bounds(p, target, witness):
    W = [[Q(a, DEN) for a in row] for row in witness["channel_counts"]]
    assert len(W) == 3 and all(len(row) == len(OUTPUTS) for row in W)
    assert all(sum(row) == 1 and min(row) >= 0 for row in W)
    assert all(not val or ALLOWED[i][j] for i,row in enumerate(W) for j,val in enumerate(row))
    depth = sum(px*sum(v*g for v,g in zip(row,GAIN)) for px,row in zip(p,W))
    assert depth >= target
    marginal = [sum(px*row[j] for px,row in zip(p,W)) for j in range(len(OUTPUTS))]
    upper = sum(px*v*log_bounds(v/marginal[j])[1] for px,row in zip(p,W)
                for j,v in enumerate(row) if v)
    if target == 3:  # equality in depth forces F=X
        lower = sum(px*log_bounds(1/px)[0] for px in p)
    else:
        t = Q(witness["tilt"])
        assert t >= 1
        q = [Q(a, DEN) for a in witness["trial_counts"]]
        assert len(q) == len(OUTPUTS) and min(q) >= 0 and sum(q) == 1
        A = [[t**g if ok else 0 for ok,g in zip(row,GAIN)] for row in ALLOWED]
        z = [sum(a*v for a,v in zip(row,q)) for row in A]
        assert min(z) > 0
        G = max(sum(px*row[j]/zx for px,row,zx in zip(p,A,z)) for j in range(len(q)))
        lower = target*log_bounds(t)[0] - sum(px*log_bounds(zx)[1] for px,zx in zip(p,z)) - log_bounds(G)[1]
    scale = 10**12
    return max(0, math.floor(lower*scale)), math.ceil(upper*scale)


def to_counts(values):
    counts = [int(v*DEN) for v in values]
    counts[max(range(len(values)), key=values.__getitem__)] += DEN - sum(counts)
    return counts


def solve(p, target):
    import numpy as np
    support = np.array(ALLOWED)
    gains = np.array(GAIN)
    law = np.array(p, dtype=float)
    g = float(target)
    full = np.array([[int(x == f) for f in OUTPUTS] for x in STATES])
    if g in (0, 3):
        channel = full if g else np.tile(np.array(OUTPUTS) == 0, (3,1))
        return dict(channel_counts=[to_counts(row) for row in channel],
                    trial_counts=to_counts(np.array(OUTPUTS) == 0),tilt="1")

    # Minimize over q the supremum over beta >= 0 of beta*g - E log Z.
    # An active-set Newton step profiles out beta, preserving the simplex.
    def evaluate(q):
        def at(beta):
            A = support*np.exp2(beta*gains)
            z = A @ q
            B = A / z[:,None]
            W = B*q
            means = W @ gains
            variance = law @ (W @ (gains*gains) - means*means)
            return W, B, means, variance, z
        lo, hi = 0., 1.
        while law @ at(hi)[2] < g:
            hi *= 2
            if hi > 128:
                raise ArithmeticError("insufficient gain on trial support")
        beta = 0.
        for _ in range(75):
            W,B,means,var,z = at(beta)
            residual = g-law @ means
            if abs(residual) < 2e-14 or (beta == 0 and residual < 0):
                break
            if residual > 0:
                lo = beta
            else:
                hi = beta
            nxt = beta + residual / (math.log(2)*var) if var > 1e-15 else hi
            beta = nxt if lo < nxt < hi else (lo+hi)/2
        value = beta*g - law @ np.log2(z)
        scores = law @ B
        hessian = (B.T*law) @ B
        if beta > 0 and var > 1e-15:
            cross = law @ (B*(gains-means[:,None]))
            hessian += np.outer(cross,cross)/var
        return value, scores, hessian, W, beta

    q = np.ones(len(OUTPUTS))/len(OUTPUTS)
    for _ in range(2000):
        value, scores, hessian, W, beta = evaluate(q)
        if max(scores) <= 1+1e-10:
            break
        active = set(np.flatnonzero(q > 0)) | {int(np.argmax(scores))}
        while True:
            idx = sorted(active)
            size = len(idx)
            system = np.ones((size+1,size+1))
            system[-1,-1] = 0
            system[:-1,:-1] = hessian[np.ix_(idx,idx)] + 1e-8*np.eye(size)
            direction = np.zeros_like(q)
            direction[idx] = np.linalg.solve(system,np.r_[scores[idx],0])[:-1]
            blocked = [j for j in idx if q[j] == 0 and direction[j] < -1e-13]
            if not blocked:
                break
            active.difference_update(blocked)
        falling = direction < -1e-13
        step = min(1.,np.min(-q[falling]/direction[falling])) if falling.any() else 1.
        for _ in range(64):
            trial = np.maximum(0,q+step*direction)
            trial[trial < 2e-16] = 0
            trial /= trial.sum()
            try:
                next_value = evaluate(trial)[0]
            except ArithmeticError:
                next_value = math.inf
            if next_value <= value + 2e-14:
                break
            step *= .5
        else:
            raise ArithmeticError("line search")
        q = trial
    else:
        raise ArithmeticError(("active set",p,target,max(scores)))
    # Add a small feasible margin before rational rounding; exact_bounds checks it.
    W = (1-1e-11)*W + 1e-11*full
    return dict(channel_counts=[to_counts(row) for row in W],trial_counts=to_counts(q),
                tilt=str(float(2**beta)))


def entropy(p):
    return -sum(x*math.log2(x) for x in p if x)


def check_analytic_point():
    p = [Q(3,5),Q(3,10),Q(1,10)]
    t = Q(7,6)
    q = [Q(7,10) if f == 0 else Q(3,10) if f == 2 else Q(0) for f in OUTPUTS]
    a = [[t**g if ok else 0 for ok,g in zip(row,GAIN)] for row in ALLOWED]
    z = [sum(v*w for v,w in zip(row,q)) for row in a]
    gradients = [sum(px*row[j]/zx for px,row,zx in zip(p,a,z)) for j in range(len(q))]
    assert max(gradients) == 1 and all(score == 1 for score,mass in zip(gradients,q) if mass)
    channel = [[mass*value/zx for mass,value in zip(q,row)] for row,zx in zip(a,z)]
    marginal = [sum(px*row[j] for px,row in zip(p,channel)) for j in range(len(q))]
    assert marginal == q
    assert sum(mass*g for mass,g in zip(marginal,GAIN)) == Q(3,10)
    assert [row[OUTPUTS.index(2)] for row in channel] == [Q(1,3), Q(1,3), Q(0)]


def run():
    rows = []
    for alpha in [Q(i,4) for i in range(5)]:
        p = [(1-alpha)/3+alpha*x for x in (Q(6,10),Q(3,10),Q(1,10))]
        slope = min(-math.log2(float(sum(px for px,ok in zip(p,col) if ok)))/g
                    for g,col in zip(GAIN,zip(*ALLOWED)) if g)
        for g in GRID:
            witness = solve(p,g)
            low,high = exact_bounds(p,g,witness)
            assert 0 <= high-low <= 3000, (p,g,low,high)
            chord = float(g/3)*entropy(p)
            if not alpha:
                assert low/1e12-1e-15 <= chord <= high/1e12+1e-15
            elif 0 < g < 3:
                assert high/1e12 < chord
            rows.append(dict(alpha=str(alpha), probabilities=list(map(str,p)), depth=str(g),
                             lower_trillionths=low,upper_trillionths=high,
                             independent_rate=chord,origin_slope=slope,witness=witness))
    return dict(outputs=OUTPUTS,states=STATES,denominator=DEN,rows=rows)


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit('Validation requires assertions. Run Python without -O.')
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check",type=Path)
    mode.add_argument("--output",type=Path)
    args = parser.parse_args()
    check_analytic_point()
    if args.check:
        stored = json.loads(args.check.read_text(encoding="utf-8-sig"))
        assert stored["outputs"] == OUTPUTS and stored["states"] == list(STATES) and stored["denominator"] == DEN
        expected = {(Q(i,4), g) for i in range(5) for g in GRID}
        for row in stored["rows"]:
            a,g = Q(row["alpha"]),Q(row["depth"])
            expected.remove((a,g))
            p = [Q(x) for x in row["probabilities"]]
            assert p == [(1-a)/3+a*x for x in (Q(6,10),Q(3,10),Q(1,10))]
            chord = float(g/3)*entropy(p)
            origin = min(-math.log2(float(sum(px for px,ok in zip(p,col) if ok)))/gain
                         for gain,col in zip(GAIN,zip(*ALLOWED)) if gain)
            assert math.isclose(row["independent_rate"],chord,rel_tol=1e-13,abs_tol=1e-14)
            assert math.isclose(row["origin_slope"],origin,rel_tol=1e-13,abs_tol=1e-14)
            low, high = exact_bounds(p,g,row["witness"])
            assert (low,high) == (row["lower_trillionths"],row["upper_trillionths"])
            assert 0 <= high-low <= 3000
        assert not expected
        print("Verified all fixed-depth upper and lower bounds.")
    else:
        result = json.dumps(run(),indent=2)
        if args.output:
            args.output.write_text(result+"\n",encoding="utf-8")
        else:
            print(result)
