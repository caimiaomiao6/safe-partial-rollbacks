# Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
# Journal: Algorithmica
# Authors: [Author names]
# Corresponding affiliation: [Department, institution, city, country]
# Corresponding email: [Corresponding email]

"""Compare the Section 6.2 fork slope with direct enumeration (standard library)."""
from fractions import Fraction as Q
from itertools import combinations
from math import log2

if not __debug__:
    raise SystemExit("Run without -O to enable checks.")

cases=0
for w,k in ((3,2),(4,2),(4,3),(5,3)):
    old_states=[sum(1<<i for i in s) for s in combinations(range(w),k)]
    for weights in ([1]*len(old_states),list(range(1,len(old_states)+1)),[2**i for i in range(len(old_states))]):
        p=[Q(v,sum(weights)) for v in weights]
        c=min(-log2(float(sum(px for s,px in zip(old_states,p) if s&t==t)))/t.bit_count()
              for t in range(1,1<<w) if t.bit_count()<k)
        h=-log2(float(max(p)))
        for b in range(7):
            states=[(s<<b)|(1<<b)-1 for s in old_states]
            order=[sum(1<<j for j in range(i+1,b+w)) if i<b else 0 for i in range(b+w)]
            masses={}
            for x,px in zip(states,p):
                f=x
                while f:
                    if all(not (order[i] & (x^f)) for i in range(w+b) if f>>i&1):
                        masses[f]=masses.get(f,0)+px
                    f=(f-1)&x
            observed=min(-log2(float(v))/f.bit_count() for f,v in masses.items())
            assert abs(observed-min(c,h/(b+k)))<1e-12
            cases+=1
assert cases==84
print(f"{cases} fork-slope comparisons passed.")
