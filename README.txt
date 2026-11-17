Article title: Computing Safe Partial Rollbacks from Compressed Checkpoint Logs
Journal: Algorithmica
Authors: Siyuan Qiu and Jianfeng Xu
Corresponding affiliation: Koguan School of Law, China Institute for Smart Justice, School of Computer Science, Shanghai Jiao Tong University, Shanghai 200030, China
Corresponding email: xujf@sjtu.edu.cn

Online Resource 1. Reproducibility bundle

Extract the files together. Run the commands below from that directory
with Python 3.11 or newer and assertions enabled (the default). All
checks use the standard library. Only the final candidate-search command
requires NumPy (tested with 2.3.5). No input downloads are required.

Sections 2, 3, 6 and 7. Semantic and counting checks
  python verify_rollback.py
  python audit_rollback.py
  Compare the first command's output with the lines after the header in
  verification-results.txt. Compare the second with audit-results.json,
  excluding article_metadata. The latter file's nonuniform_counterexample
  is a separate two-state calculation: with probabilities (0.9, 0.1),
  disclose the more likely state on a fraction 1/9 of its occurrences.
  The total disclosure probability is 0.1 for both policies compared.

Section 3. Sparse-list graph decoder
  python check_decoder.py
  104,794 cells on all naturally labelled orders of at most four events.
  Each reported region is compared with (1), and every emitted command is
  executed against each member of its input cell.

Sections 4 and 5. Block decoding and symmetry
  python check_foundations.py
  On two small sources this compares all common tagged undo words with
  the decoder for the two projected cells. Of the 526 cells, 468 are
  strict subsets of the product of their projections. An exact rational
  dual check proves a positive gap for the three-state antichain.

Section 6. Uniform-fork vertex algorithm
  python check_tangent.py
  6,325 exact comparisons against a general hull scan, plus a chain of
  length 10^100 that exercises the integer-power cutoff.

Sections 6.2 and 6.3. Nonuniform forks
  python check_fork_slope.py
  python nonuniform_rollback.py --check nonuniform-results.json
  python nonuniform_rollback.py --output reproduced.json
  The first checks 84 slopes. The second verifies 80 saved pairs of bounds
  with exact rational arithmetic and outward enclosures for logarithms.
  The third regenerates the numerical witnesses and requires NumPy.
  Regenerated JSON contains the numerical payload. The deposited copy
  also has an article_metadata field. Floating-point candidate searches
  can vary across platforms. To certify a fresh run independently, use
  python nonuniform_rollback.py --check reproduced.json
  Divide lower_trillionths and upper_trillionths by 10^12 to obtain bits per
  checkpoint. channel_counts and trial_counts use the denominator 10^14.
  The field tilt denotes theta = 2^beta in Section 6.3. Rows use alpha
  and depth g as keys. origin_slope is in bits per event and
  independent_rate is in bits per checkpoint.
  The three source masks are rAB, rAC and rBC. Bits 0 to 3 are r,A,B,C.
  The code certifies individual depths. Connecting lines in Fig. 2 are
  provided solely to aid reading.

Figures. Editable TikZ drawings are in artwork/Fig1.tex through artwork/Fig3.tex of the manuscript source package.
Fig. 1 is the two-checkpoint example in Section 2; Fig. 2 uses the 80
saved intervals; Fig. 3 uses the closed formulas in Section 7.1.
