# Day 04 Overleaf note

- Setting: VitalDB-informed reconstructed PK/PD simulation; no clinical, causal, or external-validation claim.
- Design: full development-training partition (1,642 subjects/1,680 cases), fresh internal validation (96/96), 1,800-s horizon, five paired seeds, 524,288 steps per job.
- Primary MAE (mean ± seed SD): P0S0 7.3396 ± 1.5875; P1S0 8.2575 ± 3.1697; P0S1 4.4328 ± 3.1221; P1S1 4.6484 ± 3.5379.
- Contrasts: P0_state -2.9068 (4/5 favorable; [-5.3546, +0.3341]); P1_state -3.6091 (4/5 favorable; [-7.7946, +0.9408]); S0_preprocessing +0.9179 (2/5 favorable; [-0.1297, +2.4500]); S1_preprocessing +0.2156 (2/5 favorable; [-0.0674, +0.5998]); interaction -0.7023 (3/5 favorable; [-2.3593, +0.6077]).
- State-effect language: P0 and P1 are moderate/suggestive; neither descriptive hierarchical bootstrap interval excludes zero.
- Visibility: P0 0.5441, P1 0.4129; primary actor inputs were finite, Tanh saturation was negligible, and core clipping was zero.
- Baseline PI MAE: P0 6.0052; P1 6.9496. Frozen Day 03 transfer deltas were -3.4645, -5.5104, and -1.4427 (descriptive only).
- Descriptive hierarchical bootstrap resampled paired seeds and subjects; n=5 policy seeds; original-test access was zero.
