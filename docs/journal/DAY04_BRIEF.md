# Day 04 Frozen Brief

## Question

Does the Nscale-conditioned S1 advantage and P×S pattern persist at 1,800 seconds using the full development-training partition, five new paired seeds, and a previously unused internal-validation subset?

## Frozen design

- Parent: `a391d8f5597b4b73c8479767316b5dc39fc32003`
- Training: 1,642 subjects / 1,680 cases
- Fresh internal validation: 96 subjects / 96 cases; prior 24-subject subset excluded
- Conditions: P0S0, P1S0, P0S1, P1S1
- Seeds: 48–52; requested 524,288 steps per job; 20 jobs
- Horizon: 1,800 seconds; final post-update checkpoint only
- Nscale factors: train-only pooled P0/P1, constant 1.5 and profile-specific tuned PI trajectories
- Primary endpoint: subject-mean latent-BIS MAE over 0–1,800 seconds
- Process guard: `psutil` unavailable, so one low-thread worker is the conservative fallback; reliable VS Code parent-chain detection is unavailable.
- Real-input benchmark: 588.1 transitions/s; projected core 5.70 hours; the uniform budget fallback was not invoked.

## Boundaries

This is VitalDB-informed evaluation of simulated trajectories under fixed patient profiles and remifentanil schedules. It is not clinical intervention evidence, external validation, or causal inference. Validation fits nothing. The sealed original test cohort is not accessed.
