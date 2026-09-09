# Day 05 Frozen Brief

## Question

Which nonredundant S1 feature group—cumulative dose, mechanistic concentrations, or both—best preserves the Day 04 state signal while reducing seed-dependent instability? The full S1 state is retained as a redundant-representation control.

## Frozen design

- Parent: `7db8bd0cbc16a50bf058247b9e627c48503d5c90`
- Training universe: the exact Day 04 development-training partition, 1,642 subjects / 1,680 cases
- Development evaluation: the already-examined Day 04 development-validation subset, 96 subjects / 96 cases
- Untouched reserve: 170 subjects / 170 cases remaining after the old 24 and Day 04's 96; identifiers are hashed, and no reserve bundle or outcome is accessed
- States: S0 (34-D), S_CUM (36-D), S_CONC (38-D), S_CORE (40-D), S1 (42-D)
- New training: S_CUM/S_CONC/S_CORE × P0/P1 × seeds 48–52; 30 jobs
- Reused anchors: Day 04 S0/S1 at all four matching post-update checkpoints
- Budget: 524,288 steps, 256 rollouts, 2,560 PPO epochs per new job; 15,728,640 new steps
- Horizon: 1,800 seconds; one CPU worker and one Torch thread
- Scaling: frozen Day 04 Nscale factors, SHA-256 `b0a494afb9851c4ac6eee00f85229cfbec877f54022659a9fb50cefdba7647d0`; no refitting
- Real-input S_CORE benchmark: 262.2 transitions/s; projected new-training time 16.67 hours. The frozen budget is unchanged.
- Process guard: read-only Windows sampling found no sustained external VS Code child workload; reliable continuous parent-chain attribution remains unavailable, so the conservative one-worker/one-thread setting is used.

## Boundaries

This is exploratory development screening in a VitalDB-informed reconstructed PK/PD simulation. It is not an independent confirmation cohort, external validation, clinical evidence, bedside dosing advice, or biological/clinical causal inference. No checkpoint or validation outcome selects training.
