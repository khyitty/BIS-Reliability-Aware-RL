# Day 06 Frozen Brief

## Question

Which two-coordinate view of the four model-derived mechanistic concentrations preserves the low error and seed stability of S_CONC: drug-specific propofol, drug-specific remifentanil, cross-drug plasma, or cross-drug effect-site concentrations?

## Frozen design

- Parent: `7bc18049eeea48c54dabca00c39ed7d25b50b260`
- Training universe: unchanged Day 05 development-training partition, 1,642 subjects / 1,680 cases
- Development evaluation: reused Day04 development-validation subset, 96 subjects / 96 cases
- Untouched reserve: 170 subjects / 170 cases; frozen membership counts/hashes only, with zero bundle, outcome, or statistic access
- States: S0 (34-D), S_PROP/S_REMI/S_CP/S_CE (36-D), and S_CONC (38-D)
- New training: four 36-D states × P0/P1 × seeds 48–52; 40 jobs
- Reused anchors: Day 04 S0 and Day 05 S_CONC at four matching post-update checkpoints
- Budget: 524,288 steps, 256 rollouts, 2,560 PPO epochs per new job; 20,971,520 new steps
- Horizon: 1,800 seconds; one CPU worker and one Torch thread
- Scaling: frozen Day 04 Nscale factors, SHA-256 `b0a494afb9851c4ac6eee00f85229cfbec877f54022659a9fb50cefdba7647d0`; no refitting
- Evaluation: all checkpoints on the 32-subject training-only diagnostic subset; final checkpoints only on the reused 96-subject development subset
- Selection: the predeclared representation-retention heuristic, not a statistical equivalence test

## Causal availability

At decision time t, the observation is built before the action for interval t→t+10 s is supplied. Concentrations are the simulator state produced by completed prior intervals only. Propofol uses prior policy-applied doses; recorded clinical propofol is not injected after initialization. Remifentanil uses the exogenous schedule only through completed intervals. Future BIS/SQI/infusion samples, retrospective fill, the action being chosen, and latent BIS are absent from policy features. The projection wrapper operates after scaling and changes neither dynamics nor timing.

These coordinates are exact internal states of the reconstructed PK/PD environment. They are model-derived mechanistic concentrations, not measured clinical concentrations, and may be optimistic under deployment-time PK mismatch or noisy state estimation. Day 06 does not test that mismatch.

## Boundaries

This is exploratory decomposition in a VitalDB-informed reconstructed simulation. It is not an independent confirmation cohort, external validation, clinical evidence, bedside dosing advice, a factorial estimate of individual biological feature effects, or a formal invariance/equivalence test. No validation outcome changes state definitions, budget, checkpoint choice, or the frozen retention rule.
