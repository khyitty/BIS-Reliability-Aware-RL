# Day 05 Overleaf note

- Exploratory development screening in reconstructed PK/PD simulation; not clinical, causal, or external-validation evidence.
- States: S0, S_CUM, S_CONC, S_CORE, S1; P0/P1; five paired seeds; 1,800 s; 524,288 steps per new job.
- MAE mean ± seed SD (P0 / P1): S0 7.3396 ± 1.5875 / 8.2575 ± 3.1697; S_CUM 5.4332 ± 0.2488 / 6.0290 ± 1.2550; S_CONC 2.8971 ± 0.2257 / 2.9110 ± 0.1241; S_CORE 3.6775 ± 1.7387 / 2.9053 ± 0.2419; S1 4.4328 ± 3.1221 / 4.6484 ± 3.5379.
- Concentration-only (S_CONC) was favorable versus S0 in all 10 paired profile-seed comparisons, had the lowest across-profile mean and worst-seed MAE, and removed the prespecified seed-50 underdosing pattern. S_CORE was similar under P1 but unstable in one P0 seed; redundant recent summaries in S1 reproduced seed-50 underdosing.
- All states were Pareto non-dominated when feature count and guardrails were included; frozen-rule development candidates: S_CONC, S_CORE.
- Core clipping was zero and Tanh saturation negligible; this does not establish clinical controller superiority or causal feature effects.
- Future reserve: 170 subjects / 170 cases, zero bundle/outcome access; original-test access zero.
