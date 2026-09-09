# Day 05 feature-group 개발 스크리닝 결과

## 결론

Pareto non-dominated states: S0, S_CUM, S_CONC, S_CORE, S1. 사전 지정 정렬 규칙에 따른 다음 개발 후보는 S_CONC, S_CORE이다. S_CONC가 두 profile 모두에서 가장 낮고 가장 안정적인 MAE를 보였고, S_CORE는 P1에서는 비슷했지만 P0 한 seed에서 불안정했다. 이는 재구성 simulation의 개발 추천이며 임상적 최적 상태나 생물학적 인과 효과가 아니다.

## 상태별 최종 MAE

| State | P0 mean ± SD (worst) | P1 mean ± SD (worst) | Extra features |
|---|---:|---:|---:|
| S0 | 7.3396 ± 1.5875 (10.1273) | 8.2575 ± 3.1697 (13.9188) | 0 |
| S_CUM | 5.4332 ± 0.2488 (5.8331) | 6.0290 ± 1.2550 (8.2461) | 2 |
| S_CONC | 2.8971 ± 0.2257 (3.0951) | 2.9110 ± 0.1241 (3.0442) | 4 |
| S_CORE | 3.6775 ± 1.7387 (6.7738) | 2.9053 ± 0.2419 (3.2968) | 6 |
| S1 | 4.4328 ± 3.1221 (9.9994) | 4.6484 ± 3.5379 (10.9573) | 8 |

## 고정 feature-group contrasts

| Profile | Contrast | Mean ± seed SD | Favorable seeds | Five paired seed values |
|---|---|---:|---:|---|
| P0 | cumulative_only | -1.9064 ± 1.6416 | 5/5 | -1.5991, -4.6978, -1.5399, -0.3396, -1.3558 |
| P0 | concentration_only | -4.4425 ± 1.5758 | 5/5 | -3.8871, -7.1147, -4.4776, -3.0776, -3.6555 |
| P0 | combined_core | -3.6621 ± 2.8324 | 4/5 | -3.9119, -7.3782, -3.8137, +0.6011, -3.8077 |
| P0 | cumulative_given_concentrations | +0.7804 ± 1.6599 | 3/5 | -0.0247, -0.2634, +0.6639, +3.6787, -0.1522 |
| P0 | concentration_given_cumulative | -1.7556 ± 1.5157 | 4/5 | -2.3128, -2.6803, -2.2738, +0.9407, -2.4520 |
| P0 | group_interaction | +2.6869 ± 1.4576 | 0/5 | +1.5743, +4.4344, +2.2038, +4.0183, +1.2035 |
| P0 | redundant_recent | +0.7552 ± 3.7782 | 2/5 | -0.1921, +0.6174, +6.8226, -3.6012, +0.1295 |
| P0 | original_full_state | -2.9068 ± 3.6016 | 4/5 | -4.1040, -6.7608, +3.0089, -3.0001, -3.6782 |
| P0 | cumulative_main | -0.5630 ± 1.4812 | 4/5 | -0.8119, -2.4806, -0.4380, +1.6696, -0.7540 |
| P0 | concentration_main | -3.0991 ± 1.3635 | 5/5 | -3.1000, -4.8975, -3.3757, -1.0685, -3.0537 |
| P1 | cumulative_only | -2.2285 ± 3.6359 | 4/5 | -1.5310, -8.2902, +1.5579, -1.6218, -1.2572 |
| P1 | concentration_only | -5.3465 ± 3.1121 | 5/5 | -3.7970, -10.9108, -3.9598, -4.0635, -4.0015 |
| P1 | combined_core | -5.3522 ± 2.9563 | 5/5 | -3.8770, -10.6220, -3.8438, -4.4588, -3.9596 |
| P1 | cumulative_given_concentrations | -0.0057 ± 0.2555 | 2/5 | -0.0800, +0.2888, +0.1159, -0.3953, +0.0419 |
| P1 | concentration_given_cumulative | -3.1238 ± 1.2924 | 5/5 | -2.3460, -2.3318, -5.4017, -2.8370, -2.7023 |
| P1 | group_interaction | +2.2227 ± 3.7507 | 1/5 | +1.4510, +8.5789, -1.4419, +1.2264, +1.2992 |
| P1 | redundant_recent | +1.7431 ± 3.5668 | 1/5 | -0.1628, +0.1029, +8.1129, +0.3873, +0.2751 |
| P1 | original_full_state | -3.6091 ± 5.2492 | 4/5 | -4.0397, -10.5191, +4.2690, -4.0715, -3.6845 |
| P1 | cumulative_main | -1.1171 ± 1.7679 | 4/5 | -0.8055, -4.0007, +0.8369, -1.0085, -0.6076 |
| P1 | concentration_main | -4.2351 ± 1.4699 | 5/5 | -3.0715, -6.6213, -4.6807, -3.4502, -3.3519 |

## Seed 50 prespecified diagnostic

아래 값은 Day04 development-validation subset의 final checkpoint 결과(MAE / propofol mg / BIS 40–60 fraction)다.

| State | P0 MAE / propofol / in-range | P1 MAE / propofol / in-range |
|---|---:|---:|
| S0 | 6.9904 / 240.6 / 0.790 | 6.6883 / 201.9 / 0.837 |
| S_CUM | 5.4505 / 212.5 / 0.865 | 8.2461 / 258.5 / 0.691 |
| S_CONC | 2.5128 / 216.2 / 0.939 | 2.7285 / 213.5 / 0.933 |
| S_CORE | 3.1767 / 211.3 / 0.918 | 2.8445 / 216.8 / 0.930 |
| S1 | 9.9994 / 165.4 / 0.527 | 10.9573 / 161.6 / 0.486 |

### Seed 50 train-only checkpoint trajectory

각 셀은 131k → 262k → 393k → 524k checkpoint의 `MAE / propofol mg` 순서다. 이는 32-subject train-only diagnostic이며 checkpoint 선택에 쓰지 않았다.

| State | P0 trajectory | P1 trajectory |
|---|---|---|
| S0 | 7.64/190 → 5.97/217 → 6.23/223 → 6.80/241 | 11.80/181 → 6.49/207 → 6.58/215 → 6.37/206 |
| S_CUM | 5.09/208 → 5.23/217 → 4.83/208 → 5.44/214 | 4.90/205 → 4.88/210 → 6.90/242 → 7.84/255 |
| S_CONC | 2.83/217 → 2.56/219 → 2.79/227 → 2.57/219 | 3.02/213 → 2.71/215 → 2.77/220 → 2.77/216 |
| S_CORE | 2.85/224 → 2.97/217 → 2.96/219 → 3.18/213 | 2.82/215 → 2.90/217 → 3.07/220 → 2.83/219 |
| S1 | 10.13/168 → 10.41/167 → 8.23/179 → 9.68/167 | 10.73/167 → 11.91/156 → 10.32/168 → 10.70/164 |

S_CONC와 S_CORE는 seed 50의 P0/P1 저투여 패턴을 제거했다. S1은 후기 학습에서 propofol이 감소하며 실패가 재현됐고, S_CUM은 P0에서는 개선됐지만 P1 final에서는 반대로 높은 dose와 낮은 in-range fraction을 보였다. 따라서 모든 enriched state가 같은 실패를 공유한다는 가설은 지지되지 않으며, concentration group이 가장 직접적인 안정화 신호다.

## Guardrails and PI comparison

Boundary와 diagnostic action-clip은 이 구현에서 같은 사건을 집계해 값이 일치한다. 아래 boundary는 seed mean / seed maximum이며, Tanh는 seed maximum이다.

| Profile-state | Boundary mean / max | Core clip mean | Tanh max | MAE − Day04 PI |
|---|---:|---:|---:|---:|
| P0S0 | 0.0092 / 0.0229 | 0.0 | 0.00e+00 | +1.3344 |
| P0S_CUM | 0.0165 / 0.0231 | 0.0 | 0.00e+00 | -0.5720 |
| P0S_CONC | 0.0319 / 0.0365 | 0.0 | 1.36e-05 | -3.1081 |
| P0S_CORE | 0.0308 / 0.0392 | 0.0 | 0.00e+00 | -2.3276 |
| P0S1 | 0.0294 / 0.0330 | 0.0 | 3.12e-05 | -1.5724 |
| P1S0 | 0.0028 / 0.0105 | 0.0 | 0.00e+00 | +1.3078 |
| P1S_CUM | 0.0098 / 0.0188 | 0.0 | 0.00e+00 | -0.9206 |
| P1S_CONC | 0.0338 / 0.0371 | 0.0 | 0.00e+00 | -4.0387 |
| P1S_CORE | 0.0320 / 0.0361 | 0.0 | 9.04e-07 | -4.0444 |
| P1S1 | 0.0298 / 0.0328 | 0.0 | 3.62e-06 | -2.3013 |

Day04 tuned PI baselines were P0 6.0052 and P1 6.9496 MAE. S_CONC and S_CORE improved on PI in both profiles; S_CUM improved on PI in both on average; S0 was worse. Core clipping was zero throughout, Tanh saturation was at most 3.12e-05, and the largest state/profile seed-level boundary rate was 0.0392. These are simulation guardrails, not evidence of clinical controller superiority.

Raw/physical action quantiles, scaled group magnitudes, and all checkpoint diagnostics are retained in the public CSV/JSON artifacts. Validation used only final checkpoints and selected no checkpoint.

Untouched future internal-confirmation reserve: 170 subjects / 170 cases; membership identifiers were hashed, bundle access 0, outcome access 0. Original-test access remained 0.

## Pareto interpretation

All five states are non-dominated when feature count and guardrails are included. The frozen tie-break chooses S_CONC, S_CORE. S_CONC is the primary candidate because its across-profile mean MAE, worst-seed MAE, and across-seed SD are 2.9040, 3.0696, and 0.1728. S_CORE remains secondary but its P0 seed instability argues against claiming complementarity.

## Neural Networks novelty implication

The result localizes the useful representation signal to the four concentration summaries: S_CONC improved all 10 matched profile-seed comparisons versus S0 and avoided seed 50 underdosing, while cumulative-only and redundant recent summaries were less stable. The next frozen development ablation may split Cp versus Ce and drug-specific concentration groups. Feature evidence alone does not establish architectural novelty; a Neural Networks contribution would still require a genuinely new group-structured encoder/normalizer or stability method and a later untouched-reserve confirmation. No follow-up is launched here.

## Limitations

The Day 04 development-validation subset was reused after inspection, so this is exploratory screening rather than confirmation. The source cohort is internal and reused. Contrasts compare independently retrained policies and are not biological or clinical causal effects. This is reconstructed simulation, not clinical, bedside-dosing, or external-validation evidence.

## Reproduction

30 new jobs, 15,728,640 new timesteps, 120 new checkpoints; one worker and one Torch thread; failures/retries 0/0. Wall time: 8.12 h. Resume: `.venv-journal\Scripts\python.exe scripts\journal\run_day05_feature_groups.py resume`.
