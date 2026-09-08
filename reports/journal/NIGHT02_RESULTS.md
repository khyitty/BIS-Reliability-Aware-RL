# Night 02 CPU Results

## Outcome

All 24 predeclared PPO jobs completed at 262,144 timesteps, with checkpoints every 65,536 timesteps. The lowest descriptive validation MAE was **G50_A20_S0** (12.090). This is exploratory development evidence, not a selected confirmatory model.

## Evidence boundary

These are **VitalDB-informed reconstructed simulations**. Recorded demographics, event availability, SQI, and remifentanil schedules informed the environment; BIS response and propofol concentrations were simulator-generated. Results are not patient outcomes or clinical intervention evidence. The historical test split was not accessed.

## Frozen accounting

- Development-train scaler fit: 1,642 subjects / 1,680 cases
- Bounded PPO training universe: 96 subjects / 96 cases
- Internal validation: 24 subjects / 24 cases
- Seeds: 45, 46, 47; horizon: 600 seconds
- Aggregate order: case metrics first, then equal-weight subject means

## PPO validation aggregates

| Condition | Latent BIS MAE, mean ± seed SD | Time 40–60 | Propofol mg |
|---|---:|---:|---:|
| Goff_A30_S0 | 12.227 ± 0.197 | 64.1% | 96.9 |
| Goff_A20_S0 | 12.483 ± 0.328 | 64.2% | 99.1 |
| G50_A30_S0 | 12.213 ± 0.208 | 64.0% | 94.4 |
| G50_A20_S0 | 12.090 ± 0.226 | 64.7% | 95.7 |
| Goff_A30_S1 | 25.581 ± 14.963 | 30.0% | 48.0 |
| Goff_A20_S1 | 21.923 ± 14.258 | 38.9% | 63.5 |
| G50_A30_S1 | 34.046 ± 10.294 | 12.7% | 29.1 |
| G50_A20_S1 | 29.299 ± 16.291 | 28.1% | 41.0 |

## Factor contrasts

Differences are paired by seed. Negative MAE differences favor the left condition. With three seeds these are descriptive uncertainty summaries, not confirmatory tests.

| Factor | Contrast | MAE difference, mean ± seed SD |
|---|---|---:|
| gate_G50_minus_off | G50_A20_S0 − Goff_A20_S0 | -0.393 ± 0.248 |
| gate_G50_minus_off | G50_A30_S0 − Goff_A30_S0 | -0.014 ± 0.200 |
| gate_G50_minus_off | G50_A20_S1 − Goff_A20_S1 | +7.376 ± 10.607 |
| gate_G50_minus_off | G50_A30_S1 − Goff_A30_S1 | +8.465 ± 4.875 |
| age_20_minus_30 | Goff_A20_S0 − Goff_A30_S0 | +0.256 ± 0.143 |
| age_20_minus_30 | G50_A20_S0 − G50_A30_S0 | -0.123 ± 0.321 |
| age_20_minus_30 | Goff_A20_S1 − Goff_A30_S1 | -3.657 ± 8.891 |
| age_20_minus_30 | G50_A20_S1 − G50_A30_S1 | -4.747 ± 6.011 |
| state_S1_minus_S0 | Goff_A20_S1 − Goff_A20_S0 | +9.440 ± 14.519 |
| state_S1_minus_S0 | Goff_A30_S1 − Goff_A30_S0 | +13.354 ± 15.153 |
| state_S1_minus_S0 | G50_A20_S1 − G50_A20_S0 | +17.209 ± 16.447 |
| state_S1_minus_S0 | G50_A30_S1 − G50_A30_S0 | +21.833 ± 10.429 |

The subject-paired bootstrap below averages the three seeds within each subject before 10,000 deterministic resamples. Intervals are exploratory and do not correct for multiple comparisons.

| Factor | Contrast | Mean difference | Subject-bootstrap 95% interval |
|---|---|---:|---:|
| gate_G50_minus_off | G50_A20_S0 − Goff_A20_S0 | -0.393 | [-0.729, -0.113] |
| gate_G50_minus_off | G50_A30_S0 − Goff_A30_S0 | -0.014 | [-0.139, +0.092] |
| gate_G50_minus_off | G50_A20_S1 − Goff_A20_S1 | +7.376 | [+6.377, +8.380] |
| gate_G50_minus_off | G50_A30_S1 − Goff_A30_S1 | +8.465 | [+6.844, +9.849] |
| age_20_minus_30 | Goff_A20_S0 − Goff_A30_S0 | +0.256 | [+0.072, +0.498] |
| age_20_minus_30 | G50_A20_S0 − G50_A30_S0 | -0.123 | [-0.221, -0.039] |
| age_20_minus_30 | Goff_A20_S1 − Goff_A30_S1 | -3.657 | [-4.108, -3.220] |
| age_20_minus_30 | G50_A20_S1 − G50_A30_S1 | -4.747 | [-5.915, -3.566] |
| state_S1_minus_S0 | Goff_A20_S1 − Goff_A20_S0 | +9.440 | [+7.663, +10.965] |
| state_S1_minus_S0 | Goff_A30_S1 − Goff_A30_S0 | +13.354 | [+11.847, +14.756] |
| state_S1_minus_S0 | G50_A20_S1 − G50_A20_S0 | +17.209 | [+15.221, +18.978] |
| state_S1_minus_S0 | G50_A30_S1 − G50_A30_S0 | +21.833 | [+19.093, +24.222] |

## Baselines and diagnostics

Constant, P, and PI controllers were tuned only on the frozen 12-subject training subset, frozen, then evaluated on internal validation. Controller state reset per case; missing feedback used the base action without integral update; exactly one action update occurred per transition. Every PPO checkpoint was also evaluated on that same train-only subset to produce 96 learning-trajectory rows without checkpoint selection. Full non-identifying aggregate rows are in `night02_results.csv` and machine-readable summary fields are in `night02_summary.json`.

| Condition | PPO MAE | PI MAE | PPO − PI |
|---|---:|---:|---:|
| Goff_A30_S0 | 12.227 | 11.897 | +0.330 |
| Goff_A20_S0 | 12.483 | 11.897 | +0.586 |
| G50_A30_S0 | 12.213 | 12.100 | +0.113 |
| G50_A20_S0 | 12.090 | 12.105 | -0.015 |
| Goff_A30_S1 | 25.581 | 11.897 | +13.684 |
| Goff_A20_S1 | 21.923 | 11.897 | +10.027 |
| G50_A30_S1 | 34.046 | 12.100 | +21.946 |
| G50_A20_S1 | 29.299 | 12.105 | +17.194 |

The S0 policies were close to the tuned PI reference. S1 was strongly seed-sensitive and worse than PI in every condition, so the extra pharmacology state did not provide reliable benefit at this bounded budget.

| Checkpoint | S0 train-subset MAE | S1 train-subset MAE |
|---:|---:|---:|
| 65,536 | 12.179 | 37.146 |
| 131,072 | 12.279 | 29.618 |
| 196,608 | 11.900 | 27.716 |
| 262,144 | 12.077 | 27.014 |

S1 improved over the fixed checkpoints but remained unstable at the final checkpoint. These training-subset diagnostics were not used to choose a checkpoint.

## Observation availability

Availability was summarized over the same internal validation subjects. Only aggregate reason fractions are reported; raw SQI values and event traces remain private.

| Profile | Visible | No prior observation | Low SQI | Stale |
|---|---:|---:|---:|---:|
| Goff_A30 | 19.8% | 80.2% | 0.0% | 0.0% |
| Goff_A20 | 19.8% | 80.2% | 0.0% | 0.0% |
| G50_A30 | 10.9% | 80.2% | 8.9% | 0.0% |
| G50_A20 | 10.8% | 80.2% | 9.0% | 0.0% |

Within the 600-second window, SQI gating reduced visible states from 19.8% to about 10.8%; the 20- versus 30-second age cap produced almost no availability difference because no stale states occurred. The large no-prior fraction reflects the start timing of recorded BIS availability in this bounded window.

The largest standardized S1 magnitudes under a fixed 1.5 mg/10s probe are shown below. No non-finite states occurred.

| Profile | Field | Absolute p95 | Absolute p99 | Absolute max |
|---|---|---:|---:|---:|
| Goff_A30 | propofol_cumulative_dose_mg | 85.50 | 90.00 | 90.00 |
| Goff_A20 | propofol_cumulative_dose_mg | 85.50 | 90.00 | 90.00 |
| G50_A30 | propofol_cumulative_dose_mg | 85.50 | 90.00 | 90.00 |
| G50_A20 | propofol_cumulative_dose_mg | 85.50 | 90.00 | 90.00 |
| Goff_A30 | bis_value_t-50 | 56.90 | 84.23 | 97.63 |
| Goff_A30 | bis_value_t-40 | 56.90 | 84.23 | 97.63 |
| Goff_A30 | bis_value_t-30 | 56.90 | 84.23 | 97.63 |
| Goff_A30 | bis_value_t-20 | 56.90 | 84.23 | 97.63 |

The preprocessing-neutral scaler deliberately fits BIS and propofol fields to a zero-reference distribution, leaving runtime cumulative propofol and BIS history values at large magnitudes. This is protocol-consistent, but it is a plausible optimization hazard for S1.

## Post-hoc S1 learning-rate sensitivity

The primary S1 seed instability motivated an isolated, explicitly post-hoc rerun at learning rate 3e-4. It used the same training universe, budget, seeds, and validation set and did not alter the primary analysis.

| Condition | Primary 1e-3 MAE | Post-hoc 3e-4 MAE | Post-hoc − primary |
|---|---:|---:|---:|
| Goff_A30_S1 | 25.581 ± 14.963 | 27.939 ± 18.438 | +2.358 |
| Goff_A20_S1 | 21.923 ± 14.258 | 27.870 ± 18.257 | +5.946 |
| G50_A30_S1 | 34.046 ± 10.294 | 28.611 ± 18.718 | -5.435 |
| G50_A20_S1 | 29.299 ± 16.291 | 29.395 ± 17.721 | +0.096 |

Lowering the learning rate did not resolve instability: one seed converged well while the other two produced low-dose policies with high BIS MAE, and the identity of the successful seed changed. This points to optimization sensitivity rather than a robust S1 advantage.

## Post-hoc S1 scale probe

For one representative condition (`Goff_A30_S1`), a three-seed diagnostic clipped standardized observations to ±10 while leaving the training universe, PPO budget, and all other settings unchanged. Mean validation MAE changed from 25.581 ± 14.963 to 6.522 ± 0.252; all 12 checkpoints were verified. This strongly supports an input-scale optimization problem, but because the probe was post-hoc and limited to one condition it is not primary evidence and does not establish the best general scaling rule.

## Limitations

The 600-second bounded horizon and limited development subsets were chosen for a CPU journal run. The validation set is internal to the historical TRAIN partition, seed count is small, and no causal or clinical effectiveness claim is supported.
