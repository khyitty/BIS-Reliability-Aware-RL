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

## Limitations

The 600-second bounded horizon and limited development subsets were chosen for a CPU journal run. The validation set is internal to the historical TRAIN partition, seed count is small, and no causal or clinical effectiveness claim is supported.
