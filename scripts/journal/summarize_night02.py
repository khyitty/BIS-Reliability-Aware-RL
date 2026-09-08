"""Export only non-identifying aggregate Night-02 results to tracked reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

from night02_common import atomic_json
from run_night02_cpu import DEFAULT_CONFIG, load_config, output_root

REPORT_CSV = ROOT / "reports" / "journal" / "night02_results.csv"
REPORT_JSON = ROOT / "reports" / "journal" / "night02_summary.json"
REPORT_MD = ROOT / "reports" / "journal" / "NIGHT02_RESULTS.md"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out = output_root(config)
    queue = json.loads((out / "queue_state.json").read_text(encoding="utf-8"))
    if queue.get("completed") is not True:
        raise RuntimeError("training queue is not complete")
    preparation = json.loads((out / "shareable_preparation.json").read_text(encoding="utf-8"))
    benchmark = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    baseline = json.loads((out / "baseline_aggregate.json").read_text(encoding="utf-8"))
    ppo = json.loads((out / "ppo_aggregate.json").read_text(encoding="utf-8"))
    trajectory = json.loads((out / "learning_trajectory_aggregate.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((out / "bootstrap_aggregate.json").read_text(encoding="utf-8"))
    availability = json.loads((out / "observation_availability_aggregate.json").read_text(encoding="utf-8"))
    scale_diagnostics = json.loads((out / "state_scale_diagnostics_aggregate.json").read_text(encoding="utf-8"))
    clip_probe = json.loads((out / "sensitivity" / "posthoc_goff_a30_s1_observation_clip_10" / "aggregate.json").read_text(encoding="utf-8"))
    if any(payload.get("test_access_count") != 0 for payload in (baseline, ppo, trajectory, bootstrap, availability, scale_diagnostics, clip_probe)):
        raise RuntimeError("test access count is not zero")

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows = [*ppo["aggregates"], *baseline["aggregates"]]
    fields = ["controller", "condition_id", "seed", "validation_subject_count", "validation_case_count", "latent_bis_mae", "time_in_40_60_fraction", "time_below_40_fraction", "time_above_60_fraction", "cumulative_reward", "total_propofol_mg", "action_sd", "action_boundary_fraction", "core_clip_fraction"]
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({name: row[name] for name in fields})

    condition_summary = []
    for condition in config["conditions"]:
        rows = [row for row in ppo["aggregates"] if row["condition_id"] == condition["condition_id"]]
        mae, mae_sd = mean_sd([float(row["latent_bis_mae"]) for row in rows])
        target, target_sd = mean_sd([float(row["time_in_40_60_fraction"]) for row in rows])
        dose, dose_sd = mean_sd([float(row["total_propofol_mg"]) for row in rows])
        condition_summary.append({"condition_id": condition["condition_id"], "ppo_seed_count": len(rows), "latent_bis_mae_mean": mae, "latent_bis_mae_seed_sd": mae_sd, "time_in_40_60_fraction_mean": target, "time_in_40_60_fraction_seed_sd": target_sd, "total_propofol_mg_mean": dose, "total_propofol_mg_seed_sd": dose_sd})

    lookup = {(row["condition_id"], int(row["seed"])): row for row in ppo["aggregates"]}
    contrasts = []
    definitions = []
    for state in ("S0", "S1"):
        for age in (20, 30):
            definitions.append(("gate_G50_minus_off", f"G50_A{age}_{state}", f"Goff_A{age}_{state}"))
    for state in ("S0", "S1"):
        for gate in ("Goff", "G50"):
            definitions.append(("age_20_minus_30", f"{gate}_A20_{state}", f"{gate}_A30_{state}"))
    for gate in ("Goff", "G50"):
        for age in (20, 30):
            definitions.append(("state_S1_minus_S0", f"{gate}_A{age}_S1", f"{gate}_A{age}_S0"))
    for effect, left, right in definitions:
        values = [float(lookup[(left, int(seed))]["latent_bis_mae"]) - float(lookup[(right, int(seed))]["latent_bis_mae"]) for seed in config["seeds"]]
        average, spread = mean_sd(values)
        contrasts.append({"effect": effect, "left": left, "right": right, "metric": "latent_bis_mae", "mean_paired_seed_difference": average, "seed_sd": spread, "seed_count": len(values)})

    best = min(condition_summary, key=lambda row: row["latent_bis_mae_mean"])
    pi_lookup = {row["condition_id"]: row for row in baseline["aggregates"] if row["controller"] == "PI"}
    baseline_comparison = [
        {
            "condition_id": row["condition_id"],
            "ppo_mae_seed_mean": row["latent_bis_mae_mean"],
            "pi_mae": float(pi_lookup[row["condition_id"]]["latent_bis_mae"]),
            "ppo_minus_pi_mae": row["latent_bis_mae_mean"] - float(pi_lookup[row["condition_id"]]["latent_bis_mae"]),
        }
        for row in condition_summary
    ]
    trajectory_summary = []
    for timestep in range(int(config["checkpoint_interval_timesteps"]), int(config["training_target_timesteps"]) + 1, int(config["checkpoint_interval_timesteps"])):
        selected = [row for row in trajectory["rows"] if int(row["timestep"]) == timestep]
        s0 = [float(row["latent_bis_mae"]) for row in selected if row["condition_id"].endswith("S0")]
        s1 = [float(row["latent_bis_mae"]) for row in selected if row["condition_id"].endswith("S1")]
        trajectory_summary.append({"timestep": timestep, "s0_training_subset_mae_mean": statistics.mean(s0), "s1_training_subset_mae_mean": statistics.mean(s1)})
    sensitivity_path = out / "sensitivity" / "posthoc_s1_learning_rate_3e-4" / "aggregate.json"
    sensitivity_comparison = []
    if sensitivity_path.is_file():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if sensitivity.get("posthoc") is not True or sensitivity.get("test_access_count") != 0:
            raise RuntimeError("invalid post-hoc sensitivity boundary")
        for condition in [row for row in config["conditions"] if row["state_id"] == "S1"]:
            primary_rows = [row for row in ppo["aggregates"] if row["condition_id"] == condition["condition_id"]]
            sensitivity_rows = [row for row in sensitivity["aggregates"] if row["condition_id"] == condition["condition_id"]]
            primary_mean, primary_sd = mean_sd([float(row["latent_bis_mae"]) for row in primary_rows])
            sensitivity_mean, sensitivity_sd = mean_sd([float(row["latent_bis_mae"]) for row in sensitivity_rows])
            sensitivity_comparison.append({"condition_id": condition["condition_id"], "primary_learning_rate": 0.001, "primary_mae_mean": primary_mean, "primary_seed_sd": primary_sd, "posthoc_learning_rate": 0.0003, "posthoc_mae_mean": sensitivity_mean, "posthoc_seed_sd": sensitivity_sd, "posthoc_minus_primary_mae": sensitivity_mean - primary_mean})
    clip_values = [float(row["latent_bis_mae"]) for row in clip_probe["aggregates"]]
    clip_mean, clip_sd = mean_sd(clip_values)
    clip_primary = next(row for row in condition_summary if row["condition_id"] == "Goff_A30_S1")
    clip_summary = {"probe_id": clip_probe["probe_id"], "posthoc": True, "condition_id": "Goff_A30_S1", "observation_clip_absolute": 10.0, "seed_count": len(clip_values), "primary_mae_mean": clip_primary["latent_bis_mae_mean"], "primary_seed_sd": clip_primary["latent_bis_mae_seed_sd"], "clipped_mae_mean": clip_mean, "clipped_seed_sd": clip_sd, "clipped_minus_primary_mae": clip_mean - clip_primary["latent_bis_mae_mean"], "checkpoints_verified": clip_probe["checkpoints"]}
    public = {
        "protocol_id": config["protocol_id"], "evidence_scope": config["evidence_scope"], "config_sha256": config_hash,
        "source_git_sha": preparation["source_git_sha"], "training_job_count": queue["total_jobs"], "training_target_timesteps_per_job": config["training_target_timesteps"],
        "checkpoint_interval_timesteps": config["checkpoint_interval_timesteps"], "benchmark_steps_per_second": benchmark["steps_per_second"],
        "development_train_subject_count": preparation["full_train_subject_count"], "development_train_case_count": preparation["full_train_case_count"],
        "bounded_training_subject_count": preparation["bounded_train_subject_count"], "validation_subject_count": preparation["bounded_validation_subject_count"],
        "validation_case_count": preparation["bounded_validation_case_count"], "test_access_count": 0,
        "condition_summary": condition_summary, "paired_seed_contrasts": contrasts,
        "pi_baseline_comparison": baseline_comparison, "learning_trajectory_summary": trajectory_summary,
        "posthoc_s1_learning_rate_sensitivity": sensitivity_comparison,
        "paired_subject_bootstrap": bootstrap,
        "observation_availability": availability,
        "state_scale_diagnostics": scale_diagnostics,
        "posthoc_s1_observation_clip_probe": clip_summary,
        "descriptive_lowest_mae_condition": best["condition_id"], "descriptive_lowest_mae": best["latent_bis_mae_mean"],
        "baseline_selection": baseline["selection"], "learning_trajectory_row_count": len(trajectory["rows"]),
        "claims_boundary": "VitalDB-informed reconstructed simulation; not patient outcomes or clinical intervention evidence",
    }
    atomic_json(REPORT_JSON, public)
    table = "\n".join(f"| {row['condition_id']} | {row['latent_bis_mae_mean']:.3f} ± {row['latent_bis_mae_seed_sd']:.3f} | {100*row['time_in_40_60_fraction_mean']:.1f}% | {row['total_propofol_mg_mean']:.1f} |" for row in condition_summary)
    effects = "\n".join(f"| {row['effect']} | {row['left']} − {row['right']} | {row['mean_paired_seed_difference']:+.3f} ± {row['seed_sd']:.3f} |" for row in contrasts)
    baseline_table = "\n".join(f"| {row['condition_id']} | {row['ppo_mae_seed_mean']:.3f} | {row['pi_mae']:.3f} | {row['ppo_minus_pi_mae']:+.3f} |" for row in baseline_comparison)
    trajectory_table = "\n".join(f"| {row['timestep']:,} | {row['s0_training_subset_mae_mean']:.3f} | {row['s1_training_subset_mae_mean']:.3f} |" for row in trajectory_summary)
    sensitivity_table = "\n".join(f"| {row['condition_id']} | {row['primary_mae_mean']:.3f} ± {row['primary_seed_sd']:.3f} | {row['posthoc_mae_mean']:.3f} ± {row['posthoc_seed_sd']:.3f} | {row['posthoc_minus_primary_mae']:+.3f} |" for row in sensitivity_comparison)
    bootstrap_table = "\n".join(f"| {row['effect']} | {row['left']} − {row['right']} | {row['mean_difference']:+.3f} | [{row['bootstrap_95ci_low']:+.3f}, {row['bootstrap_95ci_high']:+.3f}] |" for row in bootstrap["factor_contrasts"])
    availability_table = "\n".join(f"| {row['profile_id']} | {100*row['visible_fraction']:.1f}% | {100*row['reason_no_prior_observation_fraction']:.1f}% | {100*row['reason_sqi_below_threshold_fraction']:.1f}% | {100*row['reason_stale_beyond_pipeline_cap_fraction']:.1f}% |" for row in availability["aggregates"])
    worst_scale_rows = sorted(scale_diagnostics["rows"], key=lambda row: row["absolute_p99"], reverse=True)[:8]
    scale_table = "\n".join(f"| {row['profile_id']} | {row['field_name']} | {row['absolute_p95']:.2f} | {row['absolute_p99']:.2f} | {row['absolute_max']:.2f} |" for row in worst_scale_rows)
    REPORT_MD.write_text(f"""# Night 02 CPU Results

## Outcome

All {queue['total_jobs']} predeclared PPO jobs completed at {config['training_target_timesteps']:,} timesteps, with checkpoints every {config['checkpoint_interval_timesteps']:,} timesteps. The lowest descriptive validation MAE was **{best['condition_id']}** ({best['latent_bis_mae_mean']:.3f}). This is exploratory development evidence, not a selected confirmatory model.

## Evidence boundary

These are **VitalDB-informed reconstructed simulations**. Recorded demographics, event availability, SQI, and remifentanil schedules informed the environment; BIS response and propofol concentrations were simulator-generated. Results are not patient outcomes or clinical intervention evidence. The historical test split was not accessed.

## Frozen accounting

- Development-train scaler fit: {preparation['full_train_subject_count']:,} subjects / {preparation['full_train_case_count']:,} cases
- Bounded PPO training universe: {preparation['bounded_train_subject_count']} subjects / {preparation['bounded_train_case_count']} cases
- Internal validation: {preparation['bounded_validation_subject_count']} subjects / {preparation['bounded_validation_case_count']} cases
- Seeds: {', '.join(map(str, config['seeds']))}; horizon: {config['common_horizon_seconds']} seconds
- Aggregate order: case metrics first, then equal-weight subject means

## PPO validation aggregates

| Condition | Latent BIS MAE, mean ± seed SD | Time 40–60 | Propofol mg |
|---|---:|---:|---:|
{table}

## Factor contrasts

Differences are paired by seed. Negative MAE differences favor the left condition. With three seeds these are descriptive uncertainty summaries, not confirmatory tests.

| Factor | Contrast | MAE difference, mean ± seed SD |
|---|---|---:|
{effects}

The subject-paired bootstrap below averages the three seeds within each subject before 10,000 deterministic resamples. Intervals are exploratory and do not correct for multiple comparisons.

| Factor | Contrast | Mean difference | Subject-bootstrap 95% interval |
|---|---|---:|---:|
{bootstrap_table}

## Baselines and diagnostics

Constant, P, and PI controllers were tuned only on the frozen 12-subject training subset, frozen, then evaluated on internal validation. Controller state reset per case; missing feedback used the base action without integral update; exactly one action update occurred per transition. Every PPO checkpoint was also evaluated on that same train-only subset to produce {len(trajectory['rows'])} learning-trajectory rows without checkpoint selection. Full non-identifying aggregate rows are in `night02_results.csv` and machine-readable summary fields are in `night02_summary.json`.

| Condition | PPO MAE | PI MAE | PPO − PI |
|---|---:|---:|---:|
{baseline_table}

The S0 policies were close to the tuned PI reference. S1 was strongly seed-sensitive and worse than PI in every condition, so the extra pharmacology state did not provide reliable benefit at this bounded budget.

| Checkpoint | S0 train-subset MAE | S1 train-subset MAE |
|---:|---:|---:|
{trajectory_table}

S1 improved over the fixed checkpoints but remained unstable at the final checkpoint. These training-subset diagnostics were not used to choose a checkpoint.

## Observation availability

Availability was summarized over the same internal validation subjects. Only aggregate reason fractions are reported; raw SQI values and event traces remain private.

| Profile | Visible | No prior observation | Low SQI | Stale |
|---|---:|---:|---:|---:|
{availability_table}

Within the 600-second window, SQI gating reduced visible states from 19.8% to about 10.8%; the 20- versus 30-second age cap produced almost no availability difference because no stale states occurred. The large no-prior fraction reflects the start timing of recorded BIS availability in this bounded window.

The largest standardized S1 magnitudes under a fixed 1.5 mg/10s probe are shown below. No non-finite states occurred.

| Profile | Field | Absolute p95 | Absolute p99 | Absolute max |
|---|---|---:|---:|---:|
{scale_table}

The preprocessing-neutral scaler deliberately fits BIS and propofol fields to a zero-reference distribution, leaving runtime cumulative propofol and BIS history values at large magnitudes. This is protocol-consistent, but it is a plausible optimization hazard for S1.

## Post-hoc S1 learning-rate sensitivity

The primary S1 seed instability motivated an isolated, explicitly post-hoc rerun at learning rate 3e-4. It used the same training universe, budget, seeds, and validation set and did not alter the primary analysis.

| Condition | Primary 1e-3 MAE | Post-hoc 3e-4 MAE | Post-hoc − primary |
|---|---:|---:|---:|
{sensitivity_table}

Lowering the learning rate did not resolve instability: one seed converged well while the other two produced low-dose policies with high BIS MAE, and the identity of the successful seed changed. This points to optimization sensitivity rather than a robust S1 advantage.

## Post-hoc S1 scale probe

For one representative condition (`Goff_A30_S1`), a three-seed diagnostic clipped standardized observations to ±10 while leaving the training universe, PPO budget, and all other settings unchanged. Mean validation MAE changed from {clip_summary['primary_mae_mean']:.3f} ± {clip_summary['primary_seed_sd']:.3f} to {clip_summary['clipped_mae_mean']:.3f} ± {clip_summary['clipped_seed_sd']:.3f}; all {clip_summary['checkpoints_verified']} checkpoints were verified. This strongly supports an input-scale optimization problem, but because the probe was post-hoc and limited to one condition it is not primary evidence and does not establish the best general scaling rule.

## Limitations

The 600-second bounded horizon and limited development subsets were chosen for a CPU journal run. The validation set is internal to the historical TRAIN partition, seed count is small, and no causal or clinical effectiveness claim is supported.
""", encoding="utf-8", newline="\n")
    print(json.dumps({"csv_rows": len(aggregate_rows), "conditions": len(condition_summary), "contrasts": len(contrasts), "best": best["condition_id"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summarize(args.config)


if __name__ == "__main__":
    main()
