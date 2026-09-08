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
    if any(payload.get("test_access_count") != 0 for payload in (baseline, ppo, trajectory)):
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
    public = {
        "protocol_id": config["protocol_id"], "evidence_scope": config["evidence_scope"], "config_sha256": config_hash,
        "source_git_sha": preparation["source_git_sha"], "training_job_count": queue["total_jobs"], "training_target_timesteps_per_job": config["training_target_timesteps"],
        "checkpoint_interval_timesteps": config["checkpoint_interval_timesteps"], "benchmark_steps_per_second": benchmark["steps_per_second"],
        "development_train_subject_count": preparation["full_train_subject_count"], "development_train_case_count": preparation["full_train_case_count"],
        "bounded_training_subject_count": preparation["bounded_train_subject_count"], "validation_subject_count": preparation["bounded_validation_subject_count"],
        "validation_case_count": preparation["bounded_validation_case_count"], "test_access_count": 0,
        "condition_summary": condition_summary, "paired_seed_contrasts": contrasts,
        "pi_baseline_comparison": baseline_comparison, "learning_trajectory_summary": trajectory_summary,
        "descriptive_lowest_mae_condition": best["condition_id"], "descriptive_lowest_mae": best["latent_bis_mae_mean"],
        "baseline_selection": baseline["selection"], "learning_trajectory_row_count": len(trajectory["rows"]),
        "claims_boundary": "VitalDB-informed reconstructed simulation; not patient outcomes or clinical intervention evidence",
    }
    atomic_json(REPORT_JSON, public)
    table = "\n".join(f"| {row['condition_id']} | {row['latent_bis_mae_mean']:.3f} ± {row['latent_bis_mae_seed_sd']:.3f} | {100*row['time_in_40_60_fraction_mean']:.1f}% | {row['total_propofol_mg_mean']:.1f} |" for row in condition_summary)
    effects = "\n".join(f"| {row['effect']} | {row['left']} − {row['right']} | {row['mean_paired_seed_difference']:+.3f} ± {row['seed_sd']:.3f} |" for row in contrasts)
    baseline_table = "\n".join(f"| {row['condition_id']} | {row['ppo_mae_seed_mean']:.3f} | {row['pi_mae']:.3f} | {row['ppo_minus_pi_mae']:+.3f} |" for row in baseline_comparison)
    trajectory_table = "\n".join(f"| {row['timestep']:,} | {row['s0_training_subset_mae_mean']:.3f} | {row['s1_training_subset_mae_mean']:.3f} |" for row in trajectory_summary)
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
