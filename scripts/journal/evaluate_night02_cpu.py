"""Tune bounded baselines and evaluate completed Night-02 PPO checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

import numpy as np
from stable_baselines3 import PPO

from night02_common import aggregate_subject_rows, atomic_json, make_environment, truncate_bundle
from run_night02_cpu import DEFAULT_CONFIG, load_config, load_private, verify_checkpoint

METRICS = (
    "latent_bis_mae", "time_in_40_60_fraction", "time_below_40_fraction",
    "time_above_60_fraction", "cumulative_reward", "total_propofol_mg",
    "action_sd", "action_boundary_fraction", "core_clip_fraction",
)


@dataclass
class Controller:
    family: str
    base: float
    kp: float = 0.0
    ki: float = 0.0
    integral: float = 0.0
    calls: int = 0

    def reset(self) -> None:
        self.integral = 0.0
        self.calls = 0

    def predict_once(self, info: dict[str, Any]) -> float:
        self.calls += 1
        if not bool(info["visible_current_bis_mask"]):
            return float(np.clip(self.base, 0.0, 27.7))
        error = float(info["visible_current_bis_value"]) - 50.0
        candidate_integral = float(np.clip(self.integral + error * 10.0, -3000.0, 3000.0))
        raw = self.base + self.kp * error + self.ki * candidate_integral
        action = float(np.clip(raw, 0.0, 27.7))
        if self.family == "PI" and (action == raw or (action <= 0.0 and error > 0) or (action >= 27.7 and error < 0)):
            self.integral = candidate_integral
        return action

    def public(self) -> dict[str, Any]:
        return {"family": self.family, "base": self.base, "kp": self.kp, "ki": self.ki}


def case_metrics(latent: list[float], actions: list[float], rewards: list[float], clips: int) -> dict[str, float]:
    bis = np.asarray(latent, dtype=float)
    dose = np.asarray(actions, dtype=float)
    return {
        "latent_bis_mae": float(np.mean(np.abs(bis - 50.0))),
        "time_in_40_60_fraction": float(np.mean((bis >= 40.0) & (bis <= 60.0))),
        "time_below_40_fraction": float(np.mean(bis < 40.0)),
        "time_above_60_fraction": float(np.mean(bis > 60.0)),
        "cumulative_reward": float(np.sum(rewards)),
        "total_propofol_mg": float(np.sum(dose)),
        "action_sd": float(np.std(dose)),
        "action_boundary_fraction": float(np.mean((dose <= 1e-8) | (dose >= 27.7 - 1e-8))),
        "core_clip_fraction": float(clips / len(dose)),
    }


def run_case(store: Any, caseid: str, condition: dict[str, Any], scaler: Any, seed: int, horizon: float, policy: Callable[[np.ndarray, dict[str, Any]], float]) -> tuple[dict[str, float], int]:
    bundle = truncate_bundle(store.load_case(caseid), horizon)
    env = make_environment(bundle, condition, scaler, seed)
    observation, info = env.reset(seed=seed)
    latent: list[float] = []
    actions: list[float] = []
    rewards: list[float] = []
    clips = calls = 0
    done = False
    while not done:
        action = policy(observation, info)
        calls += 1
        observation, reward, terminated, truncated, info = env.step(np.asarray([action], dtype=np.float32))
        latent.append(float(info["latent_true_bis"]))
        actions.append(float(info["applied_action_mg_per_10s"]))
        rewards.append(float(reward))
        clips += int(info["action_was_clipped"])
        done = bool(terminated or truncated)
    env.close()
    if calls != len(actions):
        raise RuntimeError("controller call count differs from transition count")
    return case_metrics(latent, actions, rewards, clips), calls


def candidates() -> dict[str, list[Controller]]:
    constant = [Controller("constant", base) for base in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    p = [Controller("P", base, kp) for base in (1.0, 1.5, 2.0) for kp in (0.03, 0.06, 0.10, 0.15)]
    pi = [Controller("PI", base, kp, ki) for base in (1.0, 1.75) for kp in (0.04, 0.08, 0.12) for ki in (0.0002, 0.0006)]
    return {"constant": constant, "P": p, "PI": pi}


def tune_baselines(config_path: Path) -> None:
    config, _ = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    profiles: dict[tuple[Any, Any], dict[str, Any]] = {}
    for condition in config["conditions"]:
        profiles.setdefault((condition["sqi_threshold"], condition["age_seconds"]), condition)
    selection: dict[str, dict[str, Any]] = {}
    private_scores: list[dict[str, Any]] = []
    for condition in profiles.values():
        profile_id = condition["condition_id"].rsplit("_", 1)[0]
        for family, rows in candidates().items():
            scored = []
            for index, controller in enumerate(rows):
                values = []
                for caseid in membership["tune_cases"]:
                    controller.reset()
                    metrics, calls = run_case(store, caseid, condition, scalers[condition["state_id"]], 45, config["common_horizon_seconds"], lambda observation, info, c=controller: c.predict_once(info))
                    values.append(metrics["latent_bis_mae"])
                    private_scores.append({"profile_id": profile_id, "family": family, "candidate_index": index, "caseid": caseid, "mae": metrics["latent_bis_mae"], "controller_calls": calls})
                scored.append((float(np.mean(values)), index, controller))
            best_score, best_index, best = min(scored, key=lambda item: (item[0], item[1]))
            selection[f"{profile_id}/{family}"] = best.public() | {"tuning_subject_count": len(membership["tuning_subjects"]), "tuning_case_count": len(membership["tune_cases"]), "tuning_case_mean_mae": best_score, "candidate_count": len(rows)}
    atomic_json(out / "baseline_tuning_private.json", {"selection": selection, "case_scores": private_scores})

    validation_rows = []
    for condition in config["conditions"]:
        profile_id = condition["condition_id"].rsplit("_", 1)[0]
        for family in candidates():
            params = selection[f"{profile_id}/{family}"]
            for caseid in membership["validation_cases"]:
                controller = Controller(params["family"], params["base"], params["kp"], params["ki"])
                controller.reset()
                metrics, calls = run_case(store, caseid, condition, scalers[condition["state_id"]], 45, config["common_horizon_seconds"], lambda observation, info, c=controller: c.predict_once(info))
                subjectid = str(store._by_case[caseid]["subjectid"])
                validation_rows.append({"condition_id": condition["condition_id"], "seed": "deterministic", "controller": family, "caseid": caseid, "subjectid": subjectid, "controller_calls": calls, **metrics})
    atomic_json(out / "baseline_validation_private.json", {"rows": validation_rows})
    aggregates = []
    for condition in config["conditions"]:
        for family in candidates():
            rows = [row for row in validation_rows if row["condition_id"] == condition["condition_id"] and row["controller"] == family]
            aggregates.append({"condition_id": condition["condition_id"], "seed": "deterministic", "controller": family, "validation_subject_count": len({row["subjectid"] for row in rows}), "validation_case_count": len(rows), **aggregate_subject_rows(rows, list(METRICS))})
    atomic_json(out / "baseline_aggregate.json", {"selection": selection, "aggregates": aggregates, "case_to_subject_aggregation": True, "test_access_count": 0})
    print(json.dumps({"selected_controllers": len(selection), "aggregate_rows": len(aggregates), "validation_case_rows": len(validation_rows)}))


def evaluate_ppo(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    rows = []
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            directory = out / "jobs" / condition["condition_id"] / f"seed_{seed}"
            completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
            target = int(config["training_target_timesteps"])
            checkpoint = directory / f"checkpoint_{target:010d}"
            verify_checkpoint(checkpoint, {"condition_id": condition["condition_id"], "seed": seed, "timestep": target, "config_sha256": config_hash})
            model = PPO.load(str(checkpoint / "model.zip"), device="cpu")
            for caseid in membership["validation_cases"]:
                calls = 0
                def policy(observation: np.ndarray, info: dict[str, Any]) -> float:
                    nonlocal calls
                    calls += 1
                    action, _ = model.predict(observation, deterministic=True)
                    return float(np.asarray(action).reshape(-1)[0])
                metrics, transitions = run_case(store, caseid, condition, scalers[condition["state_id"]], int(seed), config["common_horizon_seconds"], policy)
                if calls != transitions:
                    raise RuntimeError("PPO predict called other than exactly once per transition")
                rows.append({"condition_id": condition["condition_id"], "seed": seed, "controller": "PPO", "caseid": caseid, "subjectid": str(store._by_case[caseid]["subjectid"]), "controller_calls": calls, **metrics})
    atomic_json(out / "ppo_validation_private.json", {"rows": rows})
    aggregates = []
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            selected = [row for row in rows if row["condition_id"] == condition["condition_id"] and row["seed"] == seed]
            aggregates.append({"condition_id": condition["condition_id"], "seed": seed, "controller": "PPO", "validation_subject_count": len({row["subjectid"] for row in selected}), "validation_case_count": len(selected), **aggregate_subject_rows(selected, list(METRICS))})
    atomic_json(out / "ppo_aggregate.json", {"aggregates": aggregates, "case_to_subject_aggregation": True, "predict_calls_per_transition": 1, "test_access_count": 0})
    print(json.dumps({"aggregate_rows": len(aggregates), "validation_case_rows": len(rows)}))


def evaluate_learning_trajectory(config_path: Path) -> None:
    """Evaluate every fixed checkpoint on the train-only tuning subset.

    This is a descriptive learning diagnostic only and never selects a model;
    the final checkpoint remains predeclared for validation.
    """
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    rows = []
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            directory = out / "jobs" / condition["condition_id"] / f"seed_{seed}"
            for timestep in range(interval, target + 1, interval):
                checkpoint = directory / f"checkpoint_{timestep:010d}"
                verify_checkpoint(checkpoint, {"condition_id": condition["condition_id"], "seed": seed, "timestep": timestep, "config_sha256": config_hash})
                model = PPO.load(str(checkpoint / "model.zip"), device="cpu")
                case_rows = []
                for caseid in membership["tune_cases"]:
                    def policy(observation: np.ndarray, info: dict[str, Any]) -> float:
                        action, _ = model.predict(observation, deterministic=True)
                        return float(np.asarray(action).reshape(-1)[0])
                    metrics, _ = run_case(store, caseid, condition, scalers[condition["state_id"]], int(seed), config["common_horizon_seconds"], policy)
                    case_rows.append({"subjectid": str(store._by_case[caseid]["subjectid"]), **metrics})
                rows.append({"condition_id": condition["condition_id"], "seed": seed, "timestep": timestep, "training_subset_subject_count": len({row["subjectid"] for row in case_rows}), **aggregate_subject_rows(case_rows, list(METRICS))})
    atomic_json(out / "learning_trajectory_aggregate.json", {"rows": rows, "selection_performed": False, "subset": "train_only_frozen_tuning_subset", "final_checkpoint_predeclared": True, "test_access_count": 0})
    print(json.dumps({"trajectory_rows": len(rows), "checkpoints_per_job": target // interval}))


def verify_all_outputs(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out, membership, _, _ = load_private(config)
    train_subjects = set(membership["full_train_subjects"])
    validation_subjects = set(membership["full_validation_subjects"])
    if train_subjects & validation_subjects:
        raise RuntimeError("private development split overlap")
    if not set(membership["tuning_subjects"]).issubset(set(membership["train_subjects"])):
        raise RuntimeError("tuning subjects are outside bounded training")
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    jobs = checkpoints = resumed = 0
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            directory = out / "jobs" / condition["condition_id"] / f"seed_{seed}"
            completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
            expected = {"condition_id": condition["condition_id"], "seed": seed, "timestep": target, "target_timesteps": target, "config_sha256": config_hash}
            for key, value in expected.items():
                if completion.get(key) != value:
                    raise RuntimeError(f"completion identity mismatch: {key}")
            if completion.get("test_access_count") != 0:
                raise RuntimeError("test access recorded in completion")
            resumed += int(bool(completion.get("resumed")))
            jobs += 1
            for timestep in range(interval, target + 1, interval):
                verify_checkpoint(directory / f"checkpoint_{timestep:010d}", expected | {"timestep": timestep})
                checkpoints += 1
    baseline = json.loads((out / "baseline_aggregate.json").read_text(encoding="utf-8"))
    ppo = json.loads((out / "ppo_aggregate.json").read_text(encoding="utf-8"))
    trajectory = json.loads((out / "learning_trajectory_aggregate.json").read_text(encoding="utf-8"))
    if len(baseline["aggregates"]) != 24 or len(ppo["aggregates"]) != 24 or len(trajectory["rows"]) != checkpoints:
        raise RuntimeError("aggregate accounting mismatch")
    if any(payload.get("test_access_count") != 0 for payload in (baseline, ppo, trajectory)):
        raise RuntimeError("test access recorded in aggregate")
    partials = [path for path in out.rglob("*") if ".partial" in path.name]
    if partials:
        raise RuntimeError("partial output remains")
    result = {"verified": True, "training_jobs": jobs, "checkpoints": checkpoints, "resumed_jobs": resumed, "full_train_subject_count": len(train_subjects), "full_validation_subject_count": len(validation_subjects), "split_overlap_count": 0, "baseline_aggregate_rows": len(baseline["aggregates"]), "ppo_aggregate_rows": len(ppo["aggregates"]), "trajectory_rows": len(trajectory["rows"]), "partial_path_count": 0, "test_access_count": 0}
    atomic_json(out / "verification.json", result)
    print(json.dumps(result, indent=2))


def bootstrap_mean(values: np.ndarray, generator: np.random.Generator, draws: int = 10_000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite one-dimensional sample")
    indices = generator.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_subject_bootstrap(config_path: Path) -> None:
    config, _ = load_config(config_path)
    out, _, _, _ = load_private(config)
    ppo_rows = json.loads((out / "ppo_validation_private.json").read_text(encoding="utf-8"))["rows"]
    baseline_rows = json.loads((out / "baseline_validation_private.json").read_text(encoding="utf-8"))["rows"]
    subjects = sorted({row["subjectid"] for row in ppo_rows})
    generator = np.random.Generator(np.random.PCG64(int(config["split_seed"])))

    def ppo_subject(condition: str, subject: str) -> float:
        values = [float(row["latent_bis_mae"]) for row in ppo_rows if row["condition_id"] == condition and row["subjectid"] == subject]
        if len(values) != len(config["seeds"]):
            raise RuntimeError("PPO subject/seed bootstrap accounting mismatch")
        return float(np.mean(values))

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
    factors = []
    for effect, left, right in definitions:
        differences = np.asarray([ppo_subject(left, subject) - ppo_subject(right, subject) for subject in subjects])
        mean, low, high = bootstrap_mean(differences, generator)
        factors.append({"effect": effect, "left": left, "right": right, "metric": "latent_bis_mae", "paired_subject_count": len(subjects), "mean_difference": mean, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high})

    baseline = []
    for condition in [row["condition_id"] for row in config["conditions"]]:
        differences = []
        for subject in subjects:
            pi = [float(row["latent_bis_mae"]) for row in baseline_rows if row["condition_id"] == condition and row["controller"] == "PI" and row["subjectid"] == subject]
            if len(pi) != 1:
                raise RuntimeError("PI subject bootstrap accounting mismatch")
            differences.append(ppo_subject(condition, subject) - pi[0])
        mean, low, high = bootstrap_mean(np.asarray(differences), generator)
        baseline.append({"condition_id": condition, "contrast": "PPO_minus_PI", "metric": "latent_bis_mae", "paired_subject_count": len(subjects), "mean_difference": mean, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high})
    result = {"method": "paired_subject_nonparametric_percentile_bootstrap", "draws": 10_000, "seed": int(config["split_seed"]), "seed_averaging_before_subject_bootstrap": True, "factor_contrasts": factors, "ppo_pi_contrasts": baseline, "test_access_count": 0}
    atomic_json(out / "bootstrap_aggregate.json", result)
    print(json.dumps({"factor_contrasts": len(factors), "ppo_pi_contrasts": len(baseline), "draws": result["draws"]}))


def observation_availability(config_path: Path) -> None:
    config, _ = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    profiles: dict[tuple[Any, Any], dict[str, Any]] = {}
    for condition in config["conditions"]:
        profiles.setdefault((condition["sqi_threshold"], condition["age_seconds"]), condition)
    reason_names = (
        "available", "no_prior_observation", "explicit_missing_event", "nonfinite_bis",
        "bis_out_of_range", "sqi_missing_exact_timestamp", "sqi_below_threshold",
        "stale_beyond_pipeline_cap",
    )
    aggregates = []
    for condition in profiles.values():
        case_rows = []
        for caseid in membership["validation_cases"]:
            bundle = truncate_bundle(store.load_case(caseid), config["common_horizon_seconds"])
            env = make_environment(bundle, condition, scalers[condition["state_id"]], 45)
            _, info = env.reset(seed=45)
            reasons = [str(info["visible_current_bis_reason"])]
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(np.asarray([1.5], dtype=np.float32))
                reasons.append(str(info["visible_current_bis_reason"]))
                done = bool(terminated or truncated)
            env.close()
            total = len(reasons)
            row = {"subjectid": str(store._by_case[caseid]["subjectid"]), **{f"reason_{name}_fraction": reasons.count(name) / total for name in reason_names}}
            if not math.isclose(sum(row[f"reason_{name}_fraction"] for name in reason_names), 1.0, abs_tol=1e-12):
                raise RuntimeError("observation reason accounting mismatch")
            row["visible_fraction"] = row["reason_available_fraction"]
            case_rows.append(row)
        metrics = ["visible_fraction", *(f"reason_{name}_fraction" for name in reason_names)]
        profile_id = condition["condition_id"].rsplit("_", 1)[0]
        aggregates.append({"profile_id": profile_id, "validation_subject_count": len(case_rows), **aggregate_subject_rows(case_rows, metrics)})
    atomic_json(out / "observation_availability_aggregate.json", {"aggregates": aggregates, "constant_probe_action_mg_per_10s": 1.5, "availability_is_action_independent": True, "raw_sqi_or_event_traces_exported": False, "test_access_count": 0})
    print(json.dumps({"profiles": len(aggregates), "validation_subjects_per_profile": len(membership["validation_subjects"])}))


def state_scale_diagnostics(config_path: Path) -> None:
    config, _ = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    rows = []
    for condition in [row for row in config["conditions"] if row["state_id"] == "S1"]:
        observations = []
        for caseid in membership["validation_cases"]:
            bundle = truncate_bundle(store.load_case(caseid), config["common_horizon_seconds"])
            env = make_environment(bundle, condition, scalers["S1"], 45)
            observation, _ = env.reset(seed=45)
            observations.append(observation)
            done = False
            while not done:
                observation, _, terminated, truncated, _ = env.step(np.asarray([1.5], dtype=np.float32))
                observations.append(observation)
                done = bool(terminated or truncated)
            env.close()
        matrix = np.asarray(observations, dtype=np.float64)
        if matrix.shape[1] != len(scalers["S1"].fields) or not np.isfinite(matrix).all():
            raise RuntimeError("invalid standardized state diagnostic matrix")
        for index, field in enumerate(scalers["S1"].fields):
            absolute = np.abs(matrix[:, index])
            rows.append({"profile_id": condition["condition_id"].rsplit("_", 1)[0], "field_name": field.field_name, "observation_count": int(matrix.shape[0]), "absolute_p95": float(np.quantile(absolute, 0.95)), "absolute_p99": float(np.quantile(absolute, 0.99)), "absolute_max": float(absolute.max())})
    atomic_json(out / "state_scale_diagnostics_aggregate.json", {"rows": rows, "fixed_probe_action_mg_per_10s": 1.5, "state_id": "S1", "nonfinite_count": 0, "test_access_count": 0})
    print(json.dumps({"rows": len(rows), "max_absolute_p99": max(row["absolute_p99"] for row in rows)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("baselines", "ppo", "trajectory", "verify", "bootstrap", "availability", "scale-diagnostics"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "baselines":
        tune_baselines(args.config)
    elif args.command == "ppo":
        evaluate_ppo(args.config)
    elif args.command == "trajectory":
        evaluate_learning_trajectory(args.config)
    elif args.command == "verify":
        verify_all_outputs(args.config)
    elif args.command == "bootstrap":
        paired_subject_bootstrap(args.config)
    elif args.command == "availability":
        observation_availability(args.config)
    else:
        state_scale_diagnostics(args.config)


if __name__ == "__main__":
    main()
