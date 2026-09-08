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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("baselines", "ppo", "trajectory"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "baselines":
        tune_baselines(args.config)
    elif args.command == "ppo":
        evaluate_ppo(args.config)
    else:
        evaluate_learning_trajectory(args.config)


if __name__ == "__main__":
    main()
