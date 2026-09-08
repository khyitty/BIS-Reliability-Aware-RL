"""Post-hoc S1 learning-rate sensitivity, isolated from the primary analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate_night02_cpu import METRICS, run_case
from night02_common import JournalSequenceEnv, aggregate_subject_rows, atomic_json, sha256_path
from run_night02_cpu import (
    DEFAULT_CONFIG, JournalCallback, _digest, load_config, load_private,
    verify_checkpoint, utc_now,
)
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model

SENSITIVITY_ID = "posthoc_s1_learning_rate_3e-4"
LEARNING_RATE = 0.0003


def train_one(config_path: Path, condition_id: str, seed: int) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    condition = next(row for row in config["conditions"] if row["condition_id"] == condition_id and row["state_id"] == "S1")
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    directory = out / "sensitivity" / SENSITIVITY_ID / condition_id / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"sensitivity_id": SENSITIVITY_ID, "posthoc": True, "learning_rate": LEARNING_RATE, "condition_id": condition_id, "seed": seed, "target_timesteps": target, "config_sha256": config_hash, "scaler_sha256": sha256_path(out / "scaler_registry.json"), "train_universe_sha256": _digest(membership["train_cases"])}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(int(config["torch_threads_per_worker"]))
    sequence = JournalSequenceEnv(store, membership["train_cases"], condition, scalers["S1"], seed, config["common_horizon_seconds"])
    vector = DummyVecEnv([lambda: sequence])
    checkpoints = sorted(directory.glob("checkpoint_*"))
    starting, resumed = 0, False
    if checkpoints:
        latest = checkpoints[-1]
        metadata = verify_checkpoint(latest, identity)
        starting = int(metadata["timestep"])
        sequence.restore_sampler(json.loads((latest / "sampler.json").read_text(encoding="utf-8")))
        model = PPO.load(str(latest / "model.zip"), env=vector, device="cpu")
        resumed = True
    else:
        candidate = replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, learning_rate=LEARNING_RATE, seed=seed, total_timesteps=target, purpose=SENSITIVITY_ID)
        model = make_ppo_model(vector, candidate)
    callback = JournalCallback(directory, interval, target, identity)
    callback.last_checkpoint = starting
    started = time.perf_counter()
    model.learn(total_timesteps=target - starting, reset_num_timesteps=not resumed, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    if int(model.num_timesteps) != target:
        raise RuntimeError("sensitivity training target mismatch")
    final = directory / f"checkpoint_{target:010d}"
    if not final.exists():
        callback._save(target)
    metadata = verify_checkpoint(final, identity | {"timestep": target})
    atomic_json(directory / "OUTPUT_COMPLETE.json", metadata | {"completed": True, "completed_utc": utc_now(), "wall_seconds_this_process": elapsed, "resumed": resumed, "test_access_count": 0})
    vector.close()


def supervise(config_path: Path) -> None:
    config, _ = load_config(config_path)
    out, _, _, _ = load_private(config)
    conditions = [row["condition_id"] for row in config["conditions"] if row["state_id"] == "S1"]
    jobs = [(condition, int(seed)) for condition in conditions for seed in config["seeds"]]
    root = out / "sensitivity" / SENSITIVITY_ID
    for index, (condition, seed) in enumerate(jobs, start=1):
        completion = root / condition / f"seed_{seed}" / "OUTPUT_COMPLETE.json"
        if not completion.exists():
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "train-one", "--config", str(config_path.resolve()), "--condition", condition, "--seed", str(seed)], cwd=ROOT)
            if result.returncode or not completion.exists():
                raise RuntimeError(f"sensitivity job failed: {condition}/seed_{seed}")
        atomic_json(root / "queue_state.json", {"completed_jobs": index, "total_jobs": len(jobs), "active": None if index == len(jobs) else "next", "completed": index == len(jobs), "updated_utc": utc_now()})


def evaluate(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    root = out / "sensitivity" / SENSITIVITY_ID
    target = int(config["training_target_timesteps"])
    rows = []
    for condition in [row for row in config["conditions"] if row["state_id"] == "S1"]:
        for seed in config["seeds"]:
            checkpoint = root / condition["condition_id"] / f"seed_{seed}" / f"checkpoint_{target:010d}"
            verify_checkpoint(checkpoint, {"sensitivity_id": SENSITIVITY_ID, "learning_rate": LEARNING_RATE, "condition_id": condition["condition_id"], "seed": seed, "timestep": target, "config_sha256": config_hash})
            model = PPO.load(str(checkpoint / "model.zip"), device="cpu")
            case_rows = []
            for caseid in membership["validation_cases"]:
                def policy(observation, info):
                    action, _ = model.predict(observation, deterministic=True)
                    return float(np.asarray(action).reshape(-1)[0])
                metrics, _ = run_case(store, caseid, condition, scalers["S1"], int(seed), config["common_horizon_seconds"], policy)
                case_rows.append({"caseid": caseid, "subjectid": str(store._by_case[caseid]["subjectid"]), **metrics})
            rows.append({"condition_id": condition["condition_id"], "seed": seed, "controller": "PPO", "learning_rate": LEARNING_RATE, "validation_subject_count": len({row["subjectid"] for row in case_rows}), "validation_case_count": len(case_rows), **aggregate_subject_rows(case_rows, list(METRICS))})
    atomic_json(root / "aggregate.json", {"sensitivity_id": SENSITIVITY_ID, "posthoc": True, "selection_effect": False, "aggregates": rows, "test_access_count": 0})
    print(json.dumps({"aggregate_rows": len(rows), "sensitivity_id": SENSITIVITY_ID}))


def verify(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out, _, _, _ = load_private(config)
    root = out / "sensitivity" / SENSITIVITY_ID
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    jobs = checkpoints = resumed = 0
    for condition in [row for row in config["conditions"] if row["state_id"] == "S1"]:
        for seed in config["seeds"]:
            directory = root / condition["condition_id"] / f"seed_{seed}"
            completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
            expected = {"sensitivity_id": SENSITIVITY_ID, "posthoc": True, "learning_rate": LEARNING_RATE, "condition_id": condition["condition_id"], "seed": seed, "target_timesteps": target, "config_sha256": config_hash}
            if completion.get("test_access_count") != 0:
                raise RuntimeError("sensitivity completion recorded test access")
            resumed += int(bool(completion.get("resumed")))
            jobs += 1
            for timestep in range(interval, target + 1, interval):
                verify_checkpoint(directory / f"checkpoint_{timestep:010d}", expected | {"timestep": timestep})
                checkpoints += 1
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    partials = [path for path in root.rglob("*") if ".partial" in path.name]
    if len(aggregate["aggregates"]) != jobs or aggregate.get("test_access_count") != 0 or partials:
        raise RuntimeError("sensitivity aggregate or partial accounting mismatch")
    result = {"verified": True, "posthoc": True, "training_jobs": jobs, "checkpoints": checkpoints, "aggregate_rows": len(aggregate["aggregates"]), "resumed_jobs": resumed, "partial_path_count": len(partials), "test_access_count": 0}
    atomic_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train-one", "supervise", "evaluate", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "train-one":
        if args.condition is None or args.seed is None:
            parser.error("train-one requires --condition and --seed")
        train_one(args.config, args.condition, args.seed)
    elif args.command == "supervise":
        supervise(args.config)
    elif args.command == "evaluate":
        evaluate(args.config)
    else:
        verify(args.config)


if __name__ == "__main__":
    main()
