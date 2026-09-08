"""Execute the privacy-aware Night-02 CPU journal experiment.

All subject/case membership, models, traces, and local paths are written below
the git-ignored output root. Only a separate aggregate exporter may create
public repository results.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from night02_common import (
    JournalSequenceEnv, atomic_json, fit_scalers, make_environment, sha256_bytes,
    sha256_path, stable_order, truncate_bundle,
)
from vitaldb_state_selection.cohort.train_runtime_inputs import (
    StateScaler, TrainRuntimeInputStore, load_scaler_registry,
)
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model

DEFAULT_CONFIG = ROOT / "configs" / "journal" / "night02_cpu.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def output_root(config: dict[str, Any]) -> Path:
    path = (ROOT / config["output_root"]).resolve()
    relative = path.relative_to(ROOT).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0
    tracked = subprocess.check_output(["git", "ls-files", relative], cwd=ROOT, text=True).strip()
    if not ignored or tracked:
        raise RuntimeError("Night-02 private output root must be ignored and untracked")
    return path


def historical_root(config: dict[str, Any], explicit: str | None = None) -> Path:
    value = explicit or os.environ.get(config["source_root_env"])
    if not value:
        raise RuntimeError(f"set {config['source_root_env']} or pass --source-root")
    root = Path(value).resolve()
    if not (root / ".git").exists():
        raise RuntimeError("historical source root is not a git checkout")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if status.strip():
        raise RuntimeError("historical source checkout is not clean")
    return root


def private_store(config: dict[str, Any], source: Path) -> TrainRuntimeInputStore:
    return TrainRuntimeInputStore(source / config["source_runtime_relative"], source)


def _digest(values: list[str]) -> str:
    return sha256_bytes(("\n".join(sorted(values)) + "\n").encode())


def prepare(config_path: Path, source_arg: str | None) -> None:
    config, config_hash = load_config(config_path)
    source = historical_root(config, source_arg)
    store = private_store(config, source)
    out = output_root(config)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(config["split_seed"])
    subject_cases: dict[str, list[str]] = {}
    for row in store.rows:
        subject_cases.setdefault(str(row["subjectid"]), []).append(str(row["caseid"]))
    subjects = stable_order(subject_cases, seed=seed, label="full-development-split")
    cut = round(len(subjects) * float(config["train_subject_fraction"]))
    full_train_subjects, full_validation_subjects = subjects[:cut], subjects[cut:]
    if set(full_train_subjects) & set(full_validation_subjects):
        raise RuntimeError("subject split leakage")

    train_subjects = stable_order(full_train_subjects, seed=seed, label="bounded-train")[:int(config["development_train_subject_limit"])]
    validation_subjects = stable_order(full_validation_subjects, seed=seed, label="bounded-validation")[:int(config["validation_subject_limit"])]
    tuning_subjects = stable_order(train_subjects, seed=seed, label="baseline-tuning")[:int(config["tuning_subject_limit"])]
    full_train_cases = [case for subject in full_train_subjects for case in subject_cases[subject]]
    train_cases = [case for subject in train_subjects for case in subject_cases[subject]]
    validation_cases = [case for subject in validation_subjects for case in subject_cases[subject]]
    tune_cases = [case for subject in tuning_subjects for case in subject_cases[subject]]
    horizon = float(config["common_horizon_seconds"])
    for caseid in train_cases + validation_cases:
        if store.load_case(caseid).episode_horizon_seconds < horizon:
            raise RuntimeError("selected case shorter than common horizon")

    started = time.perf_counter()
    registry = fit_scalers(store, full_train_cases)
    registry["split_sha256"] = _digest(full_train_subjects)
    registry["fit_subject_count"] = len(full_train_subjects)
    registry["fit_seconds"] = time.perf_counter() - started
    scaler_path = out / "scaler_registry.json"
    atomic_json(scaler_path, registry)
    membership = {
        "full_train_subjects": full_train_subjects,
        "full_validation_subjects": full_validation_subjects,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "tuning_subjects": tuning_subjects,
        "full_train_cases": full_train_cases,
        "train_cases": train_cases,
        "validation_cases": validation_cases,
        "tune_cases": tune_cases,
    }
    atomic_json(out / "private_membership.json", membership)
    local = {
        "source_root": str(source), "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "prepared_utc": utc_now(), "config_sha256": config_hash,
    }
    atomic_json(out / "local_state.json", local)
    shareable = {
        "protocol_id": config["protocol_id"], "evidence_scope": config["evidence_scope"],
        "config_sha256": config_hash, "split_seed": seed,
        "source_git_sha": local["source_git_sha"], "common_horizon_seconds": horizon,
        "full_train_subject_count": len(full_train_subjects), "full_validation_subject_count": len(full_validation_subjects),
        "full_train_case_count": len(full_train_cases), "bounded_train_subject_count": len(train_subjects),
        "bounded_train_case_count": len(train_cases), "bounded_validation_subject_count": len(validation_subjects),
        "bounded_validation_case_count": len(validation_cases), "tuning_subject_count": len(tuning_subjects),
        "full_train_subject_sha256": _digest(full_train_subjects), "full_validation_subject_sha256": _digest(full_validation_subjects),
        "bounded_train_subject_sha256": _digest(train_subjects), "bounded_validation_subject_sha256": _digest(validation_subjects),
        "scaler_registry_sha256": sha256_path(scaler_path), "test_subject_or_case_access_count": 0,
    }
    atomic_json(out / "shareable_preparation.json", shareable)
    print(json.dumps(shareable, indent=2))


def load_private(config: dict[str, Any]) -> tuple[Path, dict[str, Any], TrainRuntimeInputStore, dict[str, StateScaler]]:
    out = output_root(config)
    local = json.loads((out / "local_state.json").read_text(encoding="utf-8"))
    membership = json.loads((out / "private_membership.json").read_text(encoding="utf-8"))
    source = Path(local["source_root"])
    return out, membership, private_store(config, source), load_scaler_registry(out / "scaler_registry.json")


def benchmark(config_path: Path, steps: int) -> None:
    config, _ = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    condition = config["conditions"][0]
    seed = int(config["seeds"][0])
    torch.set_num_threads(int(config["torch_threads_per_worker"]))
    env = DummyVecEnv([lambda: JournalSequenceEnv(store, membership["train_cases"], condition, scalers[condition["state_id"]], seed, config["common_horizon_seconds"])])
    candidate = replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=seed, total_timesteps=steps, purpose="night02_actual_input_throughput_benchmark")
    model = make_ppo_model(env, candidate)
    started = time.perf_counter()
    model.learn(total_timesteps=steps, reset_num_timesteps=True, progress_bar=False)
    elapsed = time.perf_counter() - started
    result = {"actual_input": True, "steps": int(model.num_timesteps), "seconds": elapsed, "steps_per_second": model.num_timesteps / elapsed, "condition_id": condition["condition_id"], "seed": seed, "completed_utc": utc_now()}
    atomic_json(out / "benchmark.json", result)
    env.close()
    print(json.dumps(result, indent=2))


class JournalCallback(BaseCallback):
    def __init__(self, directory: Path, interval: int, target: int, identity: dict[str, Any]):
        super().__init__(0)
        self.directory, self.interval, self.target, self.identity = directory, interval, target, identity
        self.last_checkpoint = 0

    def _on_step(self) -> bool:
        for key in ("new_obs", "actions", "rewards"):
            if key in self.locals and not np.isfinite(np.asarray(self.locals[key])).all():
                raise RuntimeError(f"nonfinite PPO {key}")
        step = int(self.model.num_timesteps)
        if step and step % self.interval == 0 and step > self.last_checkpoint:
            self._save(step)
            self.last_checkpoint = step
        return step < self.target

    def _save(self, step: int) -> None:
        destination = self.directory / f"checkpoint_{step:010d}"
        if destination.exists():
            verify_checkpoint(destination, self.identity | {"timestep": step})
            return
        temporary = self.directory / f".{destination.name}.partial"
        temporary.mkdir(parents=True, exist_ok=False)
        self.model.save(str(temporary / "model"))
        sequence = unwrap(self.training_env)
        atomic_json(temporary / "sampler.json", sequence.sampler_snapshot())
        metadata = self.identity | {"timestep": step, "created_utc": utc_now(), "model_sha256": sha256_path(temporary / "model.zip"), "optimizer_in_model_archive": True, "resume_equivalence": "warm_episode_restart; PPO partial rollout buffer is not restored"}
        atomic_json(temporary / "metadata.json", metadata)
        atomic_json(temporary / "COMPLETE.json", {"complete": True, "metadata_sha256": sha256_path(temporary / "metadata.json"), "model_sha256": metadata["model_sha256"], "sampler_sha256": sha256_path(temporary / "sampler.json")})
        os.replace(temporary, destination)
        atomic_json(self.directory / "heartbeat.json", {"event": "checkpoint", "timestep": step, "utc": utc_now()})


def unwrap(vector: Any) -> JournalSequenceEnv:
    env: Any = vector.envs[0]
    while hasattr(env, "env"):
        env = env.env
    if not isinstance(env, JournalSequenceEnv):
        raise RuntimeError("unexpected sequence environment")
    return env


def verify_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("metadata_sha256") != sha256_path(path / "metadata.json") or complete.get("model_sha256") != sha256_path(path / "model.zip") or complete.get("sampler_sha256") != sha256_path(path / "sampler.json"):
        raise RuntimeError("checkpoint checksum validation failed")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch: {key}")
    return metadata


def train_one(config_path: Path, condition_id: str, seed: int) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    condition = next(row for row in config["conditions"] if row["condition_id"] == condition_id)
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    directory = out / "jobs" / condition_id / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_id": config["protocol_id"], "condition_id": condition_id, "seed": seed, "target_timesteps": target, "config_sha256": config_hash, "scaler_sha256": sha256_path(out / "scaler_registry.json"), "train_universe_sha256": _digest(membership["train_cases"])}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(int(config["torch_threads_per_worker"]))
    sequence = JournalSequenceEnv(store, membership["train_cases"], condition, scalers[condition["state_id"]], seed, config["common_horizon_seconds"])
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
        model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=seed, total_timesteps=target, purpose="night02_vitaldb_informed_reconstructed_simulation"))
    callback = JournalCallback(directory, interval, target, identity)
    callback.last_checkpoint = starting
    started = time.perf_counter()
    model.learn(total_timesteps=target - starting, reset_num_timesteps=not resumed, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    if int(model.num_timesteps) != target:
        raise RuntimeError("training target mismatch")
    final_path = directory / f"checkpoint_{target:010d}"
    if not final_path.exists():
        callback._save(target)
    verification = verify_checkpoint(final_path, identity | {"timestep": target})
    atomic_json(directory / "OUTPUT_COMPLETE.json", verification | {"completed": True, "completed_utc": utc_now(), "wall_seconds_this_process": elapsed, "resumed": resumed, "test_access_count": 0})
    vector.close()
    print(json.dumps({"condition_id": condition_id, "seed": seed, "timesteps": target, "seconds": elapsed, "resumed": resumed}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "benchmark", "train-one"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.config, args.source_root)
    elif args.command == "benchmark": benchmark(args.config, args.steps)
    else:
        if args.condition is None or args.seed is None:
            parser.error("train-one requires --condition and --seed")
        train_one(args.config, args.condition, args.seed)


if __name__ == "__main__":
    main()
