"""Post-hoc three-seed probe of S1 observation clipping for scale diagnosis."""

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

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate_night02_cpu import METRICS, case_metrics
from night02_common import JournalSequenceEnv, aggregate_subject_rows, atomic_json, make_environment, sha256_path, truncate_bundle
from run_night02_cpu import DEFAULT_CONFIG, JournalCallback, _digest, load_config, load_private, utc_now, verify_checkpoint
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model

PROBE_ID = "posthoc_goff_a30_s1_observation_clip_10"
CONDITION_ID = "Goff_A30_S1"
CLIP = 10.0


class ClipObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-CLIP, CLIP, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(observation, -CLIP, CLIP).astype(np.float32)


def train_one(config_path: Path, seed: int) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    condition = next(row for row in config["conditions"] if row["condition_id"] == CONDITION_ID)
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    directory = out / "sensitivity" / PROBE_ID / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"probe_id": PROBE_ID, "posthoc": True, "observation_clip_absolute": CLIP, "condition_id": CONDITION_ID, "seed": seed, "target_timesteps": target, "config_sha256": config_hash, "scaler_sha256": sha256_path(out / "scaler_registry.json"), "train_universe_sha256": _digest(membership["train_cases"])}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(int(config["torch_threads_per_worker"]))
    sequence = JournalSequenceEnv(store, membership["train_cases"], condition, scalers["S1"], seed, config["common_horizon_seconds"])
    vector = DummyVecEnv([lambda: ClipObservation(sequence)])
    model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=seed, total_timesteps=target, purpose=PROBE_ID))
    callback = JournalCallback(directory, interval, target, identity)
    started = time.perf_counter()
    model.learn(total_timesteps=target, reset_num_timesteps=True, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    if int(model.num_timesteps) != target:
        raise RuntimeError("clip probe target mismatch")
    final = directory / f"checkpoint_{target:010d}"
    if not final.exists():
        callback._save(target)
    metadata = verify_checkpoint(final, identity | {"timestep": target})
    atomic_json(directory / "OUTPUT_COMPLETE.json", metadata | {"completed": True, "completed_utc": utc_now(), "wall_seconds_this_process": elapsed, "resumed": False, "test_access_count": 0})
    vector.close()


def supervise(config_path: Path) -> None:
    config, _ = load_config(config_path)
    for seed in config["seeds"]:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "train-one", "--config", str(config_path.resolve()), "--seed", str(seed)], cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"clip probe failed: seed {seed}")


def evaluate_and_verify(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    out, membership, store, scalers = load_private(config)
    condition = next(row for row in config["conditions"] if row["condition_id"] == CONDITION_ID)
    target, interval = int(config["training_target_timesteps"]), int(config["checkpoint_interval_timesteps"])
    rows = []
    for seed in config["seeds"]:
        directory = out / "sensitivity" / PROBE_ID / f"seed_{seed}"
        expected = {"probe_id": PROBE_ID, "posthoc": True, "observation_clip_absolute": CLIP, "condition_id": CONDITION_ID, "seed": seed, "target_timesteps": target, "config_sha256": config_hash}
        completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
        if completion.get("test_access_count") != 0:
            raise RuntimeError("clip probe test access")
        for timestep in range(interval, target + 1, interval):
            verify_checkpoint(directory / f"checkpoint_{timestep:010d}", expected | {"timestep": timestep})
        model = PPO.load(str(directory / f"checkpoint_{target:010d}" / "model.zip"), device="cpu")
        case_rows = []
        for caseid in membership["validation_cases"]:
            env = make_environment(truncate_bundle(store.load_case(caseid), config["common_horizon_seconds"]), condition, scalers["S1"], int(seed))
            observation, _ = env.reset(seed=int(seed))
            latent, actions, rewards, clips = [], [], [], 0
            done = False
            while not done:
                action, _ = model.predict(np.clip(observation, -CLIP, CLIP).astype(np.float32), deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                latent.append(float(info["latent_true_bis"])); actions.append(float(info["applied_action_mg_per_10s"])); rewards.append(float(reward)); clips += int(info["action_was_clipped"])
                done = bool(terminated or truncated)
            env.close()
            case_rows.append({"subjectid": str(store._by_case[caseid]["subjectid"]), **case_metrics(latent, actions, rewards, clips)})
        rows.append({"condition_id": CONDITION_ID, "seed": seed, "observation_clip_absolute": CLIP, "validation_subject_count": len(case_rows), **aggregate_subject_rows(case_rows, list(METRICS))})
    root = out / "sensitivity" / PROBE_ID
    partials = [path for path in root.rglob("*") if ".partial" in path.name]
    if partials:
        raise RuntimeError("clip probe partial remains")
    atomic_json(root / "aggregate.json", {"probe_id": PROBE_ID, "posthoc": True, "training_jobs": len(rows), "checkpoints": len(rows) * target // interval, "aggregates": rows, "partial_path_count": 0, "test_access_count": 0})
    print(json.dumps({"training_jobs": len(rows), "checkpoints": len(rows) * target // interval, "aggregate_rows": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train-one", "supervise", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "train-one":
        if args.seed is None:
            parser.error("train-one requires --seed")
        train_one(args.config, args.seed)
    elif args.command == "supervise":
        supervise(args.config)
    else:
        evaluate_and_verify(args.config)


if __name__ == "__main__":
    main()
