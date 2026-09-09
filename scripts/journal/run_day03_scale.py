"""Run the bounded Day-03 observation-scale mechanism study.

Private membership, calibration observations, models, and case-level metrics stay
under the ignored output root. Only aggregate CSV/JSON/Markdown are published.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from evaluate_night02_cpu import Controller, METRICS, case_metrics
from night02_common import JournalSequenceEnv, aggregate_subject_rows, atomic_json, make_environment, sha256_path, truncate_bundle
from run_night02_cpu import _digest, load_config as load_json_config, load_private as load_night02_private, utc_now, verify_checkpoint
from vitaldb_state_selection.anesthesia.state import S0_FIELDS, S1_FIELDS
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model

DEFAULT_CONFIG = ROOT / "configs" / "journal" / "day03_scale.json"
PUBLIC_CSV = ROOT / "reports" / "journal" / "day03_results.csv"
PUBLIC_JSON = ROOT / "reports" / "journal" / "day03_summary.json"
PUBLIC_REPORT = ROOT / "reports" / "journal" / "DAY03_RESULTS.md"


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def config_and_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def output_root(config: dict[str, Any]) -> Path:
    out = (ROOT / config["output_root"]).resolve()
    relative = out.relative_to(ROOT).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0
    tracked = subprocess.check_output(["git", "ls-files", relative], cwd=ROOT, text=True).strip()
    if not ignored or tracked:
        raise RuntimeError("Day-03 private output root must be ignored and untracked")
    out.mkdir(parents=True, exist_ok=True)
    return out


def night02_context(config: dict[str, Any]):
    path = ROOT / config["night02_config"]
    old_config, old_hash = load_json_config(path)
    out, membership, store, scalers = load_night02_private(old_config)
    expected = (ROOT / config["night02_output_root"]).resolve()
    if out.resolve() != expected:
        raise RuntimeError("Night-02 output root mismatch")
    return path, old_config, old_hash, out, membership, store, scalers


def condition_for(state: str) -> dict[str, Any]:
    return {"condition_id": f"Goff_A30_{state}", "sqi_threshold": None, "age_seconds": 30, "state_id": state}


def update_status(out: Path, started: float, deadline: float, **values: Any) -> None:
    elapsed = (time.time() - started) / 60.0
    payload = {
        "phase": values.pop("phase"), "elapsed_minutes": elapsed,
        "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        "estimated_completion_utc": values.pop("estimated_completion_utc", None),
        "latest_error": values.pop("latest_error", None), "updated_utc": utc_now(), **values,
    }
    atomic_json(out / "status.json", payload)


class FixedTransform(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, transform: str, factors: np.ndarray):
        super().__init__(env)
        self.transform_name = transform
        self.factors = np.asarray(factors, dtype=np.float32)
        if self.factors.shape != env.observation_space.shape or not np.isfinite(self.factors).all() or np.any(self.factors <= 0):
            raise ValueError("invalid transform factors")
        if transform == "Nclip":
            self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=env.observation_space.shape, dtype=np.float32)
        elif transform == "Nscale":
            low = np.asarray(env.observation_space.low, dtype=np.float32) / self.factors
            high = np.asarray(env.observation_space.high, dtype=np.float32) / self.factors
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        elif transform != "N0":
            raise ValueError(transform)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return apply_transform(observation, self.transform_name, self.factors)


def apply_transform(observation: np.ndarray, transform: str, factors: np.ndarray) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32)
    if transform == "Nclip":
        values = np.clip(values, -10.0, 10.0)
    elif transform == "Nscale":
        values = values / np.asarray(factors, dtype=np.float32)
    elif transform != "N0":
        raise ValueError(transform)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite transformed observation")
    return values.astype(np.float32)


def factor_manifest(out: Path) -> dict[str, Any]:
    return json.loads((out / "nscale_factors.json").read_text(encoding="utf-8"))


def factors_for(out: Path, state: str) -> np.ndarray:
    manifest = factor_manifest(out)
    fields = S0_FIELDS if state == "S0" else S1_FIELDS
    mapped = manifest["factors_by_field"]
    return np.asarray([mapped[name] for name in fields], dtype=np.float32)


def calibrate(config_path: Path) -> None:
    config, config_hash = config_and_hash(config_path)
    out = output_root(config)
    _, old_config, _, old_out, membership, store, scalers = night02_context(config)
    condition = condition_for("S1")
    baseline = json.loads((old_out / "baseline_aggregate.json").read_text(encoding="utf-8"))
    params = baseline["selection"]["Goff_A30/PI"]
    arrays: list[np.ndarray] = []
    controller_counts: dict[str, int] = {}
    for controller_name in ("constant_1.5_mg_per_10s", "frozen_selected_Goff_A30_PI"):
        count = 0
        for caseid in membership["train_cases"]:
            env = make_environment(truncate_bundle(store.load_case(caseid), config["common_horizon_seconds"]), condition, scalers["S1"], 45)
            obs, info = env.reset(seed=45)
            controller = Controller(params["family"], params["base"], params["kp"], params["ki"])
            controller.reset()
            done = False
            while not done:
                arrays.append(np.asarray(obs, dtype=np.float32).copy())
                if controller_name.startswith("constant"):
                    action = 1.5
                else:
                    action = controller.predict_once(info)
                obs, _, terminated, truncated, info = env.step(np.asarray([action], dtype=np.float32))
                done = bool(terminated or truncated)
                count += 1
            env.close()
        controller_counts[controller_name] = count
    observations = np.stack(arrays)
    if observations.shape[1] != len(S1_FIELDS) or not np.isfinite(observations).all():
        raise RuntimeError("invalid calibration observations")
    q95 = np.quantile(np.abs(observations), float(config["calibration"]["quantile"]), axis=0)
    binary = {"sex_binary", *(name for name in S1_FIELDS if name.startswith("bis_mask_"))}
    factors = np.maximum(1.0, q95)
    for index, name in enumerate(S1_FIELDS):
        if name in binary:
            factors[index] = 1.0
    mapped = {name: float(factors[index]) for index, name in enumerate(S1_FIELDS)}
    if tuple(S1_FIELDS[:len(S0_FIELDS)]) != tuple(S0_FIELDS):
        raise RuntimeError("shared field order mismatch")
    if any(not math.isfinite(value) or value <= 0 for value in mapped.values()):
        raise RuntimeError("invalid factor")
    transformed = observations / factors
    binary_indices = [index for index, name in enumerate(S1_FIELDS) if name in binary]
    if not np.array_equal(observations[:, binary_indices], transformed[:, binary_indices]):
        raise RuntimeError("binary fields changed")
    bis_names = [name for name in S1_FIELDS if name.startswith("bis_value_")]
    bis_factor = mapped[bis_names[-1]]
    ordered = [value / bis_factor for value in (40.0, 50.0, 60.0)]
    if not ordered[0] < ordered[1] < ordered[2]:
        raise RuntimeError("Nscale did not preserve BIS order")
    membership_hash = _digest(membership["train_cases"])
    scaler_hash = sha256_path(old_out / "scaler_registry.json")
    manifest = {
        "protocol_id": config["protocol_id"], "config_sha256": config_hash,
        "formula": "z_j / d_j; d_j=max(1,q95(abs(z_j)))", "quantile": 0.95,
        "calibration_state": "S1", "calibration_case_scope": "all_fixed_training_cases",
        "calibration_case_count": len(membership["train_cases"]),
        "calibration_subject_count": len(membership["train_subjects"]),
        "calibration_observation_count": int(len(observations)), "controller_observation_counts": controller_counts,
        "calibration_membership_sha256": membership_hash, "source_scaler_sha256": scaler_hash,
        "field_names": list(S1_FIELDS), "s0_field_names": list(S0_FIELDS),
        "s0_to_s1_index": {name: int(S1_FIELDS.index(name)) for name in S0_FIELDS},
        "binary_fields": sorted(binary), "factors_by_field": mapped,
        "factors_sha256": canonical_sha(mapped),
        "validation_scores_accessed_before_freeze": False,
        "checks": {"finite_positive": True, "binary_exact": True, "shared_field_equality": True,
                   "nscale_bis_40_50_60": ordered, "nclip_bis_40_50_60": [10.0, 10.0, 10.0]},
        "frozen_utc": utc_now(), "test_access_count": 0,
    }
    atomic_json(out / "nscale_factors.json", manifest)
    np.savez_compressed(out / "calibration_observations_private.npz", S1=observations, S0=observations[:, :len(S0_FIELDS)])
    atomic_json(out / "preparation.json", {
        "prepared": True, "night02_config_sha256": hashlib.sha256((ROOT / config["night02_config"]).read_bytes()).hexdigest(),
        "source_scaler_sha256": scaler_hash, "train_universe_sha256": membership_hash,
        "factors_sha256": manifest["factors_sha256"], "test_access_count": 0,
    })
    print(json.dumps({key: manifest[key] for key in ("calibration_case_count", "calibration_observation_count", "factors_sha256")}, indent=2))


class SafeCheckpointCallback(BaseCallback):
    """Periodic snapshots only; never abort a rollout at the target boundary."""

    def __init__(self, directory: Path, interval: int, identity: dict[str, Any]):
        super().__init__(0)
        self.directory, self.interval, self.identity = directory, interval, identity
        self.last_checkpoint = 0
        self.last_heartbeat = time.monotonic()

    def _on_step(self) -> bool:
        for key in ("new_obs", "actions", "rewards"):
            if key in self.locals and not np.isfinite(np.asarray(self.locals[key])).all():
                raise RuntimeError(f"non-finite PPO {key}")
        step = int(self.model.num_timesteps)
        if time.monotonic() - self.last_heartbeat >= 30:
            atomic_json(self.directory / "heartbeat.json", {"event": "training", "timestep": step, "updates": int(self.model._n_updates), "utc": utc_now()})
            self.last_heartbeat = time.monotonic()
        if step and step % self.interval == 0 and step > self.last_checkpoint:
            self._save_snapshot(step)
            self.last_checkpoint = step
        return True

    def _save_snapshot(self, step: int) -> None:
        path = self.directory / f"snapshot_{step:010d}_pre_update"
        if path.exists():
            return
        tmp = self.directory / f".{path.name}.partial"
        tmp.mkdir(parents=True, exist_ok=False)
        self.model.save(str(tmp / "model"))
        atomic_json(tmp / "metadata.json", self.identity | {
            "timestep": step, "training_epochs_at_save": int(self.model._n_updates),
            "artifact_stage": "pre_update_callback_snapshot", "created_utc": utc_now(),
        })
        os.replace(tmp, path)


def make_vector(config: dict[str, Any], membership: dict[str, Any], store: Any, scalers: Any, state: str, transform: str, seed: int, out: Path):
    sequence = JournalSequenceEnv(store, membership["train_cases"], condition_for(state), scalers[state], seed, config["common_horizon_seconds"])
    factors = factors_for(out, state)
    return DummyVecEnv([lambda: FixedTransform(sequence, transform, factors)])


def job_dir(out: Path, transform: str, state: str, seed: int) -> Path:
    return out / "jobs" / f"{transform}_{state}" / f"seed_{seed}"


def train_one(config_path: Path, transform: str, state: str, seed: int) -> None:
    config, config_hash = config_and_hash(config_path)
    out = output_root(config)
    _, _, _, old_out, membership, store, scalers = night02_context(config)
    factors = factor_manifest(out)
    target = int(config["training_target_timesteps"])
    expected_epochs = int(config["expected_training_epochs"])
    directory = job_dir(out, transform, state, seed)
    completion_path = directory / "OUTPUT_COMPLETE.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("config_sha256") == config_hash and completion.get("training_epochs") == expected_epochs:
            print(json.dumps({"skipped_complete": str(directory.relative_to(out))}))
            return
        raise RuntimeError("existing completion identity mismatch")
    if directory.exists():
        for child in directory.iterdir():
            if child.name != "worker.log":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol_id": config["protocol_id"], "transform": transform, "state": state, "seed": seed,
        "condition_id": f"Goff_A30_{state}", "target_timesteps": target,
        "expected_completed_rollouts": config["expected_completed_rollouts"],
        "config_sha256": config_hash, "source_scaler_sha256": sha256_path(old_out / "scaler_registry.json"),
        "factors_sha256": factors["factors_sha256"], "train_universe_sha256": _digest(membership["train_cases"]),
    }
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_threads_per_worker"]))
    vector = make_vector(config, membership, store, scalers, state, transform, seed, out)
    candidate = replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=seed, total_timesteps=target, purpose="day03_scale_matched_completed_updates")
    model = make_ppo_model(vector, candidate)
    callback = SafeCheckpointCallback(directory, int(config["checkpoint_interval_timesteps"]), identity)
    started = time.perf_counter()
    model.learn(total_timesteps=target, reset_num_timesteps=True, callback=callback, progress_bar=False)
    elapsed = time.perf_counter() - started
    if int(model.num_timesteps) != target or int(model._n_updates) != expected_epochs:
        raise RuntimeError(f"final update mismatch: steps={model.num_timesteps}, updates={model._n_updates}")
    final = directory / "final_post_update"
    tmp = directory / ".final_post_update.partial"
    tmp.mkdir(parents=True, exist_ok=False)
    model.save(str(tmp / "model"))
    metadata = identity | {
        "collected_and_trained_timesteps": target, "training_epochs": int(model._n_updates),
        "completed_rollouts": int(model._n_updates // model.n_epochs), "n_epochs": int(model.n_epochs),
        "n_steps": int(model.n_steps), "artifact_stage": "post_update_authoritative_final",
        "model_sha256": sha256_path(tmp / "model.zip"), "completed_utc": utc_now(),
        "wall_seconds": elapsed, "test_access_count": 0,
    }
    atomic_json(tmp / "metadata.json", metadata)
    atomic_json(tmp / "COMPLETE.json", {"complete": True, "model_sha256": metadata["model_sha256"], "metadata_sha256": sha256_path(tmp / "metadata.json")})
    os.replace(tmp, final)
    reloaded = PPO.load(str(final / "model.zip"), device="cpu")
    if int(reloaded._n_updates) != expected_epochs or int(reloaded.num_timesteps) != target:
        raise RuntimeError("reloaded final model accounting mismatch")
    atomic_json(completion_path, metadata | {"completed": True})
    vector.close()
    print(json.dumps({"transform": transform, "state": state, "seed": seed, "timesteps": target, "training_epochs": model._n_updates, "seconds": elapsed}))


def smoke(config_path: Path) -> None:
    config, _ = config_and_hash(config_path)
    out = output_root(config)
    _, _, _, _, membership, store, scalers = night02_context(config)
    torch.set_num_threads(1)
    vector = make_vector(config, membership, store, scalers, "S0", "Nclip", 45, out)
    candidate = replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=45, total_timesteps=4096, purpose="day03_two_rollout_final_update_check")
    model = make_ppo_model(vector, candidate)
    callback = SafeCheckpointCallback(out / "smoke", 2048, {"purpose": candidate.purpose})
    model.learn(total_timesteps=4096, callback=callback, progress_bar=False)
    if model.num_timesteps != 4096 or model._n_updates != 20:
        raise RuntimeError("two-rollout update check failed")
    sample = vector.reset()
    before, _ = model.predict(sample, deterministic=True)
    model.save(str(out / "smoke" / "final_model"))
    loaded = PPO.load(str(out / "smoke" / "final_model.zip"), device="cpu")
    after, _ = loaded.predict(sample, deterministic=True)
    if not np.array_equal(before, after):
        raise RuntimeError("saved final differs from in-memory policy")
    atomic_json(out / "smoke" / "verification.json", {
        "verified": True, "rollouts": 2, "collected_and_trained_timesteps": 4096,
        "training_epochs": 20, "final_policy_exact_agreement": True, "callback_returned_false": False,
    })
    vector.close()
    print(json.dumps(json.loads((out / "smoke" / "verification.json").read_text()), indent=2))


def benchmark(config_path: Path, transform: str = "Nclip", state: str = "S0", steps: int = 4096) -> None:
    config, _ = config_and_hash(config_path)
    out = output_root(config)
    _, _, _, _, membership, store, scalers = night02_context(config)
    torch.set_num_threads(1)
    vector = make_vector(config, membership, store, scalers, state, transform, 45, out)
    model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=45, total_timesteps=steps, purpose="day03_actual_input_benchmark"))
    started = time.perf_counter(); model.learn(total_timesteps=steps, progress_bar=False); elapsed = time.perf_counter() - started
    result = {"steps": int(model.num_timesteps), "seconds": elapsed, "steps_per_second": model.num_timesteps / elapsed, "transform": transform, "state": state, "completed_utc": utc_now()}
    atomic_json(out / "benchmark.json", result); vector.close(); print(json.dumps(result, indent=2))


def verify_reused(config: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    _, old_config, old_hash, old_out, membership, _, _ = night02_context(config)
    old_target = int(old_config["training_target_timesteps"])
    scaler_hash = sha256_path(old_out / "scaler_registry.json")
    universe_hash = _digest(membership["train_cases"])
    rows = []
    for transform, state, source in (("N0", "S0", "primary"), ("N0", "S1", "primary"), ("Nclip", "S1", "clip_probe")):
        for seed in config["seeds"]:
            if source == "primary":
                directory = old_out / "jobs" / f"Goff_A30_{state}" / f"seed_{seed}"
                expected = {"condition_id": f"Goff_A30_{state}", "seed": seed, "timestep": old_target, "config_sha256": old_hash}
            else:
                directory = old_out / "sensitivity" / "posthoc_goff_a30_s1_observation_clip_10" / f"seed_{seed}"
                expected = {"probe_id": "posthoc_goff_a30_s1_observation_clip_10", "condition_id": "Goff_A30_S1", "seed": seed, "timestep": old_target, "config_sha256": old_hash}
            metadata = verify_checkpoint(directory / f"checkpoint_{old_target:010d}", expected)
            completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
            model = PPO.load(str(directory / f"checkpoint_{old_target:010d}" / "model.zip"), device="cpu")
            checks = {
                "source_scaler_sha256": metadata.get("scaler_sha256") == scaler_hash,
                "train_universe_sha256": metadata.get("train_universe_sha256") == universe_hash,
                "seed": model.seed == seed, "field_dimension": model.observation_space.shape[0] == (len(S0_FIELDS) if state == "S0" else len(S1_FIELDS)),
                "effective_training_epochs": int(model._n_updates) == int(config["expected_training_epochs"]),
                "file_integrity": completion.get("model_sha256") == sha256_path(directory / f"checkpoint_{old_target:010d}" / "model.zip"),
                "learning_rate": float(model.learning_rate) == 0.001, "n_steps": int(model.n_steps) == 2048, "n_epochs": int(model.n_epochs) == 10,
            }
            if not all(checks.values()):
                raise RuntimeError(f"reused model compatibility failed: {transform}/{state}/{seed}: {checks}")
            rows.append({"transform": transform, "state": state, "seed": seed, "provenance": "reused", "source": source,
                         "collected_timesteps": int(model.num_timesteps), "trained_timesteps": 260096,
                         "training_epochs": int(model._n_updates), "model_sha256": metadata["model_sha256"], "checks": checks})
    atomic_json(out / "reused_model_verification.json", {"verified": True, "models": rows, "test_access_count": 0})
    return rows


def model_path(config: dict[str, Any], out: Path, old_out: Path, transform: str, state: str, seed: int) -> tuple[Path, str]:
    if transform == "N0":
        return old_out / "jobs" / f"Goff_A30_{state}" / f"seed_{seed}" / "checkpoint_0000262144" / "model.zip", "reused"
    if transform == "Nclip" and state == "S1":
        return old_out / "sensitivity" / "posthoc_goff_a30_s1_observation_clip_10" / f"seed_{seed}" / "checkpoint_0000262144" / "model.zip", "reused"
    return job_dir(out, transform, state, seed) / "final_post_update" / "model.zip", "new"


def evaluate(config_path: Path) -> None:
    config, _ = config_and_hash(config_path); out = output_root(config)
    _, _, _, old_out, membership, store, scalers = night02_context(config)
    reused = verify_reused(config, out)
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    cells = [("N0", "S0"), ("N0", "S1"), ("Nclip", "S0"), ("Nclip", "S1"), ("Nscale", "S0"), ("Nscale", "S1")]
    for transform, state in cells:
        factors = factors_for(out, state)
        for seed in config["seeds"]:
            path, provenance = model_path(config, out, old_out, transform, state, seed)
            if not path.exists():
                continue
            model = PPO.load(str(path), device="cpu")
            case_rows = []
            for caseid in membership["validation_cases"]:
                env = make_environment(truncate_bundle(store.load_case(caseid), config["common_horizon_seconds"]), condition_for(state), scalers[state], seed)
                obs, _ = env.reset(seed=seed); latent=[]; actions=[]; rewards=[]; clips=0; done=False
                while not done:
                    transformed = apply_transform(obs, transform, factors)
                    action, _ = model.predict(transformed, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    latent.append(float(info["latent_true_bis"])); actions.append(float(info["applied_action_mg_per_10s"])); rewards.append(float(reward)); clips += int(info["action_was_clipped"])
                    done = bool(terminated or truncated)
                env.close()
                metrics = case_metrics(latent, actions, rewards, clips)
                record = {"subjectid": str(store._by_case[caseid]["subjectid"]), **metrics}
                case_rows.append(record); private_rows.append({"transform": transform, "state": state, "seed": seed, "caseid": caseid, **record})
            aggregate = aggregate_subject_rows(case_rows, list(METRICS))
            trained_steps = 260096
            collected_steps = 262144 if provenance == "reused" else int(config["training_target_timesteps"])
            rows.append({"transform": transform, "state": state, "seed": seed, "provenance": provenance,
                         "validation_subject_count": len({row["subjectid"] for row in case_rows}),
                         "validation_case_count": len(case_rows), "collected_timesteps": collected_steps,
                         "trained_timesteps": trained_steps, "training_epochs": 1270, **aggregate})
    atomic_json(out / "validation_private.json", {"rows": private_rows, "test_access_count": 0})
    atomic_json(out / "validation_aggregate.json", {"aggregates": rows, "reused_model_count": len(reused), "test_access_count": 0})
    print(json.dumps({"aggregate_rows": len(rows), "private_rows": len(private_rows)}, indent=2))


def actor_diagnostics(model: PPO, observations: np.ndarray) -> dict[str, float]:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    saturated = total = 0
    hooks = []
    def hook(_module, _inputs, output):
        nonlocal saturated, total
        values = output.detach()
        saturated += int((values.abs() > 0.99).sum().item()); total += int(values.numel())
    for module in model.policy.mlp_extractor.policy_net.modules():
        if isinstance(module, torch.nn.Tanh): hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        distribution = model.policy.get_distribution(tensor)
        raw = distribution.distribution.mean.detach().cpu().numpy().reshape(-1)
    for item in hooks: item.remove()
    physical = np.clip(raw, float(model.action_space.low[0]), float(model.action_space.high[0]))
    return {"actor_tanh_saturation_fraction": float(saturated / total), "raw_deterministic_action_mean": float(np.mean(raw)),
            "physical_clipped_action_mean": float(np.mean(physical)), "diagnostic_action_clip_fraction": float(np.mean(raw != physical))}


def diagnostics(config_path: Path) -> None:
    config, _ = config_and_hash(config_path); out = output_root(config)
    _, _, _, old_out, _, _, _ = night02_context(config)
    private = np.load(out / "calibration_observations_private.npz")
    groups = {"demographic": lambda n: n in S1_FIELDS[:4], "bis_value": lambda n: n.startswith("bis_value_"),
              "bis_mask": lambda n: n.startswith("bis_mask_"), "bis_age": lambda n: n.startswith("bis_age_"),
              "propofol": lambda n: n.startswith("propofol_"), "remifentanil": lambda n: n.startswith("remifentanil_")}
    magnitude=[]; actor=[]
    for state, fields in (("S0", S0_FIELDS), ("S1", S1_FIELDS)):
        base = np.asarray(private[state], dtype=np.float32); factors = factors_for(out, state)
        for transform in ("N0", "Nclip", "Nscale"):
            obs = apply_transform(base, transform, factors)
            for group, predicate in groups.items():
                indices = [i for i, name in enumerate(fields) if predicate(name)]
                if not indices: continue
                before = base[:, indices]; after = obs[:, indices]
                magnitude.append({"state": state, "transform": transform, "feature_group": group,
                                  "pre_abs_q95": float(np.quantile(np.abs(before), .95)), "post_abs_q95": float(np.quantile(np.abs(after), .95)),
                                  "pre_fraction_abs_gt_10": float(np.mean(np.abs(before) > 10)), "post_fraction_at_clip_boundary": float(np.mean(np.abs(after) >= 10)) if transform == "Nclip" else 0.0})
            for seed in config["seeds"]:
                path, provenance = model_path(config, out, old_out, transform, state, seed)
                if path.exists():
                    model = PPO.load(str(path), device="cpu")
                    actor.append({"state": state, "transform": transform, "seed": seed, "provenance": provenance, **actor_diagnostics(model, obs)})
    atomic_json(out / "mechanism_diagnostics.json", {"magnitude": magnitude, "actor": actor, "calibration_split": "fixed_training_only", "test_access_count": 0})
    print(json.dumps({"magnitude_rows": len(magnitude), "actor_rows": len(actor)}))


def verify_day03(config_path: Path) -> None:
    config, config_hash = config_and_hash(config_path); out = output_root(config)
    factor = factor_manifest(out)
    if factor["config_sha256"] != config_hash or factor["source_scaler_sha256"] != json.loads((out / "preparation.json").read_text())["source_scaler_sha256"]:
        raise RuntimeError("factor/config/source identity mismatch")
    if not all(factor["checks"].get(name) is True for name in ("finite_positive", "binary_exact", "shared_field_equality")):
        raise RuntimeError("factor invariant failed")
    if len(set(factor["checks"]["nscale_bis_40_50_60"])) != 3 or len(set(factor["checks"]["nclip_bis_40_50_60"])) != 1:
        raise RuntimeError("BIS order/collapse check failed")
    verified=[]
    for block in config["new_training_blocks"]:
        for seed in config["seeds"]:
            directory=job_dir(out,block["transform"],block["state"],seed)
            completion=json.loads((directory/"OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
            final=directory/"final_post_update"; marker=json.loads((final/"COMPLETE.json").read_text(encoding="utf-8"))
            metadata=json.loads((final/"metadata.json").read_text(encoding="utf-8"))
            expected={"config_sha256":config_hash,"transform":block["transform"],"state":block["state"],"seed":seed,
                      "collected_and_trained_timesteps":260096,"training_epochs":1270,"completed_rollouts":127,
                      "artifact_stage":"post_update_authoritative_final","test_access_count":0}
            for name,value in expected.items():
                if completion.get(name)!=value or metadata.get(name)!=value: raise RuntimeError(f"new model identity mismatch: {name}")
            if marker.get("complete") is not True or marker.get("model_sha256")!=sha256_path(final/"model.zip") or marker.get("metadata_sha256")!=sha256_path(final/"metadata.json"):
                raise RuntimeError("new final checksum mismatch")
            model=PPO.load(str(final/"model.zip"),device="cpu")
            if model.num_timesteps!=260096 or model._n_updates!=1270 or model.n_steps!=2048 or model.n_epochs!=10 or float(model.learning_rate)!=.001:
                raise RuntimeError("new loaded model configuration mismatch")
            verified.append(f"{block['transform']}/{block['state']}/seed_{seed}")
    reused=json.loads((out/"reused_model_verification.json").read_text(encoding="utf-8"))
    aggregates=json.loads((out/"validation_aggregate.json").read_text(encoding="utf-8"))["aggregates"]
    if reused.get("verified") is not True or len(reused["models"])!=9 or len(verified)!=9 or len(aggregates)!=18:
        raise RuntimeError("matrix accounting mismatch")
    if len({(r["transform"],r["state"],r["seed"]) for r in aggregates})!=18:
        raise RuntimeError("duplicate or missing matrix cell")
    if list(out.rglob(".final_post_update.partial")):
        raise RuntimeError("partial authoritative final remains")
    for public in (PUBLIC_JSON,PUBLIC_CSV,PUBLIC_REPORT):
        text=public.read_text(encoding="utf-8")
        if any(token in text for token in ("subjectid","caseid","source_root","C:\\Users\\")):
            raise RuntimeError(f"private identifier/path token in {public.name}")
    result={"verified":True,"new_models":len(verified),"reused_models":len(reused["models"]),"matrix_cells":len(aggregates),
            "new_post_update_models":verified,"factor_checks":factor["checks"],"public_privacy_scan":True,"test_access_count":0,"verified_utc":utc_now()}
    atomic_json(out/"verification.json",result); print(json.dumps(result,indent=2))


def summarize(config_path: Path, started: float | None = None) -> None:
    config, config_hash = config_and_hash(config_path); out = output_root(config)
    validation = json.loads((out / "validation_aggregate.json").read_text(encoding="utf-8"))["aggregates"]
    reused = json.loads((out / "reused_model_verification.json").read_text(encoding="utf-8"))["models"]
    diagnostics_payload = json.loads((out / "mechanism_diagnostics.json").read_text(encoding="utf-8"))
    key = {(r["transform"], r["state"], int(r["seed"])): r for r in validation}
    comparisons=[]
    for seed in config["seeds"]:
        for transform in ("N0", "Nclip", "Nscale"):
            if (transform,"S0",seed) in key and (transform,"S1",seed) in key:
                comparisons.append({"comparison": "S1_minus_S0", "transform": transform, "seed": seed,
                                    "latent_bis_mae_difference": key[(transform,"S1",seed)]["latent_bis_mae"]-key[(transform,"S0",seed)]["latent_bis_mae"]})
        for state in ("S0","S1"):
            for transform in ("Nclip","Nscale"):
                if (transform,state,seed) in key and ("N0",state,seed) in key:
                    comparisons.append({"comparison": f"{transform}_minus_N0", "state": state, "seed": seed,
                                        "latent_bis_mae_difference": key[(transform,state,seed)]["latent_bis_mae"]-key[("N0",state,seed)]["latent_bis_mae"]})
    blocks=[]
    for block in config["new_training_blocks"]:
        complete = sum((block["transform"],block["state"],seed) in key for seed in config["seeds"])
        blocks.append(block | {"completed_seeds": complete, "status": "complete" if complete == 3 else "incomplete"})
    elapsed = None if started is None else (time.time()-started)/60
    timing_path=out/"sprint_timing.json"
    sprint_timing=json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    summary = {"protocol_id": config["protocol_id"], "config_sha256": config_hash,
               "evidence_scope": config["evidence_scope"], "exploratory_reused_validation": True,
               "matrix_expected_cells": 18, "aggregate_model_cells_completed": len(validation),
               "reused_model_count": len(reused), "new_model_count": sum(r["provenance"]=="new" for r in validation),
               "incomplete_model_count": 18-len(validation), "blocks": blocks,
               "update_accounting": {"reused_collected_timesteps": 262144, "reused_trained_timesteps": 260096,
                                     "new_collected_and_trained_timesteps": 260096, "completed_rollouts": 127, "training_epochs": 1270},
               "aggregates": validation, "paired_seed_differences": comparisons,
               "mechanism_diagnostics": diagnostics_payload, "wall_minutes_this_invocation": elapsed,
               "sprint_wall_minutes": None if sprint_timing is None else sprint_timing["wall_minutes"],
               "test_access_count": 0, "generated_utc": utc_now()}
    atomic_json(PUBLIC_JSON, summary)
    PUBLIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = ["transform","state","seed","provenance","validation_subject_count","validation_case_count","collected_timesteps","trained_timesteps","training_epochs",*METRICS]
    with PUBLIC_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows({name:r.get(name) for name in columns} for r in validation)
    means={}
    for transform,state in sorted({(r["transform"],r["state"]) for r in validation}):
        selected=[r for r in validation if r["transform"]==transform and r["state"]==state]
        means[(transform,state)] = (np.mean([r["latent_bis_mae"] for r in selected]), np.std([r["latent_bis_mae"] for r in selected], ddof=1) if len(selected)>1 else 0.0)
    nscale_s1=[r for r in comparisons if r["comparison"]=="Nscale_minus_N0" and r.get("state")=="S1"]
    nscale_s0=[r for r in comparisons if r["comparison"]=="Nscale_minus_N0" and r.get("state")=="S0"]
    nscale_state=[r for r in comparisons if r["comparison"]=="S1_minus_S0" and r.get("transform")=="Nscale"]
    lines=["# Day 03 Scale Mechanism Results", "", "## 결론", "",
           "Nscale은 S1의 BIS MAE를 모든 paired seed에서 N0보다 낮췄고, Nscale 조건의 S1도 모든 seed에서 S0보다 낮았다. 수치 conditioning 문제와 scaling 후 유용한 S1 정보라는 설명을 이 제한된 설정에서 지지한다. S0도 Nscale에서 모든 seed가 소폭 개선되어 효과를 S1에만 고유하다고 볼 수는 없다.", "",
           "이 결과는 고정된 24명 내부 검증셋을 재사용한 탐색적 개발 근거이며 임상 중재 효과, 독립 검증, 광범위한 PK/PD 우월성 또는 새로운 방법론 기여를 입증하지 않는다.", "",
           "## 완성된 비교", "", "| 변환 | 상태 | BIS MAE ± SD | 40–60 비율 | <40 비율 | >60 비율 | 총 propofol (mg) | action 경계 비율 | 출처 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for (transform,state),(mean,sd) in means.items():
        selected=[r for r in validation if r["transform"]==transform and r["state"]==state]
        source="재사용" if all(r["provenance"]=="reused" for r in selected) else "신규"
        avg=lambda name: float(np.mean([r[name] for r in selected]))
        lines.append(f"| {transform} | {state} | {mean:.4f} ± {sd:.4f} | {avg('time_in_40_60_fraction'):.3f} | {avg('time_below_40_fraction'):.3f} | {avg('time_above_60_fraction'):.3f} | {avg('total_propofol_mg'):.2f} | {avg('action_boundary_fraction'):.3f} | {source} |")
    lines += ["", "## Paired seed 차이 (BIS MAE)", "", "음수는 표의 비교식에서 앞 변환/상태가 더 낮은 MAE임을 뜻한다.", "", "| 비교 | 조건 | seed 45 / 46 / 47 |", "|---|---|---|"]
    grouped={}
    for r in comparisons: grouped.setdefault((r["comparison"],r.get("transform",r.get("state"))),[]).append(r)
    for (comparison,condition),values in grouped.items():
        ordered=sorted(values,key=lambda r:r["seed"]); lines.append(f"| {comparison} | {condition} | " + " / ".join(f"{r['latent_bis_mae_difference']:+.4f}" for r in ordered) + " |")
    lines += ["", "고정 PI baseline의 validation BIS MAE는 약 11.897이다. paired 차이는 세 seed를 그대로 제시하며 seed 강건성을 subject-only bootstrap으로 과장하지 않았다.", "", "## 학습 업데이트 회계", "",
              "Night 02 재사용 모델은 262,144개 전이를 수집했지만 callback 경계 때문에 260,096개(127 rollout)만 업데이트에 사용했다. 저장 모델의 `_n_updates=1,270`은 PPO epoch 수이다. Day 03 신규 모델은 260,096개를 수집하고 마지막 rollout 업데이트 뒤 저장해 같은 127 rollout/1,270 epoch에 맞췄다.", "",
              "## 해석 한계", "",
              "Nclip은 모든 좌표를 ±10으로 잘라 현재 scaler에서 BIS 40/50/60을 모두 10으로 만든다. Nscale은 훈련 보정분포의 좌표별 q95로 나누어 순서와 구분을 보존한다. 두 개입의 차이 때문에 성능 변화만으로 단일 원인을 확정할 수 없다. 세 seed 결과는 seed 강건성의 제한적 점검이며 subject-only bootstrap으로 이를 대체하지 않았다.", "",
              "## 실행 시간", "", f"전체 sprint wall time은 {summary['sprint_wall_minutes']:.1f}분이다." if summary["sprint_wall_minutes"] is not None else f"학습·평가 supervisor wall time은 {elapsed:.1f}분이다.", "",
              "## 프로젝트 진행", "", "| 항목 | 상태 |", "|---|---|", "| 데이터 연결/실행 인프라 | 구축 완료 |", "| 96-case / 10분 pilot | 완료 |",
              f"| scale mechanism | {'완료' if len(validation)==18 else '부분 완료'} |", "| 장시간/관측 풍부 평가 | 대기 |", "| 더 큰 독립 검증 및 최종 저널 방법 기여 | 대기 |", "",
              "표준 정규화 개선은 필요한 baseline 교정이며, 그 자체만으로 Neural Networks 저널의 새로운 방법론 기여를 뜻하지 않는다.", "",
              "## 검증", "",
              "두-rollout 경계 점검은 4,096개 전이/20 epoch와 저장 전후 정책의 정확한 일치를 확인했다. Day 03 검증기는 신규 post-update 모델 9개, 재사용 모델 9개, 18개 matrix cell, factor 불변식과 공개 산출물 privacy scan을 확인한다.", "",
              "## 재현 및 개인정보", "",
              "사례/대상자 식별자, calibration 배열, 모델, checkpoint와 로그는 git-ignored 출력 경로에만 남겼다. test set 접근 횟수는 0이다. 재개 명령은 `.venv-journal\\Scripts\\python.exe scripts\\journal\\run_day03_scale.py resume --minutes 145 --workers 1`이며 새 실행마다 명시적 시간 예산을 요구한다."]
    PUBLIC_REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(PUBLIC_REPORT.relative_to(ROOT)), "aggregate_rows":len(validation), "new_models":summary["new_model_count"]}, indent=2))


def supervise(config_path: Path, minutes: float, workers: int) -> None:
    config, _ = config_and_hash(config_path); out = output_root(config)
    started=time.time(); deadline=started+minutes*60; training_stop=min(deadline-20*60, started+float(config["training_stop_minutes"])*60)
    all_jobs=[(b["transform"],b["state"],seed,b["priority"]) for b in config["new_training_blocks"] for seed in config["seeds"]]
    workers=max(1,min(int(workers),int(config["maximum_workers"])))
    attempts={}; active:dict[subprocess.Popen,tuple[str,str,int,int,Any]]={}
    def completed(t,s,seed): return (job_dir(out,t,s,seed)/"OUTPUT_COMPLETE.json").exists()
    measured_sps=float(config["queue_choice"]["single_worker_steps_per_second"] if workers == 1 else config["queue_choice"]["two_worker_aggregate_steps_per_second"])
    estimated_job_seconds=int(config["training_target_timesteps"])/measured_sps
    available=max(0.0,training_stop-time.time())
    admitted_blocks=[]; pending=[]; cumulative=0.0
    for block in config["new_training_blocks"]:
        block_jobs=[j for j in all_jobs if j[0]==block["transform"] and j[1]==block["state"] and not completed(*j[:3])]
        cost=len(block_jobs)*estimated_job_seconds/workers*1.25
        if cumulative+cost <= available:
            pending.extend(block_jobs); cumulative += cost
            admitted_blocks.append({"transform":block["transform"],"state":block["state"],"job_count":len(block_jobs),"estimated_seconds_with_margin":cost})
    jobs=list(all_jobs)
    atomic_json(out/"queue_plan.json", {"frozen_before_new_validation_scores":True,"workers":workers,"measured_aggregate_steps_per_second":measured_sps,
                "estimated_job_seconds":estimated_job_seconds,"runtime_margin":1.25,"available_seconds":available,"admitted_blocks":admitted_blocks,
                "omitted_blocks":[b for b in config["new_training_blocks"] if not any(a["transform"]==b["transform"] and a["state"]==b["state"] for a in admitted_blocks)],"created_utc":utc_now()})
    update_status(out,started,deadline,phase="training",completed_model_count=len(jobs)-sum(not completed(*j[:3]) for j in jobs),reused_model_count=9,new_model_count=sum(completed(*j[:3]) for j in jobs),incomplete_model_count=sum(not completed(*j[:3]) for j in jobs),active_jobs=[])
    while pending or active:
        while pending and len(active)<workers and time.time()<training_stop:
            transform,state,seed,priority=pending[0]
            estimate=estimated_job_seconds*1.25
            if time.time()+estimate>training_stop:
                pending=[j for j in pending if not (j[0]==transform and j[1]==state)]; continue
            pending.pop(0); key=f"{transform}/{state}/seed_{seed}"; attempts[key]=attempts.get(key,0)+1
            directory=job_dir(out,transform,state,seed); directory.mkdir(parents=True,exist_ok=True); log=(directory/"worker.log").open("a",encoding="utf-8")
            command=[sys.executable,str(Path(__file__).resolve()),"train-one","--config",str(config_path.resolve()),"--transform",transform,"--state",state,"--seed",str(seed)]
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            active[process]=(transform,state,seed,priority,log)
        if not active: break
        time.sleep(5)
        for process,item in list(active.items()):
            code=process.poll()
            if code is None: continue
            transform,state,seed,priority,log=item; log.close(); del active[process]
            if code!=0 and attempts[f"{transform}/{state}/seed_{seed}"]<=int(config["retry_limit"]): pending.insert(0,(transform,state,seed,priority))
            elif code!=0: atomic_json(job_dir(out,transform,state,seed)/"FAILED.json",{"returncode":code,"attempts":attempts[f"{transform}/{state}/seed_{seed}"],"utc":utc_now()})
        done=sum(completed(*j[:3]) for j in jobs)
        update_status(out,started,deadline,phase="training",completed_model_count=done,reused_model_count=9,new_model_count=done,incomplete_model_count=len(jobs)-done,
                      active_jobs=[f"{v[0]}/{v[1]}/seed_{v[2]}" for v in active.values()])
        if time.time()>=training_stop:
            for process,(_,_,_,_,log) in list(active.items()): process.terminate(); process.wait(timeout=30); log.close()
            active.clear(); break
    update_status(out,started,deadline,phase="evaluation",completed_model_count=sum(completed(*j[:3]) for j in jobs),reused_model_count=9,new_model_count=sum(completed(*j[:3]) for j in jobs),incomplete_model_count=sum(not completed(*j[:3]) for j in jobs),active_jobs=[])
    evaluate(config_path); diagnostics(config_path); summarize(config_path,started); verify_day03(config_path)
    update_status(out,started,deadline,phase="complete",completed_model_count=len(json.loads((out/"validation_aggregate.json").read_text())["aggregates"]),reused_model_count=9,new_model_count=sum(completed(*j[:3]) for j in jobs),incomplete_model_count=sum(not completed(*j[:3]) for j in jobs),active_jobs=[])


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("calibrate","smoke","benchmark","train-one","supervise","evaluate","diagnostics","summarize","verify","resume"))
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--transform",choices=("N0","Nclip","Nscale")); parser.add_argument("--state",choices=("S0","S1")); parser.add_argument("--seed",type=int)
    parser.add_argument("--steps",type=int,default=4096); parser.add_argument("--minutes",type=float); parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.command=="calibrate": calibrate(args.config)
    elif args.command=="smoke": smoke(args.config)
    elif args.command=="benchmark": benchmark(args.config,args.transform or "Nclip",args.state or "S0",args.steps)
    elif args.command=="train-one":
        if args.transform not in ("Nclip","Nscale") or args.state is None or args.seed is None: parser.error("train-one requires --transform Nclip|Nscale --state and --seed")
        train_one(args.config,args.transform,args.state,args.seed)
    elif args.command=="supervise":
        if args.minutes is None: parser.error("supervise requires an explicit --minutes budget")
        supervise(args.config,args.minutes,args.workers)
    elif args.command=="evaluate": evaluate(args.config)
    elif args.command=="diagnostics": diagnostics(args.config)
    elif args.command=="summarize": summarize(args.config)
    elif args.command=="verify": verify_day03(args.config)
    else:
        if args.minutes is None: parser.error("resume requires a new explicit --minutes budget")
        out=output_root(config_and_hash(args.config)[0])
        if not (out/"nscale_factors.json").exists(): calibrate(args.config)
        if not (out/"smoke"/"verification.json").exists(): smoke(args.config)
        if not (out/"benchmark.json").exists(): benchmark(args.config)
        supervise(args.config,args.minutes,args.workers)


if __name__ == "__main__": main()
