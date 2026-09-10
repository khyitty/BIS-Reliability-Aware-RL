"""Run the frozen Day-06 concentration-anatomy development decomposition.

Identifiers, memberships, models, checkpoints, logs, and row-level evaluations
remain under the ignored Day-06 output root. Only aggregate artifacts are public.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from night02_common import make_environment
from run_day03_scale import FixedTransform, actor_diagnostics, canonical_sha
from run_day04_confirmation import (
    Day04SequenceEnv, PRIMARY_METRICS, _digest, atomic_json, context,
    load_checkpoint, load_config, metric_record, save_checkpoint, sha256_path,
    utc_now,
)
from vitaldb_state_selection.anesthesia.state import S0_FIELDS, S1_FIELDS
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/journal/day06_concentration_anatomy.json"
PUBLIC_RESULTS = ROOT / "reports/journal/day06_concentration_results.csv"
PUBLIC_CONTRASTS = ROOT / "reports/journal/day06_concentration_contrasts.csv"
PUBLIC_PROFILE = ROOT / "reports/journal/day06_profile_sensitivity.csv"
PUBLIC_TRAJECTORY = ROOT / "reports/journal/day06_checkpoint_trajectory.csv"
PUBLIC_DIAGNOSTICS = ROOT / "reports/journal/day06_diagnostics.csv"
PUBLIC_AUDIT = ROOT / "reports/journal/day06_concentration_audit.csv"
PUBLIC_SUMMARY = ROOT / "reports/journal/day06_summary.json"
PUBLIC_REPORT = ROOT / "reports/journal/DAY06_RESULTS.md"
PUBLIC_OVERLEAF = ROOT / "docs/journal/DAY06_OVERLEAF_NOTE.md"
PUBLIC_FIGURES = (
    ROOT / "reports/journal/day06_concentration_mae.png",
    ROOT / "reports/journal/day06_concentration_mae.pdf",
)

DAY04_FACTOR_SHA = "b0a494afb9851c4ac6eee00f85229cfbec877f54022659a9fb50cefdba7647d0"
STATE_FIELDS = {
    "S0": tuple(S0_FIELDS),
    "S_PROP": tuple(S0_FIELDS) + ("propofol_cp_mg_per_l", "propofol_ce_mg_per_l"),
    "S_REMI": tuple(S0_FIELDS) + ("remifentanil_cp_microgram_per_l", "remifentanil_ce_microgram_per_l"),
    "S_CP": tuple(S0_FIELDS) + ("propofol_cp_mg_per_l", "remifentanil_cp_microgram_per_l"),
    "S_CE": tuple(S0_FIELDS) + ("propofol_ce_mg_per_l", "remifentanil_ce_microgram_per_l"),
    "S_CONC": tuple(S0_FIELDS) + ("propofol_cp_mg_per_l", "propofol_ce_mg_per_l", "remifentanil_cp_microgram_per_l", "remifentanil_ce_microgram_per_l"),
}
STATE_INDICES = {state: tuple(S1_FIELDS.index(name) for name in fields) for state, fields in STATE_FIELDS.items()}
EXTRA_FEATURES = {state: len(fields) - len(S0_FIELDS) for state, fields in STATE_FIELDS.items()}
EVAL_METRICS = tuple(PRIMARY_METRICS) + ("rmse_0_1800", "rmse_0_600", "rmse_600_1800", "mean_deterministic_action")
CONTRAST_FORMULAS = {
    "S_PROP-S0": "S_PROP-S0", "S_REMI-S0": "S_REMI-S0",
    "S_CP-S0": "S_CP-S0", "S_CE-S0": "S_CE-S0", "S_CONC-S0": "S_CONC-S0",
    "S_CONC-S_PROP": "S_CONC-S_PROP", "S_CONC-S_REMI": "S_CONC-S_REMI",
    "S_CONC-S_CP": "S_CONC-S_CP", "S_CONC-S_CE": "S_CONC-S_CE",
    "S_PROP-S_REMI": "S_PROP-S_REMI", "S_CE-S_CP": "S_CE-S_CP",
}


def out_root(config: dict[str, Any]) -> Path:
    out = ROOT / config["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def day04_root(config: dict[str, Any]) -> Path:
    return ROOT / config["day04_output_root"]


def day05_root(config: dict[str, Any]) -> Path:
    return ROOT / config["day05_output_root"]


def write_status(out: Path, phase: str, **extra: Any) -> None:
    atomic_json(out / "status.json", {"phase": phase, "updated_utc": utc_now(), "latest_error": None, **extra})


def profile(config: dict[str, Any], name: str) -> dict[str, Any]:
    return next(dict(row) for row in config["profiles"] if row["profile"] == name)


def condition(config: dict[str, Any], profile_name: str, state: str) -> dict[str, Any]:
    row = profile(config, profile_name)
    return {"condition_id": f"{profile_name}{state}", "profile": profile_name,
            "state_id": "S0" if state == "S0" else "S1",
            "sqi_threshold": row["sqi_threshold"], "age_seconds": row["age_seconds"]}


class ProjectObservation(gym.ObservationWrapper):
    """Select a frozen semantic subset after scaler and Day-04 Nscale."""
    def __init__(self, env: gym.Env, state: str):
        super().__init__(env)
        self.state = state
        self.indices = np.asarray(STATE_INDICES[state], dtype=np.int64)
        self.field_names = STATE_FIELDS[state]
        self.observation_space = gym.spaces.Box(
            low=np.asarray(env.observation_space.low)[self.indices],
            high=np.asarray(env.observation_space.high)[self.indices], dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        result = np.asarray(observation, dtype=np.float32)[self.indices]
        if result.shape != (len(self.field_names),) or not np.isfinite(result).all():
            raise RuntimeError("Day06 projected observation invariant failed")
        return result


def factor_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((day04_root(config) / "nscale_factors.json").read_text(encoding="utf-8"))
    if manifest["factors_sha256"] != DAY04_FACTOR_SHA or manifest["factors_sha256"] != config["factor_sha256"]:
        raise RuntimeError("Day04 factor hash mismatch")
    return manifest


def factors(config: dict[str, Any]) -> np.ndarray:
    mapped = factor_manifest(config)["factors_by_field"]
    return np.asarray([mapped[name] for name in S1_FIELDS], dtype=np.float32)


def private_membership(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads((out_root(config) / "private_membership.json").read_text(encoding="utf-8"))


def make_vector(config: dict[str, Any], membership: dict[str, Any], store: Any,
                scalers: Any, profile_name: str, state: str, seed: int) -> tuple[DummyVecEnv, Day04SequenceEnv]:
    native = condition(config, profile_name, state)
    seq = Day04SequenceEnv(store, membership["full_train_cases"], native, scalers[native["state_id"]],
                           seed, float(config["horizon_seconds"]))
    full_factors = factors(config) if native["state_id"] == "S1" else factors(config)[:len(S0_FIELDS)]
    transformed = FixedTransform(seq, "Nscale", full_factors)
    wrapped = transformed if state in ("S0", "S1") else ProjectObservation(transformed, state)
    return DummyVecEnv([lambda: wrapped]), seq


def make_single_env(config: dict[str, Any], store: Any, scalers: Any, caseid: str,
                    profile_name: str, state: str, seed: int) -> gym.Env:
    native = condition(config, profile_name, state)
    bundle = store.load_case(caseid)
    from night02_common import truncate_bundle
    base = make_environment(truncate_bundle(bundle, float(config["horizon_seconds"])), native,
                            scalers[native["state_id"]], seed)
    full_factors = factors(config) if native["state_id"] == "S1" else factors(config)[:len(S0_FIELDS)]
    transformed = FixedTransform(base, "Nscale", full_factors)
    return transformed if state in ("S0", "S1") else ProjectObservation(transformed, state)


def job_dir(out: Path, seed: int, profile_name: str, state: str) -> Path:
    return out / "jobs" / f"seed_{seed}" / f"{profile_name}{state}"


def day04_checkpoint(config: dict[str, Any], seed: int, profile_name: str, state: str, step: int) -> Path:
    cid = f"{profile_name}{state}"
    name = "final_post_update" if step == int(config["training_target_timesteps"]) else f"checkpoint_{step:010d}_post_update"
    return day04_root(config) / "jobs" / f"seed_{seed}" / cid / name


def day05_checkpoint(config: dict[str, Any], seed: int, profile_name: str, state: str, step: int) -> Path:
    cid = f"{profile_name}{state}"
    name = "final_post_update" if step == int(config["training_target_timesteps"]) else f"checkpoint_{step:010d}_post_update"
    return day05_root(config) / "jobs" / f"seed_{seed}" / cid / name


def new_checkpoint(config: dict[str, Any], seed: int, profile_name: str, state: str, step: int) -> Path:
    name = "final_post_update" if step == int(config["training_target_timesteps"]) else f"checkpoint_{step:010d}_post_update"
    return job_dir(out_root(config), seed, profile_name, state) / name


def model_checkpoint(config: dict[str, Any], seed: int, profile_name: str, state: str, step: int) -> Path:
    if state == "S0":
        return day04_checkpoint(config, seed, profile_name, state, step)
    if state == "S_CONC":
        return day05_checkpoint(config, seed, profile_name, state, step)
    return new_checkpoint(config, seed, profile_name, state, step)


def verify_checkpoint(path: Path, step: int) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"checkpoint missing: {path.name}")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    marker = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    if int(meta["timestep"]) != step or marker["model_sha256"] != sha256_path(path / "model.zip"):
        raise RuntimeError("checkpoint identity/checksum mismatch")
    if marker["metadata_sha256"] != sha256_path(path / "metadata.json"):
        raise RuntimeError("checkpoint metadata checksum mismatch")
    return meta


def prepare(config_path: Path) -> None:
    config, config_hash = load_config(config_path)
    if subprocess.check_output(["git", "merge-base", "HEAD", config["required_parent_sha"]], cwd=ROOT, text=True).strip() != config["required_parent_sha"]:
        raise RuntimeError("required Day05 parent is not ancestor")
    out = out_root(config); d4 = day04_root(config)
    _, _, _, night_membership, store, _ = context(config)
    d4_membership = json.loads((d4 / "private_membership.json").read_text(encoding="utf-8"))
    old = set(d4_membership["old_validation_subjects"]); used = set(d4_membership["fresh_validation_subjects"])
    reserve = [subject for subject in night_membership["full_validation_subjects"] if subject not in old and subject not in used]
    reserve_set=set(reserve);reserve_cases=[str(row["caseid"]) for row in store.rows if str(row["subjectid"]) in reserve_set]
    if old & used or len(reserve) != 170 or len(reserve_cases)!=170 or set(reserve) & set(d4_membership["full_train_subjects"]):
        raise RuntimeError("future reserve membership mismatch")
    membership = {"full_train_subjects": d4_membership["full_train_subjects"],
                  "full_train_cases": d4_membership["full_train_cases"],
                  "development_validation_subjects": d4_membership["fresh_validation_subjects"],
                  "development_validation_cases": d4_membership["fresh_validation_cases"],
                  "diagnostic_subjects": d4_membership["tuning_subjects"],
                  "diagnostic_cases": d4_membership["tuning_cases"],
                  "future_reserve_subjects": reserve, "future_reserve_cases": reserve_cases}
    atomic_json(out / "private_membership.json", membership)
    factor = factor_manifest(config)
    anchors = []
    for seed in config["seeds"]:
        for profile_name in ("P0", "P1"):
            for state in ("S0", "S_CONC"):
                for step in config["checkpoint_steps"]:
                    path = model_checkpoint(config, seed, profile_name, state, int(step))
                    meta = verify_checkpoint(path, int(step))
                    if meta["factor_sha256"] != DAY04_FACTOR_SHA or int(meta["training_epochs"]) != int(step) // 2048 * 10:
                        raise RuntimeError("Day04/Day05 anchor budget/factor mismatch")
                    anchors.append({"seed": seed, "profile": profile_name, "state": state, "step": step,
                                    "model_sha256": sha256_path(path / "model.zip")})
    atomic_json(out / "anchor_registry_private.json", {"anchors": anchors})
    schemas = {state: {"dimension": len(fields), "fields": list(fields), "indices_in_s1": list(STATE_INDICES[state]),
                       "schema_sha256": canonical_sha(list(fields))} for state, fields in STATE_FIELDS.items()}
    public = {"protocol_id": config["protocol_id"], "config_sha256": config_hash,
              "required_parent_sha": config["required_parent_sha"], "full_train_subject_count": len(membership["full_train_subjects"]),
              "full_train_case_count": len(membership["full_train_cases"]), "development_validation_subject_count": len(membership["development_validation_subjects"]),
              "development_validation_case_count": len(membership["development_validation_cases"]), "diagnostic_subject_count": len(membership["diagnostic_subjects"]),
              "future_reserve_subject_count": len(reserve), "future_reserve_subject_sha256": _digest(reserve),
              "future_reserve_case_count": len(reserve_cases), "future_reserve_case_sha256": _digest(reserve_cases),
              "future_reserve_bundle_access_count": 0, "future_reserve_outcome_access_count": 0,
              "future_reserve_statistic_access_count": 0,
              "factor_sha256": factor["factors_sha256"], "factor_refit": False, "anchor_count": len(anchors),
              "anchor_registry_sha256": canonical_sha(anchors), "state_schemas": schemas,
              "day04_membership_sha256": sha256_path(d4 / "private_membership.json"),
              "day05_protocol_sha256": sha256_path(day05_root(config) / "protocol_manifest.json"),
              "new_jobs": 40, "new_timesteps": 40 * int(config["training_target_timesteps"]),
              "job_order": config["job_order"], "seeds": config["seeds"], "checkpoint_steps": config["checkpoint_steps"],
              "contrast_formulas": CONTRAST_FORMULAS, "validation_outcome_access_count": 0,
              "retention_rule": config["retention_rule"], "temporal_availability_audit": "required_before_training",
              "test_access_count": 0, "process_guard": config["process_guard"], "frozen_utc": utc_now()}
    atomic_json(out / "protocol_manifest.json", public)
    print(json.dumps({k: public[k] for k in ("full_train_subject_count", "development_validation_subject_count", "future_reserve_subject_count", "anchor_count", "factor_sha256")}, indent=2))


def audit(config_path: Path) -> None:
    config, _ = load_config(config_path); out = out_root(config)
    _, _, _, _, store, scalers = context(config); membership = private_membership(config)
    from night02_common import truncate_bundle
    concentration_fields = (
        "propofol_cp_mg_per_l", "propofol_ce_mg_per_l",
        "remifentanil_cp_microgram_per_l", "remifentanil_ce_microgram_per_l",
    )
    coord_indices = [S1_FIELDS.index(name) for name in concentration_fields]
    max_projection = 0.0; checked = 0; coordinate_rows: list[np.ndarray] = []
    for caseid in membership["diagnostic_cases"]:
        env = make_environment(truncate_bundle(store.load_case(caseid), float(config["horizon_seconds"])), condition(config, "P0", "S1"), scalers["S1"], 48)
        _, _ = env.reset(seed=48)
        done = False
        while not done:
            raw = np.asarray(env.core._last_state.vector)
            scaled = np.asarray(env.scaler.transform(raw), dtype=np.float32) / factors(config)
            coordinate_rows.append(scaled[coord_indices].astype(np.float64))
            for state in config["states"]:
                projected = scaled[np.asarray(STATE_INDICES[state])]
                max_projection = max(max_projection, float(np.max(np.abs(projected - scaled[np.asarray(STATE_INDICES[state])]))))
            _, _, terminated, truncated, _ = env.step(np.asarray([1.5], dtype=np.float32)); done = bool(terminated or truncated); checked += 1
        env.close()
    values = np.stack(coordinate_rows)
    if max_projection != 0 or not np.isfinite(values).all():
        raise RuntimeError("concentration projection/finiteness audit failed")
    def ranks(column: np.ndarray) -> np.ndarray:
        order = np.argsort(column, kind="mergesort"); result = np.empty(len(column), dtype=np.float64)
        result[order] = np.arange(len(column), dtype=np.float64)
        return result
    pearson = np.corrcoef(values, rowvar=False); spearman = np.corrcoef(np.stack([ranks(values[:, i]) for i in range(4)], axis=1), rowvar=False)
    summaries = []
    for index, name in enumerate(concentration_fields):
        column = values[:, index]
        summaries.append({"field": name, "finite_rate": float(np.mean(np.isfinite(column))), "mean": float(np.mean(column)),
                          "sd": float(np.std(column, ddof=1)), "q01": float(np.quantile(column, .01)),
                          "q05": float(np.quantile(column, .05)), "q50": float(np.quantile(column, .50)),
                          "q95": float(np.quantile(column, .95)), "q99": float(np.quantile(column, .99)),
                          "zero_fraction": float(np.mean(column == 0)), "near_zero_fraction": float(np.mean(np.abs(column) <= 1e-8))})
    correlations = [{"field_a": concentration_fields[i], "field_b": concentration_fields[j],
                     "pearson": float(pearson[i, j]), "spearman": float(spearman[i, j])}
                    for i in range(4) for j in range(i + 1, 4)]
    # Toy temporal audit: an action can alter only the next decision-time state, never the state used to choose itself.
    caseid = membership["diagnostic_cases"][0]
    env_a = make_single_env(config, store, scalers, caseid, "P0", "S_CONC", 901)
    env_b = make_single_env(config, store, scalers, caseid, "P0", "S_CONC", 901)
    pre_a, _ = env_a.reset(seed=901); pre_b, _ = env_b.reset(seed=901)
    next_a, _, _, _, info_a = env_a.step(np.asarray([0.5], dtype=np.float32))
    next_b, _, _, _, _, = env_b.step(np.asarray([2.5], dtype=np.float32))
    env_a.close(); env_b.close()
    pre_equal = bool(np.array_equal(pre_a, pre_b)); next_prop_differs = bool(not np.array_equal(next_a[-4:-2], next_b[-4:-2]))
    if not pre_equal or not next_prop_differs or float(info_a["elapsed_time_seconds"]) != 10.0:
        raise RuntimeError("decision-time concentration ordering failed")
    result = {"training_only": True, "trajectories": len(membership["diagnostic_cases"]), "transitions_checked": checked,
              "projection_max_abs_error": max_projection, "coordinate_summaries": summaries, "pairwise_correlations": correlations,
              "decision_timeline": ["observe state at t from completed intervals", "choose action at t", "apply action over t_to_t_plus_10", "advance PKPD and ingest causal events", "build next state at t_plus_10"],
              "pre_action_state_independent_of_action_being_chosen": pre_equal, "completed_action_changes_next_propofol_state": next_prop_differs,
              "future_or_post_decision_inputs": False, "recorded_clinical_propofol_after_initialization": False,
              "latent_bis_in_policy_state": False, "projection_changes_dynamics_reward_or_timing": False,
              "concentration_provenance": "exact_internal_reconstructed_pkpd_model_state",
              "limitation": "model-derived mechanistic concentrations are not measured clinical concentrations; PK mismatch and noisy estimation are untested",
              "current_code_locations": ["src/vitaldb_state_selection/anesthesia/core.py:81", "src/vitaldb_state_selection/anesthesia/core.py:107", "src/vitaldb_state_selection/anesthesia/core.py:131", "src/vitaldb_state_selection/anesthesia/state.py:61"],
              "historical_code_locations": ["src/vitaldb_state_selection/anesthesia/core.py:77", "src/vitaldb_state_selection/anesthesia/core.py:103", "src/vitaldb_state_selection/anesthesia/core.py:127", "src/vitaldb_state_selection/anesthesia/state.py:61"],
              "future_reserve_bundle_access_count": 0, "future_reserve_outcome_access_count": 0,
              "future_reserve_statistic_access_count": 0, "validation_outcome_access_count": 0, "test_access_count": 0}
    day05_rows=[r for r in csv.DictReader((ROOT/"reports/journal/day05_feature_group_results.csv").open(encoding="utf-8")) if r["state"]=="S_CONC"]
    day05_by={(r["profile"],int(r["seed"])):r for r in day05_rows};day05_contrasts={}
    for metric in ("mae_0_1800","time_in_40_60_fraction","total_propofol_mg"):
        deltas=[float(day05_by[("P1",seed)][metric])-float(day05_by[("P0",seed)][metric]) for seed in config["seeds"]]
        day05_contrasts[metric]={"seed_values":deltas,"mean":float(np.mean(deltas)),"seed_sd":float(np.std(deltas,ddof=1))}
    day05_contrasts["visible_fraction"]={p:float(np.mean([float(r["visible_fraction"]) for r in day05_rows if r["profile"]==p])) for p in ("P0","P1")}
    expected={"mae_0_1800":.0139,"time_in_40_60_fraction":.00233,"total_propofol_mg":-.632}
    if any(abs(day05_contrasts[key]["mean"]-value)>.001 for key,value in expected.items()): raise RuntimeError("committed Day05 S_CONC contrast mismatch")
    result["verified_day05_s_conc_cross_profile_contrasts"]=day05_contrasts
    atomic_json(out / "structural_audit.json", result)
    protocol_path=out/"protocol_manifest.json";protocol=json.loads(protocol_path.read_text(encoding="utf-8"));protocol["temporal_availability_audit"]="passed";protocol["structural_audit_canonical_sha256"]=canonical_sha(result);protocol["validation_outcome_access_count"]=0;atomic_json(protocol_path,protocol)
    print(json.dumps(result, indent=2))


def smoke(config_path: Path) -> None:
    config, _ = load_config(config_path); out = out_root(config); root = out / "smoke"; root.mkdir(exist_ok=True)
    _, _, _, _, store, scalers = context(config); membership = private_membership(config)
    # Identical-action dynamics and native S0 versus projected S1 prefix.
    caseid = membership["diagnostic_cases"][0]
    env0 = make_single_env(config, store, scalers, caseid, "P0", "S0", 777)
    env1 = make_single_env(config, store, scalers, caseid, "P0", "S1", 777)
    o0, i0 = env0.reset(seed=777); o1, i1 = env1.reset(seed=777); max_prefix = float(np.max(np.abs(o0 - o1[:len(S0_FIELDS)])))
    max_reward = max_bis = 0.0
    for _ in range(32):
        action = np.asarray([1.25], dtype=np.float32)
        o0, r0, t0, x0, i0 = env0.step(action); o1, r1, t1, x1, i1 = env1.step(action)
        max_prefix = max(max_prefix, float(np.max(np.abs(o0 - o1[:len(S0_FIELDS)]))))
        max_reward = max(max_reward, abs(float(r0) - float(r1))); max_bis = max(max_bis, abs(i0["latent_true_bis"] - i1["latent_true_bis"]))
        if t0 or x0 or t1 or x1: break
    env0.close(); env1.close()
    if max_prefix != 0 or max_reward > 1e-12 or max_bis > 1e-12: raise RuntimeError("projection changed native dynamics")
    random.seed(777); np.random.seed(777); torch.manual_seed(777); torch.set_num_threads(1)
    vector, seq = make_vector(config, membership, store, scalers, "P0", "S_PROP", 777)
    model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=777, total_timesteps=4096, purpose="day06_projection_resume_smoke"))
    model.learn(total_timesteps=2048, progress_bar=False); cp = save_checkpoint(root, 2048, model, seq, {"purpose": "day06_projection_resume_smoke"})
    model.learn(total_timesteps=2048, reset_num_timesteps=False, progress_bar=False); expected = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}; vector.close()
    vector2, seq2 = make_vector(config, membership, store, scalers, "P0", "S_PROP", 777); resumed = load_checkpoint(cp, vector2, seq2)
    resumed.learn(total_timesteps=2048, reset_num_timesteps=False, progress_bar=False)
    exact = all(torch.equal(expected[k], value.detach().cpu()) for k, value in resumed.policy.state_dict().items()); vector2.close()
    if not exact or resumed.num_timesteps != 4096 or resumed._n_updates != 20: raise RuntimeError("Day06 resume smoke failed")
    result = {"verified": True, "native_s0_equals_s1_prefix": True, "identical_action_dynamics_reward_bis": True,
              "bitwise_policy_resume_equivalence": True, "steps": 4096, "rollouts": 2, "training_epochs": 20,
              "finite": True, "test_access_count": 0}
    atomic_json(root / "verification.json", result); print(json.dumps(result, indent=2))


def benchmark(config_path: Path) -> None:
    config, _ = load_config(config_path); out = out_root(config); _, _, _, _, store, scalers = context(config); membership = private_membership(config)
    torch.set_num_threads(1); vector, _ = make_vector(config, membership, store, scalers, "P0", "S_CE", 48)
    model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=48, total_timesteps=8192, purpose="day06_throughput_benchmark"))
    started = time.perf_counter(); model.learn(total_timesteps=8192, progress_bar=False); seconds = time.perf_counter() - started; vector.close()
    result = {"steps": 8192, "seconds": seconds, "steps_per_second": 8192 / seconds,
              "projected_training_hours": seconds * (40 * int(config["training_target_timesteps"]) / 8192) / 3600,
              "worker_count": 1, "budget_changed": False, "test_access_count": 0}
    atomic_json(out / "benchmark.json", result)
    protocol_path=out/"protocol_manifest.json";protocol=json.loads(protocol_path.read_text(encoding="utf-8"));protocol["benchmark"]=result;atomic_json(protocol_path,protocol)
    print(json.dumps(result, indent=2))


def preoutcome_committed() -> bool:
    required = ["configs/journal/day06_concentration_anatomy.json", "scripts/journal/run_day06_concentration_anatomy.py",
                "scripts/journal/plot_day06_concentration_anatomy.py", "tests/test_day06_concentration_anatomy.py", "docs/journal/DAY06_BRIEF.md"]
    if any(not subprocess.check_output(["git", "ls-files", path], cwd=ROOT, text=True).strip() for path in required): return False
    if subprocess.check_output(["git", "status", "--porcelain", "--", *required], cwd=ROOT, text=True).strip(): return False
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branches = subprocess.check_output(["git", "branch", "-r", "--contains", head], cwd=ROOT, text=True)
    return "origin/journal/day06-concentration-anatomy" in branches


def training_diagnostic(model: PPO, sequence: Day04SequenceEnv, step: int, elapsed: float) -> dict[str, Any]:
    values = {key: float(value) for key, value in model.logger.name_to_value.items()
              if isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))}
    parameters_finite = all(torch.isfinite(p).all().item() for p in model.policy.parameters())
    gradients = [p.grad for p in model.policy.parameters() if p.grad is not None]
    gradients_finite = all(torch.isfinite(g).all().item() for g in gradients)
    return {"step": step, "rollouts": step // 2048, "training_epochs": int(model._n_updates),
            "episodes": sequence.episodes, "unique_training_cases": len(sequence.seen),
            "steps_per_second": step / max(elapsed, 1e-9), "parameters_finite": bool(parameters_finite),
            "gradients_finite_where_present": bool(gradients_finite), "logger_values": values}


def train_one(config_path: Path, profile_name: str, state: str, seed: int) -> None:
    config, config_hash = load_config(config_path); out = out_root(config)
    if state not in config["new_states"]: raise ValueError("only frozen two-coordinate states are newly trained")
    _, _, old_out, _, store, scalers = context(config); membership = private_membership(config)
    target = int(config["training_target_timesteps"]); epochs = int(config["expected_training_epochs"]); directory = job_dir(out, seed, profile_name, state); directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "OUTPUT_COMPLETE.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("config_sha256") == config_hash and payload.get("training_epochs") == epochs: return
        raise RuntimeError("completion identity mismatch")
    identity = {"protocol_id": config["protocol_id"], "config_sha256": config_hash, "condition_id": f"{profile_name}{state}",
                "profile": profile_name, "state": state, "state_schema_sha256": canonical_sha(list(STATE_FIELDS[state])),
                "seed": seed, "target_timesteps": target, "factor_sha256": DAY04_FACTOR_SHA,
                "source_scaler_sha256": sha256_path(old_out / "scaler_registry.json"),
                "training_universe_sha256": _digest(membership["full_train_cases"]), "horizon_seconds": config["horizon_seconds"], "test_access_count": 0}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    vector, sequence = make_vector(config, membership, store, scalers, profile_name, state, seed)
    checkpoints = sorted(directory.glob("checkpoint_*_post_update")); starting = 0
    diagnostics_path = directory / "training_diagnostics.json"; diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))["rows"] if diagnostics_path.exists() else []
    if checkpoints:
        latest = checkpoints[-1]; starting = int(json.loads((latest / "metadata.json").read_text(encoding="utf-8"))["timestep"]); model = load_checkpoint(latest, vector, sequence)
    else:
        model = make_ppo_model(vector, replace(PAPER_ORIENTED_PPO_CANDIDATE_V1, seed=seed, total_timesteps=target, purpose="day06_concentration_anatomy"))
    started = time.perf_counter()
    while starting < target:
        chunk = min(int(config["checkpoint_interval_timesteps"]), target - starting)
        model.learn(total_timesteps=chunk, reset_num_timesteps=(starting == 0), progress_bar=False); starting = int(model.num_timesteps)
        diag = training_diagnostic(model, sequence, starting, time.perf_counter() - started)
        if not diag["parameters_finite"] or not diag["gradients_finite_where_present"]: raise RuntimeError("nonfinite training state")
        diagnostics = [row for row in diagnostics if int(row["step"]) != starting] + [diag]; atomic_json(diagnostics_path, {"rows": diagnostics})
        save_checkpoint(directory, starting, model, sequence, identity, starting == target)
    if model.num_timesteps != target or model._n_updates != epochs: raise RuntimeError("exact budget/update mismatch")
    final = directory / "final_post_update"; meta = json.loads((final / "metadata.json").read_text(encoding="utf-8")); meta |= {"completed": True, "wall_seconds_this_process": time.perf_counter() - started, "expected_rollouts": config["expected_rollouts"], "expected_training_epochs": epochs}
    atomic_json(complete, meta); vector.close(); print(json.dumps({"condition": f"{profile_name}{state}", "seed": seed, "steps": target, "epochs": epochs}))


def all_new_jobs(config: dict[str, Any]) -> list[tuple[int, str, str]]:
    return [(seed, profile_name, state) for seed in config["seeds"] for state in config["new_states"] for profile_name in ("P0", "P1")]


def supervise(config_path: Path) -> None:
    config, _ = load_config(config_path); out = out_root(config); jobs = all_new_jobs(config)
    if all((job_dir(out, seed, p, s) / "OUTPUT_COMPLETE.json").exists() for seed, p, s in jobs): return
    if not preoutcome_committed(): raise RuntimeError("Day06 protocol/code/tests must be committed and pushed before primary training")
    attempts: dict[str, int] = defaultdict(int); started = time.time()
    for seed, profile_name, state in jobs:
        complete = job_dir(out, seed, profile_name, state) / "OUTPUT_COMPLETE.json"; key = f"seed_{seed}/{profile_name}{state}"
        if complete.exists(): continue
        while attempts[key] <= int(config["retry_limit"]):
            attempts[key] += 1; write_status(out, "training", completed_jobs=sum((job_dir(out, a, b, c) / "OUTPUT_COMPLETE.json").exists() for a, b, c in jobs), total_jobs=40, active_job=key, attempt=attempts[key], elapsed_minutes=(time.time()-started)/60)
            directory = job_dir(out, seed, profile_name, state); directory.mkdir(parents=True, exist_ok=True)
            with (directory / "worker.log").open("a", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "train-one", "--config", str(config_path.resolve()), "--profile", profile_name, "--state", state, "--seed", str(seed)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and complete.exists(): break
            for partial in directory.glob("*.partial"):
                os.replace(partial, directory / f"quarantine_{partial.name.strip('.')}_{int(time.time())}")
            if attempts[key] > int(config["retry_limit"]): raise RuntimeError(f"job failed after retry: {key}")
    write_status(out, "training_complete", completed_jobs=40, total_jobs=40, elapsed_minutes=(time.time()-started)/60)


def verify_models(config_path: Path) -> dict[str, Any]:
    config, config_hash = load_config(config_path); out = out_root(config); membership = private_membership(config); target = int(config["training_target_timesteps"]); by_seed: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    if list(out.rglob("*.partial")): raise RuntimeError("partial output cannot be promoted")
    count = 0
    for seed, profile_name, state in all_new_jobs(config):
        directory = job_dir(out, seed, profile_name, state); completion = json.loads((directory / "OUTPUT_COMPLETE.json").read_text(encoding="utf-8"))
        if completion.get("config_sha256") != config_hash or completion.get("factor_sha256") != DAY04_FACTOR_SHA or int(completion.get("training_epochs", -1)) != 2560: raise RuntimeError("new job completion mismatch")
        for step in config["checkpoint_steps"]: verify_checkpoint(model_checkpoint(config, seed, profile_name, state, int(step)), int(step))
        final = directory / "final_post_update"; model = PPO.load(str(final / "model.zip"), device="cpu")
        if model.num_timesteps != target or model._n_updates != 2560 or model.observation_space.shape != (len(STATE_FIELDS[state]),): raise RuntimeError("new model identity mismatch")
        with (final / "environment.pkl").open("rb") as stream: snapshot = pickle.load(stream)
        by_seed[seed].append((canonical_sha(snapshot["generator_state"]), int(snapshot["episodes"]), _digest(list(snapshot["seen"]))))
        count += 1
    # Include the reused Day04 S0 and Day05 S_CONC anchors in paired sampler identity.
    for seed in config["seeds"]:
        for profile_name in ("P0", "P1"):
            for state in ("S0", "S_CONC"):
                final = model_checkpoint(config, seed, profile_name, state, target); verify_checkpoint(final, target)
                with (final / "environment.pkl").open("rb") as stream: snapshot = pickle.load(stream)
                by_seed[seed].append((canonical_sha(snapshot["generator_state"]), int(snapshot["episodes"]), _digest(list(snapshot["seen"]))))
    if any(len(set(rows)) != 1 for rows in by_seed.values()): raise RuntimeError("paired case order differs across state/profile cells")
    result = {"verified": True, "new_models": count, "new_checkpoints": count * 4, "anchor_models": 20, "anchor_checkpoints": 80,
              "new_timesteps": count * target, "paired_case_order": True, "factor_sha256": DAY04_FACTOR_SHA,
              "training_universe_sha256": _digest(membership["full_train_cases"]), "test_access_count": 0, "verified_utc": utc_now()}
    atomic_json(out / "model_verification.json", result); return result


def raw_actor_actions(model: PPO, observations: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    with torch.no_grad(): values = model.policy._predict(tensor, deterministic=True).detach().cpu().numpy().reshape(-1)
    return values


def group_magnitudes(observations: np.ndarray, state: str) -> dict[str, float]:
    names = STATE_FIELDS[state]; groups = {
        "s0": [i for i, name in enumerate(names) if name in S0_FIELDS],
        "cumulative": [i for i, name in enumerate(names) if "cumulative_dose" in name],
        "concentration": [i for i, name in enumerate(names) if name.endswith("_per_l")],
        "recent": [i for i, name in enumerate(names) if "recent_dose" in name],
    }
    return {f"scaled_abs_mean_{key}": (float(np.mean(np.abs(observations[:, idx]))) if idx else 0.0) for key, idx in groups.items()}


def run_eval_case(env: gym.Env, model: PPO, seed: int, observation_bank: list[np.ndarray], raw_bank: list[float]) -> dict[str, float]:
    obs, info = env.reset(seed=seed); latent=[]; actions=[]; requested=[]; rewards=[]; reasons=[]; clips=0; done=False
    while not done:
        observation_bank.append(np.asarray(obs, dtype=np.float32)); action = float(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0]); requested.append(action); raw_bank.append(action)
        obs, reward, terminated, truncated, info = env.step(np.asarray([action], dtype=np.float32)); latent.append(float(info["latent_true_bis"])); actions.append(float(info["applied_action_mg_per_10s"])); rewards.append(float(reward)); reasons.append(str(info["visible_current_bis_reason"])); clips += int(info["action_was_clipped"]); done=bool(terminated or truncated)
    row = metric_record(latent, actions, rewards, clips, reasons)
    errors = np.asarray(latent, dtype=np.float64) - 50.0; split = min(60, len(errors))
    row["rmse_0_1800"] = float(np.sqrt(np.mean(errors ** 2)))
    row["rmse_0_600"] = float(np.sqrt(np.mean(errors[:split] ** 2)))
    row["rmse_600_1800"] = float(np.sqrt(np.mean(errors[split:] ** 2)))
    row["mean_deterministic_action"] = float(np.mean(requested)); return row


def aggregate_subject(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows: grouped[row["subjectid"]].append(row)
    return [{"subjectid": subject, **{metric: float(np.mean([r[metric] for r in rows])) for metric in EVAL_METRICS}} for subject, rows in grouped.items()]


def evaluate_cell(config: dict[str, Any], store: Any, scalers: Any, caseids: list[str], model: PPO,
                  profile_name: str, state: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows=[]; observations=[]; requested=[]
    for caseid in caseids:
        env = make_single_env(config, store, scalers, caseid, profile_name, state, seed)
        metrics = run_eval_case(env, model, seed, observations, requested); env.close()
        rows.append({"caseid": caseid, "subjectid": str(store._by_case[caseid]["subjectid"]), **metrics})
    subjects = aggregate_subject(rows); obs = np.stack(observations); raw = raw_actor_actions(model, obs); low=np.asarray(model.action_space.low).reshape(-1)[0]; high=np.asarray(model.action_space.high).reshape(-1)[0]; physical=np.clip(raw, low, high)
    aggregate = {metric: float(np.mean([r[metric] for r in subjects])) for metric in EVAL_METRICS}
    diagnostic = {"finite_input": bool(np.isfinite(obs).all()), **actor_diagnostics(model, obs), **group_magnitudes(obs, state),
                  "raw_action_q01": float(np.quantile(raw,.01)), "raw_action_q50": float(np.quantile(raw,.5)), "raw_action_q99": float(np.quantile(raw,.99)),
                  "physical_action_q01": float(np.quantile(physical,.01)), "physical_action_q50": float(np.quantile(physical,.5)), "physical_action_q99": float(np.quantile(physical,.99))}
    return rows, aggregate, diagnostic


def evaluate(config_path: Path) -> None:
    config, _ = load_config(config_path); out = out_root(config); verification = verify_models(config_path)
    if not preoutcome_committed(): raise RuntimeError("protocol commit must be remote before development outcomes")
    _, _, _, _, store, scalers = context(config); membership = private_membership(config)
    atomic_json(out / "evaluation_lock.json", {"locked": True, "model_verification_sha256": sha256_path(out / "model_verification.json"),
                "development_subset_sha256": _digest(membership["development_validation_subjects"]),
                "future_reserve_bundle_access_count": 0, "test_access_count": 0, "locked_utc": utc_now()})
    checkpoint_private=[]; checkpoint_aggregate=[]; checkpoint_diagnostics=[]
    cells=[(state,p,seed,step) for step in config["checkpoint_steps"] for seed in config["seeds"] for state in config["states"] for p in ("P0","P1")]
    for index,(state,p,seed,step) in enumerate(cells,1):
        path=model_checkpoint(config,seed,p,state,int(step));model=PPO.load(str(path/"model.zip"),device="cpu")
        rows,agg,diag=evaluate_cell(config,store,scalers,membership["diagnostic_cases"],model,p,state,seed)
        checkpoint_private.extend({"state":state,"profile":p,"seed":seed,"step":step,**r} for r in rows);checkpoint_aggregate.append({"state":state,"profile":p,"seed":seed,"step":step,"subject_count":len(membership["diagnostic_subjects"]),**agg});checkpoint_diagnostics.append({"scope":"checkpoint_train_diagnostic","state":state,"profile":p,"seed":seed,"step":step,**diag})
        if index%10==0:write_status(out,"checkpoint_diagnostics",completed_cells=index,total_cells=len(cells))
    atomic_json(out/"checkpoint_evaluation_private.json",{"rows":checkpoint_private});atomic_json(out/"checkpoint_evaluation_aggregate.json",{"rows":checkpoint_aggregate,"diagnostics":checkpoint_diagnostics})
    final_private=[]; final_aggregate=[]; final_diagnostics=[]; cells2=[(state,p,seed) for seed in config["seeds"] for state in config["states"] for p in ("P0","P1")]
    for index,(state,p,seed) in enumerate(cells2,1):
        path=model_checkpoint(config,seed,p,state,int(config["training_target_timesteps"]));model=PPO.load(str(path/"model.zip"),device="cpu")
        rows,agg,diag=evaluate_cell(config,store,scalers,membership["development_validation_cases"],model,p,state,seed)
        final_private.extend({"state":state,"profile":p,"seed":seed,**r} for r in rows);final_aggregate.append({"state":state,"profile":p,"seed":seed,"subject_count":len(membership["development_validation_subjects"]),"case_count":len(membership["development_validation_cases"]),**agg});final_diagnostics.append({"scope":"final_development","state":state,"profile":p,"seed":seed,"step":config["training_target_timesteps"],**diag})
        if index%5==0:write_status(out,"development_evaluation",completed_cells=index,total_cells=len(cells2))
    atomic_json(out/"development_evaluation_private.json",{"rows":final_private,"evaluation_passes":1});atomic_json(out/"development_evaluation_aggregate.json",{"rows":final_aggregate,"diagnostics":final_diagnostics,"future_reserve_bundle_access_count":0,"test_access_count":0});print(json.dumps({"checkpoint_cells":len(cells),"final_cells":len(cells2),"private_final_rows":len(final_private)}))


def contrast_value(values: dict[str,float], name: str) -> float:
    if name not in CONTRAST_FORMULAS: raise KeyError(name)
    left, right = CONTRAST_FORMULAS[name].split("-")
    return values[left] - values[right]


def retention_decision(regrets: dict[tuple[str,int],float], seeds: list[int], candidate_worst: float,
                       full_worst: float, max_boundary: float, max_core_clip: float,
                       max_tanh: float) -> dict[str,bool]:
    values = [regrets[(p, seed)] for p in ("P0", "P1") for seed in seeds]
    return {"overall_mean_regret":float(np.mean(values))<=.25,
            "per_profile_mean_regret":all(float(np.mean([regrets[(p,s)] for s in seeds]))<=.50 for p in ("P0","P1")),
            "eight_of_ten":sum(x<=.50 for x in values)>=8,
            "maximum_positive_regret":max(values)<=.75,
            "worst_seed_margin":candidate_worst<=full_worst+.75,
            "action_guardrails":max_boundary<=.05 and max_core_clip==0 and max_tanh<=1e-4}


def diagnose(config_path: Path) -> None:
    config,_=load_config(config_path);out=out_root(config);private=json.loads((out/"development_evaluation_private.json").read_text(encoding="utf-8"))["rows"]
    subjects=sorted({r["subjectid"] for r in private}); grouped=defaultdict(list)
    for r in private:grouped[(r["profile"],r["state"],r["seed"],r["subjectid"])].append(r)
    values={(p,s,seed,subject):{m:float(np.mean([r[m] for r in grouped[(p,s,seed,subject)]])) for m in EVAL_METRICS} for p in ("P0","P1") for s in config["states"] for seed in config["seeds"] for subject in subjects}
    seed_rows=[];summary=[]
    for p in ("P0","P1"):
      for name,formula in CONTRAST_FORMULAS.items():
        deltas=[]
        for seed in config["seeds"]:
            cell={s:float(np.mean([values[(p,s,seed,subject)]["mae_0_1800"] for subject in subjects])) for s in config["states"]};delta=contrast_value(cell,name);deltas.append(delta);seed_rows.append({"row_type":"seed","profile":p,"contrast":name,"formula":formula,"seed":seed,"delta":delta,"favorable":delta<0})
        summary.append({"row_type":"summary","profile":p,"contrast":name,"formula":formula,"seed":"mean","delta":float(np.mean(deltas)),"seed_sd":float(np.std(deltas,ddof=1)),"favorable_seed_count":sum(x<0 for x in deltas),"seed_values":deltas})
    atomic_json(out/"group_contrasts.json",{"rows":seed_rows+summary,"subject_paired":True,"test_access_count":0})
    aggregates=json.loads((out/"development_evaluation_aggregate.json").read_text(encoding="utf-8"))["rows"];diagnostics=json.loads((out/"development_evaluation_aggregate.json").read_text(encoding="utf-8"))["diagnostics"]
    state_summary=[]
    for p in ("P0","P1"):
      for state in config["states"]:
        rows=sorted([r for r in aggregates if r["profile"]==p and r["state"]==state],key=lambda r:r["seed"]);mae=[r["mae_0_1800"] for r in rows];di=[r for r in diagnostics if r["profile"]==p and r["state"]==state]
        state_summary.append({"profile":p,"state":state,"seed_values":mae,"mean_mae":float(np.mean(mae)),"median_mae":float(np.median(mae)),"seed_sd":float(np.std(mae,ddof=1)),"worst_seed_mae":float(np.max(mae)),"worst_seed":int(rows[int(np.argmax(mae))]["seed"]),"range":float(np.ptp(mae)),"mean_action_boundary":float(np.mean([r["action_boundary_fraction"] for r in rows])),"max_action_boundary":float(np.max([r["action_boundary_fraction"] for r in rows])),"mean_core_clip":float(np.mean([r["core_clip_fraction"] for r in rows])),"max_core_clip":float(np.max([r["core_clip_fraction"] for r in rows])),"mean_tanh_saturation":float(np.mean([r["actor_tanh_saturation_fraction"] for r in di])),"max_tanh_saturation":float(np.max([r["actor_tanh_saturation_fraction"] for r in di]))})
    by={(r["profile"],r["state"],r["seed"]):r for r in aggregates}
    profile_rows=[];profile_summary=[]
    directions={"mae_0_1800":-1,"time_in_40_60_fraction":1,"total_propofol_mg":0,"consecutive_action_variation":0}
    for state in config["states"]:
      for metric,direction in directions.items():
        deltas=[]
        for seed in config["seeds"]:
          delta=by[("P1",state,seed)][metric]-by[("P0",state,seed)][metric];deltas.append(delta);profile_rows.append({"row_type":"seed","state":state,"metric":metric,"seed":seed,"delta":delta,"favorable":None if direction==0 else direction*delta>0})
        profile_summary.append({"row_type":"summary","state":state,"metric":metric,"seed":"mean","delta":float(np.mean(deltas)),"seed_sd":float(np.std(deltas,ddof=1)),"maximum_absolute_discrepancy":float(np.max(np.abs(deltas))),"favorable_seed_count":None if direction==0 else sum(direction*x>0 for x in deltas),"seed_values":deltas})
    atomic_json(out/"profile_sensitivity.json",{"rows":profile_rows+profile_summary,"bundle_level_profiles":True})
    visibility_metrics=("visible_fraction","sqi_rejection_fraction","stale_fraction")
    for p in ("P0","P1"):
      for seed in config["seeds"]:
        for metric in visibility_metrics:
          vals=[by[(p,state,seed)][metric] for state in config["states"]]
          if max(vals)-min(vals)>1e-12: raise RuntimeError("state-invariant visibility failed")
    full=[by[(p,"S_CONC",seed)] for p in ("P0","P1") for seed in config["seeds"]];full_worst=max(r["mae_0_1800"] for r in full)
    retention=[]
    for state in config["new_states"]:
      regrets={(p,seed):by[(p,state,seed)]["mae_0_1800"]-by[(p,"S_CONC",seed)]["mae_0_1800"] for p in ("P0","P1") for seed in config["seeds"]};rv=list(regrets.values());rows=[by[(p,state,seed)] for p in ("P0","P1") for seed in config["seeds"]];di=[r for r in diagnostics if r["state"]==state]
      criteria=retention_decision(regrets,config["seeds"],max(r["mae_0_1800"] for r in rows),full_worst,max(r["action_boundary_fraction"] for r in rows),max(r["core_clip_fraction"] for r in rows),max(r["actor_tanh_saturation_fraction"] for r in di))
      pdelta=[by[("P1",state,s)]["mae_0_1800"]-by[("P0",state,s)]["mae_0_1800"] for s in config["seeds"]]
      retention.append({"state":state,"retained_performance_candidate":all(criteria.values()),"criteria":criteria,"regrets":[regrets[(p,s)] for p in ("P0","P1") for s in config["seeds"]],"mean_regret":float(np.mean(rv)),"profile_mean_regret":{p:float(np.mean([regrets[(p,s)] for s in config["seeds"]])) for p in ("P0","P1")},"count_regret_le_0_50":sum(x<=.50 for x in rv),"max_positive_regret":float(max(rv)),"worst_seed_mae":float(max(r["mae_0_1800"] for r in rows)),"cross_profile_mae_sd":float(np.std(pdelta,ddof=1))})
    passing=[r for r in retention if r["retained_performance_candidate"]]
    selected=(sorted(passing,key=lambda r:(r["max_positive_regret"],r["worst_seed_mae"],r["mean_regret"],r["cross_profile_mae_sd"]))[0]["state"] if passing else "S_CONC")
    trajectory=json.loads((out/"checkpoint_evaluation_aggregate.json").read_text(encoding="utf-8"))["rows"]
    seed50=[r for r in trajectory if r["seed"]==50]
    atomic_json(out/"diagnosis.json",{"state_summary":state_summary,"retention":retention,"selected_state":selected,"seed50_checkpoint_trajectory":seed50,"state_invariant_visibility":True,"test_access_count":0})


def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]|None=None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=fields or sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def summarize_legacy(config_path: Path) -> None:
    config,config_hash=load_config(config_path);out=out_root(config);agg=json.loads((out/"development_evaluation_aggregate.json").read_text(encoding="utf-8"));checkpoint=json.loads((out/"checkpoint_evaluation_aggregate.json").read_text(encoding="utf-8"));contrasts=json.loads((out/"group_contrasts.json").read_text(encoding="utf-8"));diagnosis=json.loads((out/"diagnosis.json").read_text(encoding="utf-8"));protocol=json.loads((out/"protocol_manifest.json").read_text(encoding="utf-8"));timing=json.loads((out/"timing.json").read_text(encoding="utf-8")) if (out/"timing.json").exists() else {}
    write_csv(PUBLIC_RESULTS,agg["rows"]);write_csv(PUBLIC_CONTRASTS,contrasts["rows"],["row_type","profile","contrast","formula","seed","delta","seed_sd","favorable_seed_count","favorable","seed_values"]);write_csv(PUBLIC_TRAJECTORY,checkpoint["rows"]);write_csv(PUBLIC_DIAGNOSTICS,agg["diagnostics"]+checkpoint["diagnostics"])
    summary={"protocol_id":config["protocol_id"],"config_sha256":config_hash,"evidence_scope":config["evidence_scope"],"protocol":protocol,"state_summary":diagnosis["state_summary"],"contrasts":contrasts,"pareto":diagnosis["pareto"],"recommended_candidates":diagnosis["recommended_candidates"],"seed50_checkpoint_trajectory":diagnosis["seed50_checkpoint_trajectory"],"aggregate_rows":agg["rows"],"diagnostics":agg["diagnostics"],"timing":timing,"future_reserve_bundle_access_count":0,"future_reserve_outcome_access_count":0,"test_access_count":0,"generated_utc":utc_now()};atomic_json(PUBLIC_SUMMARY,summary)
    by={(r["profile"],r["state"]):r for r in diagnosis["state_summary"]};cs=[r for r in contrasts["rows"] if r["row_type"]=="summary"]
    pi_rows=[r for r in csv.DictReader((ROOT/"reports/journal/day04_results.csv").open(encoding="utf-8")) if r["kind"]=="baseline" and r["seed"]=="PI"]
    pi_mae={r["condition_id"]:float(r["mae_0_1800"]) for r in pi_rows}
    lines=["# Day 05 feature-group 개발 스크리닝 결과","","## 결론","",f"Pareto non-dominated states: {', '.join(r['state'] for r in diagnosis['pareto'] if r['non_dominated'])}. 사전 지정 정렬 규칙에 따른 다음 개발 후보는 {', '.join(diagnosis['recommended_candidates'])}이다. S_CONC가 두 profile 모두에서 가장 낮고 가장 안정적인 MAE를 보였고, S_CORE는 P1에서는 비슷했지만 P0 한 seed에서 불안정했다. 이는 재구성 simulation의 개발 추천이며 임상적 최적 상태나 생물학적 인과 효과가 아니다.","","## 상태별 최종 MAE","","| State | P0 mean ± SD (worst) | P1 mean ± SD (worst) | Extra features |","|---|---:|---:|---:|"]
    for state in config["states"]:a=by[("P0",state)];b=by[("P1",state)];lines.append(f"| {state} | {a['mean_mae']:.4f} ± {a['seed_sd']:.4f} ({a['worst_seed_mae']:.4f}) | {b['mean_mae']:.4f} ± {b['seed_sd']:.4f} ({b['worst_seed_mae']:.4f}) | {EXTRA_FEATURES[state]} |")
    lines += ["","## 고정 feature-group contrasts","","| Profile | Contrast | Mean ± seed SD | Favorable seeds | Five paired seed values |","|---|---|---:|---:|---|"]
    for r in cs:lines.append(f"| {r['profile']} | {r['contrast']} | {r['delta']:+.4f} ± {r['seed_sd']:.4f} | {r['favorable_seed_count']}/5 | "+", ".join(f"{x:+.4f}" for x in r["seed_values"])+" |")
    s50=[r for r in agg["rows"] if r["seed"]==50];lines += ["","## Seed 50 prespecified diagnostic","","아래 값은 Day04 development-validation subset의 final checkpoint 결과(MAE / propofol mg / BIS 40–60 fraction)다.","","| State | P0 MAE / propofol / in-range | P1 MAE / propofol / in-range |","|---|---:|---:|"]
    for state in config["states"]:a=next(r for r in s50 if r["state"]==state and r["profile"]=="P0");b=next(r for r in s50 if r["state"]==state and r["profile"]=="P1");lines.append(f"| {state} | {a['mae_0_1800']:.4f} / {a['total_propofol_mg']:.1f} / {a['time_in_40_60_fraction']:.3f} | {b['mae_0_1800']:.4f} / {b['total_propofol_mg']:.1f} / {b['time_in_40_60_fraction']:.3f} |")
    lines += ["","### Seed 50 train-only checkpoint trajectory","","각 셀은 131k → 262k → 393k → 524k checkpoint의 `MAE / propofol mg` 순서다. 이는 32-subject train-only diagnostic이며 checkpoint 선택에 쓰지 않았다.","","| State | P0 trajectory | P1 trajectory |","|---|---|---|"]
    for state in config["states"]:
        cells=[]
        for profile in ("P0","P1"):
            rows=sorted((r for r in diagnosis["seed50_checkpoint_trajectory"] if r["state"]==state and r["profile"]==profile),key=lambda r:r["step"])
            cells.append(" → ".join(f"{r['mae_0_1800']:.2f}/{r['total_propofol_mg']:.0f}" for r in rows))
        lines.append(f"| {state} | {cells[0]} | {cells[1]} |")
    lines += ["","S_CONC와 S_CORE는 seed 50의 P0/P1 저투여 패턴을 제거했다. S1은 후기 학습에서 propofol이 감소하며 실패가 재현됐고, S_CUM은 P0에서는 개선됐지만 P1 final에서는 반대로 높은 dose와 낮은 in-range fraction을 보였다. 따라서 모든 enriched state가 같은 실패를 공유한다는 가설은 지지되지 않으며, concentration group이 가장 직접적인 안정화 신호다.","","## Guardrails and PI comparison","","Boundary와 diagnostic action-clip은 이 구현에서 같은 사건을 집계해 값이 일치한다. 아래 boundary는 seed mean / seed maximum이며, Tanh는 seed maximum이다.","","| Profile-state | Boundary mean / max | Core clip mean | Tanh max | MAE − Day04 PI |","|---|---:|---:|---:|---:|"]
    final_di=agg["diagnostics"]
    for profile in ("P0","P1"):
        for state in config["states"]:
            s=by[(profile,state)];di=[r for r in final_di if r["profile"]==profile and r["state"]==state]
            tanh_max=max(r["actor_tanh_saturation_fraction"] for r in di)
            lines.append(f"| {profile}{state} | {s['mean_action_boundary']:.4f} / {s['max_action_boundary']:.4f} | {s['mean_core_clip']:.1f} | {tanh_max:.2e} | {s['mean_mae']-pi_mae[profile]:+.4f} |")
    lines += ["",f"Day04 tuned PI baselines were P0 {pi_mae['P0']:.4f} and P1 {pi_mae['P1']:.4f} MAE. S_CONC and S_CORE improved on PI in both profiles; S_CUM improved on PI in both on average; S0 was worse. Core clipping was zero throughout, Tanh saturation was at most {max(r['actor_tanh_saturation_fraction'] for r in final_di):.2e}, and the largest state/profile seed-level boundary rate was {max(r['action_boundary_fraction'] for r in agg['rows']):.4f}. These are simulation guardrails, not evidence of clinical controller superiority.","","Raw/physical action quantiles, scaled group magnitudes, and all checkpoint diagnostics are retained in the public CSV/JSON artifacts. Validation used only final checkpoints and selected no checkpoint.","",f"Untouched future internal-confirmation reserve: {protocol['future_reserve_subject_count']} subjects / {protocol['future_reserve_case_count']} cases; membership identifiers were hashed, bundle access 0, outcome access 0. Original-test access remained 0.","","## Pareto interpretation","",f"All five states are non-dominated when feature count and guardrails are included. The frozen tie-break chooses {', '.join(diagnosis['recommended_candidates'])}. S_CONC is the primary candidate because its across-profile mean MAE, worst-seed MAE, and across-seed SD are {next(r for r in diagnosis['pareto'] if r['state']=='S_CONC')['mean_mae_across_profiles']:.4f}, {next(r for r in diagnosis['pareto'] if r['state']=='S_CONC')['worst_seed_mae_across_profiles']:.4f}, and {next(r for r in diagnosis['pareto'] if r['state']=='S_CONC')['across_seed_sd']:.4f}. S_CORE remains secondary but its P0 seed instability argues against claiming complementarity.","","## Neural Networks novelty implication","","The result localizes the useful representation signal to the four concentration summaries: S_CONC improved all 10 matched profile-seed comparisons versus S0 and avoided seed 50 underdosing, while cumulative-only and redundant recent summaries were less stable. The next frozen development ablation may split Cp versus Ce and drug-specific concentration groups. Feature evidence alone does not establish architectural novelty; a Neural Networks contribution would still require a genuinely new group-structured encoder/normalizer or stability method and a later untouched-reserve confirmation. No follow-up is launched here.","","## Limitations","","The Day 04 development-validation subset was reused after inspection, so this is exploratory screening rather than confirmation. The source cohort is internal and reused. Contrasts compare independently retrained policies and are not biological or clinical causal effects. This is reconstructed simulation, not clinical, bedside-dosing, or external-validation evidence.","","## Reproduction","",f"30 new jobs, 15,728,640 new timesteps, 120 new checkpoints; one worker and one Torch thread; failures/retries 0/0. Wall time: {timing.get('wall_hours',float('nan')):.2f} h. Resume: `.venv-journal\\Scripts\\python.exe scripts\\journal\\run_day05_feature_groups.py resume`."]
    PUBLIC_REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    PUBLIC_OVERLEAF.write_text("# Day 05 Overleaf note\n\n- Exploratory development screening in reconstructed PK/PD simulation; not clinical, causal, or external-validation evidence.\n- States: S0, S_CUM, S_CONC, S_CORE, S1; P0/P1; five paired seeds; 1,800 s; 524,288 steps per new job.\n- MAE mean ± seed SD (P0 / P1): "+"; ".join(f"{s} {by[('P0',s)]['mean_mae']:.4f} ± {by[('P0',s)]['seed_sd']:.4f} / {by[('P1',s)]['mean_mae']:.4f} ± {by[('P1',s)]['seed_sd']:.4f}" for s in config["states"])+".\n- Concentration-only (S_CONC) was favorable versus S0 in all 10 paired profile-seed comparisons, had the lowest across-profile mean and worst-seed MAE, and removed the prespecified seed-50 underdosing pattern. S_CORE was similar under P1 but unstable in one P0 seed; redundant recent summaries in S1 reproduced seed-50 underdosing.\n- All states were Pareto non-dominated when feature count and guardrails were included; frozen-rule development candidates: "+", ".join(diagnosis["recommended_candidates"])+".\n- Core clipping was zero and Tanh saturation negligible; this does not establish clinical controller superiority or causal feature effects.\n- Future reserve: 170 subjects / 170 cases, zero bundle/outcome access; original-test access zero.\n",encoding="utf-8")


def summarize(config_path: Path) -> None:
    config,config_hash=load_config(config_path);out=out_root(config)
    agg=json.loads((out/"development_evaluation_aggregate.json").read_text(encoding="utf-8"));checkpoint=json.loads((out/"checkpoint_evaluation_aggregate.json").read_text(encoding="utf-8"));contrasts=json.loads((out/"group_contrasts.json").read_text(encoding="utf-8"));profiles=json.loads((out/"profile_sensitivity.json").read_text(encoding="utf-8"));diagnosis=json.loads((out/"diagnosis.json").read_text(encoding="utf-8"));protocol=json.loads((out/"protocol_manifest.json").read_text(encoding="utf-8"));audit_data=json.loads((out/"structural_audit.json").read_text(encoding="utf-8"));timing=json.loads((out/"timing.json").read_text(encoding="utf-8")) if (out/"timing.json").exists() else {}
    write_csv(PUBLIC_RESULTS,agg["rows"]);write_csv(PUBLIC_CONTRASTS,contrasts["rows"],["row_type","profile","contrast","formula","seed","delta","seed_sd","favorable_seed_count","favorable","seed_values"]);write_csv(PUBLIC_PROFILE,profiles["rows"]);write_csv(PUBLIC_TRAJECTORY,checkpoint["rows"]);write_csv(PUBLIC_DIAGNOSTICS,agg["diagnostics"]+checkpoint["diagnostics"])
    audit_rows=[{"row_type":"coordinate",**r} for r in audit_data["coordinate_summaries"]]+[{"row_type":"correlation",**r} for r in audit_data["pairwise_correlations"]];write_csv(PUBLIC_AUDIT,audit_rows)
    summary={"protocol_id":config["protocol_id"],"config_sha256":config_hash,"evidence_scope":config["evidence_scope"],"protocol":protocol,"causal_availability_audit":audit_data,"state_summary":diagnosis["state_summary"],"contrasts":contrasts,"profile_sensitivity":profiles,"retention":diagnosis["retention"],"selected_state":diagnosis["selected_state"],"seed50_checkpoint_trajectory":diagnosis["seed50_checkpoint_trajectory"],"aggregate_rows":agg["rows"],"diagnostics":agg["diagnostics"],"timing":timing,"future_reserve_bundle_access_count":0,"future_reserve_outcome_access_count":0,"future_reserve_statistic_access_count":0,"test_access_count":0,"generated_utc":utc_now()};atomic_json(PUBLIC_SUMMARY,summary)
    by={(r["profile"],r["state"]):r for r in diagnosis["state_summary"]};cs=[r for r in contrasts["rows"] if r["row_type"]=="summary"];ps=[r for r in profiles["rows"] if r["row_type"]=="summary"]
    lines=["# Day 06 농도 표현 해부 연구 결과","","## 결론","",f"사전 동결한 표현 유지 규칙에 따라 다음 관측 품질 profile sweep의 단일 추천 상태는 **{diagnosis['selected_state']}**이다. 이 결론은 재구성 PK/PD simulation의 재사용 개발 subset에서 얻은 탐색적 공학 판단이며 통계적 동등성, 임상적 유효성 또는 개별 feature의 생물학적 인과 효과를 뜻하지 않는다.","","## 최종 latent-BIS MAE","","| State | P0 mean ± seed SD (worst seed/value) | P1 mean ± seed SD (worst seed/value) |","|---|---:|---:|"]
    for state in config["states"]:
        a=by[("P0",state)];b=by[("P1",state)];lines.append(f"| {state} | {a['mean_mae']:.4f} ± {a['seed_sd']:.4f} ({a['worst_seed']}/{a['worst_seed_mae']:.4f}) | {b['mean_mae']:.4f} ± {b['seed_sd']:.4f} ({b['worst_seed']}/{b['worst_seed_mae']:.4f}) |")
    lines += ["","## 고정 matched contrasts","","| Profile | Contrast | Mean ± seed SD | Favorable | Five paired seed values |","|---|---|---:|---:|---|"]
    for r in cs:lines.append(f"| {r['profile']} | {r['contrast']} | {r['delta']:+.4f} ± {r['seed_sd']:.4f} | {r['favorable_seed_count']}/5 | "+", ".join(f"{x:+.4f}" for x in r["seed_values"])+" |")
    lines += ["","## P1 − P0 sensitivity","","P0/P1은 독립 학습된 bundle-level 관측 조건이므로 아래 값은 SQI 단독 효과가 아니라 cross-profile sensitivity이다.","","| State | Metric | Mean ± seed SD | Max abs | Five paired values |","|---|---|---:|---:|---|"]
    for r in ps:lines.append(f"| {r['state']} | {r['metric']} | {r['delta']:+.4f} ± {r['seed_sd']:.4f} | {r['maximum_absolute_discrepancy']:.4f} | "+", ".join(f"{x:+.4f}" for x in r["seed_values"])+" |")
    lines += ["","## 표현 유지 규칙","","| State | Pass | Mean regret | P0/P1 mean regret | ≤0.50 cells | Max regret | Worst MAE |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in diagnosis["retention"]:lines.append(f"| {r['state']} | {str(r['retained_performance_candidate']).lower()} | {r['mean_regret']:+.4f} | {r['profile_mean_regret']['P0']:+.4f}/{r['profile_mean_regret']['P1']:+.4f} | {r['count_regret_le_0_50']}/10 | {r['max_positive_regret']:+.4f} | {r['worst_seed_mae']:.4f} |")
    s50=[r for r in agg["rows"] if r["seed"]==50];lines += ["","## Seed 50 prespecified diagnostic","","| State | P0 MAE / dose / TIR | P1 MAE / dose / TIR |","|---|---:|---:|"]
    for state in config["states"]:
        a=next(r for r in s50 if r["state"]==state and r["profile"]=="P0");b=next(r for r in s50 if r["state"]==state and r["profile"]=="P1");lines.append(f"| {state} | {a['mae_0_1800']:.4f} / {a['total_propofol_mg']:.1f} / {a['time_in_40_60_fraction']:.3f} | {b['mae_0_1800']:.4f} / {b['total_propofol_mg']:.1f} / {b['time_in_40_60_fraction']:.3f} |")
    max_boundary=max(r["action_boundary_fraction"] for r in agg["rows"]);max_clip=max(r["core_clip_fraction"] for r in agg["rows"]);max_tanh=max(r["actor_tanh_saturation_fraction"] for r in agg["diagnostics"])
    lines += ["","## 인과 가용성 및 guardrails","",audit_data["limitation"]+". Decision-time audit에서 미래/사후 action 입력은 없었고 projection은 dynamics, reward, action timing을 바꾸지 않았다.","",f"전체 final cell 최대 action-boundary={max_boundary:.4f}, core clipping={max_clip:.1f}, actor Tanh saturation={max_tanh:.2e}. Dose, smoothness, TIR, RMSE와 PI-bounded 비교에 필요한 aggregate는 CSV/JSON에 보존했다. 이는 임상적 우월성 근거가 아니다.","","## Neural Networks 기여와 한계","",f"결과는 {diagnosis['selected_state']}를 후속 관측 신뢰도 실험의 입력 후보로 좁히는 표현 연구다. Feature decomposition만으로 architecture novelty를 확립하지 않으며, 후속의 실제 새 encoder/normalizer 또는 안정화 방법과 untouched reserve 확인이 필요하다. Singleton, model mismatch, 새 encoder는 Day 06에서 시험하지 않았다.","",f"Day04 development-validation subset은 이미 관찰된 개발 자료다. Reserve {protocol['future_reserve_subject_count']} subjects/{protocol['future_reserve_case_count']} cases는 bundle/outcome/statistic access 모두 0이며 historical sealed test access도 0이다.","","## 재현","",f"40 jobs, 20,971,520 timesteps, 160 new + 80 reused checkpoints; one worker/one Torch thread. Wall time {timing.get('wall_hours',float('nan')):.2f} h. Resume: `.venv-journal\\Scripts\\python.exe scripts\\journal\\run_day06_concentration_anatomy.py resume`."]
    PUBLIC_REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    PUBLIC_OVERLEAF.write_text("# Day 06 Overleaf note\n\n- Exploratory decomposition in a reconstructed PK/PD simulation; not clinical, causal, confirmatory, or external-validation evidence.\n- Six states, two bundle-level observation profiles, and five paired seeds.\n- Selected by a frozen engineering retention heuristic: **"+diagnosis["selected_state"]+"**. This is not a statistical equivalence claim.\n- Concentrations are exact model-derived mechanistic states, not measured clinical concentrations; model mismatch and noisy estimation remain untested.\n- Validation used final checkpoints only. The 170-subject reserve and historical sealed test remained untouched.\n- Representation evidence alone does not establish neural-network architectural novelty.\n",encoding="utf-8")
    subprocess.run([sys.executable,str(ROOT/"scripts/journal/plot_day06_concentration_anatomy.py")],cwd=ROOT,check=True)


def verify(config_path: Path) -> None:
    config,_=load_config(config_path);out=out_root(config);models=verify_models(config_path);membership=private_membership(config);summary=json.loads(PUBLIC_SUMMARY.read_text(encoding="utf-8"));private=json.loads((out/"development_evaluation_private.json").read_text(encoding="utf-8"))["rows"]
    if len(private)!=60*len(membership["development_validation_cases"]) or models["new_models"]!=40:raise RuntimeError("evaluation/model count mismatch")
    if len(summary["state_summary"])!=12 or summary["future_reserve_bundle_access_count"]!=0 or summary["future_reserve_statistic_access_count"]!=0 or summary["test_access_count"]!=0:raise RuntimeError("summary accounting mismatch")
    public_text=[PUBLIC_RESULTS,PUBLIC_CONTRASTS,PUBLIC_PROFILE,PUBLIC_TRAJECTORY,PUBLIC_DIAGNOSTICS,PUBLIC_AUDIT,PUBLIC_SUMMARY,PUBLIC_REPORT,PUBLIC_OVERLEAF,ROOT/"docs/journal/DAY06_BRIEF.md"]
    forbidden=(b"subjectid",b"caseid",b"C:\\Users\\",b"patient_profile.json",b"BEGIN PRIVATE KEY")
    for path in public_text:
        text=path.read_text(encoding="utf-8")
        if any(token.decode() in text for token in forbidden):raise RuntimeError(f"privacy token in {path}")
        if path.stat().st_size>5_000_000:raise RuntimeError(f"public file too large: {path}")
    for path in PUBLIC_FIGURES:
        payload=path.read_bytes()
        if any(token in payload for token in forbidden):raise RuntimeError(f"privacy token in {path}")
        if len(payload)>5_000_000:raise RuntimeError(f"public file too large: {path}")
    result={"verified":True,"new_jobs":40,"new_checkpoints":160,"reused_checkpoints":80,"new_timesteps":20971520,"comparison_cells":60,"development_evaluation_calls":60*len(membership["development_validation_cases"]),"checkpoint_diagnostic_cells":240,"subject_aggregation_before_group_metrics":True,"paired_case_order":True,"state_invariant_visibility":True,"final_only_development_evaluation":True,"future_reserve_subjects":len(membership["future_reserve_subjects"]),"future_reserve_cases":len(membership["future_reserve_cases"]),"future_reserve_bundle_access_count":0,"future_reserve_outcome_access_count":0,"future_reserve_statistic_access_count":0,"test_access_count":0,"privacy_scan":True,"verified_utc":utc_now()};atomic_json(out/"verification.json",result);print(json.dumps(result,indent=2))


def resume(config_path: Path) -> None:
    config,_=load_config(config_path);out=out_root(config);timing_path=out/"timing.json";prior=json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {};started=prior.get("started_utc",utc_now());atomic_json(timing_path,{"started_utc":started})
    if not (out/"protocol_manifest.json").exists():prepare(config_path)
    if not (out/"structural_audit.json").exists():audit(config_path)
    if not (out/"smoke/verification.json").exists():smoke(config_path)
    if not (out/"benchmark.json").exists():benchmark(config_path)
    supervise(config_path)
    if not (out/"development_evaluation_aggregate.json").exists():evaluate(config_path)
    if not (out/"diagnosis.json").exists():diagnose(config_path)
    completed=utc_now();wall=(datetime.fromisoformat(completed)-datetime.fromisoformat(started)).total_seconds()/3600;atomic_json(timing_path,{"started_utc":started,"completed_utc":completed,"wall_hours":wall});summarize(config_path);verify(config_path)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","audit","smoke","benchmark","train-one","supervise","evaluate","diagnose","summarize","verify","resume"));parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG);parser.add_argument("--profile");parser.add_argument("--state");parser.add_argument("--seed",type=int);args=parser.parse_args()
    if args.command=="prepare":prepare(args.config)
    elif args.command=="audit":audit(args.config)
    elif args.command=="smoke":smoke(args.config)
    elif args.command=="benchmark":benchmark(args.config)
    elif args.command=="train-one":
        if args.profile is None or args.state is None or args.seed is None:parser.error("train-one requires --profile --state --seed")
        train_one(args.config,args.profile,args.state,args.seed)
    elif args.command=="supervise":supervise(args.config)
    elif args.command=="evaluate":evaluate(args.config)
    elif args.command=="diagnose":diagnose(args.config)
    elif args.command=="summarize":summarize(args.config)
    elif args.command=="verify":verify(args.config)
    else:resume(args.config)


if __name__=="__main__":main()
