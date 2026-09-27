"""Phase A diagnostics using only allowed development artifacts.

Public outputs are aggregate-only. Identifier-bearing inputs and model files remain
inside ignored output roots. No reserve or sealed-test bundle is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

import numpy as np
import torch
from stable_baselines3 import PPO

import run_day04_confirmation as day04
import run_day05_feature_groups as day05
from night02_common import make_environment, truncate_bundle
from vitaldb_state_selection.cohort.train_runtime_inputs import S0_FIELDS, S1_FIELDS


DEFAULT_CONFIG = ROOT / "configs/journal/phase_a_diagnostics.json"
PUBLIC_SUMMARY = ROOT / "reports/journal/phase_a_summary.json"
PUBLIC_WINDOWS = ROOT / "reports/journal/phase_a_window_metrics.csv"
PUBLIC_CONDITIONING = ROOT / "reports/journal/phase_a_conditioning.csv"
PUBLIC_COUNTERFACTUALS = ROOT / "reports/journal/phase_a_counterfactuals.csv"
PUBLIC_BASELINES = ROOT / "reports/journal/phase_a_baselines.csv"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["test_access_count"] != 0 or any(value[key] != 0 for key in (
        "future_reserve_bundle_access_count", "future_reserve_outcome_access_count",
        "future_reserve_statistic_access_count",
    )):
        raise RuntimeError("sealed boundary counters must remain zero")
    return value


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["output_root"]


def json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value["rows"])


def aggregate_subject(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subjectid"])].append(row)
    return {
        metric: float(np.mean([np.mean([float(row[metric]) for row in group]) for group in grouped.values()]))
        for metric in metrics
    } | {"subject_count": len(grouped), "case_count": len(rows)}


def run_a4(config: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / config["day06_output_root"]
    private = json_rows(root / "development_evaluation_private.json")
    metrics = (
        "mae_0_1800", "mae_0_600", "mae_600_1800", "time_in_40_60_fraction",
        "time_above_60_fraction", "time_below_40_fraction", "total_propofol_mg",
        "action_boundary_fraction", "core_clip_fraction", "visible_fraction",
    )
    aggregate: list[dict[str, Any]] = []
    for profile in config["profiles"]:
        for state in ("S0", "S_PROP", "S_REMI", "S_CP", "S_CE", "S_CONC"):
            for seed in config["journal_sconc_seeds"]:
                selected = [r for r in private if r["profile"] == profile and r["state"] == state and int(r["seed"]) == seed]
                if len(selected) != 96:
                    raise RuntimeError(f"unexpected development row count: {profile}/{state}/{seed}: {len(selected)}")
                aggregate.append({"profile": profile, "state": state, "seed": seed, **aggregate_subject(selected, metrics)})
    public = list(csv.DictReader((ROOT / "reports/journal/day06_concentration_results.csv").open(encoding="utf-8")))
    public_by = {(r["profile"], r["state"], int(r["seed"])): r for r in public}
    max_discrepancy = max(
        abs(float(row[metric]) - float(public_by[(row["profile"], row["state"], row["seed"])][metric]))
        for row in aggregate for metric in metrics
    )
    if max_discrepancy > 1e-12:
        raise RuntimeError(f"A4 public/private aggregate mismatch: {max_discrepancy}")
    write_csv(PUBLIC_WINDOWS, aggregate)
    rankings: dict[str, Any] = {}
    for profile in config["profiles"]:
        state_means = {}
        for state in ("S0", "S_PROP", "S_REMI", "S_CP", "S_CE", "S_CONC"):
            chosen = [r for r in aggregate if r["profile"] == profile and r["state"] == state]
            state_means[state] = {
                "mae_0_1800": float(np.mean([r["mae_0_1800"] for r in chosen])),
                "mae_600_1800": float(np.mean([r["mae_600_1800"] for r in chosen])),
            }
        rankings[profile] = {
            window: sorted(state_means, key=lambda state: (state_means[state][window], state))
            for window in ("mae_0_1800", "mae_600_1800")
        }
    visibility = json.loads((ROOT / config["day04_output_root"] / "visibility_audit_aggregate.json").read_text(encoding="utf-8"))
    selected_visibility = [r for r in visibility["rows"] if r["age_seconds"] in (20, 30) and r["sqi_threshold"] in ("off", "50") and r["window"] in ("0_600", "0_1800")]
    return {
        "source_rows": len(private), "aggregate_rows": len(aggregate), "public_reconciliation_max_abs": max_discrepancy,
        "rankings": rankings, "visibility": selected_visibility,
        "alignment": {
            "template_and_remifentanil_origin": "anesthesia_start",
            "simulator_t0": "anesthesia_start_relative_zero",
            "initial_cp_ce": 0.0,
            "first_simulator_propofol": "first_policy_action_over_0_to_10_seconds",
            "clinical_first_propofol_available": False,
            "carry_forward": "last_accepted_bis_until_staleness_cap",
            "age": "query_time_minus_event_time_capped_at_30_seconds",
        },
    }


def factors(config: dict[str, Any]) -> np.ndarray:
    value = json.loads((ROOT / config["day04_output_root"] / "nscale_factors.json").read_text(encoding="utf-8"))["factors_by_field"]
    return np.asarray([value[name] for name in S1_FIELDS], dtype=np.float32)


def sample_matrix(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = ROOT / config["day04_output_root"] / "calibration_observations_private.npy"
    matrix = np.load(path, mmap_mode="r")
    steps = int(config["horizon_seconds"] // 10)
    if matrix.shape[1] != len(S1_FIELDS) or matrix.shape[0] % steps:
        raise RuntimeError("calibration matrix shape does not preserve complete trajectories")
    result = {}
    limit = int(config["diagnostic_sample_rows_per_window"])
    for name, mask in (
        ("0_600", np.arange(matrix.shape[0]) % steps < int(config["early_end_seconds"] // 10)),
        ("600_1800", np.arange(matrix.shape[0]) % steps >= int(config["early_end_seconds"] // 10)),
    ):
        available = np.flatnonzero(mask)
        chosen = available[np.linspace(0, len(available) - 1, min(limit, len(available)), dtype=int)]
        result[name] = np.asarray(matrix[chosen], dtype=np.float32) / factors(config)
    return result


def model_path(config: dict[str, Any], seed: int, profile: str, state: str) -> Path:
    if state == "S0":
        return ROOT / config["day04_output_root"] / "jobs" / f"seed_{seed}" / f"{profile}S0" / "final_post_update" / "model.zip"
    if state == "S_CONC":
        return ROOT / config["day05_output_root"] / "jobs" / f"seed_{seed}" / f"{profile}S_CONC" / "final_post_update" / "model.zip"
    raise ValueError(state)


def actor_values(model: PPO, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = model.policy.mlp_extractor.policy_net[0]
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        pre = linear(tensor).cpu().numpy()
        hidden = torch.tanh(linear(tensor)).cpu().numpy()
    action = np.asarray(model.predict(observations, deterministic=True)[0], dtype=float).reshape(-1)
    return pre, hidden, action


def summarize_delta(base_pre: np.ndarray, changed_pre: np.ndarray, base_action: np.ndarray, changed_action: np.ndarray) -> dict[str, float]:
    pre_l2 = np.linalg.norm(changed_pre - base_pre, axis=1)
    delta = changed_action - base_action
    return {
        "delta_preactivation_l2_mean": float(np.mean(pre_l2)),
        "delta_preactivation_l2_p95": float(np.quantile(pre_l2, .95)),
        "delta_action_mean": float(np.mean(delta)),
        "delta_action_abs_mean": float(np.mean(np.abs(delta))),
        "delta_action_abs_p95": float(np.quantile(np.abs(delta), .95)),
        "delta_action_max_abs": float(np.max(np.abs(delta))),
    }


def bis_off(observations: np.ndarray, fields: list[str]) -> np.ndarray:
    changed = observations.copy()
    for index, name in enumerate(fields):
        if name.startswith("bis_value_") or name.startswith("bis_mask_"):
            changed[:, index] = 0.0
    return changed


def run_a3_a1_a2(config: dict[str, Any], historical_root: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(1)
    samples = sample_matrix(config)
    conditioning_rows: list[dict[str, Any]] = []
    for profile in config["profiles"]:
        for seed in config["journal_s0_seeds"]:
            model = PPO.load(str(model_path(config, seed, profile, "S0")), device="cpu")
            for window, full in samples.items():
                obs = full[:, :len(S0_FIELDS)]
                pre, hidden, action = actor_values(model, obs)
                changed = bis_off(obs, list(S0_FIELDS))
                changed_pre, _, changed_action = actor_values(model, changed)
                conditioning_rows.append({
                    "scope": "journal_day04_s0_not_historical_conference_policy", "profile": profile, "seed": seed,
                    "window": window, "input_abs_p95": float(np.quantile(np.abs(obs), .95)),
                    "preactivation_abs_p95": float(np.quantile(np.abs(pre), .95)),
                    "tanh_abs_gt_0_95_fraction": float(np.mean(np.abs(hidden) > .95)),
                    "tanh_abs_gt_0_99_fraction": float(np.mean(np.abs(hidden) > .99)),
                    "action_mean": float(np.mean(action)), "action_boundary_fraction": float(np.mean((action <= 1e-8) | (action >= 27.7 - 1e-8))),
                    **summarize_delta(pre, changed_pre, action, changed_action),
                })
    fresh_rows = []
    base = samples["0_600"][:, :len(S0_FIELDS)]
    changed = bis_off(base, list(S0_FIELDS))
    for seed in config["fresh_initialization_seeds"]:
        torch.manual_seed(int(seed))
        layer = torch.nn.Linear(len(S0_FIELDS), 128)
        torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0)); torch.nn.init.zeros_(layer.bias)
        with torch.no_grad():
            first = layer(torch.as_tensor(base)).numpy(); second = layer(torch.as_tensor(changed)).numpy()
        fresh_rows.append(float(np.mean(np.linalg.norm(second - first, axis=1))))
    write_csv(PUBLIC_CONDITIONING, conditioning_rows)

    indices = list(day05.STATE_INDICES["S_CONC"])
    fields = [S1_FIELDS[index] for index in indices]
    counter_rows: list[dict[str, Any]] = []
    for profile in config["profiles"]:
        for seed in config["journal_sconc_seeds"]:
            model = PPO.load(str(model_path(config, seed, profile, "S_CONC")), device="cpu")
            for window, full in samples.items():
                obs = full[:, indices]
                pre, _, action = actor_values(model, obs)
                variants: dict[str, np.ndarray] = {"bis_off_value_and_mask": bis_off(obs, fields)}
                mask_only = obs.copy()
                for i, name in enumerate(fields):
                    if name.startswith("bis_mask_"): mask_only[:, i] = 0.0
                variants["bis_mask_off_values_retained"] = mask_only
                value_only = obs.copy()
                for i, name in enumerate(fields):
                    if name.startswith("bis_value_"): value_only[:, i] = 0.0
                variants["bis_values_zero_masks_retained"] = value_only
                conc = [i for i, name in enumerate(fields) if "_cp_" in name or "_ce_" in name]
                for scale in (.8, 1.2):
                    altered = obs.copy(); altered[:, conc] *= scale; variants[f"concentration_scale_{scale:.1f}"] = altered
                for bias in (-.5, .5):
                    altered = obs.copy(); altered[:, conc] += bias; variants[f"concentration_bias_{bias:+.1f}_model_units"] = altered
                for name, altered in variants.items():
                    altered_pre, _, altered_action = actor_values(model, altered)
                    counter_rows.append({"profile": profile, "seed": seed, "window": window, "perturbation": name, **summarize_delta(pre, altered_pre, action, altered_action)})
    write_csv(PUBLIC_COUNTERFACTUALS, counter_rows)
    historical_s0 = False
    if historical_root is not None:
        historical_s0 = any(historical_root.rglob("P0S0/**/model.zip")) and any(historical_root.rglob("P1S0/**/model.zip"))
    a3 = {
        "historical_p0s0_p1s0_checkpoint_available": historical_s0,
        "substitute_scope": "journal_day04_s0_final_checkpoints",
        "conditioning_rows": len(conditioning_rows),
        "fresh_initialization_scope": "same_architecture_new_untrained_initializations_not_historical_step_zero",
        "fresh_initialization_count": len(fresh_rows),
        "fresh_bis_off_delta_preactivation_l2_mean": float(np.mean(fresh_rows)),
    }
    a1 = {"rows": sum(r["perturbation"].startswith("bis_") for r in counter_rows), "scope": "one_step_same_input_counterfactual_not_closed_loop_observation_process"}
    a2 = {"rows": sum(r["perturbation"].startswith("concentration_") for r in counter_rows), "scope": "policy_input_only_physically_inconsistent_stress_test", "pd_parameter_probe_performed": False, "delay_probe_performed": False}
    return a3, a1, a2


class NominalController:
    def __init__(self, tci: dict[str, float], trim: dict[str, float] | None = None):
        self.tci, self.trim = tci, trim
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def action(self, info: dict[str, Any]) -> tuple[float, bool]:
        requested = self.tci["maintenance_mg_per_10s"] + self.tci["kp"] * (self.tci["target_ce"] - float(info["propofol_ce_mg_per_l"]))
        if self.trim is not None and float(info["visible_current_bis_mask"]) > 0.5:
            error = float(info["visible_current_bis_value"]) - 50.0
            proposal = self.integral + error * 10.0
            proposal = float(np.clip(proposal, -300.0, 300.0))
            candidate = requested + self.trim["bis_kp"] * error + self.trim["bis_ki"] * proposal
            clipped = float(np.clip(candidate, 0.0, 27.7))
            if clipped == candidate:
                self.integral = proposal
            requested = candidate
        return float(np.clip(requested, 0.0, 27.7)), bool(requested < 0.0 or requested > 27.7)


def baseline_case(store: Any, scalers: Any, caseid: str, condition: dict[str, Any], controller: NominalController, horizon: float) -> dict[str, float]:
    env = make_environment(truncate_bundle(store.load_case(caseid), horizon), condition, scalers["S1"], 45)
    _, info = env.reset(seed=45); controller.reset(); latent=[]; actions=[]; clips=0; boundaries=0; done=False
    while not done:
        action, boundary = controller.action(info); boundaries += int(boundary)
        _, _, terminated, truncated, info = env.step(np.asarray([action], dtype=np.float32))
        latent.append(float(info["latent_true_bis"])); actions.append(float(info["applied_action_mg_per_10s"])); clips += int(info["action_was_clipped"])
        done = bool(terminated or truncated)
    env.close(); bis = np.asarray(latent); act = np.asarray(actions); late = bis[60:]
    return {
        "mae_0_1800": float(np.mean(np.abs(bis - 50.0))), "mae_600_1800": float(np.mean(np.abs(late - 50.0))),
        "time_in_40_60_fraction": float(np.mean((bis >= 40) & (bis <= 60))), "time_above_60_fraction": float(np.mean(bis > 60)),
        "time_below_40_fraction": float(np.mean(bis < 40)), "total_propofol_mg": float(np.sum(act)),
        "action_boundary_fraction": boundaries / len(act), "core_clip_fraction": clips / len(act),
    }


def mean_mae(store: Any, scalers: Any, cases: list[str], conditions: list[dict[str, Any]], factory: Callable[[], NominalController], horizon: float) -> float:
    values = [baseline_case(store, scalers, caseid, condition, factory(), horizon)["mae_0_1800"] for condition in conditions for caseid in cases]
    return float(np.mean(values))


def run_a5(config: dict[str, Any], historical_root: Path | None) -> dict[str, Any]:
    if historical_root is None:
        return {"performed": False, "reason": "historical development runtime root unavailable"}
    os.environ[config["historical_root_env"]] = str(historical_root)
    day04_config, _ = day04.load_config(ROOT / "configs/journal/day04_confirmation.json")
    _, _, _, _, store, scalers = day04.context(day04_config)
    membership = json.loads((ROOT / config["day04_output_root"] / "private_membership.json").read_text(encoding="utf-8"))
    tuning_cases = list(membership["tuning_cases"][:int(config["baseline_tuning_case_limit"])])
    horizon = float(config["horizon_seconds"])
    p0 = day04.condition(day04_config, "P0S1"); p1 = day04.condition(day04_config, "P1S1")
    tci_scores=[]
    for candidate in config["tci_candidates"]:
        score=mean_mae(store,scalers,tuning_cases,[p0],lambda c=dict(candidate):NominalController(c),horizon)
        tci_scores.append((score,dict(candidate)))
    tci_score,tci=min(tci_scores,key=lambda item:(item[0],json.dumps(item[1],sort_keys=True)))
    trim_scores=[]
    for trim in config["bis_trim_candidates"]:
        score=mean_mae(store,scalers,tuning_cases,[p0,p1],lambda t=dict(tci),r=dict(trim):NominalController(t,r),horizon)
        trim_scores.append((score,dict(trim)))
    trim_score,trim=min(trim_scores,key=lambda item:(item[0],json.dumps(item[1],sort_keys=True)))
    private_rows=[]
    metrics=("mae_0_1800","mae_600_1800","time_in_40_60_fraction","time_above_60_fraction","time_below_40_fraction","total_propofol_mg","action_boundary_fraction","core_clip_fraction")
    for profile,condition in (("P0",p0),("P1",p1)):
        for family,factory in (("nominal_ce_target",lambda:NominalController(dict(tci))),("nominal_ce_target_plus_bis_pi_trim",lambda:NominalController(dict(tci),dict(trim)))):
            rows=[]
            for caseid in membership["fresh_validation_cases"]:
                result=baseline_case(store,scalers,caseid,condition,factory(),horizon)
                rows.append({"caseid":caseid,"subjectid":str(store._by_case[caseid]["subjectid"]),**result})
            private_rows.extend({"profile":profile,"family":family,**row} for row in rows)
    aggregate=[]
    for profile in config["profiles"]:
        for family in ("nominal_ce_target","nominal_ce_target_plus_bis_pi_trim"):
            selected=[r for r in private_rows if r["profile"]==profile and r["family"]==family]
            aggregate.append({"profile":profile,"family":family,**aggregate_subject(selected,metrics)})
    write_csv(PUBLIC_BASELINES,aggregate)
    private_root=output_root(config);private_root.mkdir(parents=True,exist_ok=True)
    atomic_json(private_root/"baseline_evaluation_private.json",{"rows":private_rows,"test_access_count":0,"future_reserve_access_count":0})
    return {"performed":True,"tuning_case_count":len(tuning_cases),"validation_case_count":len(membership["fresh_validation_cases"]),"tci_selection":tci,"tci_train_mae":tci_score,"trim_selection":trim,"trim_train_mae":trim_score,"aggregate":aggregate,"information_boundary":"simulator-derived nominal concentrations, causal BIS and administration history; nominal observer is not separated from true simulator"}


def verify(config: dict[str, Any]) -> dict[str, Any]:
    paths=(PUBLIC_SUMMARY,PUBLIC_WINDOWS,PUBLIC_CONDITIONING,PUBLIC_COUNTERFACTUALS,PUBLIC_BASELINES)
    missing=[str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing: raise RuntimeError(f"missing outputs: {missing}")
    summary=json.loads(PUBLIC_SUMMARY.read_text(encoding="utf-8"))
    boundary=summary["boundary"]
    if any(boundary[key] != 0 for key in boundary): raise RuntimeError("sealed access boundary changed")
    for path in paths:
        text=path.read_text(encoding="utf-8")
        if "caseid" in text.lower() or "subjectid" in text.lower(): raise RuntimeError(f"identifier leaked into {path.name}")
    return {"verified":True,"public_file_count":len(paths),"public_sha256":{path.name:sha256_path(path) for path in paths},"test_access_count":0,"future_reserve_access_count":0}


def run(config_path: Path, historical_root: Path | None) -> None:
    started=time.perf_counter();config=load_config(config_path);torch.set_num_threads(1);random.seed(20260927);np.random.seed(20260927);torch.manual_seed(20260927)
    required=str(config["required_parent_sha"])
    if subprocess.run(["git","merge-base","--is-ancestor",required,"HEAD"],cwd=ROOT).returncode != 0:
        raise RuntimeError("current HEAD is not a descendant of the frozen Phase A parent")
    out=output_root(config);out.mkdir(parents=True,exist_ok=True)
    a4=run_a4(config);a3,a1,a2=run_a3_a1_a2(config,historical_root);a5=run_a5(config,historical_root)
    summary={"protocol_id":config["protocol_id"],"source_config_sha256":sha256_path(config_path),"a4":a4,"a3":a3,"a1":a1,"a2":a2,"a5":a5,"boundary":{"future_reserve_bundle_access_count":0,"future_reserve_outcome_access_count":0,"future_reserve_statistic_access_count":0,"test_access_count":0},"phase_b_training_executed":False,"wall_seconds":time.perf_counter()-started}
    atomic_json(PUBLIC_SUMMARY,summary);atomic_json(out/"OUTPUT_COMPLETE.json",{"protocol_id":config["protocol_id"],"source_config_sha256":sha256_path(config_path),"public_summary_sha256":sha256_path(PUBLIC_SUMMARY)})
    result=verify(config);atomic_json(out/"verification.json",result);print(json.dumps({"wall_seconds":summary["wall_seconds"],**result},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("run","verify","resume"));parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG);parser.add_argument("--historical-root",type=Path)
    args=parser.parse_args();config=load_config(args.config)
    if args.command=="verify": print(json.dumps(verify(config),indent=2));return
    complete_path=output_root(config)/"OUTPUT_COMPLETE.json"
    if args.command=="resume" and complete_path.exists():
        complete=json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("source_config_sha256")==sha256_path(args.config):
            print(json.dumps({"resumed":False,"reason":"matching completed output already exists",**verify(config)},indent=2));return
    root=args.historical_root
    if root is None and os.environ.get(config["historical_root_env"]): root=Path(os.environ[config["historical_root_env"]])
    if root is not None: root=root.resolve(strict=True)
    run(args.config,root)


if __name__ == "__main__":
    main()
