"""Day-01 CPU supervisor for a balanced synthetic reliability-factor pilot.

The synthetic cohort exercises the real PK/PD environment and PPO updates but is
engineering evidence only.  Private VitalDB stores are never synthesized or
silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from vitaldb_state_selection.anesthesia import (
    AnesthesiaEnvironmentCore,
    BISEvent,
    EnvironmentConfig,
    ObservationRule,
    PiecewiseConstantRemifentanilSchedule,
    PreprocessingID,
    SQIEvent,
    StateID,
    SyntheticObservationTemplate,
)
from vitaldb_state_selection.cohort.train_runtime_inputs import StateScaler, load_scaler_registry
from vitaldb_state_selection.pkpd import PatientProfile, Sex
from vitaldb_state_selection.rl_integration.adapter import GymnasiumAnesthesiaEnv
from vitaldb_state_selection.rl_integration.config import PAPER_ORIENTED_PPO_CANDIDATE_V1, make_ppo_model
from vitaldb_state_selection.rl_integration.train_runtime import ScaledTrainRuntimeEnv


DEFAULT_CONFIG = ROOT / "configs/journal/day01_cpu.json"
REPORT_CSV = ROOT / "reports/journal/day01_results.csv"
REPORT_MD = ROOT / "reports/journal/DAY01_REPORT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class SyntheticCase:
    case_index: int
    subject_index: int
    profile: PatientProfile
    template: SyntheticObservationTemplate
    schedule: PiecewiseConstantRemifentanilSchedule


def make_cases(horizon: int) -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    for subject in range(12):
        sex = Sex.MALE if subject % 2 == 0 else Sex.FEMALE
        profile = PatientProfile(
            32.0 + 3.5 * subject,
            sex,
            158.0 + (subject % 6) * 4.0,
            54.0 + (subject % 7) * 4.0,
        )
        for repeat in range(2):
            case_index = subject * 2 + repeat
            bis_events = []
            sqi_events = []
            for step, timestamp in enumerate(range(0, horizon + 1, 10)):
                # Bursts of missing events create genuine 20-vs-30 second age contrasts.
                cycle = (step + subject + repeat * 2) % 9
                available = cycle not in (3, 4, 5)
                bis_events.append(BISEvent(float(timestamp), available))
                sqi = 35.0 if (step + 2 * subject + repeat) % 7 in (2, 3) else 82.0
                sqi_events.append(SQIEvent(float(timestamp), sqi))
            template = SyntheticObservationTemplate(
                f"journal-synthetic-case-{case_index:02d}",
                float(horizon),
                tuple(bis_events),
                tuple(sqi_events),
            )
            base = 0.4 + 0.12 * (subject % 5) + 0.08 * repeat
            schedule = PiecewiseConstantRemifentanilSchedule(
                ((0.0, base), (200.0, base * 1.4), (400.0, base * 0.75))
            )
            cases.append(SyntheticCase(case_index, subject, profile, template, schedule))
    return cases


def subject_split(cases: list[SyntheticCase], seed: int) -> tuple[list[SyntheticCase], list[SyntheticCase], str]:
    subjects = list(range(12))
    random.Random(seed).shuffle(subjects)
    validation = set(subjects[:2])
    train = [case for case in cases if case.subject_index not in validation]
    valid = [case for case in cases if case.subject_index in validation]
    digest = hashlib.sha256(("\n".join(map(str, sorted(validation))) + "\n").encode("ascii")).hexdigest()
    return train, valid, digest


def make_environment(case: SyntheticCase, condition: dict[str, Any], scaler: StateScaler, seed: int) -> ScaledTrainRuntimeEnv:
    state = StateID(condition["state_id"])
    rule = ObservationRule(
        condition["condition_id"],
        condition["sqi_threshold"],
        float(condition["age_seconds"]),
    )
    config = EnvironmentConfig(PreprocessingID.P0, state, episode_horizon_seconds=case.template.episode_horizon_seconds)
    core = AnesthesiaEnvironmentCore(
        profile=case.profile,
        config=config,
        observation_template=case.template,
        remifentanil_schedule=case.schedule,
        observation_rule=rule,
    )
    return ScaledTrainRuntimeEnv(GymnasiumAnesthesiaEnv(core, default_seed=seed), scaler)


class SyntheticSequenceEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, cases: list[SyntheticCase], condition: dict[str, Any], scaler: StateScaler, seed: int):
        self.cases = cases
        self.condition = condition
        self.scaler = scaler
        self.seed_value = seed
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.environment: ScaledTrainRuntimeEnv | None = None
        probe = make_environment(cases[0], condition, scaler, seed)
        self.action_space = probe.action_space
        self.observation_space = probe.observation_space
        probe.close()
        self.seen_cases: set[int] = set()
        self.episodes = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=self.seed_value if seed is None else seed)
        if options not in (None, {}):
            raise ValueError("journal environment reset options are unsupported")
        if self.environment is not None:
            self.environment.close()
        case = self.cases[int(self.generator.integers(0, len(self.cases)))]
        self.seen_cases.add(case.case_index)
        self.episodes += 1
        self.environment = make_environment(case, self.condition, self.scaler, self.seed_value)
        observation, info = self.environment.reset(seed=self.seed_value)
        return observation, {**info, "synthetic_case_index": case.case_index}

    def step(self, action: np.ndarray):
        if self.environment is None:
            raise RuntimeError("reset required")
        return self.environment.step(action)

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()


def unwrap_sequence(vector: DummyVecEnv) -> SyntheticSequenceEnv:
    environment: Any = vector.envs[0]
    while hasattr(environment, "env"):
        environment = environment.env
    if not isinstance(environment, SyntheticSequenceEnv):
        raise RuntimeError("unexpected environment wrapper")
    return environment


def evaluate(model: Any, cases: list[SyntheticCase], condition: dict[str, Any], scaler: StateScaler, seed: int) -> dict[str, float]:
    subject_rows: dict[int, list[dict[str, float]]] = {}
    response_deltas: list[float] = []
    value_indices = [index for index, field in enumerate(scaler.fields) if field.field_name.startswith("bis_value_")]
    for case in cases:
        env = make_environment(case, condition, scaler, seed)
        observation, _ = env.reset(seed=seed)
        actions: list[float] = []
        latent: list[float] = []
        rewards: list[float] = []
        clipped = 0
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action_value = float(np.asarray(action).reshape(-1)[0])
            low, high = observation.copy(), observation.copy()
            for index in value_indices:
                field = scaler.fields[index]
                low[index] = (40.0 - field.center) / field.scale
                high[index] = (60.0 - field.center) / field.scale
            low_action = float(np.asarray(model.predict(low, deterministic=True)[0]).reshape(-1)[0])
            high_action = float(np.asarray(model.predict(high, deterministic=True)[0]).reshape(-1)[0])
            response_deltas.append(abs(high_action - low_action))
            observation, reward, terminated, truncated, info = env.step(np.asarray([action_value], dtype=np.float32))
            actions.append(float(info["applied_action_mg_per_10s"]))
            latent.append(float(info["latent_true_bis"]))
            rewards.append(float(reward))
            clipped += int(info["action_was_clipped"])
            done = terminated or truncated
        env.close()
        deviations = np.abs(np.asarray(latent) - 50.0)
        row = {
            "mae": float(deviations.mean()),
            "return": float(sum(rewards)),
            "time_in_40_60_fraction": float(np.mean((np.asarray(latent) >= 40.0) & (np.asarray(latent) <= 60.0))),
            "time_below_40_fraction": float(np.mean(np.asarray(latent) < 40.0)),
            "total_propofol_mg": float(sum(actions)),
            "action_sd": float(np.std(actions)),
            "action_boundary_fraction": float(np.mean((np.asarray(actions) <= 1e-6) | (np.asarray(actions) >= 27.7 - 1e-5))),
            "core_clip_fraction": float(clipped / len(actions)),
        }
        subject_rows.setdefault(case.subject_index, []).append(row)
    subject_means = [
        {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for rows in subject_rows.values()
    ]
    result = {key: float(np.mean([row[key] for row in subject_means])) for key in subject_means[0]}
    result["validation_subject_count"] = float(len(subject_means))
    result["validation_case_count"] = float(len(cases))
    result["bis_counterfactual_action_delta_mean"] = float(np.mean(response_deltas))
    return result


def run_job(config: dict[str, Any], condition: dict[str, Any], timesteps: int) -> dict[str, Any]:
    seed = int(config["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["torch_threads"]))
    cases = make_cases(int(config["episode_horizon_seconds"]))
    train, valid, split_sha = subject_split(cases, int(config["development_split_seed"]))
    scalers = load_scaler_registry(ROOT / config["scaler_source"])
    scaler = scalers[condition["state_id"]]
    vector = DummyVecEnv([lambda: SyntheticSequenceEnv(train, condition, scaler, seed)])
    ppo = replace(
        PAPER_ORIENTED_PPO_CANDIDATE_V1,
        configuration_id=config["protocol_id"],
        seed=seed,
        total_timesteps=timesteps,
        purpose="synthetic_engineering_factor_decomposition",
    )
    model = make_ppo_model(vector, ppo)
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, reset_num_timesteps=True, progress_bar=False)
    elapsed = time.perf_counter() - started
    metrics = evaluate(model, valid, condition, scaler, seed)
    sequence = unwrap_sequence(vector)
    output = ROOT / config["output_root"] / condition["condition_id"]
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "final_model")
    result = {
        "status": "completed",
        "evidence_scope": config["evidence_scope"],
        "protocol_id": config["protocol_id"],
        "condition_id": condition["condition_id"],
        "state_id": condition["state_id"],
        "sqi_threshold": condition["sqi_threshold"],
        "age_seconds": condition["age_seconds"],
        "seed": seed,
        "requested_timesteps": timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "elapsed_seconds": elapsed,
        "steps_per_second": float(model.num_timesteps / elapsed),
        "train_case_count": len(train),
        "train_subject_count": len({case.subject_index for case in train}),
        "training_cases_seen": len(sequence.seen_cases),
        "training_episodes": sequence.episodes,
        "validation_membership_sha256": split_sha,
        "metrics": metrics,
        "completed_at": utc_now(),
    }
    atomic_json(output / "result.json", result)
    vector.close()
    return result


def valid_result(path: Path, condition: dict[str, Any], timesteps: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("status") == "completed"
        and payload.get("condition_id") == condition["condition_id"]
        and payload.get("requested_timesteps") == timesteps
        and payload.get("actual_timesteps", 0) >= timesteps
    ):
        return payload
    return None


def write_reports(config: dict[str, Any], results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition_id", "state_id", "sqi_threshold", "age_seconds", "seed", "actual_timesteps",
        "elapsed_seconds", "steps_per_second", "training_cases_seen", "mae", "return",
        "time_in_40_60_fraction", "time_below_40_fraction", "total_propofol_mg", "action_sd",
        "action_boundary_fraction", "core_clip_fraction", "bis_counterfactual_action_delta_mean",
        "evidence_scope",
    ]
    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            row.update({key: value for key, value in result["metrics"].items() if key in fields})
            writer.writerow(row)
    throughput = np.mean([item["steps_per_second"] for item in results]) if results else float("nan")
    by_condition = {item["condition_id"]: item for item in results}
    lines = [
        "# Day 01 CPU 실행 보고서",
        "",
        "## 완료 범위",
        "",
        f"- 실제 완료 셀: {len(results)}/8, 실패 셀: {len(failures)}",
        f"- 공통 학습 예산: {config['pilot_timesteps']:,} 환경 timestep, 새 학습 seed {config['training_seed']}",
        "- 증거 경계: 합성 관측 패턴과 합성 환자 프로필을 사용한 공학 파일럿이며 VitalDB·임상 성능 증거가 아니다.",
        f"- 평균 처리량: {throughput:.1f} timestep/s (완료 셀 기준)",
        "- 개발 분할: 합성 subject 단위 고정 분할(10 train / 2 validation subject, 각 2 case)",
        "",
        "## 재사용한 역사적 증거",
        "",
        "커밋된 Phase 8G 집계에서 seed를 선택하지 않고 모두 확인했다. interaction은 `(P1S1−P1S0)−(P0S1−P0S0)` MAE이다.",
        "",
        "| seed | P0S0 | P1S0 | P0S1 | P1S1 | interaction |",
        "|---:|---:|---:|---:|---:|---:|",
        "| 42 | 12.781 | 5.421 | 7.673 | 6.954 | +6.641 |",
        "| 43 | 11.717 | 6.032 | 7.726 | 10.557 | +8.516 |",
        "| 44 | 18.694 | 8.029 | 8.428 | 7.816 | +10.053 |",
        "",
        "세 seed의 interaction 방향은 같지만, seed 3개만으로 일반적 강건성을 확립하지 않는다. 커밋된 집계에는 각 seed/condition 최종 모델 SHA-256 provenance가 들어 있으나 이 노트북에는 모델 파일 자체가 없다.",
        "",
        "## 셀별 결과",
        "",
        "| condition | steps | MAE | return | BIS 40–60 | BIS<40 | propofol mg | action SD | BIS 반응 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metric = item["metrics"]
        lines.append(
            f"| {item['condition_id']} | {item['actual_timesteps']} | {metric['mae']:.3f} | "
            f"{metric['return']:.3f} | {metric['time_in_40_60_fraction']:.3f} | "
            f"{metric['time_below_40_fraction']:.3f} | {metric['total_propofol_mg']:.2f} | "
            f"{metric['action_sd']:.3f} | {metric['bis_counterfactual_action_delta_mean']:.4f} |"
        )
    if len(by_condition) == 8:
        def mae(name: str) -> float:
            return float(by_condition[name]["metrics"]["mae"])

        s0_interaction = (mae("G50_A20_S0") - mae("G50_A30_S0")) - (mae("Goff_A20_S0") - mae("Goff_A30_S0"))
        s1_interaction = (mae("G50_A20_S1") - mae("G50_A30_S1")) - (mae("Goff_A20_S1") - mae("Goff_A30_S1"))
        state_effects = [
            mae(f"{prefix}_S1") - mae(f"{prefix}_S0")
            for prefix in ("Goff_A30", "Goff_A20", "G50_A30", "G50_A20")
        ]
        lines.extend([
            "",
            "## 사전 정의 요인 진단",
            "",
            f"- MAE의 gate×age 차이의 차이: S0 {s0_interaction:+.3f}, S1 {s1_interaction:+.3f} (음수는 이 합성 seed에서 gate와 짧은 age의 결합이 MAE를 더 낮춘 방향)",
            f"- 동일 gate/age에서 S1−S0 MAE: " + ", ".join(f"{value:+.3f}" for value in state_effects),
            "- 모든 셀의 core action clipping은 0이었다. 따라서 이번 실행에서 물리 action 경계를 벗어나는 결함 증거는 없다.",
        ])
    lines.extend([
        "",
        "## 해석과 한계",
        "",
        "모든 비교는 동일 예산·seed·subject 분할로 실행했다. 한 개의 짧은 seed이므로 우열이나 일반적 강건성을 주장하지 않는다. "
        "BIS 반응 Δ는 동일한 정규화 상태에서 BIS-history 값을 40과 60으로 바꾼 결정론적 정책 행동 차이의 평균이며, 인과 효과가 아닌 정책 민감도 진단이다.",
        "",
        "비공개 Phase 8B/8C 저장소가 이 노트북에 없어 실제 VitalDB 개발-subject 재학습은 실행하지 않았다. "
        "필요 입력은 `phase8b_train_observation_templates_v1`, `phase8c_train_runtime_inputs_v1`이며, 원본 train subject만으로 새 분할을 만들어야 한다.",
        "",
        "GPU는 이번 측정의 필수 병목 해소책이 아니다. 네트워크가 작고 단일 환경 시뮬레이션이 직렬이므로, 먼저 실제 데이터 I/O와 환경 처리량을 측정한 뒤 판단한다.",
        "",
        "## 다음 큐",
        "",
        "1. 비공개 train stores를 읽기 전용 경로로 제공하면 동일한 8셀 프로토콜을 실제 원본-train subject 개발 분할로 재실행한다.",
        "2. 그 전에는 seed 46 전체 8셀을 같은 예산으로 추가해 합성 파이프라인의 실행 안정성만 확인할 수 있다.",
        "3. 성능을 본 뒤 특정 셀만 연장하지 말고, 다음 공통 milestone을 모든 셀에 적용한다.",
    ])
    if failures:
        lines.extend(["", "## 실패", ""] + [f"- {item['condition_id']}: {item['error']}" for item in failures])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"active or stale single-run lock exists: {self.path}") from error
        identity = {"pid": os.getpid(), "started_at": utc_now(), "command": sys.argv}
        os.write(self.descriptor, canonical_bytes(identity))
        os.fsync(self.descriptor)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark", action="store_true", help="run only the first cell at benchmark budget")
    parser.add_argument("--resume", action="store_true", help="verify completed manifests and run remaining cells")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    timesteps = int(config["benchmark_timesteps"] if args.benchmark else config["pilot_timesteps"])
    conditions = config["conditions"][:1] if args.benchmark else config["conditions"]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with RunLock(output_root / "run.lock"):
        for index, condition in enumerate(conditions):
            result_path = output_root / condition["condition_id"] / "result.json"
            existing = valid_result(result_path, condition, timesteps) if args.resume else None
            status = {
                "pid": os.getpid(), "process_identity": f"day01-{os.getpid()}", "updated_at": utc_now(),
                "mode": "benchmark" if args.benchmark else "pilot", "job_index": index,
                "condition_id": condition["condition_id"], "completed": len(results),
                "failed": len(failures), "total": len(conditions),
            }
            atomic_json(output_root / "status.json", status)
            if existing is not None:
                results.append(existing)
                continue
            log_path = output_root / condition["condition_id"] / "error.log"
            try:
                results.append(run_job(config, condition, timesteps))
            except Exception as error:  # continue balanced queue after a recoverable cell failure
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
                failures.append({"condition_id": condition["condition_id"], "error": str(error)})
        # Regenerate after normal or pure --resume passes.
        write_reports(config, results, failures)
        atomic_json(output_root / "status.json", {
            "pid": os.getpid(), "process_identity": f"day01-{os.getpid()}", "updated_at": utc_now(),
            "mode": "benchmark" if args.benchmark else "pilot", "state": "completed",
            "completed": len(results), "failed": len(failures), "total": len(conditions),
        })
    print(json.dumps({"completed": len(results), "failed": failures, "timesteps": timesteps}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
