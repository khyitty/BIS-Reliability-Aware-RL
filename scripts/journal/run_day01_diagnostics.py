"""Focused constant-rate/PID and trained-policy diagnostics for Day 01."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from run_day01_cpu import evaluate, make_cases, subject_split
from vitaldb_state_selection.cohort.train_runtime_inputs import StateScaler, load_scaler_registry


CONFIG = ROOT / "configs/journal/day01_cpu.json"
OUTPUT = ROOT / "reports/journal/day01_controller_diagnostics.csv"
REPORT = ROOT / "reports/journal/DAY01_DIAGNOSTICS.md"


class ConstantModel:
    def __init__(self, action: float):
        self.action = float(action)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return np.asarray([self.action], dtype=np.float32), None


class PIDLikeModel:
    """Transparent proportional engineering comparator, not a tuned clinical PID."""

    def __init__(self, scaler: StateScaler, base: float = 1.5, gain: float = 0.08):
        names = [field.field_name for field in scaler.fields]
        self.value_index = names.index("bis_value_t+0")
        self.mask_index = names.index("bis_mask_t+0")
        self.field = scaler.fields[self.value_index]
        self.base = base
        self.gain = gain

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        array = np.asarray(observation).reshape(-1)
        visible = float(array[self.value_index]) * self.field.scale + self.field.center
        mask = float(array[self.mask_index])
        action = self.base if mask < 0.5 else self.base + self.gain * (visible - 50.0)
        return np.asarray([np.clip(action, 0.0, 27.7)], dtype=np.float32), None


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cases = make_cases(int(config["episode_horizon_seconds"]))
    _, validation, split_sha = subject_split(cases, int(config["development_split_seed"]))
    scalers = load_scaler_registry(ROOT / config["scaler_source"])
    rows: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        scaler = scalers[condition["state_id"]]
        for controller, model in (
            ("constant_1.5_mg_per_10s", ConstantModel(1.5)),
            ("p_like_base1.5_gain0.08", PIDLikeModel(scaler)),
        ):
            metrics = evaluate(model, validation, condition, scaler, 45)
            rows.append({
                "condition_id": condition["condition_id"],
                "state_id": condition["state_id"],
                "sqi_threshold": condition["sqi_threshold"],
                "age_seconds": condition["age_seconds"],
                "controller": controller,
                "validation_membership_sha256": split_sha,
                **metrics,
                "evidence_scope": config["evidence_scope"],
            })
    fields = tuple(rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Day 01 controller 진단",
        "",
        "고정 1.5 mg/10s와 비튜닝 proportional comparator(`base=1.5`, `gain=0.08`)를 동일한 합성 validation subject에 적용했다. 임상 PID나 공정한 최적 baseline으로 주장하지 않는다.",
        "",
        "| condition | constant MAE | P-like MAE | constant BIS 40–60 | P-like BIS 40–60 |",
        "|---|---:|---:|---:|---:|",
    ]
    lookup = {(row["condition_id"], row["controller"]): row for row in rows}
    for condition in config["conditions"]:
        name = condition["condition_id"]
        constant = lookup[(name, "constant_1.5_mg_per_10s")]
        pid = lookup[(name, "p_like_base1.5_gain0.08")]
        lines.append(
            f"| {name} | {constant['mae']:.3f} | {pid['mae']:.3f} | "
            f"{constant['time_in_40_60_fraction']:.3f} | {pid['time_in_40_60_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "고정-rate 결과가 모든 관측 조건에서 같아야 한다는 불변식과 core clip fraction 0을 확인했다. P-like 차이는 오직 각 gate/age 규칙이 제공한 현재 BIS/mask에서 생긴다. 이 진단은 PPO 비교의 우월성 검정이 아니다.",
    ])
    constant_mae = {round(float(row["mae"]), 12) for row in rows if row["controller"].startswith("constant")}
    if len(constant_mae) != 1 or any(float(row["core_clip_fraction"]) != 0.0 for row in rows):
        raise RuntimeError("controller diagnostic invariant failed")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "constant_mae": next(iter(constant_mae)), "status": "completed"}))


if __name__ == "__main__":
    main()
