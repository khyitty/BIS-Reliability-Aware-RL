"""Validate and summarize the three matched-budget Day-01 engineering seeds."""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/journal/day01"
REPORT_CSV = ROOT / "reports/journal/day01_multiseed_results.csv"
REPORT_MD = ROOT / "reports/journal/DAY01_MULTISEED.md"
SEEDS = (45, 46, 47)
CONDITIONS = (
    "Goff_A30_S0", "Goff_A20_S0", "G50_A30_S0", "G50_A20_S0",
    "Goff_A30_S1", "Goff_A20_S1", "G50_A30_S1", "G50_A20_S1",
)
def load(budget: int) -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        suffix = "" if budget == 32768 else f"_t{budget}"
        run_root = OUTPUT / f"synthetic_seed{seed}{suffix}"
        status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed" or status.get("completed") != 8 or status.get("failed") != 0:
            raise RuntimeError(f"seed {seed} is not a complete eight-cell batch")
        for condition in CONDITIONS:
            path = run_root / condition / "result.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("seed") != seed or item.get("condition_id") != condition:
                raise RuntimeError(f"provenance mismatch: {seed}/{condition}")
            if item.get("actual_timesteps") != budget or item.get("status") != "completed":
                raise RuntimeError(f"budget/status mismatch: {seed}/{condition}")
            if item.get("evidence_scope") != "synthetic_engineering_only_not_vitaldb_or_clinical_evidence":
                raise RuntimeError(f"claim-boundary mismatch: {seed}/{condition}")
            rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, choices=(32768, 65536, 131072, 262144), default=32768)
    args = parser.parse_args()
    budget = args.budget
    rows = load(budget)
    report_csv = REPORT_CSV if budget == 32768 else REPORT_CSV.with_name(f"day01_multiseed_t{budget}_results.csv")
    report_md = REPORT_MD if budget == 32768 else REPORT_MD.with_name(f"DAY01_MULTISEED_T{budget}.md")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "condition_id", "state_id", "sqi_threshold", "age_seconds", "seed", "actual_timesteps",
        "elapsed_seconds", "steps_per_second", "mae", "return", "time_in_40_60_fraction",
        "time_below_40_fraction", "total_propofol_mg", "action_sd", "action_boundary_fraction",
        "core_clip_fraction", "bis_counterfactual_action_delta_mean", "evidence_scope",
    )
    with report_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for item in rows:
            flat = dict(item)
            flat.update(item["metrics"])
            writer.writerow({name: flat.get(name) for name in fields})

    lookup = {(item["condition_id"], item["seed"]): item for item in rows}
    interactions = {}
    for seed in SEEDS:
        mae = lambda name: float(lookup[(name, seed)]["metrics"]["mae"])
        interactions[(seed, "S0")] = (mae("G50_A20_S0") - mae("G50_A30_S0")) - (mae("Goff_A20_S0") - mae("Goff_A30_S0"))
        interactions[(seed, "S1")] = (mae("G50_A20_S1") - mae("G50_A30_S1")) - (mae("Goff_A20_S1") - mae("Goff_A30_S1"))

    lines = [
        "# Day 01 합성 다중-seed 공학 요약",
        "",
        f"이 문서는 실제 VitalDB 또는 임상 성능 결과가 아니다. 동일 합성 cohort, subject 분할, {budget:,}-step 예산으로 새 seed 45/46/47을 모두 보고한다.",
        "",
        "| condition | seed 45 MAE | seed 46 MAE | seed 47 MAE | mean | SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = [float(lookup[(condition, seed)]["metrics"]["mae"]) for seed in SEEDS]
        lines.append(
            f"| {condition} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
            f"{statistics.mean(values):.3f} | {statistics.stdev(values):.3f} |"
        )
    lines.extend([
        "",
        "## Gate×age interaction 진단",
        "",
        "interaction은 `(G50_A20−G50_A30)−(Goff_A20−Goff_A30)` MAE이다. 짧은 예산의 부호 변동 자체가 단일 seed 해석의 불안정성을 보여준다.",
        "",
        "| seed | S0 interaction | S1 interaction |",
        "|---:|---:|---:|",
    ])
    for seed in SEEDS:
        lines.append(f"| {seed} | {interactions[(seed, 'S0')]:+.3f} | {interactions[(seed, 'S1')]:+.3f} |")
    total_steps = sum(int(item["actual_timesteps"]) for item in rows)
    total_elapsed = sum(float(item["elapsed_seconds"]) for item in rows)
    clip_max = max(float(item["metrics"]["core_clip_fraction"]) for item in rows)
    if not math.isfinite(total_elapsed) or clip_max != 0.0:
        raise RuntimeError("aggregate runtime or action-clipping invariant failed")
    lines.extend([
        "",
        "## 실행 사실",
        "",
        f"- 완료: 24/24 seed-condition cell, 총 {total_steps:,} timestep, 실패 0",
        f"- 순수 학습 경과시간 합: {total_elapsed:.1f}초, 평균 처리량 {total_steps / total_elapsed:.1f} timestep/s",
        "- core action clipping 최대값: 0",
        "- 각 validation 결과는 case를 먼저 계산한 뒤 반복 case를 subject 안에서 평균하고, 마지막에 subject 평균을 계산했다.",
        "- 세 seed와 짧은 합성 실행은 일반 강건성 또는 방법 우월성을 확립하지 않는다.",
    ])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(rows), "timesteps": total_steps, "failed": 0}))


if __name__ == "__main__":
    main()
