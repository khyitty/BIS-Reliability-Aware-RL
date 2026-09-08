"""One-command verification/resume for all completed Day-01 CPU batches."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/journal/run_day01_cpu.py"
SUMMARY = ROOT / "scripts/journal/summarize_day01.py"
SEEDS = (45, 46, 47)
MILESTONES = (32768, 65536, 131072, 262144)


def main() -> None:
    for milestone in MILESTONES:
        for seed in SEEDS:
            subprocess.run(
                [sys.executable, str(RUNNER), "--seed", str(seed), "--timesteps", str(milestone), "--resume"],
                cwd=ROOT,
                check=True,
            )
        subprocess.run([sys.executable, str(SUMMARY), "--budget", str(milestone)], cwd=ROOT, check=True)
    print("verified/resumed 4 milestones x 3 seeds x 8 conditions")


if __name__ == "__main__":
    main()
