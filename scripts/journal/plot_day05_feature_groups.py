"""Create the publication-safe Day-05 seed-linked MAE figure."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/journal/day05_feature_group_results.csv"
PNG = ROOT / "reports/journal/day05_feature_group_mae.png"
PDF = ROOT / "reports/journal/day05_feature_group_mae.pdf"
STATES = ("S0", "S_CUM", "S_CONC", "S_CORE", "S1")
SEEDS = (48, 49, 50, 51, 52)


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True, constrained_layout=True)
    x = np.arange(len(STATES))
    colors = {seed: ("#c23b22" if seed == 50 else "#7393b3") for seed in SEEDS}
    for axis, profile in zip(axes, ("P0", "P1")):
        matrix = np.asarray([[float(next(r["mae_0_1800"] for r in rows if r["profile"] == profile and r["state"] == state and int(r["seed"]) == seed)) for state in STATES] for seed in SEEDS])
        for seed, values in zip(SEEDS, matrix):
            axis.plot(x, values, marker="o", linewidth=1.0, markersize=4,
                      color=colors[seed], alpha=0.95 if seed == 50 else 0.55,
                      label=f"seed {seed}" if profile == "P0" else None)
        mean = matrix.mean(axis=0); sd = matrix.std(axis=0, ddof=1)
        axis.errorbar(x, mean, yerr=sd, fmt="D", color="black", capsize=4,
                      linewidth=1.8, markersize=5, label="mean ± seed SD" if profile == "P0" else None)
        axis.set_title(profile); axis.set_xticks(x, STATES, rotation=25, ha="right")
        axis.set_xlabel("State representation"); axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Latent-BIS MAE, 0–1,800 s")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    fig.suptitle("Day 05 exploratory development screening (reconstructed simulation)", y=1.08)
    for path in (PNG, PDF): fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Creator": "deterministic Day05 plotting script"})
    plt.close(fig)


if __name__ == "__main__": main()
