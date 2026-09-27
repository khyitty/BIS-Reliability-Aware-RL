from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase_a", ROOT / "scripts/journal/run_phase_a_diagnostics.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phase_a_config_keeps_sealed_boundaries_closed() -> None:
    config = json.loads((ROOT / "configs/journal/phase_a_diagnostics.json").read_text(encoding="utf-8"))
    assert config["test_access_count"] == 0
    assert config["future_reserve_bundle_access_count"] == 0
    assert config["future_reserve_outcome_access_count"] == 0
    assert config["future_reserve_statistic_access_count"] == 0


def test_bis_off_changes_only_value_and_mask_coordinates() -> None:
    fields = ["x", "bis_value_t+0", "bis_mask_t+0", "bis_age_seconds_t+0", "y"]
    original = np.arange(15, dtype=np.float32).reshape(3, 5)
    changed = MODULE.bis_off(original, fields)
    assert np.array_equal(changed[:, [0, 3, 4]], original[:, [0, 3, 4]])
    assert np.array_equal(changed[:, [1, 2]], np.zeros((3, 2), dtype=np.float32))
    assert np.array_equal(original, np.arange(15, dtype=np.float32).reshape(3, 5))


def test_nominal_controller_antiwindup_and_bounds() -> None:
    controller = MODULE.NominalController(
        {"target_ce": 3.0, "maintenance_mg_per_10s": 1.5, "kp": 3.0},
        {"bis_kp": 10.0, "bis_ki": 10.0},
    )
    info = {"propofol_ce_mg_per_l": 0.0, "visible_current_bis_mask": 1.0, "visible_current_bis_value": 100.0}
    action, clipped = controller.action(info)
    assert action == 27.7
    assert clipped is True
    assert controller.integral == 0.0


def test_public_output_names_are_aggregate_only() -> None:
    names = {path.name for path in (MODULE.PUBLIC_WINDOWS, MODULE.PUBLIC_CONDITIONING, MODULE.PUBLIC_COUNTERFACTUALS, MODULE.PUBLIC_BASELINES)}
    assert names == {
        "phase_a_window_metrics.csv", "phase_a_conditioning.csv",
        "phase_a_counterfactuals.csv", "phase_a_baselines.csv",
    }
