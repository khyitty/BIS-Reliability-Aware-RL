from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "journal"))

from night02_common import RunningStatistic, schedule_integral, stable_order
from run_night02_cpu import verify_checkpoint
from evaluate_night02_cpu import Controller


def test_stable_order_is_deterministic_and_label_separated() -> None:
    values = ["subject-c", "subject-a", "subject-b"]
    assert stable_order(values, seed=20260908, label="train") == stable_order(values, seed=20260908, label="train")
    train_key = hashlib.sha256(b"20260908:train:subject-a").digest()
    validation_key = hashlib.sha256(b"20260908:validation:subject-a").digest()
    assert train_key != validation_key


def test_running_statistic_and_schedule_integral() -> None:
    stat = RunningStatistic()
    for value in (1.0, 2.0, 3.0):
        stat.add(value)
    assert stat.triple() == pytest.approx((3, 2.0, 1.0))
    assert schedule_integral(((0.0, 2.0), (30.0, 4.0)), 0.0, 60.0) == pytest.approx(3.0)


def test_checkpoint_validation_rejects_identity_substitution(tmp_path: Path) -> None:
    from night02_common import atomic_json, sha256_path
    directory = tmp_path / "checkpoint_0000000064"
    directory.mkdir()
    (directory / "model.zip").write_bytes(b"model")
    atomic_json(directory / "sampler.json", {"state": 1})
    metadata = {"condition_id": "Goff_A30_S0", "seed": 45, "timestep": 64}
    atomic_json(directory / "metadata.json", metadata)
    atomic_json(directory / "COMPLETE.json", {
        "complete": True, "metadata_sha256": sha256_path(directory / "metadata.json"),
        "model_sha256": sha256_path(directory / "model.zip"), "sampler_sha256": sha256_path(directory / "sampler.json"),
    })
    assert verify_checkpoint(directory, metadata) == metadata
    with pytest.raises(RuntimeError, match="seed"):
        verify_checkpoint(directory, {"seed": 46})


def test_pi_controller_resets_and_does_not_integrate_missing_feedback() -> None:
    controller = Controller("PI", 1.0, 0.1, 0.001)
    missing = {"visible_current_bis_mask": 0.0, "visible_current_bis_value": 0.0}
    assert controller.predict_once(missing) == 1.0
    assert controller.integral == 0.0
    visible = {"visible_current_bis_mask": 1.0, "visible_current_bis_value": 55.0}
    assert controller.predict_once(visible) > 1.0
    assert controller.integral == 50.0
    controller.reset()
    assert controller.integral == 0.0 and controller.calls == 0
