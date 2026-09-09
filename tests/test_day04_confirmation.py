import inspect
import json
from pathlib import Path

import numpy as np

from run_day03_scale import apply_transform
from run_day04_confirmation import (
    CONTRAST_FORMULAS, Day04SequenceEnv, PRIMARY_METRICS, contrast_value,
    evaluate, metric_record, prepare, subject_rows, supervise, resume,
    verify, verify_models,
)
from vitaldb_state_selection.anesthesia import ObservationRule
from vitaldb_state_selection.anesthesia.state import S0_FIELDS, S1_FIELDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/journal/day04_confirmation.json").read_text(encoding="utf-8"))


def test_exact_factorial_mapping_seed_order_and_budget():
    assert [(x["condition_id"], x["profile"], x["state_id"], x["sqi_threshold"], x["age_seconds"]) for x in CONFIG["conditions"]] == [
        ("P0S0", "P0", "S0", None, 30), ("P1S0", "P1", "S0", 50, 20),
        ("P0S1", "P0", "S1", None, 30), ("P1S1", "P1", "S1", 50, 20),
    ]
    assert CONFIG["seeds"] == [48, 49, 50, 51, 52]
    assert CONFIG["training_target_timesteps"] == 524288 == 256 * 2048
    assert CONFIG["expected_rollouts"] == 256
    assert CONFIG["expected_training_epochs"] == 2560


def test_audit_age_grid_and_historical_rules_coexist():
    for age in (10, 20, 30, 60):
        assert ObservationRule(str(age), None, age).staleness_seconds == age


def test_scaling_prefix_factor_sharing_and_binary_invariants():
    assert tuple(S1_FIELDS[:len(S0_FIELDS)]) == tuple(S0_FIELDS)
    base = np.arange(len(S1_FIELDS), dtype=np.float32)
    factors = np.maximum(1, base + 1)
    for i, name in enumerate(S1_FIELDS):
        if name == "sex_binary" or name.startswith("bis_mask_"):
            factors[i] = 1
    scaled = apply_transform(base, "Nscale", factors)
    binary = [i for i, name in enumerate(S1_FIELDS) if name == "sex_binary" or name.startswith("bis_mask_")]
    np.testing.assert_array_equal(base[binary], scaled[binary])
    np.testing.assert_array_equal(scaled[:len(S0_FIELDS)], apply_transform(base[:len(S0_FIELDS)], "Nscale", factors[:len(S0_FIELDS)]))


def test_contrast_direction_and_interaction():
    values = {"P0S0": 10.0, "P1S0": 8.0, "P0S1": 7.0, "P1S1": 4.0}
    assert {name: contrast_value(values, name) for name in CONTRAST_FORMULAS} == {
        "P0_state": -3.0, "P1_state": -4.0, "S0_preprocessing": -2.0,
        "S1_preprocessing": -3.0, "interaction": -1.0,
    }


def test_window_metrics_and_subject_aggregation():
    latent = [40.0] * 60 + [60.0] * 120
    row = metric_record(latent, [1.0] * 180, [0.1] * 180, 0, ["available"] * 180)
    assert row["mae_0_1800"] == row["mae_0_600"] == row["mae_600_1800"] == 10.0
    rows = [{"subjectid": "private-a", **{m: 1.0 for m in PRIMARY_METRICS}}, {"subjectid": "private-a", **{m: 3.0 for m in PRIMARY_METRICS}}]
    assert subject_rows(rows)[0]["mae_0_1800"] == 2.0


def test_preoutcome_freeze_resume_idempotence_and_no_partial_promotion_are_explicit():
    supervisor = inspect.getsource(supervise)
    runner = inspect.getsource(resume)
    verifier = inspect.getsource(verify_models)
    assert "preoutcome_committed" in supervisor
    assert "if not" in runner and "supervise(config_path)" in runner
    assert "final_post_update" in verifier and "OUTPUT_COMPLETE.json" in verifier
    assert ".partial" in verifier and "partial artifact cannot be promoted" in verifier
    assert supervisor.index("if all(") < supervisor.index("if not preoutcome_committed()")


def test_paired_sampler_is_seeded_and_recovery_capable():
    source = inspect.getsource(Day04SequenceEnv)
    assert "self.generator" in source and "recovery_snapshot" in source and "restore_recovery" in source


def test_test_access_is_frozen_zero_and_public_paths_are_generic():
    assert CONFIG["test_access_count"] == 0
    assert CONFIG["output_root"] == "outputs/journal/day04/confirmation_v1"
    assert "C:\\Users\\" not in json.dumps(CONFIG)


def test_runtime_membership_evaluation_and_publication_contracts_are_explicit():
    preparation = inspect.getsource(prepare)
    evaluation = inspect.getsource(evaluate)
    verification = inspect.getsource(verify)
    model_verification = inspect.getsource(verify_models)
    assert "membership overlap" in preparation
    assert preparation.index('atomic_json(out/"private_membership.json"') < preparation.index("min_fresh=")
    assert "verify_models(config_path)" in evaluation and "evaluation_lock.json" in evaluation
    assert "expected=20*len" in verification and '"evaluation_calls":len(primary)' in verification
    assert "subject_aggregation_before_group_metrics" in verification
    assert "privacy token" in verification and '"test_access_count":0' in verification
    assert "paired case order mismatch" in model_verification
