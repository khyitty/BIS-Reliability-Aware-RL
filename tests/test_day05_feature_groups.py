import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from run_day05_feature_groups import (
    CONTRAST_FORMULAS, DAY04_FACTOR_SHA, ProjectObservation, STATE_FIELDS,
    STATE_INDICES, aggregate_subject, audit, contrast_value, evaluate, factors,
    prepare, resume, smoke, supervise, verify, verify_models,
)
from vitaldb_state_selection.anesthesia.config import StateID
from vitaldb_state_selection.anesthesia.observation import BISReason, VisibleBIS
from vitaldb_state_selection.anesthesia.state import (
    CompletedDrugInterval, S0_FIELDS, S1_FIELDS, build_state,
)
from vitaldb_state_selection.pkpd import PatientProfile, Sex


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/journal/day05_feature_groups.json").read_text(encoding="utf-8"))


class _Processor:
    def query(self, _: float) -> VisibleBIS:
        return VisibleBIS(50.0, 1.0, 0.0, BISReason.AVAILABLE)


class _VectorEnv(gym.Env):
    action_space = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-100.0, 100.0, (42,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.arange(42, dtype=np.float32), {}

    def step(self, action):
        return np.arange(42, dtype=np.float32), 0.0, False, True, {}


def test_state_schema_order_dimensions_and_projection_equality():
    assert {key: len(value) for key, value in STATE_FIELDS.items()} == {
        "S0": 34, "S_CUM": 36, "S_CONC": 38, "S_CORE": 40, "S1": 42,
    }
    assert STATE_FIELDS["S0"] == tuple(S0_FIELDS)
    assert STATE_FIELDS["S1"] == tuple(S1_FIELDS)
    full = np.arange(42, dtype=np.float32)
    for state, indices in STATE_INDICES.items():
        wrapper = ProjectObservation(_VectorEnv(), state)
        np.testing.assert_array_equal(wrapper.observation(full), full[list(indices)])
        assert tuple(S1_FIELDS[index] for index in indices) == STATE_FIELDS[state]


def test_recent_summaries_are_exact_redundant_identities():
    intervals = tuple(CompletedDrugInterval(float(10*i), float(i), float(2*i), float(i/3)) for i in range(1, 8))
    profile = PatientProfile(age_years=40, sex=Sex.FEMALE, height_cm=165, weight_kg=60)
    state = build_state(state_id=StateID.S1, profile=profile, elapsed_seconds=70.0,
                        bis_processor=_Processor(), intervals=intervals, transition=None)
    values = dict(zip(state.field_names, state.vector))
    prop_history = sum(values[name] for name in S0_FIELDS if name.startswith("propofol_dose_mg_"))
    remi_history = sum(values[name] for name in S0_FIELDS if name.startswith("remifentanil_rate_microgram_per_min_"))
    assert values["propofol_recent_dose_60s_mg"] == prop_history
    assert values["remifentanil_recent_dose_60s_microgram"] == (10 / 60) * remi_history


def test_frozen_budget_order_factor_and_anchor_contracts():
    assert CONFIG["factor_sha256"] == DAY04_FACTOR_SHA
    assert CONFIG["new_states"] == ["S_CUM", "S_CONC", "S_CORE"]
    assert CONFIG["seeds"] == [48, 49, 50, 51, 52]
    assert CONFIG["training_target_timesteps"] == 524288
    assert CONFIG["expected_rollouts"] == 256 and CONFIG["expected_training_epochs"] == 2560
    assert CONFIG["checkpoint_steps"] == [131072, 262144, 393216, 524288]
    source = inspect.getsource(prepare) + inspect.getsource(factors)
    assert "factor_refit" in source and "anchor budget/factor mismatch" in source
    assert "verify_checkpoint" in source and "future_reserve_bundle_access_count" in source
    assert "future_reserve_case_count" in source


def test_identical_dynamics_pairing_final_only_and_resume_contracts():
    assert "projection changed native dynamics" in inspect.getsource(smoke)
    assert "paired case order differs" in inspect.getsource(verify_models)
    assert 'config["training_target_timesteps"]' in inspect.getsource(evaluate)
    assert "checkpoint_evaluation_private" in inspect.getsource(evaluate)
    supervisor = inspect.getsource(supervise)
    assert supervisor.index("if all(") < supervisor.index("preoutcome_committed")
    assert "quarantine_" in supervisor and "*.partial" in supervisor
    assert "supervise(config_path)" in inspect.getsource(resume)


def test_subject_aggregation_and_all_contrast_algebra():
    rows = [{"subjectid": "a", **{metric: 1.0 for metric in ("mae_0_1800",)}}]
    # Exercise the real aggregator with its complete metric contract.
    from run_day05_feature_groups import EVAL_METRICS
    rows = [{"subjectid": "a", **{metric: value for metric in EVAL_METRICS}} for value in (1.0, 3.0)]
    assert aggregate_subject(rows)[0]["mae_0_1800"] == 2.0
    values = {"S0": 10.0, "S_CUM": 8.0, "S_CONC": 7.0, "S_CORE": 4.0, "S1": 5.0}
    expected = {
        "cumulative_only": -2.0, "concentration_only": -3.0, "combined_core": -6.0,
        "cumulative_given_concentrations": -3.0, "concentration_given_cumulative": -4.0,
        "group_interaction": -1.0, "redundant_recent": 1.0, "original_full_state": -5.0,
        "cumulative_main": -2.5, "concentration_main": -3.5,
    }
    assert set(CONTRAST_FORMULAS) == set(expected)
    assert {name: contrast_value(values, name) for name in expected} == expected


def test_access_privacy_and_accounting_are_explicit():
    assert CONFIG["test_access_count"] == 0
    assert "future_reserve_bundle_access_count" in inspect.getsource(verify)
    assert "development_evaluation_calls" in inspect.getsource(verify)
    assert "subject_aggregation_before_group_metrics" in inspect.getsource(verify)
    assert "privacy token" in inspect.getsource(verify)
    assert "store.load_case" not in inspect.getsource(prepare)
    assert "C:\\Users\\" not in json.dumps(CONFIG)
