import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from run_day06_concentration_anatomy import (
    CONTRAST_FORMULAS, DAY04_FACTOR_SHA, ProjectObservation, STATE_FIELDS,
    STATE_INDICES, aggregate_subject, audit, contrast_value, evaluate, factors,
    prepare, resume, retention_decision, smoke, supervise, verify, verify_models,
)
from vitaldb_state_selection.anesthesia.core import AnesthesiaEnvironmentCore
from vitaldb_state_selection.anesthesia.state import S0_FIELDS, S1_FIELDS, build_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/journal/day06_concentration_anatomy.json").read_text(encoding="utf-8"))


class _VectorEnv(gym.Env):
    action_space = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-100.0, 100.0, (42,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.arange(42, dtype=np.float32), {}

    def step(self, action):
        return np.arange(42, dtype=np.float32), 0.0, False, True, {}


def test_exact_state_fields_order_dimensions_and_projection_equality():
    assert {state: len(fields) for state, fields in STATE_FIELDS.items()} == {
        "S0": 34, "S_PROP": 36, "S_REMI": 36, "S_CP": 36, "S_CE": 36, "S_CONC": 38,
    }
    expected = {
        "S_PROP": ("propofol_cp_mg_per_l", "propofol_ce_mg_per_l"),
        "S_REMI": ("remifentanil_cp_microgram_per_l", "remifentanil_ce_microgram_per_l"),
        "S_CP": ("propofol_cp_mg_per_l", "remifentanil_cp_microgram_per_l"),
        "S_CE": ("propofol_ce_mg_per_l", "remifentanil_ce_microgram_per_l"),
    }
    full = np.arange(42, dtype=np.float32)
    assert STATE_FIELDS["S0"] == tuple(S0_FIELDS)
    for state, extras in expected.items():
        assert STATE_FIELDS[state] == tuple(S0_FIELDS) + extras
    for state, indices in STATE_INDICES.items():
        projected = ProjectObservation(_VectorEnv(), state).observation(full)
        np.testing.assert_array_equal(projected, full[list(indices)])
        assert tuple(S1_FIELDS[index] for index in indices) == STATE_FIELDS[state]


def test_causal_decision_order_is_pre_action_and_has_no_latent_bis_feature():
    source = inspect.getsource(AnesthesiaEnvironmentCore.step)
    assert source.index("application = apply_propofol_action") < source.index("self._simulator.advance")
    assert source.index("self._simulator.advance") < source.index("self._last_state = self._build_state()")
    assert "latent_true_bis" not in S1_FIELDS
    state_source = inspect.getsource(build_state)
    assert "transition.propofol_cp_mg_per_l" in state_source
    assert "transition.remifentanil_ce_microgram_per_l" in state_source
    assert "future" not in state_source.lower()
    assert "recorded" not in source.lower()
    assert "future_or_post_decision_inputs" in inspect.getsource(audit)


def test_frozen_budget_order_factor_anchor_and_access_contracts():
    assert CONFIG["factor_sha256"] == DAY04_FACTOR_SHA
    assert CONFIG["new_states"] == ["S_PROP", "S_REMI", "S_CP", "S_CE"]
    assert CONFIG["seeds"] == [48, 49, 50, 51, 52]
    assert CONFIG["training_target_timesteps"] == 524288
    assert CONFIG["expected_rollouts"] == 256 and CONFIG["expected_training_epochs"] == 2560
    assert CONFIG["checkpoint_steps"] == [131072, 262144, 393216, 524288]
    assert CONFIG["test_access_count"] == 0
    source = inspect.getsource(prepare) + inspect.getsource(factors)
    assert "factor_refit" in source and "Day04/Day05 anchor budget/factor mismatch" in source
    assert "future_reserve_statistic_access_count" in source and "validation_outcome_access_count" in source


def test_identical_dynamics_pairing_final_only_resume_and_quarantine_contracts():
    assert "projection changed native dynamics" in inspect.getsource(smoke)
    assert "paired case order differs" in inspect.getsource(verify_models)
    assert 'config["training_target_timesteps"]' in inspect.getsource(evaluate)
    assert "checkpoint_evaluation_private" in inspect.getsource(evaluate)
    supervisor = inspect.getsource(supervise)
    assert supervisor.index("if all(") < supervisor.index("preoutcome_committed")
    assert "quarantine_" in supervisor and "*.partial" in supervisor
    assert "supervise(config_path)" in inspect.getsource(resume)


def test_subject_aggregation_and_all_fixed_contrast_algebra():
    from run_day06_concentration_anatomy import EVAL_METRICS
    rows = [{"subjectid": "a", **{metric: value for metric in EVAL_METRICS}} for value in (1.0, 3.0)]
    assert aggregate_subject(rows)[0]["mae_0_1800"] == 2.0
    values = {"S0": 10.0, "S_PROP": 8.0, "S_REMI": 9.0, "S_CP": 7.0, "S_CE": 6.0, "S_CONC": 5.0}
    assert len(CONTRAST_FORMULAS) == 11
    for name, formula in CONTRAST_FORMULAS.items():
        left, right = formula.split("-")
        assert contrast_value(values, name) == values[left] - values[right]


def test_retention_rule_inclusive_boundaries_and_failures():
    seeds = [48, 49, 50, 51, 52]
    boundary = {(p, seed): 0.25 for p in ("P0", "P1") for seed in seeds}
    assert all(retention_decision(boundary, seeds, 3.75, 3.0, .05, 0.0, 1e-4).values())
    too_large = dict(boundary); too_large[("P1", 52)] = .7500001
    assert not retention_decision(too_large, seeds, 3.75, 3.0, .05, 0.0, 1e-4)["maximum_positive_regret"]
    assert not retention_decision(boundary, seeds, 3.75, 3.0, .050001, 0.0, 1e-4)["action_guardrails"]
    assert not retention_decision(boundary, seeds, 3.75, 3.0, .05, 1e-12, 1e-4)["action_guardrails"]


def test_privacy_accounting_and_expected_counts_are_explicit():
    source = inspect.getsource(verify)
    assert "development_evaluation_calls" in source
    assert "subject_aggregation_before_group_metrics" in source
    assert "privacy token" in source
    assert "future_reserve_statistic_access_count" in source
    assert "new_jobs\":40" in source and "checkpoint_diagnostic_cells\":240" in source
    assert "store.load_case" not in inspect.getsource(prepare)
    assert "C:\\Users\\" not in json.dumps(CONFIG)
