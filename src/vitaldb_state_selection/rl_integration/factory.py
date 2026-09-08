"""Four-condition synthetic Gymnasium environment factory."""

from __future__ import annotations

from vitaldb_state_selection.anesthesia import (
    AnesthesiaEnvironmentCore,
    ConditionID,
    EnvironmentConfig,
    FOUR_CONDITION_CONFIGS,
    ObservationRule,
    SyntheticObservationTemplate,
)
from vitaldb_state_selection.anesthesia.schedule import ConstantRemifentanilSchedule, PiecewiseConstantRemifentanilSchedule
from vitaldb_state_selection.pkpd import PatientProfile

from .adapter import GymnasiumAnesthesiaEnv


Schedule = ConstantRemifentanilSchedule | PiecewiseConstantRemifentanilSchedule


def make_gymnasium_environment(
    *,
    condition_id: ConditionID | str,
    patient_profile: PatientProfile,
    observation_template: SyntheticObservationTemplate,
    remifentanil_schedule: Schedule,
    seed: int,
    episode_horizon_seconds: float | None = None,
    observation_rule: ObservationRule | None = None,
) -> GymnasiumAnesthesiaEnv:
    condition = condition_id if isinstance(condition_id, ConditionID) else ConditionID(condition_id)
    base = FOUR_CONDITION_CONFIGS[condition.value]
    horizon = base.episode_horizon_seconds if episode_horizon_seconds is None else float(episode_horizon_seconds)
    config = EnvironmentConfig(base.preprocessing_id, base.state_id, episode_horizon_seconds=horizon)
    core = AnesthesiaEnvironmentCore(
        profile=patient_profile,
        config=config,
        observation_template=observation_template,
        remifentanil_schedule=remifentanil_schedule,
        observation_rule=observation_rule,
    )
    return GymnasiumAnesthesiaEnv(core, default_seed=seed)
