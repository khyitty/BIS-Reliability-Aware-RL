"""Dependency-free closed-loop anesthesia environment core for Stage II."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vitaldb_state_selection.pkpd import DualDrugSimulator, PatientProfile, deterministic_bis

from .action import ActionApplication, apply_propofol_action
from .config import EnvironmentConfig
from .observation import BISObservationProcessor, ObservationRule, SyntheticObservationTemplate
from .schedule import ConstantRemifentanilSchedule, PiecewiseConstantRemifentanilSchedule
from .state import BuiltState, CompletedDrugInterval, build_state


RemifentanilSchedule = ConstantRemifentanilSchedule | PiecewiseConstantRemifentanilSchedule


def latent_bis_reward(*, target_bis: float, latent_next_bis: float, alpha: float = 1.0) -> float:
    reward = 1.0 / (abs(float(target_bis) - float(latent_next_bis)) + float(alpha))
    if not math.isfinite(reward) or reward <= 0:
        raise RuntimeError("reward invariant violated")
    return reward


class AnesthesiaEnvironmentCore:
    """Stateful reset/step interface without Gymnasium, SB3, Torch, or PPO."""

    def __init__(
        self,
        *,
        profile: PatientProfile,
        config: EnvironmentConfig,
        observation_template: SyntheticObservationTemplate,
        remifentanil_schedule: RemifentanilSchedule | None = None,
        observation_rule: ObservationRule | None = None,
    ) -> None:
        if observation_template.episode_horizon_seconds < config.episode_horizon_seconds:
            raise ValueError("template horizon cannot be shorter than environment horizon")
        self.profile = profile
        self.config = config
        self.template = observation_template
        self.schedule = remifentanil_schedule or ConstantRemifentanilSchedule(0.0)
        self.observation_rule = observation_rule
        self._simulator: DualDrugSimulator
        self._processor: BISObservationProcessor
        self._intervals: list[CompletedDrugInterval]
        self._last_transition = None
        self._elapsed = 0.0
        self._done = False
        self._saturation_count = 0
        self._step_count = 0
        self._last_state: BuiltState
        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        if options not in (None, {}):
            raise ValueError("Stage II reset options are not implemented")
        self._simulator = DualDrugSimulator.from_profile(self.profile)
        self._processor = BISObservationProcessor(
            self.config.preprocessing_id, self.template, self.observation_rule
        )
        self._intervals = []
        self._last_transition = None
        self._elapsed = 0.0
        self._done = False
        self._saturation_count = 0
        self._step_count = 0
        baseline = deterministic_bis(0.0, 0.0)
        for event in self.template.bis_events:
            if event.timestamp_seconds == 0.0:
                self._processor.ingest(event, baseline)
        self._last_state = self._build_state()
        return self._last_state.vector.copy(), self._info(None, baseline, seed=seed)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("reset is required after episode completion")
        application = apply_propofol_action(
            action,
            minimum=self.config.action_min_mg_per_10s,
            maximum=self.config.action_max_mg_per_10s,
        )
        if application.action_was_clipped:
            self._saturation_count += 1
        start = self._elapsed
        end = start + self.config.control_interval_seconds
        bis_events = self.template.bis_between(start, end)
        sqi_events = self.template.sqi_between(start, end)
        boundaries = sorted({
            *(event.timestamp_seconds for event in bis_events),
            *(event.timestamp_seconds for event in sqi_events),
            *self.schedule.change_times(start, end),
            end,
        })
        current = start
        remifentanil_dose = 0.0
        for boundary in boundaries:
            duration = boundary - current
            if duration > 0:
                remi_rate = self.schedule.rate_microgram_per_min(current)
                transition = self._simulator.advance(
                    duration,
                    application.propofol_rate_mg_per_min,
                    remi_rate,
                )
                self._simulator = transition.next_simulator
                self._last_transition = transition
                remifentanil_dose += remi_rate * duration / 60.0
            for event in bis_events:
                if event.timestamp_seconds == boundary:
                    self._processor.ingest(event, self._last_transition.deterministic_bis_index)
            current = boundary
        transition = self._last_transition
        if transition is None or transition.elapsed_seconds != end:
            raise RuntimeError("transition endpoint invariant violated")
        average_remi_rate = remifentanil_dose * 60.0 / self.config.control_interval_seconds
        self._intervals.append(CompletedDrugInterval(
            end_seconds=end,
            propofol_dose_mg=application.applied_action_mg_per_10s,
            remifentanil_rate_microgram_per_min=average_remi_rate,
            remifentanil_dose_microgram=remifentanil_dose,
        ))
        self._elapsed = end
        self._step_count += 1
        self._last_state = self._build_state()
        latent = transition.deterministic_bis_index
        reward = latent_bis_reward(
            target_bis=self.config.target_bis,
            latent_next_bis=latent,
            alpha=self.config.reward_alpha,
        )
        terminated = not self._scientific_state_valid(transition)
        truncated = self._elapsed >= self.config.episode_horizon_seconds
        self._done = terminated or truncated
        return self._last_state.vector.copy(), reward, terminated, truncated, self._info(application, latent)

    @property
    def bis_audit_events(self) -> tuple[Any, ...]:
        """Expose immutable event-level P0/P1 audit evidence outside the state vector."""

        return self._processor.audit_events

    def _build_state(self) -> BuiltState:
        return build_state(
            state_id=self.config.state_id,
            profile=self.profile,
            elapsed_seconds=self._elapsed,
            bis_processor=self._processor,
            intervals=tuple(self._intervals),
            transition=self._last_transition,
        )

    @staticmethod
    def _scientific_state_valid(transition: Any, tolerance: float = 1e-10) -> bool:
        values = (
            transition.propofol_a1_mg, transition.propofol_a2_mg, transition.propofol_a3_mg,
            transition.propofol_cp_mg_per_l, transition.propofol_ce_mg_per_l,
            transition.remifentanil_a1_microgram, transition.remifentanil_a2_microgram,
            transition.remifentanil_a3_microgram, transition.remifentanil_cp_microgram_per_l,
            transition.remifentanil_ce_microgram_per_l, transition.deterministic_bis_index,
        )
        return all(math.isfinite(value) for value in values) and all(value >= -tolerance for value in values[:-1])

    def _info(self, application: ActionApplication | None, latent: float, *, seed: int | None = None) -> dict[str, Any]:
        visible = self._last_state.current_visible_bis
        transition = self._last_transition
        remi_rate = self.schedule.rate_microgram_per_min(self._elapsed)
        return {
            "elapsed_time_seconds": self._elapsed,
            "target_bis": self.config.target_bis,
            "latent_true_bis": latent,
            "visible_current_bis_value": visible.value,
            "visible_current_bis_mask": visible.mask,
            "visible_current_bis_age_seconds": visible.age_seconds,
            "visible_current_bis_reason": visible.reason.value,
            "propofol_cp_mg_per_l": 0.0 if transition is None else transition.propofol_cp_mg_per_l,
            "propofol_ce_mg_per_l": 0.0 if transition is None else transition.propofol_ce_mg_per_l,
            "remifentanil_cp_microgram_per_l": 0.0 if transition is None else transition.remifentanil_cp_microgram_per_l,
            "remifentanil_ce_microgram_per_l": 0.0 if transition is None else transition.remifentanil_ce_microgram_per_l,
            "raw_action_mg_per_10s": None if application is None else application.raw_action_mg_per_10s,
            "applied_action_mg_per_10s": None if application is None else application.applied_action_mg_per_10s,
            "action_was_clipped": False if application is None else application.action_was_clipped,
            "propofol_rate_mg_per_min": 0.0 if application is None else application.propofol_rate_mg_per_min,
            "remifentanil_rate_microgram_per_min": remi_rate,
            "preprocessing_id": self.config.preprocessing_id.value,
            "observation_rule_id": (
                None if self.observation_rule is None else self.observation_rule.rule_id
            ),
            "state_id": self.config.state_id.value,
            "template_id": self.template.template_id,
            "action_saturation_count": self._saturation_count,
            "completed_control_steps": self._step_count,
            "bis_audit_event_count": len(self._processor.audit_events),
            "seed": seed,
        }
