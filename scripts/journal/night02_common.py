"""Shared, privacy-aware Night-02 real-input preparation and environments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]

import gymnasium as gym
import numpy as np

from vitaldb_state_selection.anesthesia import (
    AnesthesiaEnvironmentCore,
    EnvironmentConfig,
    ObservationRule,
    PreprocessingID,
    StateID,
)
from vitaldb_state_selection.anesthesia.recorded_observation import RecordedObservationTemplate
from vitaldb_state_selection.cohort.train_runtime_inputs import (
    S0_FIELDS,
    S1_FIELDS,
    StateScaler,
    TrainRuntimeBundle,
    TrainRuntimeInputStore,
    make_scaler_fields,
    state_schema_sha256,
)
from vitaldb_state_selection.pkpd import DualDrugSimulator
from vitaldb_state_selection.rl_integration.adapter import GymnasiumAnesthesiaEnv
from vitaldb_state_selection.rl_integration.train_runtime import ScaledTrainRuntimeEnv


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stable_order(values: Iterable[str], *, seed: int, label: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{label}:{value}".encode("utf-8")).digest())


@dataclass
class RunningStatistic:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite scaler source")
        self.count += 1
        delta = numeric - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (numeric - self.mean)

    def triple(self) -> tuple[int, float, float]:
        return self.count, self.mean, math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0


def schedule_integral(knots: tuple[tuple[float, float], ...], start: float, end: float) -> float:
    total = 0.0
    for index, (timestamp, rate) in enumerate(knots):
        next_time = knots[index + 1][0] if index + 1 < len(knots) else end
        left, right = max(start, timestamp), min(end, next_time)
        if right > left:
            total += rate * (right - left) / 60.0
    return total


def fit_scalers(store: TrainRuntimeInputStore, caseids: list[str]) -> dict[str, Any]:
    stats = {name: RunningStatistic() for name in S1_FIELDS}
    rate_names = [name for name in S1_FIELDS if name.startswith("remifentanil_rate_")]
    fixed_zero = [name for name in S1_FIELDS if name.startswith("bis_value_") or name.startswith("propofol_")]
    age_names = [name for name in S1_FIELDS if name.startswith("bis_age_seconds_")]
    mask_names = [name for name in S1_FIELDS if name.startswith("bis_mask_")]
    dynamic = (
        "remifentanil_recent_dose_60s_microgram", "remifentanil_cumulative_dose_microgram",
        "remifentanil_cp_microgram_per_l", "remifentanil_ce_microgram_per_l",
    )
    for caseid in caseids:
        bundle = store.load_case(caseid)
        profile = bundle.profile
        for name, value in (
            ("age_years", profile.age_years), ("sex_binary", 1.0 if profile.sex.value == "male" else 0.0),
            ("height_cm", profile.height_cm), ("weight_kg", profile.weight_kg),
        ):
            stats[name].add(value)
        for name in fixed_zero:
            stats[name].add(0.0)
        for name in age_names:
            stats[name].add(30.0)
        for name in mask_names:
            stats[name].add(0.0)
        knots = bundle.remifentanil_schedule.knots
        for _, rate in knots:
            for name in rate_names:
                stats[name].add(rate)
        for name in dynamic:
            stats[name].add(0.0)
        horizon = bundle.episode_horizon_seconds
        cumulative = schedule_integral(knots, 0.0, horizon)
        recent = schedule_integral(knots, max(0.0, horizon - 60.0), horizon)
        average_rate = cumulative * 60.0 / horizon
        transition = DualDrugSimulator.from_profile(profile).advance(horizon, 0.0, average_rate)
        stats["remifentanil_recent_dose_60s_microgram"].add(recent)
        stats["remifentanil_cumulative_dose_microgram"].add(cumulative)
        stats["remifentanil_cp_microgram_per_l"].add(transition.remifentanil_cp_microgram_per_l)
        stats["remifentanil_ce_microgram_per_l"].add(transition.remifentanil_ce_microgram_per_l)
    triples = {name: statistic.triple() for name, statistic in stats.items()}
    s0 = StateScaler("S0", make_scaler_fields("S0", triples), state_schema_sha256(S0_FIELDS))
    s1 = StateScaler("S1", make_scaler_fields("S1", triples), state_schema_sha256(S1_FIELDS))
    return {
        "registry_id": "journal_night02_devtrain_scaler_v1",
        "fit_case_count": len(caseids),
        "fit_split": "development_train_only",
        "test_case_count_used": 0,
        "preprocessing_condition_used_for_fit": False,
        "p0_p1_share_same_scaler_for_each_state": True,
        "binary_and_mask_fields_unchanged": True,
        "source": "historical_preprocessing_neutral_recipe_refit_on_journal_development_train",
        "scalers": {"S0": s0.as_manifest(), "S1": s1.as_manifest()},
    }


def truncate_bundle(bundle: TrainRuntimeBundle, horizon: float) -> TrainRuntimeBundle:
    if bundle.episode_horizon_seconds < horizon:
        raise ValueError("bundle is shorter than frozen journal horizon")
    template = RecordedObservationTemplate(
        template_id=bundle.observation_template.template_id,
        episode_horizon_seconds=horizon,
        bis_events=tuple(event for event in bundle.observation_template.bis_events if event.timestamp_seconds <= horizon),
        sqi_events=tuple(event for event in bundle.observation_template.sqi_events if event.timestamp_seconds <= horizon),
        source_type=bundle.observation_template.source_type,
    )
    return TrainRuntimeBundle(
        bundle.caseid, bundle.subjectid, bundle.profile, template,
        bundle.remifentanil_schedule, horizon, bundle.bundle_id,
    )


def make_environment(bundle: TrainRuntimeBundle, condition: dict[str, Any], scaler: StateScaler, seed: int) -> ScaledTrainRuntimeEnv:
    state = StateID(condition["state_id"])
    rule = ObservationRule(condition["condition_id"], condition["sqi_threshold"], float(condition["age_seconds"]))
    config = EnvironmentConfig(PreprocessingID.P0, state, episode_horizon_seconds=bundle.episode_horizon_seconds)
    core = AnesthesiaEnvironmentCore(
        profile=bundle.profile,
        config=config,
        observation_template=bundle.observation_template,
        remifentanil_schedule=bundle.remifentanil_schedule,
        observation_rule=rule,
    )
    return ScaledTrainRuntimeEnv(GymnasiumAnesthesiaEnv(core, default_seed=seed), scaler)


class JournalSequenceEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, store: TrainRuntimeInputStore, caseids: list[str], condition: dict[str, Any], scaler: StateScaler, seed: int, horizon: float):
        self.store, self.caseids, self.condition, self.scaler = store, caseids, condition, scaler
        self.seed_value, self.horizon = seed, horizon
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.cache: dict[str, TrainRuntimeBundle] = {}
        self.environment: ScaledTrainRuntimeEnv | None = None
        self.seen: set[str] = set()
        self.episodes = 0
        probe = self._bundle(caseids[0])
        env = make_environment(probe, condition, scaler, seed)
        self.action_space, self.observation_space = env.action_space, env.observation_space
        env.close()

    def sampler_snapshot(self) -> dict[str, Any]:
        return {
            "bit_generator_state": self.generator.bit_generator.state,
            "episodes_started": self.episodes,
            "seen_case_count": len(self.seen),
            "resume_boundary": "next_episode; active_episode_and_rollout_buffer_are_not_restored",
        }

    def restore_sampler(self, snapshot: dict[str, Any]) -> None:
        self.generator.bit_generator.state = snapshot["bit_generator_state"]
        self.episodes = int(snapshot["episodes_started"])

    def _bundle(self, caseid: str) -> TrainRuntimeBundle:
        if caseid not in self.cache:
            self.cache[caseid] = truncate_bundle(self.store.load_case(caseid), self.horizon)
        return self.cache[caseid]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=self.seed_value if seed is None else seed)
        if options not in (None, {}):
            raise ValueError("reset options unsupported")
        if self.environment is not None:
            self.environment.close()
        caseid = self.caseids[int(self.generator.integers(0, len(self.caseids)))]
        self.seen.add(caseid)
        self.episodes += 1
        self.environment = make_environment(self._bundle(caseid), self.condition, self.scaler, self.seed_value)
        observation, info = self.environment.reset(seed=self.seed_value)
        return observation, info

    def step(self, action: np.ndarray):
        if self.environment is None:
            raise RuntimeError("reset required")
        return self.environment.step(action)

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()


def aggregate_subject_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subjectid"], []).append(row)
    subject_means = [
        {name: float(np.mean([float(row[name]) for row in values])) for name in metric_names}
        for values in grouped.values()
    ]
    return {name: float(np.mean([row[name] for row in subject_means])) for name in metric_names}
