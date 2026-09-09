"""Synthetic observation events and causal P0/P1 BIS processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .config import PreprocessingID


class BISReason(str, Enum):
    AVAILABLE = "available"
    NO_PRIOR = "no_prior_observation"
    EXPLICIT_MISSING = "explicit_missing_event"
    NONFINITE = "nonfinite_bis"
    OUT_OF_RANGE = "bis_out_of_range"
    SQI_MISSING = "sqi_missing_exact_timestamp"
    SQI_LOW = "sqi_below_threshold"
    STALE = "stale_beyond_pipeline_cap"


@dataclass(frozen=True, slots=True)
class BISEvent:
    timestamp_seconds: float
    available: bool = True


@dataclass(frozen=True, slots=True)
class SQIEvent:
    timestamp_seconds: float
    value: float


@dataclass(frozen=True, slots=True)
class SyntheticObservationTemplate:
    template_id: str
    episode_horizon_seconds: float
    bis_events: tuple[BISEvent, ...] = ()
    sqi_events: tuple[SQIEvent, ...] = ()
    source_type: str = "synthetic"

    def __post_init__(self) -> None:
        if self.source_type != "synthetic" or not self.template_id:
            raise ValueError("Stage II accepts named synthetic templates only")
        horizon = float(self.episode_horizon_seconds)
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("template horizon must be finite and positive")
        bis_events = tuple(sorted(self.bis_events, key=lambda e: e.timestamp_seconds))
        sqi_events = tuple(sorted(self.sqi_events, key=lambda e: e.timestamp_seconds))
        for event in (*bis_events, *sqi_events):
            if not math.isfinite(event.timestamp_seconds) or not 0 <= event.timestamp_seconds <= horizon:
                raise ValueError("event timestamp outside template horizon")
        if len({e.timestamp_seconds for e in bis_events}) != len(bis_events):
            raise ValueError("BIS event timestamps must be unique")
        if len({e.timestamp_seconds for e in sqi_events}) != len(sqi_events):
            raise ValueError("SQI event timestamps must be unique")
        object.__setattr__(self, "episode_horizon_seconds", horizon)
        object.__setattr__(self, "bis_events", bis_events)
        object.__setattr__(self, "sqi_events", sqi_events)

    def bis_between(self, start: float, end: float) -> tuple[BISEvent, ...]:
        return tuple(e for e in self.bis_events if start < e.timestamp_seconds <= end)

    def sqi_between(self, start: float, end: float) -> tuple[SQIEvent, ...]:
        return tuple(e for e in self.sqi_events if start < e.timestamp_seconds <= end)

    def sqi_exact(self, timestamp: float) -> float | None:
        for event in self.sqi_events:
            if event.timestamp_seconds == timestamp:
                return event.value
        return None


@dataclass(frozen=True, slots=True)
class VisibleBIS:
    value: float
    mask: float
    age_seconds: float
    reason: BISReason


@dataclass(frozen=True, slots=True)
class BISAuditEvent:
    timestamp: float
    value: float | None
    reason: BISReason


@dataclass(frozen=True, slots=True)
class ObservationRule:
    """Journal-only factorization of SQI gating and accepted-event age."""

    rule_id: str
    sqi_threshold: float | None
    staleness_seconds: float

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("observation rule ID is required")
        if self.sqi_threshold is not None and (not math.isfinite(self.sqi_threshold) or self.sqi_threshold < 0.0):
            raise ValueError("SQI threshold must be finite and nonnegative")
        if self.staleness_seconds not in (10.0, 20.0, 30.0, 60.0):
            raise ValueError("journal observation age must be 10, 20, 30, or 60 seconds")


class BISObservationProcessor:
    def __init__(
        self,
        preprocessing_id: PreprocessingID,
        template: SyntheticObservationTemplate,
        observation_rule: ObservationRule | None = None,
    ):
        self.preprocessing_id = preprocessing_id
        self.template = template
        self.observation_rule = observation_rule
        self._events: list[BISAuditEvent] = []

    @property
    def audit_events(self) -> tuple[BISAuditEvent, ...]:
        """Return immutable event-level acceptance and rejection evidence."""

        return tuple(self._events)

    @property
    def staleness_cap(self) -> float:
        if self.observation_rule is not None:
            return self.observation_rule.staleness_seconds
        return 30.0 if self.preprocessing_id is PreprocessingID.P0 else 20.0

    def ingest(self, event: BISEvent, latent_bis: float) -> BISReason:
        if not event.available:
            reason, value = BISReason.EXPLICIT_MISSING, None
        elif not math.isfinite(latent_bis):
            reason, value = BISReason.NONFINITE, None
        elif not 0.0 <= latent_bis <= 100.0:
            reason, value = BISReason.OUT_OF_RANGE, None
        elif self.observation_rule is not None and self.observation_rule.sqi_threshold is not None:
            sqi = self.template.sqi_exact(event.timestamp_seconds)
            if sqi is None:
                reason, value = BISReason.SQI_MISSING, None
            elif not math.isfinite(sqi) or sqi < self.observation_rule.sqi_threshold:
                reason, value = BISReason.SQI_LOW, None
            else:
                reason, value = BISReason.AVAILABLE, float(latent_bis)
        elif self.preprocessing_id is PreprocessingID.P1:
            sqi = self.template.sqi_exact(event.timestamp_seconds)
            if sqi is None:
                reason, value = BISReason.SQI_MISSING, None
            elif not math.isfinite(sqi) or sqi < 50.0:
                reason, value = BISReason.SQI_LOW, None
            else:
                reason, value = BISReason.AVAILABLE, float(latent_bis)
        else:
            reason, value = BISReason.AVAILABLE, float(latent_bis)
        self._events.append(BISAuditEvent(event.timestamp_seconds, value, reason))
        return reason

    def query(self, timestamp: float) -> VisibleBIS:
        causal = [event for event in self._events if event.timestamp <= timestamp]
        if not causal:
            return VisibleBIS(0.0, 0.0, 30.0, BISReason.NO_PRIOR)
        latest_raw = causal[-1]
        accepted = next((event for event in reversed(causal) if event.value is not None), None)
        raw_age = min(max(timestamp - latest_raw.timestamp, 0.0), 30.0)
        if accepted is None:
            return VisibleBIS(0.0, 0.0, raw_age, latest_raw.reason)
        accepted_age = max(timestamp - accepted.timestamp, 0.0)
        if accepted_age <= self.staleness_cap:
            return VisibleBIS(float(accepted.value), 1.0, min(accepted_age, 30.0), BISReason.AVAILABLE)
        reason = latest_raw.reason if latest_raw.timestamp > accepted.timestamp else BISReason.STALE
        return VisibleBIS(0.0, 0.0, raw_age, reason)
