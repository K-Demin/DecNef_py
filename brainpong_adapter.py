from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
import math
import statistics
import time


@dataclass
class BrainPongScoreState:
    volume_index: Optional[int]
    timestamp: float
    raw_score: Optional[float]
    score_z: Optional[float]
    signed_score: Optional[float]
    normalized_score: Optional[float]
    feedback_value: float
    source_volume_index: Optional[int]
    score_missing: bool
    score_delayed: bool
    reg_ready: Optional[bool]
    details: dict[str, Any] = field(default_factory=dict)


class BrainPongScoreAdapter:
    """Convert realtime decoder messages into signed Brain-pong feedback values."""

    def __init__(self, decoder_cfg: dict[str, Any], feedback_cfg: dict[str, Any]):
        self.decoder_cfg = decoder_cfg
        self.feedback_cfg = feedback_cfg
        self.initial_ignore_volumes = int(feedback_cfg.get("initial_ignore_volumes", 0) or 0)
        self.feedback_delay_volumes = int(feedback_cfg.get("feedback_delay_volumes", 0) or 0)
        self.smoothing_window = max(1, int(feedback_cfg.get("smoothing_window", 1) or 1))
        self.score_buffer_length = max(
            self.feedback_delay_volumes + self.smoothing_window + 2,
            int(feedback_cfg.get("score_buffer_length", 256) or 256),
        )
        self.zscore_mode = str(feedback_cfg.get("zscore_mode", "none")).lower()
        self.feedback_transform = str(feedback_cfg.get("feedback_transform", "identity")).lower()
        self.clip_min = float(feedback_cfg.get("clip_min", -1.0))
        self.clip_max = float(feedback_cfg.get("clip_max", 1.0))
        self.invert_score = bool(decoder_cfg.get("invert_score", False))

        self._score_buffer: deque[dict[str, Any]] = deque(maxlen=self.score_buffer_length)
        self._baseline_values: list[float] = []
        self._last_state = BrainPongScoreState(
            volume_index=None,
            timestamp=time.time(),
            raw_score=None,
            score_z=None,
            signed_score=None,
            normalized_score=None,
            feedback_value=0.0,
            source_volume_index=None,
            score_missing=True,
            score_delayed=False,
            reg_ready=None,
        )

    def direction_sign(self) -> float:
        category_a_direction = str(self.decoder_cfg.get("category_a_direction", "up")).lower()
        sign = 1.0 if category_a_direction == "up" else -1.0
        return -sign if self.invert_score else sign

    def update(
        self,
        raw_score: float,
        volume_index: int,
        timestamp: Optional[float] = None,
        score_z: Optional[float] = None,
        **meta: Any,
    ) -> BrainPongScoreState:
        if timestamp is None:
            timestamp = time.time()
        raw_signed = float(raw_score) * self.direction_sign()
        if self.zscore_mode in {"reference", "pipeline"}:
            if score_z is None or not math.isfinite(float(score_z)):
                raise ValueError(
                    "zscore_mode='reference' requires a finite score_z from the "
                    "reference run. Check --rs and the reference scores.csv."
                )
            signed = float(score_z) * self.direction_sign()
        else:
            signed = raw_signed
        item = {
            "volume_index": int(volume_index),
            "timestamp": float(timestamp),
            "raw_score": float(raw_score),
            "score_z": float(score_z) if score_z is not None else None,
            "signed_score": signed,
            **meta,
        }
        self._score_buffer.append(item)

        if int(volume_index) <= self.initial_ignore_volumes:
            self._baseline_values.append(raw_signed)
            return self._state_from_missing(item, ignored=True)

        item["normalized_score"] = self._transform(self._normalize(signed))

        state = self.current_state(now=timestamp)
        self._last_state = state
        return state

    def current_state(self, now: Optional[float] = None) -> BrainPongScoreState:
        if now is None:
            now = time.time()
        usable = [
            item
            for item in self._score_buffer
            if int(item["volume_index"]) > self.initial_ignore_volumes
        ]
        if not usable:
            return BrainPongScoreState(
                volume_index=None,
                timestamp=now,
                raw_score=None,
                score_z=None,
                signed_score=None,
                normalized_score=None,
                feedback_value=self._last_state.feedback_value,
                source_volume_index=None,
                score_missing=True,
                score_delayed=bool(self.feedback_delay_volumes),
                reg_ready=None,
            )

        source_index = max(0, len(usable) - 1 - self.feedback_delay_volumes)
        source = usable[source_index]
        window_start = max(0, source_index - self.smoothing_window + 1)
        window = usable[window_start : source_index + 1]
        signed_values = [float(item["signed_score"]) for item in window]
        normalized_values = [float(item["normalized_score"]) for item in window]
        signed = float(sum(signed_values) / len(signed_values))
        normalized = float(sum(normalized_values) / len(normalized_values))
        clipped = max(self.clip_min, min(self.clip_max, normalized))

        return BrainPongScoreState(
            volume_index=int(usable[-1]["volume_index"]),
            timestamp=now,
            raw_score=float(source["raw_score"]),
            score_z=float(source["score_z"]) if source.get("score_z") is not None else None,
            signed_score=signed,
            normalized_score=normalized,
            feedback_value=clipped,
            source_volume_index=int(source["volume_index"]),
            score_missing=False,
            score_delayed=source_index != len(usable) - 1,
            reg_ready=source.get("reg_ready"),
            details={
                "smoothing_n": len(window),
                "newest_volume_index": int(usable[-1]["volume_index"]),
                "clip_min": self.clip_min,
                "clip_max": self.clip_max,
                "zscore_mode": self.zscore_mode,
                "feedback_transform": self.feedback_transform,
            },
        )

    def _state_from_missing(self, item: dict[str, Any], ignored: bool = False) -> BrainPongScoreState:
        return BrainPongScoreState(
            volume_index=int(item["volume_index"]),
            timestamp=float(item["timestamp"]),
            raw_score=float(item["raw_score"]),
            score_z=float(item["score_z"]) if item.get("score_z") is not None else None,
            signed_score=float(item["signed_score"]),
            normalized_score=None,
            feedback_value=0.0,
            source_volume_index=None,
            score_missing=True,
            score_delayed=False,
            reg_ready=item.get("reg_ready"),
            details={"ignored_initial_volume": ignored},
        )

    def _normalize(self, signed: float) -> float:
        if self.zscore_mode in {"reference", "pipeline"}:
            return signed
        if self.zscore_mode in {"none", "", "null"}:
            return signed
        if self.zscore_mode == "baseline":
            ref = self._baseline_values
        elif self.zscore_mode == "running":
            ref = [float(item["signed_score"]) for item in self._score_buffer]
        else:
            raise ValueError(f"Unsupported zscore_mode={self.zscore_mode!r}")
        if len(ref) < 2:
            return 0.0
        mean = statistics.fmean(ref)
        std = statistics.stdev(ref)
        if not math.isfinite(std) or std < 1e-8:
            return 0.0
        return (signed - mean) / std

    def _transform(self, value: float) -> float:
        if self.feedback_transform in {"identity", "none", ""}:
            return value
        if self.feedback_transform == "signed_percentile":
            return math.erf(value / math.sqrt(2.0))
        raise ValueError(f"Unsupported feedback_transform={self.feedback_transform!r}")


class PaddleDynamics:
    """Update a normalized paddle position from signed feedback."""

    def __init__(self, feedback_cfg: dict[str, Any], task_cfg: dict[str, Any]):
        self.gain = float(feedback_cfg.get("gain", 1.0))
        self.dead_zone = abs(float(feedback_cfg.get("dead_zone", 0.0)))
        self.max_speed = abs(float(task_cfg.get("max_paddle_speed", 1.0)))
        limits = task_cfg.get("position_limits", [-0.9, 0.9])
        self.min_pos = float(limits[0])
        self.max_pos = float(limits[1])
        self.movement_mode = str(task_cfg.get("movement_mode", "velocity")).lower()
        self.stop_at_peak = bool(task_cfg.get("stop_at_peak", False))
        self.return_to_center = bool(task_cfg.get("return_to_center", False))
        self.return_rate = abs(float(task_cfg.get("return_rate", 0.4)))
        self.position = 0.0
        self.velocity = 0.0

    def reset(self) -> None:
        self.position = 0.0
        self.velocity = 0.0

    def update(self, feedback_value: float, dt: float, target_direction: Optional[str] = None) -> tuple[float, float]:
        value = float(feedback_value)
        if abs(value) < self.dead_zone:
            value = 0.0

        if self.stop_at_peak and target_direction:
            if target_direction == "up" and self.position >= self.max_pos:
                value = min(0.0, value)
            elif target_direction == "down" and self.position <= self.min_pos:
                value = max(0.0, value)

        if self.movement_mode == "position":
            desired = self.gain * value
            self.velocity = (desired - self.position) / max(dt, 1e-6)
            self.position = desired
        elif self.movement_mode == "velocity":
            self.velocity = max(-self.max_speed, min(self.max_speed, self.gain * value))
            self.position += self.velocity * dt
        else:
            raise ValueError(f"Unsupported movement_mode={self.movement_mode!r}")

        if self.return_to_center and value == 0.0:
            if self.position > 0:
                self.position = max(0.0, self.position - self.return_rate * dt)
            elif self.position < 0:
                self.position = min(0.0, self.position + self.return_rate * dt)

        clipped = max(self.min_pos, min(self.max_pos, self.position))
        if clipped != self.position:
            self.position = clipped
            self.velocity = 0.0
        return self.position, self.velocity
