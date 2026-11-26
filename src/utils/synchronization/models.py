"""Reusable synchronization models helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.synchronization.constants import (
    NS_PER_SECOND,
)


@dataclass(frozen=True)
class MotiveExport:
    path: Path
    frame_rate_hz: float
    frame_ids: tuple[int, ...]
    first_frame: int
    capture_start_frame: int | None
    take_name: str | None


@dataclass(frozen=True)
class ClockAnchor:
    companion_ns: int
    ubuntu_ns: int
    stable: bool
    sequence: int | None


@dataclass(frozen=True)
class EventAnchor:
    motive_frame: int
    motive_elapsed_s: float
    companion_ns: int
    ubuntu_ns: int
    source: str


@dataclass(frozen=True)
class TimedSignal:
    time_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AutomaticAlignment:
    events: tuple[EventAnchor, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class AffineMap:
    """Ubuntu Unix nanoseconds = intercept_ns + slope * Motive elapsed ns."""

    slope: float
    intercept_ns: int
    residuals_ms: tuple[float, ...]

    def ubuntu_ns(self, motive_elapsed_s: float) -> int:
        return self.intercept_ns + round(self.slope * motive_elapsed_s * NS_PER_SECOND)
