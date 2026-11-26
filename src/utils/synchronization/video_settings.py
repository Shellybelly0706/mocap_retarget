"""Reusable synchronization video settings helpers."""

from __future__ import annotations

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT
from utils.synchronization.constants import NS_PER_SECOND

PROJECT_ROOT = _REPOSITORY_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sync"

__all__ = ["NS_PER_SECOND", "PROJECT_ROOT", "DEFAULT_OUTPUT_DIR"]
