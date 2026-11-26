"""Shared locations relative to the repository checkout."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data"
ASSETS_DIR = REPOSITORY_ROOT / "assets"
MOSHPP_DIR = REPOSITORY_ROOT / "src" / "moshpp"
