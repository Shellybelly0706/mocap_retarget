"""Programmatic OptiTrack loading and visualization without CLI parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .c3d import C3DMotion
    from .csv import CsvExport


def capture_path(capture: str | Path) -> Path:
    """Resolve and validate a supported capture path."""
    path = Path(capture).expanduser().resolve()
    if path.suffix.lower() not in {".c3d", ".csv"}:
        raise ValueError("capture must be a .c3d or .csv file")
    if not path.is_file():
        raise FileNotFoundError(f"capture does not exist: {path}")
    return path


def load_capture(capture: str | Path) -> C3DMotion | CsvExport:
    """Load a capture for processing; no MuJoCo or GUI imports are needed."""
    path = capture_path(capture)
    if path.suffix.lower() == ".c3d":
        from .c3d import load_c3d

        return load_c3d(path)
    from .csv import load_csv_export

    return load_csv_export(path)


def visualize_capture(
    capture: str | Path,
    *,
    speed: float = 1.0,
    no_loop: bool = False,
    check_only: bool = False,
) -> None:
    """Play or validate a C3D/CSV capture using the existing viewer defaults."""
    path = capture_path(capture)
    if speed <= 0:
        raise ValueError("playback speed must be greater than zero")
    options = {"speed": speed, "no_loop": no_loop, "check_only": check_only}
    if path.suffix.lower() == ".c3d":
        from .c3d import DEFAULT_C3D_READER
        from .c3d_playback import visualize

        args = SimpleNamespace(
            **options,
            c3d=path,
            c3d_reader=DEFAULT_C3D_READER,
            marker_size=0.012,
            joint_size=0.016,
            bone_width=0.008,
            no_skeleton=False,
            short_labels=False,
        )
    else:
        from .csv_playback import visualize

        args = SimpleNamespace(
            **options, take=str(path), csv_dir=path.parent, list=False
        )
    visualize(args)
