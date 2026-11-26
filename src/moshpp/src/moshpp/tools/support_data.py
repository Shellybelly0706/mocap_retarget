"""Locate MoSh++ support files after moving the checkout under root src/."""

import sys
from pathlib import Path


def get_support_data_dir(current_fname=__file__):
    """Use the nearest support_data directory, rather than the first 'src'."""
    location = Path(current_fname).resolve()
    start = location if location.is_dir() else location.parent
    for ancestor in (start, *start.parents):
        candidate = ancestor / "support_data"
        if candidate.is_dir():
            return str(candidate)
    installed = Path(sys.prefix) / "moshpp" / "support_data"
    if installed.is_dir():
        return str(installed)
    raise FileNotFoundError(f"MoSh++ support_data directory not found near {location}")
