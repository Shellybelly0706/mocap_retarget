#!/usr/bin/env python3
"""Build bundled native packages after creating the unified environment.yml."""

import importlib
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import numpy as np

    if sys.version_info[:2] != (3, 10) or np.__version__ != "1.23.5":
        raise SystemExit(
            "Activate mocap_data created from environment.yml (Python 3.10 / NumPy 1.23.5)."
        )
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(root / "src/mesh"),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [python, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=root / "src/moshpp/src/moshpp/scan2mesh/mesh_distance",
        check=True,
    )
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(root / "src/moshpp"),
        ],
        cwd=root,
        check=True,
    )
    print("Bundled packages built in", sys.prefix)
    try:
        importlib.import_module("psbody.smpl.fast_derivatives.smplcpp_chumpy")
        print("The external psbody.smpl component imports successfully.")
    except (ImportError, OSError) as exc:
        print(
            "MoSh++ optimization still needs the separately supplied psbody.smpl component:",
            exc,
        )


if __name__ == "__main__":
    main()
