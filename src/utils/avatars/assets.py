"""Reusable avatars assets helpers."""

from __future__ import annotations

import numpy as np

from utils.avatars.models import (
    AvatarAsset,
    AvatarConfig,
)
from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

PROJECT_ROOT = _REPOSITORY_ROOT


AVATAR_CONFIGS = {
    "baldman": AvatarConfig(
        name="Baldman",
        source=PROJECT_ROOT / "assets" / "Baldman.vrm",
        skin=PROJECT_ROOT / "assets" / "baldman" / "upper_body_skin.npz",
        texture=PROJECT_ROOT / "assets" / "baldman" / "avatar_texture.png",
    ),
    "cookieman": AvatarConfig(
        name="Cookieman",
        source=PROJECT_ROOT / "assets" / "Cookieman.vrm",
        skin=PROJECT_ROOT / "assets" / "cookieman" / "upper_body_skin.npz",
        texture=PROJECT_ROOT / "assets" / "cookieman" / "avatar_texture.png",
    ),
}


DEFAULT_AVATAR = "baldman"


def load_avatar_asset(config: AvatarConfig) -> AvatarAsset:
    path = config.skin
    if not path.is_file():
        raise FileNotFoundError(
            f"Avatar asset not found: {path}. Generate it with "
            f"utils.avatars.vrm.convert(Path({str(config.source)!r}), "
            f"Path({str(config.skin)!r}), upper_body=True)."
        )
    with np.load(path, allow_pickle=False) as archive:
        return AvatarAsset(
            vertices=archive["vertices"],
            texcoords=archive["texcoords"],
            faces=archive["faces"],
            weights=archive["weights"],
            joint_names=archive["joint_names"].tolist(),
            bind_positions=archive["bind_positions"],
            bind_rotations=archive["bind_rotations"],
        )
