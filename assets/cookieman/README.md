# Cookieman upper-body asset

`upper_body_skin.npz` and `avatar_texture.png` are generated from
`../Cookieman.vrm` for the reusable avatar helpers in `src/utils/avatars/`.

Regenerate the derived files from the project root:

```bash
conda activate mocap_data
PYTHONPATH=src python - <<'PYTHON'
from pathlib import Path
from utils.avatars.vrm import convert

convert(
    Path("assets/Cookieman.vrm"),
    Path("assets/cookieman/upper_body_skin.npz"),
    upper_body=True,
)
PYTHON
```

The extracted PNG is Cookieman's original texture. Leg-weighted triangles are
removed, while non-humanoid end-joint weights are folded into their nearest VRM
humanoid parent. Cookieman has no humanoid finger bones, so its hands are
retargeted as rigid parts.
