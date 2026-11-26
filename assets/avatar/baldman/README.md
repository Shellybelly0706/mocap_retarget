# Baldman upper-body asset

`upper_body_skin.npz` and `avatar_texture.png` are generated from
`../Baldman.vrm` for the reusable avatar helpers in `src/utils/avatars/`.

The source avatar is **Baldman** from the Open Source Avatars 100Avatars R1
collection. That collection is published under CC0.

- Registry: https://www.opensourceavatars.com/en
- Catalog metadata:
  https://raw.githubusercontent.com/ToxSam/open-source-avatars/main/data/avatars/100avatars-r1.json
- Collection license metadata:
  https://raw.githubusercontent.com/ToxSam/open-source-avatars/main/data/projects.json

Regenerate the derived files from the project root:

```bash
conda activate mocap_data
PYTHONPATH=src python - <<'PYTHON'
from pathlib import Path
from utils.avatars.vrm import convert

convert(
    Path("assets/avatar/Baldman.vrm"),
    Path("assets/avatar/baldman/upper_body_skin.npz"),
    upper_body=True,
)
PYTHON
```

The extracted PNG is Baldman's original texture. Leg-weighted triangles are
removed, while non-humanoid end-joint weights are folded into their nearest VRM
humanoid parent.
