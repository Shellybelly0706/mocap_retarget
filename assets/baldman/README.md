# Baldman upper-body asset

`upper_body_skin.npz` and `avatar_texture.png` are generated from
`../Baldman.vrm` for `src/utils/1029_comic_avatar_test/visualize_upper_body.py`.

The source avatar is **Baldman** from the Open Source Avatars 100Avatars R1
collection. That collection is published under CC0.

- Registry: https://www.opensourceavatars.com/en
- Catalog metadata:
  https://raw.githubusercontent.com/ToxSam/open-source-avatars/main/data/avatars/100avatars-r1.json
- Collection license metadata:
  https://raw.githubusercontent.com/ToxSam/open-source-avatars/main/data/projects.json

Regenerate the derived files from the project root:

```bash
conda run --no-capture-output -n mocap_data \
  python src/utils/1029_comic_avatar_test/convert_vrm_avatar.py \
  assets/Baldman.vrm assets/baldman/upper_body_skin.npz \
  --upper-body
```

The extracted PNG is Baldman's original texture. Leg-weighted triangles are
removed, while non-humanoid end-joint weights are folded into their nearest VRM
humanoid parent.
