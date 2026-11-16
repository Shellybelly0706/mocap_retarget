# Cookieman upper-body asset

`upper_body_skin.npz` and `avatar_texture.png` are generated from
`../Cookieman.vrm` for `scripts/visualize_upper_body.py`.

Regenerate the derived files from the project root:

```bash
conda run --no-capture-output -n mocap_data \
  python scripts/convert_vrm_avatar.py \
  assets/Cookieman.vrm assets/cookieman/upper_body_skin.npz \
  --upper-body
```

The extracted PNG is Cookieman's original texture. Leg-weighted triangles are
removed, while non-humanoid end-joint weights are folded into their nearest VRM
humanoid parent. Cookieman has no humanoid finger bones, so its hands are
retargeted as rigid parts.
