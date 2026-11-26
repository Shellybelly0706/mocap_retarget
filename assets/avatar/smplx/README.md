# SMPL-X model assets

Imported from the user's `main` commit `390216a` using Git LFS. Model version:
SMPL-X 1.0. This directory groups the model assets with the other avatars.

- `SMPLX_NEUTRAL.npz`: the neutral model used by the current NumPy/MuJoCo viewer.
  Verified topology: 10,475 vertices, 20,908 triangles, 55 joints.
- `SMPLX_FEMALE.npz`, `SMPLX_MALE.npz`: alternative gender models.
- Corresponding `.pkl` files and `smplx_npz.zip`: original supplied variants.
- `SMPLX_NEUTRAL_moshpp.pkl` and `moshpp_support/`: assets for legacy MoSh++
  fitting in the shared `mocap_data` environment with a marker-to-surface layout.
- `marker_layout/`: the three original layouts imported from main commit
  `77c22c3`, with their backup files and provenance unchanged. Their checksum
  matches `SMPLX_NEUTRAL.pkl`; its mesh matches the converted fitting model
  and the viewer NPZ. The first 1126 pass uses `Zekun_upper_25` as transferred
  initialization, leaving custom-marker tracks out until separately mapped.

Download the LFS objects after cloning:

```bash
git lfs pull
```

Display the imported neutral model with the first aligned 1126 recording:

```bash
conda activate mocap_data
python tests/visualize_recording.py data/1126/aligned/recording_1 \
  --smplx-model assets/avatar/smplx/SMPLX_NEUTRAL.npz --imu-orientation
```

The current viewer directly retargets solved Motive bones. It does not yet
perform marker-based SMPL-X shape/pose fitting. Untracked legs and face retain
neutral pose, and the glasses-to-nose translation remains a visual estimate.
