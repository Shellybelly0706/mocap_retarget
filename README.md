# Motion-capture retargeting with MoSh++ and SMPL-X

This repository contains an end-to-end workflow for fitting labeled OptiTrack
motion capture to a neutral SMPL-X body, inspecting the fit, and playing the
result in MuJoCo. It also includes utilities for Motive CSV/C3D inspection,
marker-layout authoring, and experimental avatar visualization.

The current fitting pipeline is configured for the `Zekun_upper` captures in
`mocap_data/data/1112`. Those captures are local data and are not committed to
Git.

## Workflow

1. Inspect a C3D capture and create or validate its marker-to-SMPL-X layout.
2. Run MoSh++ Stage I (body shape) and Stage II (per-frame pose).
3. Export an AMASS-style SMPL-X NPZ at 30 Hz.
4. Render a marker/mesh preview or play the skinned result in MuJoCo.

The source C3D files are recorded at 120 Hz in a Z-up coordinate system. The
fitting script rotates them to SMPL-X's Y-up coordinates and downsamples them
by four to 30 Hz.

## Repository layout

```text
.
├── scripts/
│   ├── fit_c3d_to_smplx.py       # C3D -> MoSh++ -> AMASS-style NPZ
│   ├── render_moshpp_result.py    # headless mesh/marker MP4 renderer
│   └── visualize_smplx_mujoco.py # skinned SMPL-X MuJoCo viewer
├── neon_realtime/                 # Pupil Labs Neon Wi-Fi live receiver
├── mocap_data/
│   ├── environment.yml           # Python 3.12 data/viewer environment
│   ├── ENVIRONMENT.md            # data tooling notes
│   ├── scripts/                  # conversion, inspection, and layout tools
│   └── data/                     # local captures and SMPL-X assets (ignored)
├── moshpp/                       # bundled, locally adapted MoSh++ source
├── mesh/                         # bundled psbody.mesh source
└── moshpp_outputs/               # generated fitting results
```

## Environments

Two environments are useful because legacy MoSh++/Chumpy requires Python 3.10
and NumPy older than 1.24, while the data utilities use modern Python and
NumPy.

### Data inspection and MuJoCo

```bash
conda env create -f mocap_data/environment.yml
conda activate mocap_data
python -c "import numpy, pandas, scipy, h5py, matplotlib, mujoco; print('environment OK')"
```

See [`mocap_data/ENVIRONMENT.md`](mocap_data/ENVIRONMENT.md) for Motive CSV and
TAK notes.

### MoSh++ fitting and rendering

Create a separate Python 3.10 environment and follow the complete installation
instructions in [`moshpp/README.md`](moshpp/README.md). The key constraints are:

- Python 3.10 or older;
- `numpy<1.24` and `cython<3`;
- Chumpy installed without build isolation;
- `psbody.mesh`, `psbody.smpl`, and the MoSh++ scan-to-mesh extension.

The fitting and headless rendering commands below assume that this environment
is active and that the local `moshpp` package is importable.

## Required local assets

The fitting profiles expect these files relative to the repository root:

```text
mocap_data/data/1112/c3d/*.c3d
mocap_data/data/1112/c3d/marker_layout/*/marker_layout.json
mocap_data/data/smplx/SMPLX_NEUTRAL_moshpp.pkl
mocap_data/data/smplx/SMPLX_NEUTRAL.npz
mocap_data/data/smplx/moshpp_support/pose_body_prior.pkl
mocap_data/data/smplx/moshpp_support/pose_hand_prior.npz
```

SMPL-X model files require acceptance of the SMPL-X license and must be
obtained separately. Capture data, licensed model files, and generated results
should not be committed.

> **Path note:** `fit_c3d_to_smplx.py` and `render_moshpp_result.py` currently
> define `ROOT` as `/media/fiona/LinuxData/mocap_retarget`. Update that constant
> if the repository is moved.

## Fit C3D to SMPL-X

Run all commands from the repository root. First validate the selected inputs
and resolved MoSh++ configuration without starting optimization:

```bash
conda activate moshpp
python scripts/fit_c3d_to_smplx.py \
  --sequence upper \
  --mode smoke \
  --check-only
```

Then run a short smoke fit or the complete sequence:

```bash
# First 120 source frames, sampled to 30 fitted frames
python scripts/fit_c3d_to_smplx.py --sequence upper --mode smoke

# Entire capture at 30 Hz
python scripts/fit_c3d_to_smplx.py --sequence upper --mode full
```

Available profiles are:

| Profile | Capture | Markers | Finger optimization |
| --- | --- | ---: | --- |
| `upper` | `Zekun_upper_1.c3d` | 25 | No |
| `righthand` | `Zekun_upper_righthand_1.c3d` | 32 unique labels | Yes |
| `righthand-added` | `Zekun_upper_righthand_added_1.c3d` | 32 | Yes |

For example, fit the capture with the added right-hand markers:

```bash
python scripts/fit_c3d_to_smplx.py \
  --sequence righthand-added \
  --mode full
```

Each run writes Stage-I and Stage-II artifacts under a directory such as:

```text
moshpp_outputs/Zekun_upper_righthand_added_1_fingers_full/1112/c3d/
└── Skeleton_Zixuan/
    ├── neutral_stagei.npz
    ├── Zekun_upper_righthand_added_1_stageii.pkl
    └── Zekun_upper_righthand_added_1_stageii.npz
```

The Stage-II NPZ contains the SMPL-X translation, root/body/hand pose, shape,
observed markers, fitted markers, and corrected 30 Hz timing metadata.

## Inspect the result

### Render an MP4 without OpenGL

The renderer reconstructs the SMPL-X mesh and draws front and side views with
observed markers in red and fitted markers in green:

```bash
conda activate moshpp
python scripts/render_moshpp_result.py \
  --result moshpp_outputs/Zekun_upper_1_full/1112/c3d/Skeleton_Zixuan/Zekun_upper_1_stageii.npz \
  --output /tmp/Zekun_upper_1_preview.mp4
```

Use `--stride 4` for faster rendering or `--max-frames 120` for a short
preview. If `--output` is omitted, the video is written next to the input NPZ.

### Play the skinned body in MuJoCo

Validate the model and result without opening a GUI:

```bash
conda activate mocap_data
python scripts/visualize_smplx_mujoco.py \
  --result moshpp_outputs/Zekun_upper_righthand_added_1_fingers_full/1112/c3d/Skeleton_Zixuan/Zekun_upper_righthand_added_1_stageii.npz \
  --validate-only
```

Remove `--validate-only` to launch the interactive viewer. The visualization
converts SMPL-X Y-up coordinates to MuJoCo Z-up, renders the 55-joint skinned
body, and places the fitted motion in a table-and-cup scene. Add
`--animate-cup-pickup` to enable the experimental kinematic pickup animation.

## Supporting tools

Common utilities under `mocap_data/scripts` include:

- `visualize_c3d_markers.py`: inspect C3D markers and skeleton data;
- `convert_motive_csv_to_mosh_npz.py`: convert a Motive CSV to marker NPZ;
- `inspect_mosh_marker_npz.py`: report marker quality and create previews;
- `edit_smplx_marker_layout.py`: assign capture labels to SMPL-X vertices;
- `compare_1105_exports.py`: inspect the six November 5 Motive exports.

Every script supports `--help`; inspection tools also provide a non-GUI
`--check-only` or `--validate-only` mode where applicable.

## Receive Pupil Labs Neon data over Wi-Fi

The standalone receiver under [`neon_realtime/`](neon_realtime/) records live
gaze, IMU, eye-event, and scene-camera streams from the Neon Companion App.
See [`neon_realtime/README.md`](neon_realtime/README.md) for installation,
network setup, output formats, and examples. A minimal session is:

```bash
python -m pip install -r neon_realtime/requirements.txt
python -m neon_realtime --streams gaze imu
```

## Licensing and citation

The bundled MoSh++ and mesh components retain their upstream licenses. MoSh++
is licensed for non-commercial scientific research; read
[`moshpp/LICENSE`](moshpp/LICENSE) before use. If this workflow contributes to
published research, cite AMASS/MoSh++ and the original MoSh work as described
in [`moshpp/README.md`](moshpp/README.md#citation).
