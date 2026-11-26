# Egocentric data collection with OptiTrack and Pupil Labs

We are building an **egocentric data collection pipeline** that pairs
**upper-body tracking from OptiTrack** with **front-facing scene-camera video
from Pupil Labs Neon**. The goal is to collect and process recordings that
connect what a participant sees with how their upper body moves over time.

| Source | Data | Role in the pipeline |
| --- | --- | --- |
| OptiTrack / Motive | Upper-body marker and skeleton tracking, exported as CSV or C3D | Describe the participant's upper-body motion |
| Pupil Labs Neon | Front-facing scene/world-camera video and frame timestamps | Record the participant's egocentric view |
| Clock measurements and synchronization events | Time Echo logs, measured clock offsets, and event anchors where available | Establish the temporal relationship between motion samples and video frames |

The intended result is a paired session with motion and video on a documented
time basis, while retaining the original recordings, timestamps, frame IDs,
units, and coordinate conventions. SMPL-X fitting, MuJoCo visualization, and
avatar retargeting support motion reconstruction and quality inspection within
this larger collection pipeline.

## Collection and processing workflow

1. Record upper-body motion in Motive and scene-camera video with Pupil Labs
   Neon during the same session. Run the Neon clock-offset monitor alongside
   recording.
2. Import the Motive CSV/C3D exports, Neon video, video-frame timestamps, and
   available clock measurements under the root `data/` directory.
3. Inspect tracking coverage, marker labels, video timing, units, and coordinate
   conventions before further processing.
4. Align the motion and video timelines using clock measurements and, where
   needed, common recorded events or motion-based alignment.
5. Inspect a synchronized motion/video preview and retain the alignment method,
   measured offsets, and remaining timing uncertainty with the session.
6. Optionally fit the markers to SMPL-X and produce derived motion or avatar
   visualizations for further inspection.

## Current implementation

The repository provides reusable capture loaders, an OptiTrack visualization
entry point, a Neon clock-offset monitor, offline synchronization and paired
video-preview utilities, and optional MoSh++/SMPL-X fitting tools. These are
building blocks for the collection pipeline; a unified collection and session
export workflow is still being developed.

Neon recording is performed through the Companion App / Neon Monitor and
exported afterward. The code in `src/neon/` monitors clock offsets; it does not
receive or record the scene-camera video itself.

The existing fitting profiles target the `Zekun_upper` captures in `data/1112`.
Those source C3D captures use 120 Hz and Z-up coordinates. The fitting workflow
rotates them to SMPL-X's Y-up coordinates and samples every fourth frame for
30 Hz output. These are profile-specific settings, not requirements for every
future collection session. Capture data and licensed model assets are local
and are not committed to Git.

## Repository layout

```text
.
├── assets/                       # avatars, textures, and derived skins
├── data/                         # imported sessions and model assets (ignored)
├── AGENTS.md                     # project purpose and agent working guidance
├── environment.yml               # Python 3.12 data/viewer environment
├── ENVIRONMENT.md                # data tooling notes
├── src/
│   ├── utils/                    # reusable pipeline helpers
│   │   ├── optitrack/            # capture IO, skeletons, scenes, and playback
│   │   ├── markers/              # conversion, quality analysis, and layouts
│   │   ├── smplx/                # fitting, kinematics, and rendering
│   │   ├── synchronization/      # clocks, alignment, and video helpers
│   │   └── avatars/              # asset decoding, retargeting, and scenes
│   ├── neon/                     # Pupil Labs Neon clock-offset monitor
│   ├── moshpp/                   # bundled, locally adapted MoSh++ source
│   └── mesh/                     # bundled psbody.mesh source
├── tests/
└── moshpp_outputs/               # generated fitting results (ignored)
```

## Environments

Two environments are useful because legacy MoSh++/Chumpy requires Python 3.10
and NumPy older than 1.24, while the data utilities use modern Python and
NumPy.

### Data inspection and MuJoCo

```bash
conda env create -f environment.yml
conda activate mocap_data
python -c "import numpy, pandas, scipy, h5py, matplotlib, mujoco; print('environment OK')"
```

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for Motive CSV and
TAK notes.

### MoSh++ fitting and rendering

Create a separate Python 3.10 environment and follow the complete installation
instructions in [`src/moshpp/README.md`](src/moshpp/README.md). The key constraints are:

- Python 3.10 or older;
- `numpy<1.24` and `cython<3`;
- Chumpy installed without build isolation;
- `psbody.mesh`, `psbody.smpl`, and the MoSh++ scan-to-mesh extension.

The fitting and headless rendering commands below assume that this environment
is active and that the local `moshpp` package is importable.

## Required local assets

The fitting profiles expect these files relative to the repository root:

```text
data/1112/c3d/*.c3d
data/1112/c3d/marker_layout/*/marker_layout.json
data/smplx/SMPLX_NEUTRAL_moshpp.pkl
data/smplx/SMPLX_NEUTRAL.npz
data/smplx/moshpp_support/pose_body_prior.pkl
data/smplx/moshpp_support/pose_hand_prior.npz
```

SMPL-X model files require acceptance of the SMPL-X license and must be
obtained separately. Capture data, licensed model files, and generated results
should not be committed.

Scripts resolve asset, capture, and output paths relative to the repository
root. Install bundled libraries from `src/mesh` and `src/moshpp`; existing
editable installations need to be reinstalled after this move.

## Synchronize motion with egocentric video

Synchronization is provided through reusable functions in
[`src/utils/synchronization/`](src/utils/synchronization/):

- `clocks` reads Time Echo anchors and converts between Neon and Ubuntu clocks.
- `motive` and `motive_motion` load exported tracking data and its timing.
- `neon_video` loads the exported scene video and frame timestamps.
- `alignment` fits temporal mappings from shared events or motion signals.
- `video_rendering` produces paired motion/video previews from prepared inputs.

Use these helpers from Python with `src/` on the import path (`PYTHONPATH=src`).
The unified collection and synchronization entry point remains to be built.
Use exported scene-camera frame timestamps when available and preserve the
original timestamps alongside derived alignment results.

Time Echo measures the offset between the Ubuntu computer and Neon Companion.
It does not measure the clock offset of the Windows computer running Motive.
Global-timestamp alignment therefore also needs a known relationship between
the Motive and Ubuntu clocks, or an explicitly documented assumption. The
current tools do not guarantee sub-millisecond synchronization.

## Optional SMPL-X processing

Reusable helpers under [`src/utils/smplx/`](src/utils/smplx/) support derived
human-body motion and quality inspection:

- `fitting.build_overrides(mode, sequence)` builds MoSh++ configuration for the
  existing `upper`, `righthand`, and `righthand-added` capture profiles.
- `reconstruction.reconstruct_vertices(...)` reconstructs meshes from fitted
  SMPL-X parameters.
- `kinematics` computes poses, joint transforms, and skinned vertices.
- `scene`, `mujoco_model`, and `playback` construct and display a MuJoCo scene.
- `rendering` draws mesh/marker previews, and `head_markers` provides diagnostics.

Fitting and reconstruction require the local licensed model assets and the
appropriate environment described above. Import these helpers directly when
building a workflow; the former fitting, rendering, and viewer CLI wrappers
have been removed.

## Supporting tools

Visualize an OptiTrack C3D or Motive CSV capture with the entry point in
`tests/`:

```bash
conda activate mocap_data
python tests/visualize_optitrack.py data/1112/c3d/Zekun_upper_1.c3d
python tests/visualize_optitrack.py data/1105/csv/take.csv --check-only
```

The file extension selects the existing viewer in `src/utils/`. Use
`--speed 0.5` for half-speed playback, `--no-loop` to stop at the final frame,
or `--check-only` to validate the capture without opening a window.

Other processing is available through reusable helper packages:

- `utils.markers`: CSV-to-NPZ conversion, quality analysis, frame selection,
  marker-layout editing, and previews.
- `utils.smplx`: fitting configuration, kinematics, reconstruction, and rendering.
- `utils.synchronization`: clock conversion, motion/video alignment, and previews.
- `utils.avatars`: avatar decoding, conversion, and retargeting.

See [`src/utils/README.md`](src/utils/README.md) for the module map and Python
examples. Runnable workflow entry points belong in `tests/`; the current
OptiTrack entry point imports the helper API directly. Run regression checks
with `python -m unittest discover -s tests/unit -v` in the data/viewer environment.

## Monitor Pupil Labs Neon clock offsets

The monitor under [`src/neon/`](src/neon/) records the clock offset between
Ubuntu and the Neon Companion App for offline timestamp synchronization.
See [`src/neon/README.md`](src/neon/README.md) for setup and output details.
Run it from the repository root:

```bash
python -m pip install -r src/neon/requirements.txt
PYTHONPATH=src python -m neon
```

## Licensing and citation

The bundled MoSh++ and mesh components retain their upstream licenses. MoSh++
is licensed for non-commercial scientific research; read
[`src/moshpp/LICENSE`](src/moshpp/LICENSE) before use. If this workflow contributes to
published research, cite AMASS/MoSh++ and the original MoSh work as described
in [`src/moshpp/README.md`](src/moshpp/README.md#citation).
