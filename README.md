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
video-preview utilities, a shared-transport MuJoCo/video player, and optional
MoSh++/SMPL-X fitting tools. These are
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
├── assets/
│   ├── avatar/                   # avatar VRM files, textures, and derived skins
│   ├── christmas/                # scene textures
│   └── neon/                     # Neon CAD, STL meshes, and calibrated camera
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
│   │   ├── recording/            # aligned-session scene and paired player
│   │   ├── neon/                 # CAD, camera calibration and head mounting
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

### Main collection and visualization environment

```bash
conda env create -f environment.yml
conda activate mocap_data
python -c "import numpy, pandas, scipy, h5py, matplotlib, mujoco, cv2, PIL, pupil_labs.realtime_api; print('environment OK')"
```

The environment file includes the Neon real-time API, OpenCV video processing,
and Pillow image export as well as the scientific and MuJoCo dependencies.

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

## Align and inspect the first 1126 recording

Use the existing `mocap_data` environment. Preprocessing writes a new derived
directory and preserves the original recordings:

```bash
conda activate mocap_data
python tests/align_recording.py \
  --motive "data/1126/mocap/csv/Take 2026-11-26 11.50.55 AM.csv" \
  --neon data/1126/neon/1 \
  --output data/1126/aligned/recording_1
python tests/visualize_recording.py data/1126/aligned/recording_1
```

The first export is already available locally: **450 paired frames over
14.971 seconds**. Skip the alignment command when that output directory exists;
choose a new directory to make another export.

Preprocessing interprets Motive's capture start in `Europe/Berlin` and adds
each CSV row's recorded `Time (Seconds)`. It intersects this timeline with the
native Neon scene-camera `.time` timestamps, cuts the video to those exact
frames, and selects the nearest OptiTrack row for each retained camera frame.
It applies **no IMU-based correction, clock offset, interpolation, or gap
filling**. Motive's exported time precision contributes to sample timing;
the first take's largest selected-sample difference is **4.491 ms**. This
does not measure clock synchronization accuracy: agreement between Motive's
computer clock and Neon's clock is assumed, not verified.

The derived directory contains:

- `front_camera_aligned.mp4`: exactly the retained camera frames, reencoded
  with their relative timestamp spacing.
- `mocap_aligned.csv`: one source motion row per video frame, retaining
  source Frame/Time cells, units, quaternions, and missing values.
- `frame_alignment.csv`: canonical aligned index and Unix nanosecond timestamp,
  both source indices/IDs, source timestamps, and signed Motive-minus-Neon
  sample differences. Use this mapping for aligned playback; the sampled Motive
  CSV retains its original capture/export rate metadata and can have gaps in IDs.
- `alignment.json`: sources and checksums, crop bounds, method, clock assumption,
  counts, units, original metadata, and sample timing statistics.

The player has a MuJoCo viewport on the left and the recorded front video on
the right. One frame index drives both. Play/pause, previous/next, restart,
speed, looping and the timeline control both panels. Drag the left view to
orbit, Shift-drag to pan, and scroll to zoom. Space toggles playback, arrows
step, and Home restarts. Debug controls show the solved skeleton, raw markers,
and camera optical axes: **X right/red, Y down/green, Z forward/blue**.

Without a licensed model, the player explicitly shows the solved skeleton
with a translucent head reference. This is **not a SMPL-X avatar**. The neutral
model loading and visual-retargeting path is prepared; its appearance and nose
fit remain unverified until the local licensed model is available:

```bash
python tests/visualize_recording.py data/1126/aligned/recording_1 \
  --smplx-model data/models/SMPLX_NEUTRAL.npz
```

This path uses the neutral template, zero shape coefficients and linear blend
skinning with solved OptiTrack upper-body/finger motion. It is not a MoSh++ fit;
legs and facial joints are untracked and retain neutral pose relative to their
parents. A pose reference 0.5 seconds into the retained recording sets the
visual rig alignment; it does not trim or shift the recorded timeline.

The glasses follow the head rigidly. Their default mount aligns an approximate
CAD bridge point to an estimated nose bridge. With SMPL-X, this estimate uses
the [official nose-tip vertex landmark](https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py)
plus an up/back bridge offset. **Focus head** and **Glasses fit…** let you inspect
and adjust translation/rotation; save to `neon_mount.json` and reload with
`--mount data/1126/aligned/recording_1/neon_mount.json`. This is a visual mount,
not calibrated camera-to-head or OptiTrack extrinsics. The attached XML camera
retains module `370641` factory intrinsics and its visible optical frame;
MuJoCo renders pinhole projection, while the recorded video retains lens
distortion.

To use recorded IMU orientation for the glasses:

```bash
python tests/visualize_recording.py data/1126/aligned/recording_1 --imu-orientation
```

The IMU is in the **top bar of the Neon module**. Its XML frame uses local X
right, Y forward, Z up and the manufacturer's nominal camera-to-IMU transform.
The player estimates the magnetic-world-to-OptiTrack heading difference from
paired head motion; it does not treat these worlds as identical. It samples
native quaternions at the existing video timestamps without shifting the
timeline. Recording 1 has a 0.99° median fit residual and 1.82° held-out
90th-percentile residual. Separate `imu_attachment.json`/`.csv` preserve the
method, source hashes, interpolation and residuals. The UI can toggle IMU
orientation and its sensor axes. Nose translation remains a geometric fit;
the nominal IMU origin inherits the CAD camera reference's uncertainty. See
the [Neon asset notes](assets/neon/README.md#imu-installation-and-attachment)
for transform directions, placement and validation limits.

Validate or export a preview without opening the desktop player:

```bash
python tests/visualize_recording.py data/1126/aligned/recording_1 --check-only
MUJOCO_GL=osmesa python tests/visualize_recording.py \
  data/1126/aligned/recording_1 --frame 100 --focus-head \
  --render data/1126/aligned/recording_1/head_preview.png \
  --export-scene data/1126/aligned/recording_1/scene.xml
```

## Supporting tools

The official Neon reference glasses and a MuJoCo scene are available under
[`assets/neon/`](assets/neon/README.md), including the original STEP CAD,
converted STL meshes, a visible optical frame, and factory camera intrinsics
for module `370641`:

```bash
python tests/visualize_neon.py
python tests/visualize_neon.py --camera scene
python tests/visualize_neon.py --check-only
```

The camera pose uses the CAD lens-opening reference; accurate camera-to-MoCap
alignment still requires measured extrinsics. See the asset README for units,
coordinate conventions, calibration details and previews.

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
