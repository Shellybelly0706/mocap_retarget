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
│   ├── background/               # scene backgrounds, grouped by theme
│   │   └── christmas/            # Christmas scene textures
│   └── neon/                     # Neon CAD, STL meshes, and calibrated camera
├── data/                         # imported sessions and model assets (ignored)
├── AGENTS.md                     # project purpose and agent working guidance
├── environment.yml               # unified Python 3.10 project environment
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

## Environment

Use **one environment, `mocap_data`**, for collection, processing, visualization,
CAD conversion and legacy MoSh++ reconstruction. `environment.yml` is the
single dependency definition. Python 3.10 and NumPy 1.23.5 support Chumpy while
the viewer/video dependencies are pinned to compatible versions.

```bash
conda env create -f environment.yml
conda activate mocap_data
python tests/setup_environment.py
python -c "import numpy, pandas, scipy, h5py, matplotlib, mujoco, cv2, PIL, pupil_labs.realtime_api; print('environment OK')"
```

The environment file includes the Neon API, OpenCV, PyAV, MuJoCo, CPU PyTorch,
Chumpy, Human Body Prior and CAD libraries. The setup command builds the
bundled mesh and scan-to-mesh extensions and links the bundled MoSh++ source.

See [`ENVIRONMENT.md`](ENVIRONMENT.md) for Motive CSV and
TAK notes.

For an existing checkout, update with
`conda env update -n mocap_data -f environment.yml --prune`, activate
`mocap_data`, and rerun `python tests/setup_environment.py`.

MoSh++ uses this same environment. Its separately supplied `psbody.smpl`
component remains an external prerequisite; consolidating dependencies does
not supply that missing solver binary.

## Required local assets

The fitting profiles expect these files relative to the repository root:

```text
data/1112/c3d/*.c3d
assets/avatar/smplx/marker_layout/*/marker_layout.json
assets/avatar/smplx/SMPLX_NEUTRAL_moshpp.pkl
assets/avatar/smplx/SMPLX_NEUTRAL.npz
assets/avatar/smplx/moshpp_support/pose_body_prior.pkl
assets/avatar/smplx/moshpp_support/pose_hand_prior.npz
```

The user supplied the licensed SMPL-X models and fitting support files through
Git LFS on `main` (commit `390216a`), imported into `assets/avatar/smplx/`.
Run `git lfs pull` after cloning to retrieve actual models rather than pointer
files. Capture data and generated results remain local under ignored `data/`.

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
shared `mocap_data` environment described above. Import these helpers directly when
building a workflow; the former fitting, rendering, and viewer CLI wrappers
have been removed.

The legacy MoSh++ fitting workflow is now available through
`tests/fit_smplx.py`, with reusable orchestration in `utils.smplx.pipeline`:

```bash
conda activate mocap_data
python tests/fit_smplx.py --check-only
```

The user selected this marker-based reconstruction workflow for 1126 first.
The fitting environment and imported model/priors are prepared. Main commit
`77c22c3` supplied the layouts under `assets/avatar/smplx/marker_layout/`.
Their model checksum and neutral topology have been validated. The default
1126 job reuses the complete 25-marker body layout; all 25 labels are present,
and 1,370 frames meet the 95% availability threshold. The 14 custom markers
are excluded from this first pass. Reusing the legacy layout is an
initialization; new-subject Stage I calibration is still needed. Optimization
is pending only `smpl-fast-derivatives` / `psbody.smpl`. Preflight reports that
blocker and does not produce fitted parameters. Once ready, use `--mode smoke` or
`--mode full`; outputs include body shape, per-frame pose/translation, source
sample rows and C3D-relative times. See
[`ENVIRONMENT.md`](ENVIRONMENT.md#legacy-mosh-reconstruction)
for setup, layout reuse and the Stage I/II workflow.

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

The player defaults to the solved skeleton with a translucent head reference.
This is **not a SMPL-X avatar**. The imported neutral model has been validated
and rendered with Neon CAD and IMU axes. Select it explicitly:

```bash
python tests/visualize_recording.py data/1126/aligned/recording_1 \
  --smplx-model assets/avatar/smplx/SMPLX_NEUTRAL.npz
```

This path uses the neutral template, zero shape coefficients and linear blend
skinning with solved OptiTrack upper-body/finger motion. It is not a MoSh++ fit;
legs and facial joints are untracked and retain neutral pose relative to their
parents. A pose reference 0.5 seconds into the retained recording sets the
visual rig alignment; it does not trim or shift the recorded timeline.

The glasses follow the head rigidly. With SMPL-X, their default CAD brow-bar
anchor is placed 8 mm above the midpoint of the model's eye landmarks. A ray
through the actual nasal surface sets its depth, with a 14 mm forward standoff
for the CAD bar/nose-pad geometry. This replaces the lower nose-tip offset that
placed the frame across the cheeks. **Focus head** and **Glasses fit…** let you inspect
and adjust translation/rotation; save to `neon_mount.json` and reload with
`--mount data/1126/aligned/recording_1/neon_mount.json`. This is a visual mount,
not calibrated camera-to-head or OptiTrack extrinsics. The attached XML camera
retains module `370641` factory intrinsics and its visible optical frame;
MuJoCo renders pinhole projection, while the recorded video retains lens
distortion.

The CAD's separate rigid temple arms are opened about nominal hinges to clear
the neutral head, without scaling the frame or changing its front camera/IMU
assembly. This is an approximate visual fit based on the avatar surface, not a
measured participant fit. Subject-specific head shape and camera/head extrinsics
still require fitting and calibration.

For the first 1126 take, IMU and OptiTrack head orientation disagree by about
27 degrees at aligned frame 0. The player shows the per-frame residual in IMU
mode. Turn off **IMU orientation** to inspect the rigid head attachment; changing
the visual mount does not resolve that orientation discrepancy or change timing.

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
with `python -m unittest discover -s tests/unit -v` in `mocap_data`.

## Monitor Pupil Labs Neon clock offsets

The monitor under [`src/neon/`](src/neon/) records the clock offset between
Ubuntu and the Neon Companion App for offline timestamp synchronization.
See [`src/neon/README.md`](src/neon/README.md) for setup and output details.
Run it from the repository root:

```bash
conda activate mocap_data
PYTHONPATH=src python -m neon
```

## Licensing and citation

The bundled MoSh++ and mesh components retain their upstream licenses. MoSh++
is licensed for non-commercial scientific research; read
[`src/moshpp/LICENSE`](src/moshpp/LICENSE) before use. If this workflow contributes to
published research, cite AMASS/MoSh++ and the original MoSh work as described
in [`src/moshpp/README.md`](src/moshpp/README.md#citation).
