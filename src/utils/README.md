# Reusable egocentric data pipeline helpers

These helpers support the project's goal of pairing OptiTrack upper-body
tracking with Pupil Labs Neon front-facing scene-camera video. Loading,
synchronization, and quality inspection connect the two streams; SMPL-X fitting
and avatar retargeting provide optional derived motion representations.

Import helpers with `src/` on the Python path (`PYTHONPATH=src` when running
from the repository root). Modules are grouped by responsibility:

| Package | Helpers |
| --- | --- |
| `optitrack` | C3D/CSV loading, skeleton reconstruction, MuJoCo scenes and playback |
| `markers` | CSV conversion, NPZ validation, quality analysis, frame selection, marker layouts and preview export |
| `smplx` | MoSh++ configuration, body reconstruction, kinematics, scene building and rendering |
| `synchronization` | Motive/Neon inputs, clock interpolation, signal alignment, projection and video rendering |
| `avatars` | glTF/VRM decoding, avatar conversion, pose retargeting, obstacles and scenes |
| `neon` | Reference CAD conversion, factory scene-camera calibration, MJCF generation and viewing |
| `recording` | Aligned-session IO, head-mounted camera scenes, paired preview and shared-transport desktop playback |
| `paths.py` | Repository, data, asset and bundled MoSh++ locations |

Loading, analysis, kinematics, scene construction and playback live in separate
files. Data-processing helpers can be imported without loading GUI or fitting
dependencies. Rendering and fitting use the shared `mocap_data` environment
and the appropriate local model assets.

## OptiTrack capture loading and visualization

```python
from utils.optitrack.visualization import load_capture, visualize_capture

motion = load_capture("data/capture.c3d")
print(motion.positions.shape, motion.valid.sum())
visualize_capture("data/capture.c3d", speed=0.5, check_only=True)
```

`load_capture` accepts C3D and CSV paths and returns `C3DMotion` or `CsvExport`.
Use `optitrack.c3d.load_c3d` and `optitrack.csv.load_csv_export` when the format
is already known. Capture loaders preserve missing samples; they do not fill
gaps. C3D positions are converted to metres, while the CSV viewer loader keeps
the export's coordinates. Use marker conversion below to normalize Motive
CSV marker coordinates to metres.

## Marker conversion and analysis

```python
from pathlib import Path
from utils.markers.conversion import convert_csv
from utils.markers.npz import write_npz
from utils.markers.data import load_and_validate_npz
from utils.markers.analysis import analyze
from utils.markers.selection import select_stage1_candidates

converted = convert_csv(Path("data/take.csv"), skeleton="Body")
write_npz(Path("data/take.npz"), converted)
motion = load_and_validate_npz(Path("data/take.npz"))
quality = analyze(motion, jump_floor_m=0.1, long_gap_seconds=0.5, swap_proximity_m=0.02)
candidates = select_stage1_candidates(motion, quality, requested_count=12, anomaly_margin_seconds=0.25)
```

Marker-layout IO, validation, the interactive editor and reporting are also
separate modules: `layout_io`, `layout`, `layout_editor` and `layout_reporting`.

## Other reusable APIs

- `synchronization.preprocessing.align_recording(csv, neon_dir, output_dir)`
  crops by shared global timestamps and maps each retained native video frame
  to its nearest motion row. It writes an explicit mapping and provenance;
  source Frame/Time cells and missing data remain intact. No clock correction
  is applied. See the root README for the clock assumption and output formats.
- `recording.session.load_session(directory)` reads that derived mapping/CSV.
- `recording.scene.RecordingScene(session, model_path=None, mount_path=None)`
  constructs a labeled skeleton or optional local neutral SMPL-X skin together
  with head-mounted Neon CAD, real camera intrinsics and optical debug axes.
- `recording.player.RecordingPlayer(scene).run()` opens the paired desktop
  player. `recording.rendering.render_preview(scene, path, frame)` exports a PNG.
- `smplx.neutral` handles the optional licensed neutral NPZ and direct visual
  retargeting. It does not estimate shape or fill untracked limb motion.
- `neon.mount` merges CAD/camera assets under a tracked head and computes an
  editable visual nose-bridge mount; it does not calibrate extrinsics.
- `neon.frame_fit` opens the CAD's rigid temple arms about nominal hinges to
  accommodate the neutral head, preserving front-frame/camera/IMU geometry.
- `neon.imu` defines the nominal scene-camera-to-module-IMU transform, loads
  native WXYZ quaternions, samples with SLERP at unchanged video timestamps,
  and validates a head/IMU world-heading registration. The paired scene can use
  IMU rotation while keeping nose position geometric. Source hashes and fit
  diagnostics are saved separately from the original time alignment.

- `smplx.fitting.build_overrides(mode, sequence)` builds a MoSh++ configuration.
- `smplx.fitting.build_capture_overrides(...)` applies those legacy settings
  to an explicitly supplied capture/layout, including 1126.
- `smplx.pipeline.preflight(...)` checks marker assignments, real observation
  coverage and the legacy runtime without starting optimization.
- `smplx.pipeline.run_legacy_fit(...)` runs the existing MoSh++ Stage I/II
  solver and saves AMASS-style parameters with explicit source-frame metadata.
  It uses the same `mocap_data` environment as the viewers, plus completed
  surface assignments; the imported 25-marker legacy body layout is validated for
  initialization on 1126. Only the separate `psbody.smpl` component remains
  pending before fitting can start.
- `smplx.kinematics.global_joint_motion(...)` computes joint transforms from poses.
- `smplx.reconstruction.reconstruct_vertices(...)` reconstructs a fitted body.
- `smplx.head_markers.diagnose_head_markers(result_path, model_path, output_path)` creates diagnostics without CLI parsing.
- `synchronization.clocks.make_clock_converter(...)` interpolates clock anchors using integer nanoseconds.
- `synchronization.alignment.fit_affine_map(events)` fits offset and drift from synchronization events.
- `avatars.gltf.read_glb(path)` and `read_accessor(...)` decode avatar assets.
- `avatars.vrm.convert(...)` generates a skin from a humanoid VRM asset.

Import the domain modules directly. The legacy script wrappers and experimental
command directory have been removed. Deliberately chosen workflow entry points
live under `tests/`; `tests/visualize_optitrack.py` runs OptiTrack visualization.

Run helper regression checks in the data/viewer environment:

```bash
python -m unittest discover -s tests/unit -v
```
