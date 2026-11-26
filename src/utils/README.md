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
| `paths.py` | Repository, data, asset and bundled MoSh++ locations |

Loading, analysis, kinematics, scene construction and playback live in separate
files. Data-processing helpers can be imported without loading GUI or fitting
dependencies. Rendering and fitting functions require their respective
environments and local model assets.

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

- `smplx.fitting.build_overrides(mode, sequence)` builds a MoSh++ configuration.
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
