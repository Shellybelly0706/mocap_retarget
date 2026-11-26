# Pupil Labs Neon reference glasses for MuJoCo

The official **Just act natural** reference assembly is included as its
original STEP CAD and converted binary STL meshes. The assembly contains the
frame, Neon module, nest and mounting hardware. Source revision, URLs and
SHA-256 checksums are recorded in [source.json](source.json). Upstream usage
terms are preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

## View the model

From the repository root, in a Python environment with MuJoCo and NumPy:

```bash
python tests/visualize_neon.py
python tests/visualize_neon.py --camera scene
python tests/visualize_neon.py --check-only
```

Press **C** to switch between the orbit view and the scene camera, or **F** to
toggle the camera/IMU axes. Orbit and scene-camera previews are available
in [preview.png](preview.png) and [scene_camera_preview.png](scene_camera_preview.png).

The MJCF files can also be opened directly:

- [neon.xml](neon.xml): glasses geometry, optical frame and calibrated camera.
- [scene.xml](scene.xml): inspection scene with floor, lighting and view targets.

For headless PNG output, select an available MuJoCo rendering backend:

```bash
MUJOCO_GL=osmesa python tests/visualize_neon.py --render /tmp/neon.png
MUJOCO_GL=osmesa python tests/visualize_neon.py --camera scene --render /tmp/neon_scene.png
```

PNG export also requires Pillow. EGL can be used instead of OSMesa where the
graphics driver supports it.

## Scene-camera intrinsics

Factory calibration was retrieved for **Neon module 370641**, confirmed by the
user. The original calibration fields are in
[scene_camera_370641.json](scene_camera_370641.json). The image size is
**1600 × 1200**. The OpenCV camera matrix, in pixels, is:

```text
K = [[883.379628805954,   0,                801.970848833916],
     [  0,              883.038257515406, 598.653751909991],
     [  0,                0,                1              ]]
```

The camera named `neon_scene_camera` uses these focal lengths directly.
MuJoCo's `principalpixel` uses offsets from the image centre with:

```text
principalpixel = (width / 2 - cx, height / 2 - cy)
               = (-1.970848833916, 1.346248090009)
```

The eight OpenCV distortion coefficients are preserved both in the calibration
JSON and in the MJCF custom numeric `neon_distortion_opencv`. Native MuJoCo
renders an **undistorted pinhole image**; it does not apply these coefficients.
Compare its output with video undistorted using the same `K`, or apply lens
distortion to rendered images separately. A replacement calibration must use
the same module and recording resolution as the real data being compared.

## Camera frame and geometry units

- The STEP source uses millimetres. STL vertices are converted to **metres**;
  no additional `scale="0.001 0.001 0.001"` should be applied.
- CAD Y-up coordinates are rotated +90° about X to obtain a Z-up MuJoCo scene.
  Component positions within the original assembly are preserved.
- `neon_scene_camera_optical` uses **OpenCV axes**: X right, Y down, Z forward.
  The visible axes are red, green and blue respectively.
- The child `neon_scene_camera` applies a 180° rotation around X for MuJoCo's
  **OpenGL axes**: X right, Y up, negative Z forward.
- The frame origin is the CAD lens-opening centre, at CAD coordinates
  `(0, 52.1650169862, 14.8096489865)` mm. The optical axis comes from the circular
  aperture and tilts downward by 12° relative to CAD +Z.

**The CAD lens-opening centre is a reference pose, not a calibrated optical
centre.** Factory intrinsics do not establish the camera-to-glasses or
camera-to-OptiTrack extrinsics. Use a measured rigid transform when accurate
spatial alignment with motion capture is required.

## IMU installation and attachment

Pupil Labs locates the IMU **inside the top bar of the Neon sensor module**.
Its local axes when worn are **X right, Y forward, Z up**. This is a separate
frame from the camera optical axes. Module quaternions rotate IMU coordinates
into a magnetic/gravity world: +Y magnetic north and −Z gravity. That world's
heading is not the OptiTrack world's heading.

The manufacturer's nominal camera-to-IMU point transform is recorded in
[`imu_reference.json`](imu_reference.json), with its source URLs:

```text
p_imu = Rx(-102 degrees) @ p_camera + [0, -0.0013, -0.00662]  # metres
```

Using the shipped CAD camera reference, this places `neon_imu` at CAD
**(0, 58.7850169862, 16.1096489865) mm**, with CAD-from-IMU rotation:

```text
[[-1, 0, 0],
 [ 0, 0, 1],
 [ 0, 1, 0]]
```

This places the reference in the module's upper bar. Its absolute CAD location
inherits the approximate lens-opening camera origin; it is **not a surveyed
per-device chip centre**. The published relative transform is nominal. The XML
shows the IMU origin in magenta and camera origin in yellow, each with RGB
axes. Site group 4 toggles IMU axes; group 5 toggles camera/nose references.

Use the recorded module IMU to orient the glasses in the paired player:

```bash
python tests/visualize_recording.py data/1126/aligned/recording_1 --imu-orientation
```

`utils.neon.imu` reads **`imu ps1.raw` and `imu.dtype`**, not the separate
`extimu` stream. It normalizes valid WXYZ quaternions and uses SLERP at the
unchanged canonical video timestamps, rejecting extrapolation and gaps over
50 ms. It estimates a constant head-to-IMU rotation and unknown world yaw
jointly, assuming both worlds are Z-up. The first 0.5 s is excluded only from
the parameter fit, not from playback. A held-out time segment and a numerical
observability check reject poor registrations.

For recording 1, the estimated world yaw is **65.22°**, the post-startup median
orientation residual is **0.99°**, and the held-out 90th-percentile residual is
**1.82°**. These are orientation consistency diagnostics, not independent
ground-truth accuracy. The native timestamps show approximately **124.8 Hz**
median sampling for this capture; use the recorded timestamps rather than a
nominal fixed IMU rate.

The active IMU mode drives glasses rotation while OptiTrack supplies head/nose
translation. The glasses rotate about the estimated nose-bridge contact point;
the IMU does not determine that contact point or provide position. Manual fit
rotations become extra corrections to the IMU pose. The IMU cannot establish
absolute glasses-to-nose translation through double integration here.

The player writes `imu_attachment.json` and `.csv` under the derived session,
including IMU source hashes, interpolation brackets, registration matrices,
fit/held-out diagnostics and per-frame residuals. Existing video/motion
alignment files and all raw recordings remain unchanged. In the GUI, **IMU
orientation** switches back to OptiTrack orientation for comparison, and **IMU
frame (top bar)** independently toggles the sensor axes.

Sources: [sensor location and axes](https://docs.pupil-labs.com/neon/data-collection/data-streams/#movement-imu-data),
[camera-to-IMU transformation](https://docs.pupil-labs.com/alpha-lab/imu-transformations/).

Meshes are visual-only: no collision shapes, articulated hinges or measured
inertial properties are supplied. The large frame mesh is split into two STL
files to stay within MuJoCo's 200,000-face-per-STL limit. No surface triangles
are removed except degenerate faces.

## Regenerate the meshes and XML

Loading the shipped assets needs no CAD dependency. To regenerate them, use a
separate Python 3.11 environment with NumPy and
`cadquery-ocp==7.8.1.1.post1` (which also installs `vtk==9.3.1`):

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from utils.neon.cad import convert_reference_cad
from utils.neon.model import build_neon_model

folder = Path("assets/neon")
convert_reference_cad(folder / "cad/Neon_reference.STEP", folder / "meshes")
build_neon_model(folder, folder / "scene_camera_370641.json")
PY
```

`geometry.json` records tessellation tolerances, triangle counts, units, bounds,
and the CAD camera reference. `src/utils/neon/` contains reusable conversion,
calibration, model-building and viewing helpers. `tests/visualize_neon.py` is
the runnable entry point.

## References

- [Official Neon CAD repository](https://github.com/pupil-labs/neon-geometry)
- [Neon recording format and factory scene-camera calibration](https://docs.pupil-labs.com/neon/data-collection/data-format/)
- [Scene-camera resolution and data streams](https://docs.pupil-labs.com/neon/data-collection/data-streams/)
- [MuJoCo camera conventions and intrinsics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)
