# Unified `mocap_data` environment

Run commands from the repository root. Create and verify the environment with:

```bash
conda env create -f environment.yml
conda activate mocap_data
python tests/setup_environment.py
python -c "import olefile, numpy, pandas, scipy, h5py, matplotlib, mujoco, cv2, PIL, pupil_labs.realtime_api; print('environment OK')"
```

Use this one environment for the whole project. Root `environment.yml` is the
single dependency definition for collection, processing, viewing, CAD conversion
and legacy MoSh++ fitting. It pins Python 3.10, NumPy 1.23.5 and SciPy 1.10.1
for Chumpy compatibility, with matching versions of pandas, OpenCV, PyAV and
Matplotlib. MuJoCo and the Neon API share this stack.

`tests/setup_environment.py` builds the bundled `psbody.mesh` and scan-to-mesh
extensions and links the MoSh++ source in the active environment. The external
`psbody.smpl` solver component must still be supplied separately.

Update an existing installation with:

```bash
conda env update -n mocap_data -f environment.yml --prune
conda activate mocap_data
python tests/setup_environment.py
```

Activation disables user-site packages and clears the inherited `PYTHONPATH`
for this environment, preventing the shell's ROS Python packages from leaking
into the project. Conda restores the previous values on deactivation. Commands
that import helpers directly can still use `PYTHONPATH=src python ...`.

The environment supports:

- inspecting and extracting the OLE streams inside a `.tak` file with `olefile`;
- parsing an OptiTrack CSV and writing CSV, pickle, NPZ, or HDF5 data;
- coordinate and quaternion conversion;
- MuJoCo loading, playback, and retargeting code;
- Neon clock-offset monitoring and offline video synchronization;
- rendering model previews and exporting PNG files;
- interactive SMPL-X marker-layout editing with Matplotlib's Tk backend;
- paired MuJoCo/Neon desktop playback with Tk, and exact timestamp-based video
  cropping/reencoding with PyAV;
- legacy MoSh++ model/prior utilities and native geometry components;
- Neon STEP-to-STL regeneration using the included CAD libraries.

The first 1126 recording can be opened with:

```bash
conda activate mocap_data
python tests/visualize_recording.py data/1126/aligned/recording_1
```

See the root README for preprocessing, shared playback controls, optional local
`SMPLX_NEUTRAL.npz` loading, and the provisional glasses mount. The default view
uses the solved skeleton unless the imported SMPL-X model is selected with
`--smplx-model assets/avatar/smplx/SMPLX_NEUTRAL.npz`.

Validate the Neon model and run the helper checks:

```bash
python tests/visualize_neon.py --check-only
python -m unittest discover -s tests/unit -v
```

Headless rendering requires an available OpenGL backend, for example
`MUJOCO_GL=osmesa`. Fitting, CAD conversion and viewing all use `mocap_data`.

## View the 1105 CSV takes in MuJoCo

Open a CSV take by its path using the OptiTrack visualization entry point:

```bash
conda run --no-capture-output -n mocap_data \
  python tests/visualize_optitrack.py data/1105/csv/take.csv
conda run --no-capture-output -n mocap_data \
  python tests/visualize_optitrack.py data/1105/csv/take.csv --check-only
```

Replace `take.csv` with the imported capture's filename. The same entry point
accepts C3D files. Use `--check-only` to validate a take without opening a window.
During CSV playback, Space pauses, R restarts, keys 1 through 6 toggle object
categories, and the left/right arrow keys step one frame while paused.

## Legacy MoSh++ reconstruction

The unified environment contains Chumpy, CPU PyTorch, Human Body Prior,
body_visualizer and the tools needed by the existing MoSh++ Stage I/II solver.
The legacy layouts have been imported and validated. The
`smpl-fast-derivatives` / `psbody.smpl` component remains pending a later upload;
optimization has not started.

The compiled `psbody.smpl` component must be installed alongside `psbody.mesh`
and verified against this Python/Linux environment. It is supplied separately
through the [SOMA download portal](https://soma.is.tue.mpg.de/).

Check the first 1126 take without fitting:

```bash
conda activate mocap_data
python tests/fit_smplx.py --check-only
```

The default input is the first 1126 C3D with the imported body layout
`assets/avatar/smplx/marker_layout/Zekun_upper_25/marker_layout.json`. All 25
labels occur in the capture; 14 CustomMarker tracks are excluded. The layouts
retain their original provenance, including the 1112 head-marker calibration;
Stage I must recalibrate the new subject. Legacy hand layouts preserve their
nonzero skin offsets, which are accepted by the fitting reader without changing
the surface editor's zero-offset policy.

Preflight exits with status 2 while required
inputs are missing and saves `preflight_report.json` under
`data/1126/reconstruction/recording_1/smoke/`.

Once the solver component is ready, run
`python tests/fit_smplx.py --mode smoke` for the first 120 source
frames at a sampling step of 4 (30 output frames). Use `--stagei-only` to inspect
shape/marker calibration first, then run without it to resume Stage II. Use
`--mode full` for the full take. The AMASS-style NPZ retains explicit source
sample rows, C3D frame numbers and relative times. Global Neon alignment is
left unchanged; the fitted sequence must subsequently be joined to its frame
mapping. The user-supplied layout is preserved; stage I uses a working copy.
Override `--layout PATH` to select another completed layout.

## TAK to CSV limitation

`optitrack-motive-file-converter` is a C# wrapper around OptiTrack's proprietary
NMotive library. NMotive and its native Qt dependencies are Windows binaries;
they are not Python or Conda packages and do not run natively on this Linux
host. Consequently, the actual `.tak` to CSV decoding must be performed on a
Windows machine or Windows VM with the converter and its matching NMotive
dependencies.

Run the converter on a copy because its current implementation reconstructs,
solves, and saves the input take:

```powershell
Copy-Item "take.tak" "take.conversion-copy.tak"
converter.exe "take.conversion-copy.tak" "take.csv" 0
```

Move the resulting CSV back into this repository, then use the `mocap_data`
environment for normalization, serialization, and MuJoCo retargeting.
