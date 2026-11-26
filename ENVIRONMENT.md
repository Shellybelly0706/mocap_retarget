# `mocap_data` environment

Run commands from the repository root. Create and verify the environment with:

```bash
conda env create -f environment.yml
conda activate mocap_data
python -c "import olefile, numpy, pandas, scipy, h5py, matplotlib, mujoco, cv2, PIL, pupil_labs.realtime_api; print('environment OK')"
```

This is the main environment for the egocentric collection pipeline. Its
definition includes the Neon real-time API, OpenCV video processing, Pillow
image export, and the scientific/MuJoCo tools; no extra pip install is needed
after creating it from `environment.yml`.

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
- interactive SMPL-X marker-layout editing with Matplotlib's Tk backend.
- paired MuJoCo/Neon desktop playback with Tk, and exact timestamp-based video
  cropping/reencoding with PyAV.

The first 1126 recording can be opened with:

```bash
conda activate mocap_data
python tests/visualize_recording.py data/1126/aligned/recording_1
```

See the root README for preprocessing, shared playback controls, optional local
`SMPLX_NEUTRAL.npz` loading, and the provisional glasses mount. The default view
uses the solved skeleton while that licensed model is unavailable.

Validate the Neon model and run the helper checks:

```bash
python tests/visualize_neon.py --check-only
python -m unittest discover -s tests/unit -v
```

Headless rendering requires an available OpenGL backend, for example
`MUJOCO_GL=osmesa`. MoSh++ fitting and CAD-to-STL regeneration use separate
environments, as described in the root and Neon asset READMEs.

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
