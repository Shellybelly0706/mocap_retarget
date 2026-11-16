# `mocap_data` environment

Create and verify the environment with:

```bash
conda env create -f environment.yml
conda activate mocap_data
python -c "import olefile, numpy, pandas, scipy, h5py, mujoco; print('environment OK')"
```

The environment supports:

- inspecting and extracting the OLE streams inside a `.tak` file with `olefile`;
- parsing an OptiTrack CSV and writing CSV, pickle, NPZ, or HDF5 data;
- coordinate and quaternion conversion;
- MuJoCo loading, playback, and retargeting code.

## View the 1105 CSV takes in MuJoCo

List the six CSV takes, then open exactly one by its 1-based number:

```bash
conda run --no-capture-output -n mocap_data \
  python scripts/compare_1105_exports.py --list
conda run --no-capture-output -n mocap_data \
  python scripts/compare_1105_exports.py 1
```

The viewer displays CSV points and rigid bodies without skeleton links. Use
`--check-only` to print and validate a take without opening a window. During
playback, Space pauses, R restarts, keys 1 through 5 toggle the CSV object
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
