# Project purpose

This repository is building an **egocentric data collection pipeline** combining:

- **OptiTrack / Motive upper-body tracking**, including marker and skeleton
  motion exported as CSV or C3D.
- **Pupil Labs Neon front-facing scene/world-camera video**, with its exported
  frame timestamps.

The central objective is to pair the participant's egocentric view with their
upper-body motion over time. Motion fitting, visualization, and avatar
retargeting are supporting capabilities. Do not describe the project as only
a MoSh++/SMPL-X fitting or robot-retargeting project.

Read `README.md` for the intended collection workflow and current capabilities.
Read `TODO.md` for open calibration work and unresolved accuracy limitations.
Distinguish implemented tools from the unified collection/session-export
workflow that is still being developed.

## Data and synchronization

- Treat motion and scene-camera video as the two core streams. Gaze and IMU
  data can support related work and alignment, but are not substitutes for
  either core stream.
- Preserve original recordings, timestamps, frame IDs, units, and coordinate
  conventions. Keep derived alignment and fitted motion identifiable as
  derived data.
- Make clock domains, offset direction, resampling, and alignment assumptions
  explicit in changes that affect synchronization. Retain missing-marker and
  missing-frame information rather than silently filling gaps.
- Neon Time Echo measures Ubuntu minus Neon Companion time. It does not
  measure the Motive Windows computer's clock offset. Do not assume that all
  devices share a clock or claim sub-millisecond accuracy without evidence.
- `tests/align_recording.py` performs the requested global-timestamp crop and
  nearest-row sampling onto native Neon video frames. It applies no IMU clock
  correction. The output mapping is authoritative; sampled Motive CSV Frame/Time
  cells and rate metadata retain their source meaning. The two device clocks'
  agreement is an explicit, unverified assumption.
- `src/neon/` is our clock-offset monitor. Recording happens through Neon
  Companion / Neon Monitor; the monitor does not capture scene video itself.
- Store imported sessions and local model files under root `data/`. Its
  contents are ignored by Git except `.gitkeep`; do not commit captures,
  licensed models, or generated session results.

## Code organization

- Put reusable implementations in `src/utils/`, grouped by responsibility:
  `optitrack`, `markers`, `smplx`, `synchronization`, `avatars`, `neon`, and
  `recording` (paired-session IO, scenes and playback).
- Separate loading, processing, scene construction, and playback. Keep GUI and
  fitting dependencies out of data-only helpers where possible.
- Group avatar models and their derived textures/skins under `assets/avatar/`.
  Neon geometry and camera calibration live under `assets/neon/`.
- `tests/visualize_recording.py` shares one transport between its MuJoCo left
  panel and recorded Neon video right panel. Its default is a labeled skeleton
  until the user's licensed neutral SMPL-X NPZ is available. Do not call the
  skeleton or another avatar SMPL-X. The user has the file on another computer
  and asked to supply it later; do not repeatedly request it during this work.
- The optional neutral SMPL-X skin path is visual retargeting, not MoSh++
  fitting. Its real-model appearance and nose fit still need validation. Legs
  and face are untracked neutral pose. Neon nose-bridge placement is provisional
  and adjustable; factory camera intrinsics do not calibrate head-camera
  extrinsics. Keep optical axes visible for debugging and retain that distinction.
- Neon's IMU is inside the module's top bar, with local X right, Y forward,
  Z up. Use `assets/neon/imu_reference.json` and `utils.neon.imu` for the nominal
  camera-to-IMU transform; do not co-locate the two frame origins. The IMU's
  magnetic/gravity world is not the OptiTrack world. `--imu-orientation` fits
  a validated world-yaw/mount registration and samples native quaternions on
  the existing video timeline, without changing the global timestamp alignment.
  IMU mode drives rotation about the geometric nose bridge, not translation.
  Its CAD IMU origin remains nominal/approximate, not a surveyed chip location.
- Use `src/utils/paths.py` for shared repository locations. Avoid machine-specific
  absolute paths and the removed `mocap_data/` directory. `mocap_data` remains
  the Conda environment name in `environment.yml`.
- Use `tests/` for deliberately chosen runnable workflow entry points, such as
  `tests/visualize_optitrack.py`; keep their implementation in reusable helpers.
  Do not automatically create one launcher per helper.
- `tests/unit/` contains automated helper regression checks. Legacy command
  wrappers have been removed from `src/utils/`; import domain helpers directly
  and keep runnable workflow entry points in `tests/`.
- `src/mesh/` and `src/moshpp/` contain bundled third-party sources with local
  adaptations. `src/neon/` and `src/utils/` contain project-specific code.

See `src/utils/README.md` for helper APIs and `ENVIRONMENT.md` for data tooling.
The data/viewer and legacy MoSh++ environments have different Python and NumPy
requirements; use the appropriate environment for each workflow.

## Verification

For helper changes, run the relevant checks in the data/viewer environment:

```bash
python -m unittest discover -s tests/unit -v
python tests/visualize_optitrack.py --help
```

Use `--check-only` or `--validate-only` when validating available local captures
without a GUI. Report when a full recording, visualization, or fitting run
could not be checked because hardware, captures, or licensed assets were absent.
