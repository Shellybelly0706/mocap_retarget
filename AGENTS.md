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
- Store imported sessions and generated results under root `data/`. Its
  contents are ignored by Git except `.gitkeep`; do not commit captures or
  generated session results. The user supplied licensed SMPL-X assets through
  Git LFS on `main`; these now live under `assets/avatar/smplx/`. Keep additional
  private models in ignored `data/` unless the user explicitly directs otherwise.

## Code organization

- Put reusable implementations in `src/utils/`, grouped by responsibility:
  `optitrack`, `markers`, `smplx`, `synchronization`, `avatars`, `neon`, and
  `recording` (paired-session IO, scenes and playback).
- Separate loading, processing, scene construction, and playback. Keep GUI and
  fitting dependencies out of data-only helpers where possible.
- Group avatar models and their derived textures/skins under `assets/avatar/`.
  Neon geometry and camera calibration live under `assets/neon/`.
- Group scene backgrounds under `assets/background/<theme>/`, such as
  `assets/background/christmas/`. Add future backgrounds as sibling themes
  and use `BACKGROUND_ASSETS_DIR` from `utils.paths` for their shared root.
- `tests/visualize_recording.py` shares one transport between its MuJoCo left
  panel and recorded Neon video right panel. Its default is a labeled skeleton;
  use `--smplx-model assets/avatar/smplx/SMPLX_NEUTRAL.npz` for the imported,
  validated neutral model. Do not call the skeleton or another avatar SMPL-X.
- The optional neutral SMPL-X skin path is visual retargeting, not MoSh++
  fitting. The real-model skin has been rendered; nose fit remains provisional. The
  default SMPL-X glasses anchor uses eye height and the nasal surface, with a CAD
  brow-bar standoff; do not restore the low nose-tip offset. `neon.frame_fit`
  opens the separate rigid CAD arms about nominal hinges for approximate skin
  clearance. It does not scale the glasses or alter the front camera/IMU assembly.
  Do not change the subject's head shape or call this a calibrated physical fit. Legs
  and face are untracked neutral pose. Neon nose-bridge placement is provisional
  and adjustable; factory camera intrinsics do not calibrate head-camera
  extrinsics. Keep optical axes visible for debugging and retain that distinction.
- The user's next SMPL-X objective is a saved reconstruction of the OptiTrack
  motion as SMPL-X parameters, not only displaying an avatar. Treat the current
  direct-retargeting viewer as a preview; do not describe it as completed
  marker-based body-shape/pose fitting.
- The user selected the legacy MoSh++ workflow first. `tests/fit_smplx.py`
  restores that workflow using reusable fitting/pipeline helpers. Use the
  shared `mocap_data` environment. The layouts arrived in main commit `77c22c3` and
  live under `assets/avatar/smplx/marker_layout/`. Their model checksum and
  neutral topology have been verified. The first 1126 pass uses the complete
  legacy 25-marker body layout; its 14 CustomMarker tracks are explicitly
  excluded. Treat this as transferred initialization, with fresh Stage I
  calibration still required. The 39-marker draft remains unset for future
  custom-marker mapping. Only `smpl-fast-derivatives` / `psbody.smpl` is still
  pending a later upload; do not claim the fit ran or repeatedly request it.
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
Use one project environment, `mocap_data`, defined only by root
`environment.yml`: Python 3.10 / NumPy 1.23.5 with compatible viewer, video,
Neon, CAD and fitting dependencies. `tests/setup_environment.py` builds the
bundled native components in that environment. Do not reintroduce a separate
MoSh++ environment YAML or requirements file. The external `psbody.smpl`
component is still pending; do not claim environment consolidation completed
the fit.

## Verification

For helper changes, run the relevant checks in the shared `mocap_data` environment:

```bash
python -m unittest discover -s tests/unit -v
python tests/visualize_optitrack.py --help
```

Use `--check-only` or `--validate-only` when validating available local captures
without a GUI. Report when a full recording, visualization, or fitting run
could not be checked because hardware, captures, or licensed assets were absent.
