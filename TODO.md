# TODO

## Calibrate Neon camera and IMU pose in OptiTrack world coordinates

Recorded: 2026-11-26. Status: open.

- [ ] Establish and validate the rigid transform between a tracked glasses
  marker body and the Neon scene camera / IMU.

The current viewer does **not** provide a calibrated IMU XYZ position in the
OptiTrack world. Both orientation modes derive position from the tracked head
and an approximate nose/glasses offset. IMU mode uses recorded orientation
after estimating the heading relationship between the magnetic/gravity world
and OptiTrack; this orientation fit does not calibrate translation. The IMU's
CAD location uses a manufacturer nominal camera-to-IMU transform and an
approximate CAD camera origin.

Required relationship, where `T_A_B` maps coordinates from B into A:

```text
T_world_imu = T_world_glasses_body @ T_glasses_body_imu
```

Next steps:

1. Define a stable OptiTrack rigid body using markers attached to the glasses,
   and document its axes, origin and relationship to the tracked head.
2. Record a calibration target of known dimensions visible to the Neon scene
   camera and localized in OptiTrack space. An AprilTag board with tracked
   corners is a suitable approach; tags with unknown world pose alone do not
   establish the camera's OptiTrack world position.
3. Estimate the camera-to-glasses-body transform, then compose it with the
   documented nominal camera-to-IMU transform. Retain the uncertainty of that
   nominal internal transform. Respect this capture's Z-up convention.
4. Save transform directions, units, marker geometry, calibration recordings,
   timing assumptions and validation results. Check held-out target poses and
   report position, orientation and reprojection errors before labeling the
   resulting world poses calibrated. Check or repeat calibration if the
   marker configuration or glasses-to-head fit changes.

Spatial calibration is separate from clock synchronization. Keep the existing
global-timestamp alignment unchanged unless a timing correction is separately
measured and documented. Until this task is complete, keep the viewer's nose
placement and IMU world position labeled approximate.

Reference: [Pupil Labs motion-capture integration workflow](https://docs.pupil-labs.com/alpha-lab/mocap/).
