"""Check sensor placement, native quaternion sampling and independent-world fitting."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.neon.imu import (
    camera_to_imu,
    imu_pose_in_cad,
    register_imu_world,
    sample_native_imu,
)


class NeonImuTests(unittest.TestCase):
    def test_documented_transform_places_imu_in_module_upper_bar(self):
        camera_position = np.array([0, 0.0521650169862149, 0.014809648986464719])
        c, s = np.cos(np.deg2rad(12)), np.sin(np.deg2rad(12))
        rotation_a_c = np.array([[-1, 0, 0], [0, -c, -s], [0, -s, c]])
        position, rotation = imu_pose_in_cad(camera_position, rotation_a_c)
        np.testing.assert_allclose(
            position, [0, 0.0587850169862149, 0.01610964898646472], atol=1e-12
        )
        np.testing.assert_allclose(
            rotation, [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], atol=1e-12
        )
        r_i_c, t_i_c = camera_to_imu()
        np.testing.assert_allclose(
            rotation.T @ (camera_position - position), t_i_c, atol=1e-12
        )
        # Camera forward in IMU axes: forward and 12 degrees down.
        np.testing.assert_allclose(r_i_c @ [0, 0, 1], [0, c, -s], atol=1e-12)

    def test_native_wxyz_slerp_uses_exact_timestamps_and_rejects_large_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = [("timestamp_ns", "<i8")] + [
                (f"quaternion_{axis}", "<f4") for axis in "wxyz"
            ]
            (root / "imu.dtype").write_text(repr(fields))
            data = np.zeros(3, dtype=np.dtype(fields))
            epoch = 1795690256000000000
            data["timestamp_ns"] = epoch + np.array([0, 10_000_000, 20_000_000])
            quats = Rotation.from_euler("z", [[0], [20], [40]], degrees=True).as_quat()[
                :, [3, 0, 1, 2]
            ]
            quats[1] *= -1  # Equivalent signs must not cause a full-turn interpolation.
            for index, axis in enumerate("wxyz"):
                data[f"quaternion_{axis}"] = quats[:, index]
            data.tofile(root / "imu ps1.raw")
            targets = epoch + np.array([5_000_000, 15_000_000])
            result = sample_native_imu(root, targets)
            np.testing.assert_allclose(
                result.rotations,
                Rotation.from_euler("z", [[10], [30]], degrees=True).as_matrix(),
                atol=1e-7,
            )
            np.testing.assert_array_equal(result.source_left, [0, 1])
            np.testing.assert_array_equal(result.source_right, [1, 2])
            self.assertEqual(result.report["clock_correction_ns"], 0)
            with self.assertRaisesRegex(ValueError, "cover"):
                sample_native_imu(root, epoch + np.array([-1, 0]))
            data["timestamp_ns"] = epoch + np.array([0, 100_000_000, 200_000_000])
            data.tofile(root / "imu ps1.raw")
            with self.assertRaisesRegex(ValueError, "gap"):
                sample_native_imu(root, targets)

    def test_world_yaw_and_mount_are_recovered_without_identifying_world_frames(self):
        time = np.linspace(0, 12, 361)
        head = Rotation.from_euler(
            "xyz",
            np.column_stack(
                [20 * np.sin(time), 15 * np.cos(0.7 * time), 40 * np.sin(0.4 * time)]
            ),
            degrees=True,
        ).as_matrix()
        mount = Rotation.from_euler("xyz", [15, -8, 35], degrees=True).as_matrix()
        world = Rotation.from_euler("z", 72, degrees=True).as_matrix()
        imu = world.T @ head @ mount
        registered, report = register_imu_world(head, imu, time)
        np.testing.assert_allclose(registered, head @ mount, atol=1e-8)
        np.testing.assert_allclose(report["rotation_head_from_imu"], mount, atol=1e-8)
        self.assertAlmostEqual(report["world_yaw_degrees"], 72, places=6)
        self.assertLess(report["heldout"]["p90_residual_degrees"], 1e-6)
        self.assertLess(report["condition_number"], 100)
        stationary = np.repeat(np.eye(3)[None], len(time), axis=0)
        with self.assertRaisesRegex(ValueError, "observable"):
            register_imu_world(stationary, stationary, time)


if __name__ == "__main__":
    unittest.main()
