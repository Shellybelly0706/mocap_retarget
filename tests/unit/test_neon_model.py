"""Regression checks for the Neon CAD asset and calibrated camera frame."""

import hashlib
import json
import os
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.neon.camera import SceneCameraCalibration
from utils.neon.viewer import NEON_ASSET_DIR, validate_neon_model


class NeonModelTests(unittest.TestCase):
    def test_downloaded_cad_matches_provenance(self):
        source = json.loads((NEON_ASSET_DIR / "source.json").read_text())
        path = NEON_ASSET_DIR / source["cad"]["file"]
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), source["cad"]["sha256"]
        )

    def test_stl_meshes_are_binary_and_within_mujoco_face_limit(self):
        geometry = json.loads((NEON_ASSET_DIR / "geometry.json").read_text())
        for name, component in geometry["components"].items():
            with self.subTest(component=name):
                path = NEON_ASSET_DIR / "meshes" / component["file"]
                with path.open("rb") as stream:
                    stream.seek(80)
                    count = struct.unpack("<I", stream.read(4))[0]
                self.assertEqual(path.stat().st_size, 84 + count * 50)
                self.assertGreater(count, 0)
                self.assertLessEqual(count, 200000)
                self.assertEqual(count, component["triangles"])

    def test_compiled_model_preserves_metres_optical_axes_and_factory_intrinsics(self):
        report = validate_neon_model()
        self.assertEqual(report["serial"], "370641")
        self.assertEqual(report["resolution"], [1600, 1200])
        self.assertGreater(report["triangles"], 100000)

    def test_principal_point_is_an_offset_from_image_centre(self):
        calibration = SceneCameraCalibration(
            "test",
            np.array([[500.0, 0, 350.0], [0, 600.0, 200.0], [0, 0, 1.0]]),
            np.zeros(8),
            (640, 480),
        )
        attrs = calibration.mjcf_attributes()
        np.testing.assert_allclose(
            np.fromstring(attrs["focalpixel"], sep=" "), [500, 600]
        )
        np.testing.assert_allclose(
            np.fromstring(attrs["principalpixel"], sep=" "), [-30, 40]
        )

    @unittest.skipUnless(
        os.environ.get("MUJOCO_GL") in {"osmesa", "egl"},
        "Set a headless MuJoCo backend to check rendered pixel projection",
    )
    def test_rendered_projection_matches_opencv_camera_matrix(self):
        import mujoco

        calibration = SceneCameraCalibration(
            "test",
            np.array([[500.0, 0, 350.0], [0, 600.0, 200.0], [0, 0, 1.0]]),
            np.zeros(8),
            (640, 480),
        )
        attrs = " ".join(
            f'{name}="{value}"' for name, value in calibration.mjcf_attributes().items()
        )
        # CV point (0.1, 0.05, 1) -> OpenGL point (0.1, -0.05, -1).
        model = mujoco.MjModel.from_xml_string(
            f"""<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody><camera name="test" {attrs}/><geom type="sphere" size="0.008" pos="0.1 -0.05 -1" rgba="1 0 0 1"/></worldbody></mujoco>"""
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="test")
            pixels = renderer.render()
        ys, xs = np.where(
            (pixels[:, :, 0] > 80) & (pixels[:, :, 1] < pixels[:, :, 0] * 0.3)
        )
        self.assertGreater(xs.size, 0)
        # Pixel-centre coordinates account for OpenGL's half-pixel convention.
        np.testing.assert_allclose(
            [xs.mean() + 0.5, ys.mean() + 0.5], [400, 230], atol=1
        )


if __name__ == "__main__":
    unittest.main()
