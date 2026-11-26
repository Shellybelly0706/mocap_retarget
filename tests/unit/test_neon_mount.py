"""Check that CAD bridge, optical axes and real intrinsics follow the head."""

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.neon.frame_fit import attach_temple_hinges, frame_parts
from utils.neon.mount import CAD_BRIDGE_M, attach_neon, mount_transform


class NeonMountTests(unittest.TestCase):
    def test_opening_rigid_arms_preserves_sensor_frames_and_all_cad_triangles(self):
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", inertiafromgeom="false")
        head = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="head")
        attach_neon(root, head, np.array([0, -0.09, 0.06]), {})
        original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        before = mujoco.MjData(original)
        mujoco.mj_forward(original, before)
        fit = {"left": {"opening_deg": 8}, "right": {"opening_deg": 9}}
        attach_temple_hinges(root, fit)
        opened = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        after = mujoco.MjData(opened)
        mujoco.mj_forward(opened, after)
        self.assertEqual(original.mesh_facenum.sum(), opened.mesh_facenum.sum())
        for name in ("neon_glasses", "neon_scene_camera_optical", "neon_imu"):
            np.testing.assert_allclose(before.body(name).xpos, after.body(name).xpos)
            np.testing.assert_allclose(before.body(name).xmat, after.body(name).xmat)
        np.testing.assert_array_equal(original.cam_intrinsic, opened.cam_intrinsic)
        _, parts = frame_parts()
        cad_rotation = after.body("neon_glasses").xmat.reshape(3, 3)
        cad_origin = after.body("neon_glasses").xpos
        for side, (part, pivot) in parts.items():
            body = after.body(f"neon_temple_{side}")
            # Opening must leave each hinge fixed and move its rear end outward.
            np.testing.assert_allclose(body.xpos, cad_origin + cad_rotation @ pivot)
            point = part.vertices[np.argmin(part.vertices[:, 2])]
            world = body.xpos + body.xmat.reshape(3, 3) @ (point - pivot)
            moved = cad_rotation.T @ (world - cad_origin)
            self.assertGreater(abs(moved[0]), abs(point[0]) + 0.01)

    def test_bridge_anchor_and_camera_follow_translated_rotated_head(self):
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", inertiafromgeom="false")
        head = ET.SubElement(
            ET.SubElement(root, "worldbody"), "body", name="head", mocap="true"
        )
        anchor = np.array([0, -0.08, 0.11])
        settings = {
            "translation_m": [0.001, 0.002, 0.003],
            "rotation_deg_xyz": [2, 3, 4],
        }
        attach_neon(root, head, anchor, settings)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        position, quat = mount_transform(anchor, settings)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(
            position + rotation @ CAD_BRIDGE_M, anchor + settings["translation_m"]
        )
        mujoco.mj_forward(model, data)
        local_optical = data.body("neon_scene_camera_optical").xpos.copy()
        local_axes = data.body("neon_scene_camera_optical").xmat.reshape(3, 3).copy()
        head_rotation = Rotation.from_euler("xyz", [20, -15, 60], degrees=True)
        translation = np.array([1, 2, 1.5])
        data.mocap_pos[0] = translation
        data.mocap_quat[0] = head_rotation.as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.body("neon_scene_camera_optical").xpos,
            translation + head_rotation.apply(local_optical),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            data.body("neon_scene_camera_optical").xmat.reshape(3, 3),
            head_rotation.as_matrix() @ local_axes,
            atol=1e-10,
        )
        camera = model.camera("neon_scene_camera").id
        np.testing.assert_array_equal(model.cam_resolution[camera], [1600, 1200])
        self.assertEqual(model.ncam, 1)


if __name__ == "__main__":
    unittest.main()
