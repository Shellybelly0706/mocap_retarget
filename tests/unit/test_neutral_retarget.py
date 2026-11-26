"""Rig mathematics with a synthetic fixture; no licensed SMPL-X data needed."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.avatars.pose import anatomical_frames
from utils.smplx.neutral import NeutralAvatar, retarget_neutral


class NeutralRetargetTests(unittest.TestCase):
    def test_rigid_capture_preserves_global_rotation_and_bind_lengths(self):
        # Deliberately synthetic joint tree and vertices, not a human mesh.
        joints = np.random.default_rng(7).normal(0, 0.15, (55, 3))
        joints[0] = [0, 0, 0.9]
        joints[15] = [0, 0, 1.5]
        joints[13], joints[14] = [0.15, 0, 1.35], [-0.15, 0, 1.35]
        parents = np.zeros(55, dtype=int)
        parents[0] = -1
        mapping = {
            0: "Root",
            3: "Ab",
            6: "Ab",
            9: "Chest",
            12: "Neck",
            13: "LShoulder",
            14: "RShoulder",
            15: "Head",
            16: "LUArm",
            17: "RUArm",
            18: "LFArm",
            19: "RFArm",
            20: "LHand",
            21: "RHand",
        }
        # Ab drives two spine joints; make their bind positions agree in this
        # rigid-motion fixture so both rest directions have the same solution.
        joints[6] = joints[3]
        for start, side in ((25, "L"), (40, "R")):
            for finger_index, finger in enumerate(
                ("Index", "Middle", "Pinky", "Ring", "Thumb")
            ):
                for segment in range(3):
                    mapping[start + finger_index * 3 + segment] = (
                        f"{side}{finger}{segment+1}"
                    )
        rotation = Rotation.from_euler("xyz", [[8, 5, 25], [-10, 12, 75]], degrees=True)
        matrices = rotation.as_matrix()
        translation = np.array([[1, 2, 0], [2, 3, 0.1]])
        target = np.einsum("fij,bj->fbi", matrices, joints) + translation[:, None, :]
        bones = {}
        for joint, name in mapping.items():
            bones[name] = np.column_stack((rotation.as_quat(), target[:, joint]))
        body = anatomical_frames(joints[13] - joints[14], joints[15] - joints[0])[0]
        head_top = target[:, 15] + np.einsum("fij,j->fi", matrices, body[:, 2] * 0.15)
        head_front = target[:, 15] + np.einsum("fij,j->fi", matrices, body[:, 0] * 0.1)
        avatar = NeutralAvatar(
            np.array([[0, 0, 1.65]]),
            np.empty((0, 3), int),
            np.empty((1, 55)),
            joints,
            parents,
            np.zeros(3),
        )
        positions, actual_rotation, scale = retarget_neutral(
            avatar, bones, head_top, head_front, reference=1
        )
        self.assertAlmostEqual(scale, 1.0)
        np.testing.assert_allclose(
            actual_rotation, np.repeat(matrices[:, None], 55, axis=1), atol=1e-10
        )
        np.testing.assert_allclose(positions, target, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
