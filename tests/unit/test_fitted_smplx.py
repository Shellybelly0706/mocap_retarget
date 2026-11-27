"""Check exact source-frame mapping for fitted SMPL-X paired playback."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.smplx.fitted import match_source_frames, register_fitted_world


class FittedSmplxTests(unittest.TestCase):
    def test_exact_source_frames_preserve_requested_order_and_repeats(self):
        indices = match_source_frames(
            np.array([1, 5, 9, 13]), np.array([1, 9, 9, 13])
        )
        np.testing.assert_array_equal(indices, [0, 2, 2, 3])

    def test_missing_source_frames_are_not_silently_interpolated(self):
        with self.assertRaisesRegex(ValueError, r"5.*1 missing"):
            match_source_frames(np.array([1, 9]), np.array([1, 5, 9]))

    def test_duplicate_fitted_source_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            match_source_frames(np.array([1, 5, 5]), np.array([1, 5]))

    def test_world_registration_recovers_export_axis_rotation(self):
        fitted = np.array(
            [
                [[-0.6, -2.5, 1.2], [-0.9, -2.4, 1.6], [-0.7, -2.7, 1.0]],
                [[-0.5, -2.4, 1.3], [-0.8, -2.3, 1.7], [-0.6, -2.6, 1.1]],
            ]
            * 5
        )
        expected_rotation = np.diag([-1.0, -1.0, 1.0])
        expected_translation = np.array([0.01, -0.02, 0.03])
        reference = fitted @ expected_rotation.T + expected_translation
        rotation, translation, median, p95 = register_fitted_world(
            fitted,
            np.array(["A", "B", "C"]),
            reference,
            ["A", "B", "C"],
        )
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
        self.assertLess(median, 1e-12)
        self.assertLess(p95, 1e-12)


if __name__ == "__main__":
    unittest.main()
