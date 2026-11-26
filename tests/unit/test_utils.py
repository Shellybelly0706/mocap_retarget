"""Regression checks for reusable helpers and their cross-module workflows."""

import contextlib
import csv
import io
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.avatars.gltf import read_accessor
from utils.markers.analysis import analyze
from utils.markers.conversion import convert_csv
from utils.markers.data import load_and_validate_npz
from utils.markers.npz import verify_npz, write_npz
from utils.markers.selection import select_stage1_candidates
from utils.optitrack.c3d import load_c3d, load_c3d_module
from utils.optitrack.skeleton import SKELETON_NODE_MARKERS, build_skeleton_motion
from utils.optitrack.visualization import load_capture, visualize_capture
from utils.paths import MOSHPP_DIR
from utils.smplx.kinematics import axis_angle_to_matrix, global_joint_motion
from utils.synchronization.alignment import fit_affine_map
from utils.synchronization.clocks import make_clock_converter, parse_timestamp_ns
from utils.synchronization.models import ClockAnchor, EventAnchor


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def csv_capture(self):
        path = self.folder / "capture.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows(
                [
                    ["Export Frame Rate", "120", "Length Units", "Millimeters"],
                    [],
                    ["Type", "", *(["Bone Marker"] * 6)],
                    ["Name", "", *(["Body:HeadTop"] * 3), *(["Body:HeadFront"] * 3)],
                    ["ID", "", "1", "1", "1", "2", "2", "2"],
                    ["", "", *(["Position"] * 6)],
                    ["Frame", "Time", "X", "Y", "Z", "X", "Y", "Z"],
                ]
            )
            for frame in range(60):
                writer.writerow(
                    [
                        frame,
                        frame / 120,
                        100 + frame,
                        200,
                        1000,
                        200 + frame,
                        "" if frame == 3 else 300,
                        1000,
                    ]
                )
        return path

    def c3d_capture(self):
        path = self.folder / "capture.c3d"
        c3d = load_c3d_module(MOSHPP_DIR / "src/moshpp/tools/c3d.py")
        labels = sorted(
            {label for labels in SKELETON_NODE_MARKERS.values() for label in labels}
        )
        points = np.zeros((len(labels), 5))
        points[:, :3] = [100, 200, 1000]
        points[:, 4] = 1
        missing = points.copy()
        missing[labels.index("HeadTop"), 3:] = -1
        writer = c3d.Writer(point_rate=120, point_units="mm")
        writer.add_frames([(points, np.empty((0, 0))), (missing, np.empty((0, 0)))])
        with warnings.catch_warnings(), path.open("wb") as stream:
            warnings.simplefilter("ignore")
            writer.write(stream, labels)
        return path

    def test_csv_conversion_analysis_and_selection_preserve_missing_samples(self):
        result = convert_csv(self.csv_capture(), "Body")
        np.testing.assert_allclose(result.markers[0, 0], [0.1, 0.2, 1.0])
        self.assertTrue(np.isnan(result.markers[3, 1]).all())
        output = self.folder / "capture.npz"
        write_npz(output, result)
        verify_npz(output, result)
        data = load_and_validate_npz(output)
        analysis = analyze(data, 0.1, 0.005, 0.01)
        candidates = select_stage1_candidates(data, analysis, 4, 0.0)
        self.assertEqual(analysis.summaries[1].longest_missing_frames, 1)
        self.assertGreater(len(candidates), 0)
        self.assertNotIn(3, [candidate.frame_index for candidate in candidates])

    def test_c3d_units_and_missing_markers_propagate_to_virtual_skeleton(self):
        motion = load_c3d(self.c3d_capture())
        np.testing.assert_allclose(motion.positions[0, 0], [0.1, 0.2, 1.0])
        skeleton = build_skeleton_motion(motion)
        head = skeleton.node_names.index("head")
        self.assertTrue(skeleton.valid[0, head])
        self.assertFalse(skeleton.valid[1, head])
        self.assertTrue(np.isnan(skeleton.positions[1, head]).all())

    def test_public_visualization_api_validates_both_formats_without_a_window(self):
        for path in (self.csv_capture(), self.c3d_capture()):
            with self.subTest(suffix=path.suffix), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                loaded = load_capture(path)
                self.assertEqual(loaded.path, path)
                visualize_capture(path, check_only=True)
                self.assertIn("Validation complete", output.getvalue())

    def test_entry_point_runs_outside_the_repository(self):
        path = self.csv_capture()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/visualize_optitrack.py"),
                str(path),
                "--check-only",
            ],
            cwd=self.folder,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Validation complete", result.stdout)

    def test_capture_api_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            load_capture(self.folder / "capture.txt")
        with self.assertRaises(FileNotFoundError):
            load_capture(self.folder / "missing.c3d")
        with self.assertRaises(ValueError):
            visualize_capture(self.csv_capture(), speed=0)


class ProcessingTests(unittest.TestCase):
    def test_clock_interpolation_preserves_nanosecond_precision(self):
        origin = 1_700_000_000_000_000_000
        anchors = [
            ClockAnchor(origin, origin + 1_000_000, True, 0),
            ClockAnchor(origin + 1000, origin + 1_001_100, True, 1),
        ]
        convert = make_clock_converter(
            anchors,
            source_attr="companion_ns",
            target_attr="ubuntu_ns",
            allow_extrapolation=False,
        )
        self.assertEqual(convert(origin + 500), origin + 1_000_550)
        with self.assertRaises(ValueError):
            convert(origin - 1)
        self.assertEqual(
            parse_timestamp_ns("2023-11-14T22:13:20.000000123Z"), origin + 123
        )

    def test_event_alignment_models_clock_drift(self):
        origin = 1_700_000_000_000_000_000
        events = [
            EventAnchor(
                frame, seconds, origin, origin + round(seconds * 1.001e9), "manual"
            )
            for frame, seconds in ((0, 0.0), (120, 1.0), (240, 2.0))
        ]
        mapping = fit_affine_map(events)
        self.assertAlmostEqual(mapping.slope, 1.001)
        self.assertEqual(mapping.ubuntu_ns(3.0), origin + 3_003_000_000)

    def test_joint_motion_composes_parent_rotation_and_translation(self):
        poses = np.zeros((1, 2, 3))
        poses[0, 0, 2] = np.pi / 2
        positions, rotations = global_joint_motion(
            poses,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([-1, 0]),
            np.array([[10.0, 20.0, 30.0]]),
        )
        np.testing.assert_allclose(positions[0, 1], [10.0, 21.0, 30.0], atol=1e-12)
        np.testing.assert_allclose(rotations[0, 1], axis_angle_to_matrix(poses[0, 0]))

    def test_gltf_accessor_handles_stride_and_normalized_values(self):
        document = {
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5121,
                    "type": "VEC2",
                    "count": 2,
                    "normalized": True,
                }
            ],
            "bufferViews": [{"byteOffset": 1, "byteStride": 3}],
        }
        values = read_accessor(document, bytes([99, 0, 255, 99, 127, 255, 99]), 0)
        np.testing.assert_allclose(values, [[0.0, 1.0], [127 / 255, 1.0]])

    def test_data_helpers_import_without_visualization_or_fitting_dependencies(self):
        code = """
import importlib, sys
for name in ('mujoco', 'cv2', 'torch', 'human_body_prior'):
    sys.modules[name] = None
for name in ('utils.optitrack.visualization', 'utils.markers.conversion',
             'utils.markers.analysis', 'utils.smplx.fitting',
             'utils.smplx.kinematics', 'utils.smplx.head_markers',
             'utils.synchronization.alignment', 'utils.avatars.gltf'):
    importlib.import_module(name)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n" + code,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
