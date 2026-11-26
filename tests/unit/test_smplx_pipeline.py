"""Check legacy-fit preflight timing, missing observations and input preservation."""

import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.optitrack.c3d import load_c3d_module
from utils.paths import MOSHPP_DIR
from utils.smplx.pipeline import preflight, run_legacy_fit


class SmplxPipelineTests(unittest.TestCase):
    def test_legacy_skin_offsets_are_accepted_without_changing_the_layout(self):
        value = json.loads(self.layout.read_text())
        value["markersets"][0]["distance_from_skin"] = 0.0002
        self.layout.write_text(json.dumps(value))
        before = self.layout.read_bytes()
        report = preflight(
            self.capture,
            self.layout,
            model_dir=self.folder / "missing_models",
            subject_prefix="Body",
        )
        self.assertEqual(report["assigned_markers"], ["HeadTop"])
        self.assertEqual(self.layout.read_bytes(), before)

    def test_support_data_lookup_handles_nested_src_directories(self):
        path = MOSHPP_DIR / "src/moshpp/tools/support_data.py"
        spec = importlib.util.spec_from_file_location("_moshpp_support_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "src/moshpp/src/moshpp/tools/file.py"
            source.parent.mkdir(parents=True)
            source.touch()
            (project / "support_data").mkdir()
            expected = project / "src/moshpp/support_data"
            expected.mkdir()
            self.assertEqual(Path(module.get_support_data_dir(source)), expected)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.capture = self.folder / "capture.c3d"
        c3d = load_c3d_module(MOSHPP_DIR / "src/moshpp/tools/c3d.py")
        frames = []
        for index in range(20):
            point = np.array([[100.0, 200.0, 1000.0, 0.0, 1.0]])
            if index == 7:
                point[0, 3:] = -1
            frames.append((point, np.empty((0, 0))))
        writer = c3d.Writer(point_rate=120, point_units="mm")
        writer.add_frames(frames)
        with warnings.catch_warnings(), self.capture.open("wb") as stream:
            warnings.simplefilter("ignore")
            writer.write(stream, ["Body:HeadTop"])
        self.layout = self.folder / "layout.json"
        self.layout.write_text(
            json.dumps(
                {
                    "surface_model_type": "smplx",
                    "markersets": [
                        {
                            "type": "body",
                            "distance_from_skin": 0,
                            "indices": {"HeadTop": 100, "HeadFront": None},
                        }
                    ],
                }
            )
        )

    def test_preflight_retains_source_frame_mapping_and_missing_marker_coverage(self):
        report = preflight(
            self.capture,
            self.layout,
            model_dir=self.folder / "missing_models",
            subject_prefix="Body",
            mode="full",
            sample_step=4,
        )
        self.assertEqual(report["source_units"], "mm")
        self.assertEqual(report["source_frame_rate"], 120)
        self.assertEqual(report["output_frame_rate"], 30)
        self.assertEqual(report["source_sample_rows"], [0, 4, 8, 12, 16])
        self.assertEqual(report["source_frame_numbers"], [1, 5, 9, 13, 17])
        np.testing.assert_allclose(report["source_relative_time_s"], np.arange(5) / 30)
        self.assertAlmostEqual(report["marker_valid_fraction"]["HeadTop"], 0.95)
        self.assertEqual(report["stagei_eligible_frames"], 19)
        self.assertEqual(report["unassigned_markers"], ["HeadFront"])
        self.assertEqual(report["missing_markers"], ["HeadFront"])
        self.assertFalse(report["ready"])
        self.assertFalse(report["fit_started"])

    def test_unready_fit_writes_diagnostics_without_changing_layout_or_claiming_results(
        self,
    ):
        before = self.layout.read_bytes()
        output = self.folder / "output"
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            run_legacy_fit(
                self.capture,
                self.layout,
                output,
                model_dir=self.folder / "missing_models",
                subject_prefix="Body",
            )
        self.assertEqual(self.layout.read_bytes(), before)
        report = json.loads((output / "preflight_report.json").read_text())
        self.assertFalse(report["fit_started"])
        self.assertFalse((output / "fitting_report.json").exists())
        self.assertEqual(list(output.glob("*.npz")), [])


if __name__ == "__main__":
    unittest.main()
