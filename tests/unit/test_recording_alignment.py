"""Check timestamp boundaries, source identity, missing data and exact video crop."""

import csv
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.synchronization.motive_motion import parse_motive_start_time
from utils.synchronization.preprocessing import align_recording, nearest_samples


class RecordingAlignmentTests(unittest.TestCase):
    def test_nearest_preserves_nanosecond_precision_ties_and_bounds(self):
        origin = 1_795_690_000_000_000_000
        source = origin + np.array([0, 8, 20], dtype=np.int64)
        target = origin + np.array([-10, 4, 5, 14, 19, 25], dtype=np.int64)
        np.testing.assert_array_equal(
            nearest_samples(source, target), [0, 0, 1, 1, 2, 2]
        )
        with self.assertRaises(ValueError):
            nearest_samples(source[::-1], target)

    def test_global_crop_keeps_source_rows_missing_cells_and_video_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            neon = root / "native"
            neon.mkdir()
            video = neon / "Neon Scene Camera test.mp4"
            with av.open(str(video), "w") as writer:
                stream = writer.add_stream("libx264", rate=10)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                for index in range(8):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((32, 32, 3), index * 25, dtype=np.uint8), format="rgb24"
                    )
                    frame.pts = index
                    frame.time_base = Fraction(1, 10)
                    for packet in stream.encode(frame):
                        writer.mux(packet)
                for packet in stream.encode():
                    writer.mux(packet)
            start_text = "2026-11-26 11.50.56.621 AM"
            start = parse_motive_start_time(start_text, "Europe/Berlin")
            stamps = start + np.array(
                [
                    -150_000_000,
                    -50_000_000,
                    50_000_000,
                    151_000_000,
                    249_000_000,
                    350_000_000,
                    450_000_000,
                    550_000_000,
                ],
                dtype=np.int64,
            )
            stamps.astype("<i8").tofile(video.with_suffix(".time"))
            source = root / "motive.csv"
            header = [
                [
                    "Capture Start Time",
                    start_text,
                    "Length Units",
                    "Meters",
                    "Export Frame Rate",
                    "120",
                    "Total Exported Frames",
                    "4",
                ],
                [],
                [],
                [],
                [],
                [],
                ["Frame", "Time (Seconds)", "missing", "value"],
            ]
            rows = [
                ["0", "0.0", "", "100"],
                ["12", "0.1", "", "101"],
                ["24", "0.2", "", "102"],
                ["36", "0.3", "", "103"],
            ]
            with source.open("w", newline="") as stream:
                csv.writer(stream).writerows(header + rows)
            original = source.read_bytes()
            output = root / "aligned"
            report = align_recording(source, neon, output)
            self.assertEqual(report["frame_count"], 3)
            self.assertEqual(report["neon_source_frame_range"], [2, 4])
            self.assertEqual(report["clock_correction_ns"], 0)
            self.assertEqual(source.read_bytes(), original)
            with (output / "mocap_aligned.csv").open(newline="") as stream:
                exported = list(csv.reader(stream))[7:]
            self.assertEqual(exported, [rows[0], rows[2], rows[2]])
            with (output / "frame_alignment.csv").open(newline="") as stream:
                mapping = list(csv.DictReader(stream))
            self.assertEqual(
                [int(row["timestamp_ns"]) for row in mapping], stamps[2:5].tolist()
            )
            with av.open(str(output / "front_camera_aligned.mp4")) as reader:
                frames = list(reader.decode(video=0))
            self.assertEqual(len(frames), 3)
            np.testing.assert_allclose(
                [float(f.pts * f.time_base) for f in frames],
                [0, 0.101, 0.199],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                [f.to_ndarray(format="rgb24").mean() for f in frames],
                [50, 75, 100],
                atol=4,
            )
            self.assertEqual(
                json.loads((output / "alignment.json").read_text())["source_counts"],
                {"motive": 4, "neon": 8},
            )
            with self.assertRaises(FileExistsError):
                align_recording(source, neon, output)


if __name__ == "__main__":
    unittest.main()
