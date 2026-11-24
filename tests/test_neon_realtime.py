import datetime as dt
import json
import struct
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from neon_realtime.receiver import (
    ExportCompatibleImuWriter,
    JsonlWriter,
    build_parser,
    to_jsonable,
    validate_args,
)


class JsonConversionTests(unittest.TestCase):
    def test_nested_named_tuples_are_serialized(self) -> None:
        Point = namedtuple("Point", "x y")
        Datum = namedtuple("Datum", "point worn timestamp")
        value = Datum(Point(12.5, 8.0), True, dt.datetime(2026, 1, 2, tzinfo=dt.UTC))

        self.assertEqual(
            to_jsonable(value),
            {
                "point": {"x": 12.5, "y": 8.0},
                "worn": True,
                "timestamp": "2026-01-02T00:00:00+00:00",
            },
        )

    def test_non_finite_float_becomes_null(self) -> None:
        self.assertIsNone(to_jsonable(float("nan")))

    def test_jsonl_writer_adds_envelope(self) -> None:
        Gaze = namedtuple("Gaze", "x y worn")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gaze.jsonl"
            with JsonlWriter(path) as writer:
                writer.write("gaze", Gaze(10.0, 20.0, True))
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["stream"], "gaze")
        self.assertEqual(record["data_type"], "Gaze")
        self.assertEqual(record["data"]["x"], 10.0)
        self.assertIsInstance(record["received_at_unix_ns"], int)

    def test_export_compatible_imu_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            with ExportCompatibleImuWriter(output) as writer:
                writer.write(
                    b"\x08\x01",
                    1_234_567_890,
                    (1.0, 2.0, 3.0),
                    (4.0, 5.0, 6.0),
                    (0.7, 0.1, 0.2, 0.3),
                )

            decoded = (output / "imu" / "imu ps1.raw").read_bytes()
            timestamp = (output / "imu" / "imu ps1.time").read_bytes()
            encoded = (output / "imu" / "extimu ps1.raw").read_bytes()
            record = json.loads((output / "imu.jsonl").read_text(encoding="utf-8"))

        values = struct.unpack("<q10f", decoded)
        self.assertEqual(len(decoded), 48)
        self.assertEqual(values[0], 1_234_567_890)
        self.assertEqual(struct.unpack("<q", timestamp)[0], 1_234_567_890)
        self.assertEqual(encoded, b"\x02\x00\x08\x01")
        self.assertEqual(record["data"]["timestamp_unix_ns"], 1_234_567_890)
        self.assertAlmostEqual(record["data"]["quaternion"]["w"], 0.7)


class CliTests(unittest.TestCase):
    def test_default_streams(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        validate_args(parser, args)
        self.assertEqual(args.streams, ["gaze", "imu"])
        self.assertEqual(args.imu_mode, "direct")

    def test_multiple_streams(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--streams", "gaze", "imu", "events", "scene"])
        validate_args(parser, args)
        self.assertIn("scene", args.streams)


if __name__ == "__main__":
    unittest.main()

