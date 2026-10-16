"""CPU media fixtures for the successor MP4 validator; no model or GPU calls."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imageio
import numpy as np

from chaoyang.pipeline import propainter_mp4_acceptance_v2 as mp4


class ProPainterMp4AcceptanceV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        cls.root = Path(cls.temp.name)
        image = np.zeros((720, 960, 3), np.uint8)
        for count in (131, 284):
            with imageio.get_writer(str(cls.root / f"{count}.mp4"), fps=30, quality=7) as writer:
                for _ in range(count):
                    writer.append_data(image)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_normal_131_and_284_have_exact_stream_ticks_and_quantized_movie_duration(self):
        for count in (131, 284):
            with self.subTest(count=count):
                result = mp4.verify_propainter_mp4(self.root / f"{count}.mp4", count, 30)
                self.assertEqual(result["decoded_frames"], count)
                self.assertEqual(result["stream_time_base"], "1/15360")
                self.assertEqual(result["frame_period_ticks"], 512)
                self.assertEqual(result["stream_duration_ticks"], count * 512)
                self.assertEqual(result["movie_timescale"], 1000)
                self.assertEqual(result["movie_duration_ticks"],
                                 4367 if count == 131 else 9467)
                self.assertEqual(result["time_scope"],
                                 "ENCODED_CFR_DISPLAY_ONLY_NOT_CAPTURE_TIME")
                self.assertEqual(mp4.verify_propainter_mp4(
                    self.root / f"{count}.mp4", count, 30,
                    expected_ref=result["artifact"])["artifact"], result["artifact"])

    def test_missing_or_extra_frame_is_rejected(self):
        for count, expected in ((131, 130), (131, 132), (284, 283), (284, 285)):
            with self.subTest(count=count, expected=expected):
                with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "STREAM_OR_FRAME_COUNT"):
                    mp4.verify_propainter_mp4(self.root / f"{count}.mp4", expected, 30)

    def test_wrong_fps_and_noninteger_period_are_rejected(self):
        path = self.root / "131.mp4"
        with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FRAME_RATE"):
            mp4.verify_propainter_mp4(path, 131, 25)
        record = mp4._probe(path)
        record["streams"][0]["time_base"] = "1/1000"
        with patch.object(mp4, "_probe", return_value=record):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FRAME_PERIOD_NOT_INTEGER_TICKS"):
                mp4.verify_propainter_mp4(path, 131, 30)

    def test_bad_frame_pts_and_frame_duration_are_rejected(self):
        path = self.root / "131.mp4"
        original = mp4._probe(path)
        wrong = copy.deepcopy(original)
        wrong["frames"][75]["best_effort_timestamp"] += 1
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FRAME_PTS:75"):
                mp4.verify_propainter_mp4(path, 131, 30)
        wrong = copy.deepcopy(original)
        wrong["frames"][75]["best_effort_timestamp"] = wrong["frames"][74]["best_effort_timestamp"]
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FRAME_PTS:75"):
                mp4.verify_propainter_mp4(path, 131, 30)
        wrong = copy.deepcopy(original)
        wrong["frames"][75]["pkt_duration"] += 1
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FRAME_DURATION:75"):
                mp4.verify_propainter_mp4(path, 131, 30)

    def test_bad_stream_or_movie_duration_is_rejected(self):
        path = self.root / "131.mp4"
        original = mp4._probe(path)
        wrong = copy.deepcopy(original)
        wrong["streams"][0]["duration_ts"] -= 512
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "STREAM_DURATION_TICKS"):
                mp4.verify_propainter_mp4(path, 131, 30)
        wrong = copy.deepcopy(original)
        wrong["streams"][0]["duration"] = "4.400000"
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "STREAM_DURATION_DECIMAL"):
                mp4.verify_propainter_mp4(path, 131, 30)
        with patch.object(mp4, "_movie_header", return_value=(1000, 4368)):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "MOVIE_DURATION_QUANTIZATION"):
                mp4.verify_propainter_mp4(path, 131, 30)
        wrong = copy.deepcopy(original)
        wrong["format"]["duration"] = "4.368000"
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FORMAT_DURATION_DECIMAL"):
                mp4.verify_propainter_mp4(path, 131, 30)
        wrong = copy.deepcopy(original)
        wrong["format"]["duration"] = "4.4"
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "FORMAT_DURATION_DECIMAL"):
                mp4.verify_propainter_mp4(path, 131, 30)

    def test_additional_stream_is_rejected(self):
        path = self.root / "131.mp4"
        wrong = mp4._probe(path)
        wrong["streams"].append(copy.deepcopy(wrong["streams"][0]))
        with patch.object(mp4, "_probe", return_value=wrong):
            with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "STREAM_OR_FRAME_COUNT"):
                mp4.verify_propainter_mp4(path, 131, 30)

    def test_truncation_or_sha_drift_is_rejected(self):
        path = self.root / "131.mp4"
        truncated = self.root / "truncated.mp4"
        payload = path.read_bytes()
        truncated.write_bytes(payload[:len(payload) // 2])
        with self.assertRaises(mp4.Mp4AcceptanceError):
            mp4.verify_propainter_mp4(truncated, 131, 30)
        result = mp4.verify_propainter_mp4(path, 131, 30)
        wrong_ref = dict(result["artifact"], sha256="0" * 64)
        with self.assertRaisesRegex(mp4.Mp4AcceptanceError, "MP4_SHA_OR_BYTES_DRIFT"):
            mp4.verify_propainter_mp4(path, 131, 30, expected_ref=wrong_ref)


if __name__ == "__main__":
    unittest.main()
