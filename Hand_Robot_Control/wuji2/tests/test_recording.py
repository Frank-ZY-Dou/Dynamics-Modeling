"""Recording tests use fake cameras and generated video, never physical devices."""

import importlib.util
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wuji2_control.recording import (
    CameraRecorder,
    RecordingError,
    _atomic_json,
    _capture_frames,
    _resample_indices,
    export_video,
)


class FakeFrame:
    shape = (4, 6, 3)
    dtype = SimpleNamespace(name="uint8")

    def tobytes(self):
        return b"x" * 72


class FakeCapture:
    def __init__(self, stop, *, fail_read=False):
        self.stop, self.fail_read = stop, fail_read
        self.released = False
        self.reading = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        self.reading = True
        self.stop.set()
        self.reading = False
        return (False, None) if self.fail_read else (True, FakeFrame())

    def release(self):
        assert not self.reading, "A capture cannot be released while read() runs"
        self.released = True


class FakeWriter:
    def __init__(self, *, opened=True, fail_write=False):
        self.opened, self.fail_write = opened, fail_write
        self.released = False
        self.path = None

    def isOpened(self):
        return self.opened

    def write(self, _):
        if self.fail_write:
            raise OSError("disk write failed")
        with self.path.open("ab") as stream:
            stream.write(b"frame")

    def get(self, _):
        return 5.0

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    CAP_OPENCV_MJPEG = 6
    VIDEOWRITER_PROP_FRAMEBYTES = 7

    def __init__(self, capture, writer):
        self.capture, self.writer = capture, writer

    def VideoCapture(self, _):
        return self.capture

    def VideoWriter(self, path, *_):
        self.writer.path = Path(path)
        self.writer.path.touch()
        return self.writer

    def VideoWriter_fourcc(self, *_):
        return 0

    def imencode(self, *_):
        return True, FakeFrame()


class FakeProcess:
    def __init__(self, alive=True):
        self.alive, self.exitcode = alive, None if alive else 0
        self.terminated = self.closed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        assert timeout <= 2

    def terminate(self):
        self.terminated, self.alive, self.exitcode = True, False, -15

    def close(self):
        self.closed = True


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_worker(self, *, fail_read=False, opened=True, fail_write=False):
        stop = threading.Event()
        capture = FakeCapture(stop, fail_read=fail_read)
        writer = FakeWriter(opened=opened, fail_write=fail_write)
        cv2 = FakeCV2(capture, writer)
        _capture_frames(cv2, "fake", self.directory, 6, 4, 30, stop)
        state = json.loads((self.directory / "camera_recording.json").read_text())
        return state, capture, writer

    def healthy_recorder(self):
        recorder = CameraRecorder("fake", self.directory)
        recorder._process = FakeProcess()
        recorder._stop = threading.Event()
        _atomic_json(
            self.directory / "camera_recording.json",
            {
                "status": "recording",
                "frame_count": 1,
                "last_frame_monotonic": time.monotonic(),
            },
        )
        return recorder

    def test_constructing_does_not_start_camera_or_process(self):
        with patch("wuji2_control.recording.multiprocessing.get_context") as context:
            recorder = CameraRecorder("fake", self.directory)
            context.assert_not_called()
            with self.assertRaises(RecordingError):
                recorder.check()
            recorder.close()

    def test_worker_finalizes_frames_and_releases_owned_resources(self):
        state, capture, writer = self.run_worker()
        self.assertEqual(state["status"], "complete")
        times = json.loads((self.directory / "camera_frame_times.json").read_text())
        journal = [
            json.loads(line)
            for line in (self.directory / "camera_frame_times.jsonl").read_text().splitlines()
        ]
        self.assertEqual(times, journal)
        self.assertEqual(len(times), 1)
        self.assertTrue(capture.released and writer.released)
        self.assertTrue((self.directory / "latest.jpg").exists())

    def test_read_failure_marks_recording_incomplete(self):
        state, capture, _ = self.run_worker(fail_read=True)
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("stopped supplying frames", state["error"])
        self.assertTrue(capture.released)
        self.assertEqual(state["frame_count"], 0)

    def test_unopened_writer_never_announces_ready(self):
        state, capture, writer = self.run_worker(opened=False)
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("writer is not open", state["error"])
        self.assertNotIn("last_frame_monotonic", state)
        self.assertTrue(capture.released and writer.released)

    def test_write_failure_does_not_count_an_unrecorded_frame(self):
        state, capture, writer = self.run_worker(fail_write=True)
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("disk write failed", state["error"])
        self.assertEqual(state["frame_count"], 0)
        self.assertTrue(capture.released and writer.released)

    def test_missing_encoded_frame_is_not_reported_as_healthy(self):
        with patch.object(FakeWriter, "get", return_value=0):
            state, _, _ = self.run_worker()
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("did not encode", state["error"])
        self.assertEqual(state["frame_count"], 0)

    def test_silent_disk_stall_exceeding_buffer_allowance_is_rejected(self):
        with patch.object(FakeWriter, "get", return_value=2 * 1024 * 1024):
            with patch.object(FakeWriter, "write", return_value=None):
                state, _, _ = self.run_worker()
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("no disk output", state["error"])
        self.assertEqual(state["frame_count"], 0)

    def test_check_detects_stale_frames(self):
        recorder = self.healthy_recorder()
        recorder.check()
        state = recorder._state()
        state["last_frame_monotonic"] -= 1
        _atomic_json(self.directory / "camera_recording.json", state)
        with self.assertRaisesRegex(RecordingError, "stale"):
            recorder.check()

    def test_check_detects_exited_worker_despite_recent_frame(self):
        recorder = self.healthy_recorder()
        recorder._process.alive = False
        with self.assertRaisesRegex(RecordingError, "exited"):
            recorder.check()

    def test_check_detects_writer_error_despite_recent_frame(self):
        recorder = self.healthy_recorder()
        state = recorder._state()
        state["error"] = "Video writer rejected a frame"
        _atomic_json(self.directory / "camera_recording.json", state)
        with self.assertRaisesRegex(RecordingError, "writer rejected"):
            recorder.check()

    def test_hung_worker_is_terminated_without_parent_releasing_capture(self):
        recorder = self.healthy_recorder()
        with self.assertRaisesRegex(RecordingError, "did not finalize"):
            recorder.close()
        self.assertTrue(recorder._process.terminated)
        self.assertTrue(recorder._process.closed)
        self.assertTrue(recorder._stop.is_set())
        self.assertEqual(recorder._state()["status"], "incomplete")
        recorder.close()

    def test_snapshot_rejects_paths_and_existing_files(self):
        recorder = self.healthy_recorder()
        (self.directory / "latest.jpg").write_bytes(b"jpeg")
        for name in ("../escape", "/absolute", "a/b", "a\\b", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                recorder.snapshot(name)
        path = recorder.snapshot("open_palm")
        self.assertEqual(path.read_bytes(), b"jpeg")
        with self.assertRaises(FileExistsError):
            recorder.snapshot("open_palm")


class TimingTests(unittest.TestCase):
    def test_ten_fps_camera_retains_elapsed_time_at_thirty_fps_export(self):
        self.assertEqual(_resample_indices([0, 0.1, 0.2], 30), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_irregular_frames_hold_until_next_acquisition(self):
        self.assertEqual(_resample_indices([0, 0.1, 0.4], 10), [0, 1, 1, 1, 2, 2])

    def test_absolute_monotonic_clock_offset_does_not_set_video_duration(self):
        self.assertEqual(_resample_indices([1234, 1234.25, 1234.5], 4), [0, 1, 2])

    def test_clock_offset_does_not_shift_a_frame_boundary(self):
        self.assertEqual(
            _resample_indices([100000, 100000.1, 100000.2], 30),
            _resample_indices([0, 0.1, 0.2], 30),
        )

    def test_invalid_timestamps_are_rejected(self):
        for values in ([], [0], [0, 0], [1, 0], [0, float("nan")], [0, float("inf")]):
            with self.subTest(values=values), self.assertRaises(RecordingError):
                _resample_indices(values, 30)

    def test_invalid_export_rate_is_rejected(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                _resample_indices([0, 1], rate)


@unittest.skipUnless(
    importlib.util.find_spec("cv2") and shutil.which("ffmpeg"),
    "Generated-video tests require OpenCV and ffmpeg",
)
class ExportTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        writer = cv2.VideoWriter(
            str(self.directory / "camera.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        try:
            for value in (20, 80, 140, 200):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        _atomic_json(self.directory / "camera_frame_times.json", [0, 0.1, 0.2, 0.3])
        _atomic_json(
            self.directory / "camera_recording.json", {"status": "complete", "frame_count": 4}
        )

    def test_export_resamples_generated_video_and_refuses_overwrite(self):
        import cv2

        output = export_video(self.directory, self.directory / "measured.mp4")
        video = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 12)
            self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 30)
            means = []
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                means.append(float(frame.mean()))
            self.assertEqual(len(means), 12)
            for index, mean in enumerate(means):
                self.assertAlmostEqual(mean, (20, 80, 140, 200)[index // 3], delta=5)
        finally:
            video.release()
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            export_video(self.directory, output)
        self.assertEqual(output.read_bytes(), original)

    def test_header_timestamp_mismatch_does_not_publish_output(self):
        _atomic_json(self.directory / "camera_frame_times.json", [0, 0.1, 0.2])
        _atomic_json(
            self.directory / "camera_recording.json", {"status": "complete", "frame_count": 3}
        )
        output = self.directory / "bad.mp4"
        with self.assertRaisesRegex(RecordingError, "counts differ"):
            export_video(self.directory, output)
        self.assertFalse(output.exists())

    def test_decode_failure_rejects_video_even_when_header_count_matches(self):
        capture = SimpleNamespace(
            isOpened=lambda: True,
            get=lambda _: 4,
            release=lambda: None,
        )
        frames = iter([(True, FakeFrame()), (True, FakeFrame()), (False, None)])
        capture.read = lambda: next(frames)
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FRAME_COUNT=7)
        output = self.directory / "truncated.mp4"
        with patch("wuji2_control.recording.importlib.import_module", return_value=cv2):
            with self.assertRaisesRegex(RecordingError, "ended before"):
                export_video(self.directory, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.directory.glob(".wuji2-export-*")), [])

    def test_incomplete_recording_cannot_be_exported(self):
        _atomic_json(
            self.directory / "camera_recording.json", {"status": "incomplete", "frame_count": 4}
        )
        with self.assertRaisesRegex(RecordingError, "incomplete"):
            export_video(self.directory, self.directory / "bad.mp4")

    def test_low_detail_recording_allows_avi_buffering(self):
        import cv2
        import numpy as np

        directory = self.directory / "buffered"
        directory.mkdir()
        stop = threading.Event()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        frame_number = [0]
        capture = SimpleNamespace(isOpened=lambda: True, set=lambda *_: True, release=lambda: None)

        def read():
            frame_number[0] += 1
            if frame_number[0] >= 30:
                stop.set()
            return True, image

        capture.read = read
        fake_cv2 = SimpleNamespace(
            **{
                name: getattr(cv2, name)
                for name in (
                    "CAP_PROP_FOURCC",
                    "CAP_PROP_FRAME_WIDTH",
                    "CAP_PROP_FRAME_HEIGHT",
                    "CAP_PROP_FPS",
                    "CAP_PROP_BUFFERSIZE",
                    "CAP_OPENCV_MJPEG",
                    "VIDEOWRITER_PROP_FRAMEBYTES",
                    "VideoWriter",
                    "VideoWriter_fourcc",
                    "imencode",
                )
            },
            VideoCapture=lambda _: capture,
        )
        # Three measured seconds of a 10 fps, nearly uniform stream. The real
        # writer buffers all frames until release(), which is healthy behavior.
        with patch(
            "wuji2_control.recording.time.monotonic", side_effect=lambda: frame_number[0] / 10
        ):
            _capture_frames(fake_cv2, "fake", directory, 64, 48, 30, stop)
        state = json.loads((directory / "camera_recording.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["frame_count"], 30)


if __name__ == "__main__":
    unittest.main()
