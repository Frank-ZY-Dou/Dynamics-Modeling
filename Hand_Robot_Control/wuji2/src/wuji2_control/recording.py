"""Camera recording with a separate capture process and measured frame timing.

OpenCV and ffmpeg are needed only when recording or exporting. The capture
process owns every camera and writer handle; the controlling process never
releases a handle while a camera read may still be running.
"""

from __future__ import annotations

import importlib
import json
import math
import multiprocessing
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


class RecordingError(RuntimeError):
    """Recording is unavailable, unhealthy, or could not be finalized."""


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _capture_frames(
    cv2: Any,
    device: str,
    directory: Path,
    width: int,
    height: int,
    fps: float,
    stop: Any,
) -> None:
    """Worker implementation, accepting an OpenCV implementation for testing."""
    capture = writer = None
    timestamps: list[float] = []
    error: str | None = None
    dimensions: tuple[int, int] | None = None
    state: dict[str, Any] = {"status": "starting", "frame_count": 0}
    video = directory / "camera.avi"
    try:
        capture = cv2.VideoCapture(int(device) if device.isdecimal() else device)
        if not capture.isOpened():
            raise RecordingError(f"Cannot open camera {device}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_size, buffered_bytes = 0, 0.0
        with (directory / "camera_frame_times.jsonl").open("x", encoding="utf-8") as journal:
            while not stop.is_set():
                ok, frame = capture.read()
                acquired = time.monotonic()
                if not ok or frame is None:
                    raise RecordingError("Camera stopped supplying frames")
                shape = frame.shape
                if len(shape) != 3 or shape[2] != 3 or frame.dtype.name != "uint8":
                    raise RecordingError("Camera must supply uint8 BGR frames")
                size = (shape[1], shape[0])
                if writer is None:
                    dimensions = size
                    # This backend reports encoded bytes for each frame; the
                    # default FFmpeg backend exposes no write acknowledgement.
                    writer = cv2.VideoWriter(
                        str(video), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), fps, size
                    )
                if dimensions != size:
                    raise RecordingError("Camera frame dimensions changed during recording")
                if not writer.isOpened():
                    raise RecordingError("Video writer is not open")
                written = writer.write(frame)
                if written is False or not writer.isOpened():
                    raise RecordingError("Video writer rejected a frame")
                encoded_bytes = writer.get(cv2.VIDEOWRITER_PROP_FRAMEBYTES)
                if not math.isfinite(encoded_bytes) or encoded_bytes <= 0:
                    raise RecordingError("Video writer did not encode a frame")
                buffered_bytes += encoded_bytes
                # Low-detail images may take many seconds to fill an AVI
                # buffer. Allow one MiB of buffering instead of assuming that
                # healthy writers increase file size on a fixed time schedule.
                size_on_disk = video.stat().st_size if video.exists() else 0
                if size_on_disk > last_size:
                    last_size, buffered_bytes = size_on_disk, 0.0
                elif buffered_bytes > 1024 * 1024:
                    raise RecordingError(
                        "Video writer produced no disk output after encoding one MiB"
                    )
                timestamps.append(acquired)
                journal.write(json.dumps(acquired) + "\n")
                journal.flush()
                encoded, jpeg = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise RecordingError("Could not encode the camera preview")
                temporary = directory / "latest.jpg.tmp"
                temporary.write_bytes(jpeg.tobytes())
                temporary.replace(directory / "latest.jpg")
                state.update(
                    status="recording",
                    frame_count=len(timestamps),
                    last_frame_monotonic=acquired,
                    nominal_fps=fps,
                    width=size[0],
                    height=size[1],
                )
                _atomic_json(directory / "camera_recording.json", state)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for resource in (writer, capture):
            if resource is not None:
                try:
                    resource.release()
                except Exception as exc:
                    error = error or f"Could not release camera resources: {exc}"
        if not timestamps:
            error = error or "No camera frames were recorded"
        if not video.exists() or video.stat().st_size == 0:
            error = error or "The recorded video is empty"
        _atomic_json(directory / "camera_frame_times.json", timestamps)
        state.update(
            status="incomplete" if error else "complete", frame_count=len(timestamps), error=error
        )
        _atomic_json(directory / "camera_recording.json", state)


def _record_worker(
    device: str, directory: Path, width: int, height: int, fps: float, stop: Any
) -> None:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        _atomic_json(
            directory / "camera_recording.json",
            {
                "status": "incomplete",
                "frame_count": 0,
                "error": "OpenCV is required; install the camera extra",
            },
        )
        return
    _capture_frames(cv2, device, directory, width, height, fps, stop)


class CameraRecorder:
    """Record until closed, raising when capture or recording becomes unhealthy.

    Construction does not open the camera. Use this object as a context manager
    or call :meth:`start` explicitly. Call :meth:`check` during every control
    iteration. ``latest.jpg`` is replaced atomically after each recorded frame.
    A forced shutdown leaves ``camera_recording.json`` marked incomplete.
    """

    def __init__(
        self,
        device: str,
        output_dir: Path,
        width: int = 1280,
        height: int = 720,
        fps: float = 30,
        max_age: float = 0.75,
        startup_timeout: float = 4,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a nonempty string")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (width, height)):
            raise ValueError("width and height must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (fps, max_age, startup_timeout)):
            raise ValueError("fps, max_age, and startup_timeout must be finite and positive")
        self.device, self.output_dir = device, Path(output_dir)
        self.width, self.height, self.fps = width, height, fps
        self.max_age, self.startup_timeout = max_age, startup_timeout
        self._process: Any = None
        self._stop: Any = None
        self._closed = False

    def _state(self) -> dict[str, Any]:
        try:
            state = json.loads(
                (self.output_dir / "camera_recording.json").read_text(encoding="utf-8")
            )
            if not isinstance(state, dict):
                raise ValueError("expected a JSON object")
            return state
        except (OSError, ValueError) as exc:
            raise RecordingError(f"Cannot read camera recording status: {exc}") from exc

    def start(self) -> CameraRecorder:
        if self._closed or self._process is not None:
            raise RecordingError("A recorder can only be started once")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "camera.avi",
            "camera_frame_times.json",
            "camera_frame_times.jsonl",
            "camera_recording.json",
            "latest.jpg",
        ):
            if (self.output_dir / name).exists():
                raise FileExistsError(self.output_dir / name)
        # Exclusive creation also prevents two recorders from claiming one
        # session directory between the existence checks above.
        with (self.output_dir / "camera_recording.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "starting", "frame_count": 0}, stream)
        context = multiprocessing.get_context("spawn")
        self._stop = context.Event()
        self._process = context.Process(
            target=_record_worker,
            args=(
                self.device,
                self.output_dir,
                self.width,
                self.height,
                self.fps,
                self._stop,
            ),
            name="wuji2-camera",
            daemon=True,
        )
        try:
            self._process.start()
        except BaseException:
            self._closed = True
            _atomic_json(
                self.output_dir / "camera_recording.json",
                {
                    "status": "incomplete",
                    "frame_count": 0,
                    "error": "Camera worker could not be started",
                },
            )
            raise
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                state = self._state()
                if state.get("error"):
                    raise RecordingError(state["error"])
                if not self._process.is_alive():
                    raise RecordingError("Camera worker exited during startup")
                if state.get("status") == "recording":
                    self.check()
                    return self
                time.sleep(0.02)
            raise RecordingError("Camera supplied no recorded frame before the startup timeout")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> CameraRecorder:
        return self.start()

    def check(self) -> None:
        if self._closed or self._process is None:
            raise RecordingError("Camera recorder is not running")
        state = self._state()
        if state.get("error"):
            raise RecordingError(str(state["error"]))
        if not self._process.is_alive():
            raise RecordingError(f"Camera worker exited with code {self._process.exitcode}")
        if state.get("status") != "recording":
            raise RecordingError("Camera recording is not active")
        timestamp = state.get("last_frame_monotonic", float("nan"))
        age = time.monotonic() - timestamp
        if not math.isfinite(age) or age < 0 or age > self.max_age:
            raise RecordingError(f"Camera recording is stale ({age:.3f} seconds)")

    def snapshot(self, name: str) -> Path:
        """Copy the latest recorded frame to a new JPEG within the session."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Snapshot name must be a simple filename without path separators")
        self.check()
        filename = name if name.lower().endswith((".jpg", ".jpeg")) else name + ".jpg"
        destination = self.output_dir / filename
        data = (self.output_dir / "latest.jpg").read_bytes()
        with destination.open("xb") as stream:
            stream.write(data)
        return destination

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        self._stop.set()
        self._process.join(timeout=2)
        forced = self._process.is_alive()
        if forced:
            self._process.terminate()
            self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        try:
            state = self._state()
        except RecordingError:
            state = {"frame_count": 0}
        if forced or self._process.exitcode != 0 or state.get("status") != "complete":
            message = state.get("error") or "Camera worker did not finalize the recording"
            state.update(status="incomplete", error=message)
            _atomic_json(self.output_dir / "camera_recording.json", state)
            if not self._process.is_alive():
                self._process.close()
            raise RecordingError(message)
        self._process.close()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _resample_indices(timestamps: Sequence[float], fps: float) -> list[int]:
    """Hold each source frame until the next timestamp at a uniform output rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Export fps must be finite and positive")
    if len(timestamps) < 2 or any(not math.isfinite(t) for t in timestamps):
        raise RecordingError("At least two finite camera timestamps are required")
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if min(intervals) <= 0:
        raise RecordingError("Camera timestamps must be strictly increasing")
    relative = [t - timestamps[0] for t in timestamps]
    duration = relative[-1] + statistics.median(intervals)
    count = max(1, math.ceil(duration * fps - 1e-9))
    indices, source = [], 0
    for n in range(count):
        output_time = timestamps[0] + n / fps
        while source + 1 < len(timestamps) and timestamps[source + 1] <= output_time:
            source += 1
        indices.append(source)
    return indices


def export_video(recording_dir: Path, output: Path, fps: float = 30) -> Path:
    """Export a complete recording as H.264, preserving measured elapsed time.

    The final frame lasts the median source-frame interval. All source frames
    are decoded and checked against the timestamp count before publication.
    An existing output is never replaced, including one created during export.
    """
    recording_dir, output = Path(recording_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((recording_dir / "camera_recording.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RecordingError("Cannot export an incomplete recording")
    timestamps = json.loads((recording_dir / "camera_frame_times.json").read_text(encoding="utf-8"))
    indices = _resample_indices(timestamps, fps)
    if state.get("frame_count") != len(timestamps):
        raise RecordingError("Recording status and timestamp counts differ")
    encoder_path = shutil.which("ffmpeg")
    if encoder_path is None:
        raise RecordingError("ffmpeg is required to export video")
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RecordingError("OpenCV is required; install the camera extra") from exc
    capture = cv2.VideoCapture(str(recording_dir / "camera.avi"))
    temporary: Path | None = None
    encoder: Any = None
    try:
        if not capture.isOpened():
            raise RecordingError("Cannot open the recorded video")
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(timestamps):
            raise RecordingError("Video header and timestamp counts differ")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RecordingError("Cannot decode the first recorded frame")
        shape = frame.shape
        if len(shape) != 3 or shape[2] != 3 or frame.dtype.name != "uint8":
            raise RecordingError("Recorded video must decode to uint8 BGR frames")
        height, width = shape[:2]
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".wuji2-export-", suffix=".mp4", dir=output.parent)
        os.close(handle)
        temporary = Path(name)
        with tempfile.TemporaryFile() as errors:
            encoder = subprocess.Popen(
                [
                    encoder_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "bgr24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    format(fps, ".10g"),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
                stderr=errors,
            )
            assert encoder.stdin is not None
            source = 0
            for wanted in indices:
                while source < wanted:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RecordingError("Video ended before its timestamps")
                    if frame.shape != shape or frame.dtype.name != "uint8":
                        raise RecordingError("Video frame format changed")
                    source += 1
                encoder.stdin.write(frame.tobytes())
            # A low export rate may omit source frames at the end. Decode those
            # too, so count and format validation do not depend on output fps.
            while True:
                ok, remaining = capture.read()
                if not ok:
                    break
                if remaining is None or remaining.shape != shape or remaining.dtype.name != "uint8":
                    raise RecordingError("Video frame format changed")
                source += 1
            if source + 1 != len(timestamps):
                raise RecordingError("Decoded video and timestamp counts differ")
            encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                errors.seek(0)
                raise RecordingError(
                    "Video encoder failed: " + errors.read(4096).decode(errors="replace")
                )
        if temporary.stat().st_size == 0:
            raise RecordingError("Video encoder produced an empty file")
        os.link(temporary, output)
        return output
    except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
        raise RecordingError(f"Video encoder failed: {exc}") from exc
    finally:
        capture.release()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=2)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait(timeout=2)
        if encoder is not None and encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        if temporary is not None:
            temporary.unlink(missing_ok=True)
