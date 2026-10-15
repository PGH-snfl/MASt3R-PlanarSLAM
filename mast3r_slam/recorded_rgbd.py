"""Sequential SDK playback with backpressure; never seeks or opens a camera."""
from pathlib import Path
import time
import pyrealsense2 as rs


class RecordingEnded(EOFError):
    pass


class RecordedRGBDSource:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        self.pipeline = rs.pipeline()
        self.running = False
        cfg = rs.config()
        cfg.enable_device_from_file(str(self.path), repeat_playback=False)
        cfg.enable_stream(rs.stream.color)
        cfg.enable_stream(rs.stream.depth)
        profile = cfg.resolve(rs.pipeline_wrapper(self.pipeline))
        # Set before start: loading weights/slow processing must not drop input.
        profile.get_device().as_playback().set_real_time(False)
        try:
            self.profile = self.pipeline.start(cfg)
            self.running = True
            self.playback = self.profile.get_device().as_playback()
            self.playback.set_real_time(False)
            self.duration_s = self.playback.get_duration().total_seconds()
        except BaseException:
            self.close()
            raise

    def next_frames(self, timeout_ms):
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            ready, frames = self.pipeline.try_wait_for_frames(100)
            # Drain buffered frames before interpreting the stopped status.
            if ready:
                return frames
            if self.playback.current_status() == rs.playback_status.stopped:
                raise RecordingEnded(str(self.path))
        raise TimeoutError(f'Recording playback stalled: {self.path}')

    def close(self):
        if self.running:
            self.running = False
            self.pipeline.stop()
