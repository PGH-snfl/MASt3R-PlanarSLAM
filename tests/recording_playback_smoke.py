"""Read a real SDK .db3 without a camera; compare decoded frame counts to SQLite.

Usage: python tests/recording_playback_smoke.py /path/to/recording.db3
Does not modify the recording or run model inference.
"""
import argparse
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r_slam.recorded_rgbd import RecordedRGBDSource, RecordingEnded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    args = parser.parse_args()
    path = args.recording.resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        expected = dict(db.execute('''SELECT t.name, count(*) FROM messages m
          JOIN topics t ON t.id=m.topic_id WHERE t.type='sensor_msgs/msg/Image'
          GROUP BY t.id''').fetchall())
    source = RecordedRGBDSource(path)
    colors, depths = set(), set()
    previous = [None, None]
    backwards = [0, 0]
    started = time.monotonic()
    try:
        time.sleep(.5)  # Simulates slow model loading, longer than SDK buffering.
        while True:
            try:
                frames = source.next_frames(5000)
            except RecordingEnded:
                break
            for j, (frame, seen) in enumerate(((frames.get_color_frame(), colors),
                                              (frames.get_depth_frame(), depths))):
                if frame and frame.get_frame_number() not in seen:
                    seen.add(frame.get_frame_number())
                    timestamp = frame.get_timestamp()
                    backwards[j] += int(previous[j] is not None and timestamp <= previous[j])
                    previous[j] = timestamp
            if len(depths) and len(depths) % 2000 == 0:
                print('decoded', len(colors), len(depths), flush=True)
    finally:
        source.close()
    color_count = sum(n for name, n in expected.items() if '/Color_' in name)
    depth_count = sum(n for name, n in expected.items() if '/Depth_' in name)
    print(dict(recorded_rgb=color_count, decoded_rgb=len(colors), recorded_depth=depth_count,
               decoded_depth=len(depths), timestamp_errors=backwards,
               elapsed_s=round(time.monotonic()-started, 2)), flush=True)
    # A single unpaired boundary image may be omitted by the SDK synchronizer.
    assert color_count-1 <= len(colors) <= color_count
    assert depth_count-1 <= len(depths) <= depth_count
    assert backwards == [0, 0]
    print('PASS: sequential playback, slow consumer, stream counts and EOF')


if __name__ == '__main__':
    main()
