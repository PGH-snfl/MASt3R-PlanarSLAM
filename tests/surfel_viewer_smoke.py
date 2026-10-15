"""Render a user's saved map read-only, capture the viewport, then close our window.

Usage: python tests/surfel_viewer_smoke.py SESSION --mode surfels --output /tmp/view.png
Requires a working desktop OpenGL session. Does not start SLAM or a camera.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--mode', choices=('points', 'surfels'), default='surfels')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--default-view', action='store_true', help='Test the actual startup view without a fixed test camera')
    args = parser.parse_args()
    import view_saved_map
    from mast3r_slam.visualization import Window
    from mast3r_slam.frame import Mode
    session = args.session.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    started = time.monotonic()
    rendered = []
    original_init = Window.__init__
    original_render = Window._render_locked
    # Fixed first-camera viewpoint allows comparing an actively growing map.
    T = np.array(json.loads((session/'map/poses.json').read_text())['matrices'][0], np.float32)
    T[:3, :3] /= np.linalg.norm(T[:3, 0])
    offset = np.eye(4, dtype=np.float32);offset[2, 3] = -1.
    camera = np.diag([1., -1., -1., 1.]).astype('f4') @ np.linalg.inv(T @ offset)
    def init(window, *a, **kw):
        original_init(window, *a, **kw)
        if not args.default_view:
            window.follow_cam = False
            window.camera.T_CW = camera
            window.show_keyframe = window.show_axis = False
    def render(window, *a, **kw):
        begin = time.monotonic()
        original_render(window, *a, **kw)
        if time.monotonic()-started > 45:
            window.states.set_mode(Mode.TERMINATED)
            return
        ready = window.disk_map_count > 0
        if args.mode == 'surfels':
            ready &= len(window.disk_surfel_textures) > 0
            assert len(window.disk_surfel_textures) <= 12
        if ready:
            window.ctx.finish()
            rendered.append((time.monotonic()-begin)*1000)
        if len(rendered) == 15:
            screen = window.viewport.screen
            width, height = screen.size
            pixels = np.frombuffer(screen.read(components=3, alignment=1), np.uint8).reshape(height, width, 3)
            assert pixels.std() > 5, 'Viewport appears empty'
            Image.fromarray(pixels[::-1].copy()).save(args.output)
            report.update(mode=args.mode, session=str(session), stats=window.disk_map_stats,
                          texture_keyframes=len(window.disk_surfel_textures),
                          texture_bytes=sum(t.size[0]*t.size[1]*t.components*4
                                            for textures in window.disk_surfel_textures.values() for t in textures),
                          render_median_ms=float(np.median(rendered[1:])),
                          gl_error=window.ctx.error, screenshot=str(args.output))
            window.states.set_mode(Mode.TERMINATED)
    with patch.object(Window, '__init__', init), patch.object(Window, '_render_locked', render), \
         patch.object(Window, 'window_size', (1280, 720)), \
         patch.object(sys, 'argv', ['view_saved_map.py', str(session), '--render-mode', args.mode]):
        view_saved_map.main()
    assert report and report['gl_error'] == 'GL_NO_ERROR', report
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print('PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
