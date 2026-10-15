"""Open a saved map in the existing viewer without camera or model inference."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_session(path=None):
    if path:
        session = Path(path).expanduser().resolve()
    else:
        candidates = sorted(p.parent.parent for p in (ROOT/'maps').glob('*/map/index.jsonl'))
        if not candidates:
            raise FileNotFoundError('No saved map under maps/. Run ./run_d457.sh first.')
        session = candidates[-1]
    if not (session/'map/index.jsonl').is_file():
        raise FileNotFoundError(f'No disk map index in {session}')
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', help='Session directory, default latest maps/ session')
    parser.add_argument('--inspect', action='store_true', help='Print stored map counts without opening a window')
    parser.add_argument('--render-mode', choices=('points', 'surfels'), default='points',
                        help='surfels: original shader on full keyframes plus auxiliary point cloud')
    parser.add_argument('--surfel-keyframes', type=int, default=12, help='Maximum full keyframes resident in the surfel viewer')
    parser.add_argument('--start-view', choices=('first', 'last'), default='first',
                        help='Initial saved camera viewpoint (default: first)')
    args = parser.parse_args()
    if not 1 <= args.surfel_keyframes <= 64:
        parser.error('--surfel-keyframes must be between 1 and 64')
    session = resolve_session(args.session)
    os.chdir(ROOT)
    from mast3r_slam.disk_map import DiskMapReader
    from mast3r_slam.config import config, load_config
    load_config(str(session/'config.yaml') if (session/'config.yaml').is_file() else 'config/d457_planar.yaml')
    config.setdefault('map_storage', {}).update(view_render_mode=args.render_mode,
                                              view_surfel_keyframes=args.surfel_keyframes)
    reader = DiskMapReader(session, config.get('map_storage', {}))
    try:
        reader.refresh()
        entries, pose_vectors = reader.entries, reader.pose_vectors
    finally:
        reader.close()
    if args.inspect:
        print(json.dumps(dict(session=str(session), frames=len(entries),
              keyframes=sum(e['kind'] == 'keyframe' for e in entries),
              nonkeyframes=sum(e['kind'] == 'nonkeyframe' for e in entries),
              saved_points=sum(e['points'] for e in entries)), indent=2))
        return
    keyframes = [e for e in entries if e['kind'] == 'keyframe' and
                 (session/'keyframes'/f"{e['keyframe_id']:08d}.pt").exists()]
    if not keyframes:
        raise RuntimeError('No fully saved keyframe is available yet; wait for the writer to finish.')
    import torch
    import torch.multiprocessing as mp
    import lietorch
    from mast3r_slam.keyframe_archive import unpack_frame
    from mast3r_slam.frame import SharedStates, SharedKeyframes, Mode
    from mast3r_slam.visualization import run_visualization
    entry = keyframes[0] if args.start_view == 'first' else keyframes[-1]
    payload = torch.load(session/'keyframes'/f"{entry['keyframe_id']:08d}.pt", map_location='cpu', weights_only=True)
    frame = unpack_frame(payload, 'cpu')
    # The .pt pose may predate a graph correction; the disk map pose table wins.
    if entry['keyframe_id'] < len(pose_vectors):
        frame.T_WC = lietorch.Sim3(torch.tensor(pose_vectors[entry['keyframe_id']], dtype=torch.float32))
    torch.set_num_threads(4)
    mp.set_start_method('spawn', force=True)
    h, w = map(int, frame.img_shape.reshape(-1).tolist())
    with mp.Manager() as manager:
        states = SharedStates(manager, h, w, device='cpu')
        view = SharedKeyframes(manager, h, w, buffer=1, device='cpu')
        view.set_intrinsics(frame.K)
        view.append(frame)
        view.map_directory = str(session)
        view.map_view_only = True
        states.set_frame(frame)
        states.set_mode(Mode.TRACKING)
        print(f'Viewing saved map: {session}', flush=True)
        print(f"Starting at saved keyframe {entry['keyframe_id']}; use Reset saved view if the map leaves the screen.", flush=True)
        run_visualization(config, states, view, manager.Queue(), manager.Queue())


if __name__ == '__main__':
    main()
