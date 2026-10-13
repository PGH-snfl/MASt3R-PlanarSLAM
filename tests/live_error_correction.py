"""Bounded actual-camera check: dense backend, then stable global-graph drain.

The first 12 frames request keyframes, after which normal keyframe rules apply.
Only metrics/events are kept; temporary maps and images are removed on exit.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_d457
from mast3r_slam.tracker import FrameTracker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-viz', action='store_true')
    args=parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    track=FrameTracker.track
    def tracked(self, frame):
        self.cfg['match_frac_thresh']=1.01 if frame.frame_id < 12 else .333
        return track(self, frame)
    with tempfile.TemporaryDirectory(prefix='planar-correction-live-') as root:
        session=Path(root)/'map'
        argv=['run_d457.py','--max-frames','32','--profile','--archive-dir',str(session)]
        if args.no_viz: argv.append('--no-viz')
        with patch.object(sys,'argv',argv), patch.object(FrameTracker,'track',tracked):
            run_d457.main()
        rows=[json.loads(x) for x in (session/'performance.jsonl').read_text().splitlines()]
        events=[json.loads(x) for x in (session/'events.jsonl').read_text().splitlines()]
        summary=json.loads((session/'map/summary.json').read_text())
        assert len(rows)==32
        assert all(r['mode']=='TRACKING' for r in rows)
        assert summary['keyframes']>8, summary
        for f in (session/'keyframes').glob('*.pt'):
            data=torch.load(f,weights_only=True,map_location='cpu')
            assert data['depth'].shape==(480,640)
            assert data['X_canon'].shape==(307200,3)
            sensor=data['depth_source']==1
            delta=data['surface_depth_delta']
            plane=data['plane_depth_delta']
            expected=data['raw_depth']+(0 if plane is None else plane)+(0 if delta is None else delta)
            torch.testing.assert_close(data['depth'][sensor],expected[sensor])
            assert (data['depth'][sensor]-data['raw_depth'][sensor]).abs().max()<=.010001
            torch.testing.assert_close(data['X_canon'][:,2].reshape(480,640),data['depth'])
            if data['surface_depth_delta'] is not None:
                assert data['surface_depth_delta'].abs().max()<=.004001
        report=dict(summary=summary,rows=rows,
                    events=[e for e in events if e['event'].startswith('global_')],
                    note='Real camera, first 12 frames force keyframes; no ground-truth accuracy claim.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print('CORRECTION_EVENTS',json.dumps(report['events']))
        print('PASS: actual RGB-D, tracking, surface correction, archived geometry and bounded keyframes')


if __name__=='__main__': main()
