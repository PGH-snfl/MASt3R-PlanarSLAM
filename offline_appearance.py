"""Bounded offline RGB-D Gaussian appearance experiment. Never edits source maps.

Fixed saved camera poses, train-only point initialization, held-out RGB evaluation.
This is not a reproduction of Photo-SLAM and does not change live SLAM geometry.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def rigid_pose(matrix):
    """Sim3 local point s*R*x+t becomes SE3 R*(s*x)+t."""
    matrix = np.asarray(matrix, np.float64)
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid saved Sim3 scale')
    pose = matrix.copy()
    pose[:3, :3] /= scale
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-4):
        raise ValueError('Saved pose is not a scaled rotation')
    return pose.astype(np.float32), scale


def load_frames(session, width):
    poses = json.loads((session / 'map/poses.json').read_text())['matrices']
    frames = []
    for i, mat in enumerate(poses):
        p = torch.load(session / 'keyframes' / f'{i:08d}.pt', weights_only=True, map_location='cpu')
        pose, scale = rigid_pose(mat)
        rgb = p['uimg'].float()
        h, w = rgb.shape[:2]
        height = round(h * width / w)
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None], (height, width), mode='area')[0].permute(1, 2, 0)
        depth = F.interpolate(p['depth'][None, None].float(), (height, width), mode='nearest-exact')[0, 0] * scale
        source = F.interpolate(p['depth_source'][None, None].float(), (height, width), mode='nearest-exact')[0, 0]
        conf = F.interpolate(p['C'].reshape(h, w)[None, None], (height, width), mode='nearest-exact')[0, 0]
        K = p['K'].float().clone()
        K[0] *= width / w
        K[1] *= height / h
        K[0, 2] += (width / w - 1) * .5
        K[1, 2] += (height / h - 1) * .5
        frames.append(dict(id=i, rgb=rgb, depth=depth, source=source, conf=conf,
                           K=K, pose=torch.from_numpy(pose), view=torch.from_numpy(np.linalg.inv(pose)), scale=scale))
    return frames


def initial_cloud(frames, voxel, cap):
    points, colors, sizes, quats, scores = [], [], [], [], []
    for f in frames:
        d, K = f['depth'].numpy(), f['K'].numpy()
        h, w = d.shape
        yy, xx = np.mgrid[:h:2, :w:2]
        z = d[::2, ::2]
        valid = np.isfinite(z) & (z > .2) & (z < 8) & (f['conf'].numpy()[::2, ::2] >= 1.5)
        # Avoid depth discontinuities, where a broad splat can join unrelated surfaces.
        dy, dx = np.gradient(d)
        valid &= (np.abs(dx[::2, ::2]) + np.abs(dy[::2, ::2])) < .12
        xyz = np.stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z), -1)[valid]
        pose = f['pose'].numpy()
        points.append(xyz @ pose[:3, :3].T + pose[:3, 3])
        colors.append(f['rgb'].numpy()[::2, ::2][valid])
        footprint = np.clip(z[valid] / K[0, 0] * 1.4, .003, .05)
        sizes.append(np.stack((footprint, footprint, footprint * .3), -1))
        xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        quats.append(np.tile(xyzw[[3, 0, 1, 2]], (len(xyz), 1)))
        scores.append(f['conf'].numpy()[::2, ::2][valid])
    points, colors, sizes, quats, scores = map(np.concatenate, (points, colors, sizes, quats, scores))
    order = np.argsort(-scores, kind='stable')
    _, inds = np.unique(np.floor(points[order] / voxel).astype(np.int64), axis=0, return_index=True)
    inds = order[inds]
    if len(inds) > cap:
        inds = np.random.default_rng(42).choice(inds, cap, replace=False)
    if not len(inds):
        raise ValueError('No usable geometry in training frames')
    return [torch.from_numpy(x[inds].astype(np.float32)).cuda() for x in (points, colors, sizes, quats)]


def image(array):
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def train(args):
    from gsplat import rasterization
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for training')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    session = Path(args.session).resolve()
    frames = load_frames(session, args.width)
    if len(frames) < 10:
        raise ValueError('At least 10 saved keyframes needed for this train/holdout experiment')
    held = [f for f in frames if f['id'] % 8 == 4]
    training = [f for f in frames if f['id'] % 8 != 4]
    base, rgb, size, quat = initial_cloud(training, args.voxel, args.max_points)
    print(f'{len(training)} training / {len(held)} held-out frames; {len(base)} Gaussians', flush=True)
    params = torch.nn.ParameterDict(dict(
        offset=torch.nn.Parameter(torch.zeros_like(base)),
        color=torch.nn.Parameter(torch.logit(rgb.clamp(.01, .99))),
        log_scale=torch.nn.Parameter(size.log()),
        quat=torch.nn.Parameter(quat),
        opacity=torch.nn.Parameter(torch.full((len(base),), float(np.log(.7/.3)), device='cuda'))))
    optim = torch.optim.Adam([{'params': [params[k]], 'lr': lr} for k, lr in
                             [('offset', .015), ('color', .025), ('log_scale', .003), ('quat', .001), ('opacity', .015)]], eps=1e-8)
    for f in frames:
        for k in ('rgb', 'depth', 'source', 'conf', 'K', 'view'):
            f[k] = f[k].cuda()
    h, w = frames[0]['depth'].shape

    def values():
        return (base + .03 * params['offset'].tanh(), F.normalize(params['quat'], dim=-1),
                params['log_scale'].exp().clamp(.0008, .08), params['opacity'].sigmoid(), params['color'].sigmoid())

    def render(f):
        return rasterization(*values(), f['view'][None], f['K'][None], w, h,
                             render_mode='RGB+ED', near_plane=.1, far_plane=10, packed=True)[:2]

    @torch.no_grad()
    def evaluate(label):
        records = []
        for f in held:
            pred, alpha = render(f)
            pred = pred[0, ..., :3].clamp(0, 1)
            mse = (pred-f['rgb']).square().mean().item()
            records.append(dict(id=f['id'], psnr=-10*np.log10(max(mse, 1e-12)),
                                coverage=float((alpha > .5).float().mean())))
            image(pred.cpu().numpy()).save(out / f'{label}_{f["id"]:03d}.png')
            if label == 'initial':
                image(f['rgb'].cpu().numpy()).save(out / f'reference_{f["id"]:03d}.png')
        return records

    before = evaluate('initial')
    started = time.monotonic()
    history = []
    rng = np.random.default_rng(42)
    order = []
    for step in range(args.steps):
        if not order:
            order = rng.permutation(len(training)).tolist()
        f = training[order.pop()]
        pred, alpha = render(f)
        color_loss = (pred[0, ..., :3] - f['rgb']).abs().mean()
        valid = torch.isfinite(f['depth']) & (f['depth'] > .2) & (f['depth'] < 8) & (f['source'] == 1) & (alpha[0, ..., 0].detach() > .3)
        depth_loss = ((pred[0, ..., 3][valid] - f['depth'][valid]).abs() / f['depth'][valid]).clamp(max=.25).mean() if valid.any() else color_loss*0
        loss = color_loss + .1 * depth_loss + .0005 * params['offset'].tanh().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            params['log_scale'].clamp_(np.log(.0008), np.log(.08))
        if (step+1) % 100 == 0:
            entry = dict(step=step+1, rgb_l1=color_loss.item(), depth_relative_l1=depth_loss.item(), seconds=time.monotonic()-started)
            history.append(entry)
            print(json.dumps(entry), flush=True)
    after = evaluate('trained')
    means, qs, scales, opacities, colors = [x.detach().cpu().numpy() for x in values()]
    rots = Rotation.from_quat(qs[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
    rs = rots * scales[:, None, :]
    cov = rs @ rs.transpose(0, 2, 1)
    np.savez_compressed(out/'gaussians.npz', centers=means, covariances=cov, rgbs=colors, opacities=opacities[:, None],
                        scales=scales, quats=qs, poses=np.stack([f['pose'].numpy() for f in frames]))
    # Standard degree-zero Gaussian PLY, readable by common splat viewers.
    from plyfile import PlyData, PlyElement
    names = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    data = np.concatenate((means, np.zeros_like(means), (colors-.5)/.28209479177387814,
                           np.log(np.clip(opacities, 1e-6, 1-1e-6)/np.clip(1-opacities, 1e-6, 1))[:,None],
                           np.log(scales), qs), -1).astype('f4')
    structured = np.empty(len(data), dtype=[(n,'f4') for n in names])
    for i, n in enumerate(names):
        structured[n] = data[:, i]
    PlyData([PlyElement.describe(structured, 'vertex')]).write(out/'gaussians.ply')
    sheet = Image.new('RGB', (w*3, (h+24)*len(held)))
    for row, f in enumerate(held):
        for col, label in enumerate(('reference', 'initial', 'trained')):
            sheet.paste(Image.open(out/f'{label}_{f["id"]:03d}.png'), (col*w, row*(h+24)+24))
            ImageDraw.Draw(sheet).text((col*w+4, row*(h+24)+4), f'KF {f["id"]}: {label}', fill='white')
    sheet.save(out/'comparison.jpg')
    report = dict(source=str(session), width=w, height=h, train_ids=[f['id'] for f in training], heldout_ids=[f['id'] for f in held],
                  gaussian_count=len(base), steps=args.steps, voxel=args.voxel, before=before, after=after,
                  mean_psnr_before=float(np.mean([r['psnr'] for r in before])), mean_psnr_after=float(np.mean([r['psnr'] for r in after])),
                  train_seconds=time.monotonic()-started, peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                  pose_scale_range=[min(f['scale'] for f in frames), max(f['scale'] for f in frames)], history=history,
                  method='fixed-camera RGB-D Gaussian appearance, bounded 3cm offsets, degree-zero color, no densification',
                  limitations=['held-out views use SLAM poses estimated with all frames: appearance holdout, not independent SLAM accuracy',
                               'training keyframes only initialize geometry; no held-out images or depths in training',
                               'no global pose correction; dynamic objects, blur and missing observations remain'])
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ('mean_psnr_before','mean_psnr_after','train_seconds','peak_cuda_allocated_mb')}), flush=True)
    print(f'Result: {out}', flush=True)


def view(args):
    import viser
    data = np.load(Path(args.result)/'gaussians.npz')
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('-y')
    server.scene.add_gaussian_splats('/appearance', centers=data['centers'], covariances=data['covariances'],
                                     rgbs=data['rgbs'], opacities=data['opacities'])
    poses = data['poses']
    choice = server.gui.add_slider('Camera keyframe', min=0, max=len(poses)-1, step=1, initial_value=0)
    button = server.gui.add_button('Move to saved camera')
    server.gui.add_markdown('Offline Gaussian appearance preview. Drag to look; scroll to move. Original saved map is unchanged.')

    def set_camera(client):
        p = poses[choice.value]
        client.camera.position = p[:3, 3]
        client.camera.up_direction = -p[:3, 1]
        client.camera.look_at = p[:3, 3] + p[:3, 2]
        client.camera.fov = 1.1

    @server.on_client_connect
    def on_connect(client):
        set_camera(client)

    @button.on_click
    def on_click(event):
        if event.client is not None:
            set_camera(event.client)
    print(f'Viewer: http://127.0.0.1:{args.port}', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('session')
    t.add_argument('--output', required=True)
    t.add_argument('--width', type=int, default=384)
    t.add_argument('--steps', type=int, default=1800)
    t.add_argument('--voxel', type=float, default=.015)
    t.add_argument('--max-points', type=int, default=180000)
    v = sub.add_parser('view')
    v.add_argument('result')
    v.add_argument('--port', type=int, default=8090)
    args = p.parse_args()
    train(args) if args.command == 'train' else view(args)
