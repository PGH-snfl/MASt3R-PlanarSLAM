"""Offline, RTAB-Map-style point filtering / Poisson / projective texturing.

Uses the saved MASt3R poses, not RTAB-Map pose optimization or a RTAB-Map DB.
Open3D supplies screened Poisson; photo selection and gain adjustment are local
implementations. No Gaussian training, pose updates, or live SLAM modification.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import trimesh

from offline_appearance import load_frames


def log(message):
    print(message, flush=True)


def manifest_hash(root):
    """Hash every input byte; verifies originals are unchanged by the run."""
    digest = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():
            digest.update(str(p.relative_to(root)).encode())
            with p.open('rb') as stream:
                while block := stream.read(2**20):
                    digest.update(block)
    return digest.hexdigest()


def frames_numpy(root):
    frames = load_frames(root, 512)
    for f in frames:
        for k, v in f.items():
            if torch.is_tensor(v):
                f[k] = v.numpy()
        f['sharpness'] = float(cv2.Laplacian(cv2.cvtColor(f['rgb'], cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    return frames


def project(points, f):
    xyz = points @ f['view'][:3, :3].T + f['view'][:3, 3]
    z = xyz[:, 2]
    uv = xyz[:, :2] / np.maximum(z[:, None], 1e-6)
    uv = uv * np.array([f['K'][0, 0], f['K'][1, 1]]) + f['K'][:2, 2]
    h, w = f['depth'].shape
    good = (z > .2) & (uv[:, 0] >= 2) & (uv[:, 0] < w-3) & (uv[:, 1] >= 2) & (uv[:, 1] < h-3)
    px = np.clip(np.rint(uv[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.rint(uv[:, 1]).astype(int), 0, h-1)
    d = f['depth'][py, px]
    good &= np.isfinite(d) & (d > .2) & (np.abs(d-z) < .055+.012*z)
    return uv, z, good


def point_cloud(frames, voxel, out, stats, normal_radius=0.):
    xs, ns, cs = [], [], []
    for f in frames:
        d, K = f['depth'], f['K']
        yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
        xyz = np.stack(((xx-K[0, 2])*d/K[0, 0], (yy-K[1, 2])*d/K[1, 1], d), -1)
        dy, dx = np.gradient(xyz, axis=(0, 1))
        normal = np.cross(dx, dy)
        length = np.linalg.norm(normal, axis=-1)
        normal /= np.maximum(length[..., None], 1e-10)
        normal[(normal*xyz).sum(-1) > 0] *= -1
        dz_y, dz_x = np.gradient(d)
        valid = np.isfinite(d) & (d > .25) & (d < 5.) & (f['conf'] >= 1.5) & (length > 1e-8)
        valid &= (np.abs(dz_x)+np.abs(dz_y)) < .06+.01*d
        valid[:2] = valid[-2:] = False
        valid[:, :2] = valid[:, -2:] = False
        valid[1::2, :] = False
        valid[:, 1::2] = False
        pose = f['pose']
        xs.append((xyz[valid] @ pose[:3, :3].T + pose[:3, 3]).astype('f4'))
        ns.append((normal[valid] @ pose[:3, :3].T).astype('f4'))
        cs.append(f['rgb'][valid])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(xs))
    cloud.normals = o3d.utility.Vector3dVector(np.concatenate(ns))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(cs))
    stats['input_sampled_points'] = len(cloud.points)
    cloud = cloud.voxel_down_sample(voxel)
    stats['voxel_points'] = len(cloud.points)
    cloud, _ = cloud.remove_radius_outlier(nb_points=7, radius=voxel*3.)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.)
    n = np.asarray(cloud.normals)
    cloud = cloud.select_by_index(np.flatnonzero(np.linalg.norm(n, axis=1) > .25))
    cloud.normalize_normals()
    if normal_radius > 0:
        # Pixel derivatives are noisy on a depth camera. Fit local PCA planes,
        # preserving each normal's observed camera-facing orientation.
        reference = np.asarray(cloud.normals).copy()
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60))
        normals = np.asarray(cloud.normals)
        normals[np.sum(normals*reference, axis=1) < 0] *= -1
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud.normalize_normals()
    stats['filtered_points'] = len(cloud.points)
    if len(cloud.points) < 1000:
        raise ValueError('Insufficient filtered geometry')
    if not o3d.io.write_point_cloud(str(out/'filtered_cloud.ply'), cloud):
        raise IOError('Failed writing filtered point cloud')
    log(f'Point cloud: {stats["input_sampled_points"]:,} -> {len(cloud.points):,}')
    return cloud


def poisson_mesh(cloud, args, out, stats):
    log('Poisson reconstruction started')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=args.poisson_depth, scale=1.05, linear_fit=False, n_threads=8)
    stats['poisson_faces'] = len(mesh.triangles)
    tree = cKDTree(np.asarray(cloud.points))
    dist, _ = tree.query(np.asarray(mesh.vertices), workers=4)
    densities = np.asarray(densities)
    # Reject unsupported extrapolated surfaces rather than filling unseen space.
    mesh.remove_vertices_by_mask((densities < np.quantile(densities, .01)) | (dist > args.voxel*3.))
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    if len(mesh.triangles) > args.max_faces:
        mesh = mesh.simplify_quadric_decimation(args.max_faces)
    labels, counts, areas = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 100)
    mesh.remove_unreferenced_vertices().compute_vertex_normals()
    if len(mesh.triangles) < 100:
        raise ValueError('No useful surface after mesh cleanup')
    _, nn = tree.query(np.asarray(mesh.vertices), workers=4)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[nn])
    if not o3d.io.write_triangle_mesh(str(out/'colored_mesh.ply'), mesh):
        raise IOError('Failed writing mesh')
    stats['mesh_vertices'] = len(mesh.vertices)
    stats['mesh_faces'] = len(mesh.triangles)
    log(f'Mesh: {stats["poisson_faces"]:,} -> {len(mesh.triangles):,} triangles')
    return mesh


def gain_compensation(cloud, frames):
    points = np.asarray(cloud.points)[::max(1, len(cloud.points)//12000)]
    observed = []
    for f in frames:
        uv, _, good = project(points, f)
        values = cv2.remap(f['rgb'], uv[:, 0].astype('f4')[:, None], uv[:, 1].astype('f4')[:, None], cv2.INTER_LINEAR)[:, 0]
        good &= (values.min(1) > .06) & (values.max(1) < .94)
        observed.append((values, good))
    equations, rhs = [], []
    for i in range(len(frames)):
        for j in range(i+1, len(frames)):
            valid = observed[i][1] & observed[j][1]
            if valid.sum() < 100:
                continue
            ratio = np.log(observed[j][0][valid]) - np.log(observed[i][0][valid])
            median = np.median(ratio, axis=0)
            if np.median(np.abs(ratio-median)) > .15:
                continue
            a = np.zeros(len(frames)); a[i], a[j] = 1, -1
            equations.append(a); rhs.append(median)
    # Weak neutral prior also handles disconnected camera groups.
    A = np.vstack([np.asarray(equations).reshape(-1, len(frames)), .2*np.eye(len(frames))])
    b = np.vstack([np.asarray(rhs).reshape(-1, 3), np.zeros((len(frames), 3))])
    gains = np.exp(np.linalg.lstsq(A, b, rcond=None)[0]).clip(.8, 1.25)
    log(f'Exposure: {len(equations)} overlapping camera pairs; gain {gains.min():.3f}..{gains.max():.3f}')
    return gains, len(equations)


def ray_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def assign_photos(mesh, frames, scene):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    tri = vertices[faces]
    centers = tri.mean(1)
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    scores = np.zeros((len(frames), len(faces)), dtype='f4')
    median_sharp = np.median([f['sharpness'] for f in frames])
    for i, f in enumerate(frames):
        _, z, good = project(centers, f)
        for j in range(3):
            good &= project(tri[:, j], f)[2]
        direction = f['pose'][:3, 3] - centers
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        cosine = np.sum(normals*direction, axis=1)
        good &= cosine > .2
        ids = np.flatnonzero(good)
        if len(ids):
            origin = np.broadcast_to(f['pose'][:3, 3], (len(ids), 3))
            rays = np.concatenate((origin, centers[ids]-origin), axis=1).astype('f4')
            hits = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            ids = ids[np.isfinite(hits) & (np.abs(hits-1.) < .005)]
            sharp = np.clip(f['sharpness']/max(median_sharp, 1e-6), .4, 1.4)
            scores[i, ids] = cosine[ids]**2 / np.maximum(z[ids]**2, .1) * sharp
        if (i+1) % 10 == 0:
            log(f'Photo visibility {i+1}/{len(frames)}')
    labels = scores.argmax(0)
    valid = scores.max(0) > 0
    labels[~valid] = -1
    # Local label agreement reduces isolated face-by-face photo switches.
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    neighbors = np.full((len(faces), 3), -1, dtype=int)
    count = np.zeros(len(faces), dtype=int)
    for a, b in tm.face_adjacency:
        if count[a] < 3: neighbors[a, count[a]] = b; count[a] += 1
        if count[b] < 3: neighbors[b, count[b]] = a; count[b] += 1
    index = np.arange(len(faces))
    for _ in range(4):
        neighbor_labels = labels[np.maximum(neighbors, 0)]
        neighbor_labels[neighbors < 0] = -1
        candidates = np.column_stack((labels, neighbor_labels))
        best_cost = np.full(len(faces), np.inf)
        next_labels = labels.copy()
        best_score = np.maximum(scores.max(0), 1e-12)
        for col in range(4):
            candidate = candidates[:, col]
            score = scores[np.maximum(candidate, 0), index]
            cost = -np.log(np.maximum(score, 1e-12)/best_score)
            cost += .3*((candidate[:, None] != neighbor_labels) & (neighbor_labels >= 0)).sum(1)
            use = (candidate >= 0) & (score > 0) & (cost < best_cost)
            next_labels[use] = candidate[use]
            best_cost[use] = cost[use]
        labels = next_labels
    log(f'Photos assigned: {(labels >= 0).sum():,}/{len(labels):,} faces')
    return labels


def texture_export(mesh, frames, gains, labels, out):
    h, w = frames[0]['depth'].shape
    pad = 4
    columns = math.ceil(math.sqrt(len(frames)*(h+2*pad)/(w+2*pad)))
    rows = math.ceil(len(frames)/columns)
    aw, ah = columns*(w+2*pad), rows*(h+2*pad)+4
    atlas = np.full((ah, aw, 3), 150, dtype=np.uint8)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    uvs = np.zeros((len(faces), 3, 2), dtype='f4')
    uvs[:] = [(aw-.5)/aw, .5/ah]  # dedicated gray row for unsupported faces
    for i, f in enumerate(frames):
        rgb = (np.clip(f['rgb']*gains[i], 0, 1)*255).astype('uint8')
        x, y = (i%columns)*(w+2*pad), (i//columns)*(h+2*pad)
        atlas[y:y+h+2*pad, x:x+w+2*pad] = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        ids = np.flatnonzero(labels == i)
        if len(ids):
            uv = project(vertices[faces[ids]].reshape(-1, 3), f)[0].reshape(-1, 3, 2)
            uv[..., 0] = (uv[..., 0]+x+pad+.5)/aw
            uv[..., 1] = 1-(uv[..., 1]+y+pad+.5)/ah
            uvs[ids] = uv
    picture = Image.fromarray(atlas)
    picture.save(out/'texture_atlas.jpg', quality=95)
    with (out/'textured_mesh.mtl').open('w') as f:
        f.write('newmtl captured_rgb\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd texture_atlas.jpg\n')
    with (out/'textured_mesh.obj').open('w') as f:
        f.write('mtllib textured_mesh.mtl\nusemtl captured_rgb\n')
        for v in vertices: f.write('v %.7f %.7f %.7f\n' % tuple(v))
        for uv in uvs.reshape(-1, 2): f.write('vt %.8f %.8f\n' % tuple(uv))
        for i, face in enumerate(faces):
            f.write('f '+' '.join(f'{v+1}/{i*3+j+1}' for j, v in enumerate(face))+'\n')
    visual = trimesh.visual.texture.TextureVisuals(uv=uvs.reshape(-1, 2), image=picture)
    textured = trimesh.Trimesh(vertices=vertices[faces].reshape(-1, 3), faces=np.arange(len(faces)*3).reshape(-1, 3), visual=visual, process=False)
    export_photo_glb(textured, out/'textured_mesh.glb')
    np.savez_compressed(out/'mesh_data.npz', vertices=vertices.astype('f4'), faces=faces, triangle_uvs=uvs,
                        photo_ids=labels, poses=np.stack([f['pose'] for f in frames]), gains=gains)
    return atlas, uvs


def export_photo_glb(mesh, path):
    # Captured RGB already contains illumination. Do not add a second light to
    # the photo texture in the browser. Double-sided surfaces permit inspection.
    def unlit(tree):
        tree.setdefault('extensionsUsed', []).append('KHR_materials_unlit')
        for material in tree.get('materials', []):
            material.setdefault('extensions', {})['KHR_materials_unlit'] = {}
            material['doubleSided'] = True
    path.write_bytes(trimesh.exchange.gltf.export_glb(mesh, tree_postprocessor=unlit))


def render_mesh(scene, mesh, atlas, uvs, f):
    h, w = f['depth'].shape
    yy, xx = np.mgrid[:h, :w]
    K = f['K']
    dirs = np.stack(((xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)), -1)
    dirs = dirs @ f['pose'][:3, :3].T
    origin = np.broadcast_to(f['pose'][:3, 3], dirs.shape)
    hit = scene.cast_rays(o3d.core.Tensor(np.concatenate((origin, dirs), -1).astype('f4')))
    ids = hit['primitive_ids'].numpy()
    good = np.isfinite(hit['t_hit'].numpy())
    bary = hit['primitive_uvs'].numpy()
    weights = np.stack((1-bary[..., 0]-bary[..., 1], bary[..., 0], bary[..., 1]), -1)
    uv = np.zeros((h, w, 2), dtype='f4')
    uv[good] = (uvs[ids[good]]*weights[good, :, None]).sum(1)
    ah, aw = atlas.shape[:2]
    textured = cv2.remap(atlas, (uv[..., 0]*aw-.5).astype('f4'), ((1-uv[..., 1])*ah-.5).astype('f4'), cv2.INTER_LINEAR)
    textured[~good] = 25
    normals = hit['primitive_normals'].numpy()
    view = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    shade = (.25+.65*np.abs((normals*view).sum(-1)))*255
    gray = np.repeat(shade[..., None], 3, -1).astype('uint8')
    gray[~good] = 25
    return textured, gray, hit['t_hit'].numpy(), good


def render_points(points, colors, f):
    h, w = f['depth'].shape
    xyz = points @ f['view'][:3, :3].T + f['view'][:3, 3]
    good = xyz[:, 2] > .2
    xyz, color = xyz[good], colors[good]
    uv = xyz[:, :2]/xyz[:, 2:3] * [f['K'][0, 0], f['K'][1, 1]] + f['K'][:2, 2]
    xy = np.rint(uv).astype(int)
    good = (xy[:, 0] >= 1) & (xy[:, 0] < w-1) & (xy[:, 1] >= 1) & (xy[:, 1] < h-1)
    xy, z, color = xy[good], xyz[good, 2], color[good]
    pix = np.concatenate([(xy[:, 1]+dy)*w+xy[:, 0]+dx for dy, dx in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]])
    zz = np.tile(z, 5)
    cc = np.tile(color, (5,1))
    sort = np.lexsort((zz, pix))
    _, first = np.unique(pix[sort], return_index=True)
    picks = sort[first]
    rgb = np.full((h*w, 3), 25, dtype='uint8')
    rgb[pix[picks]] = (cc[picks]*255).clip(0,255).astype('uint8')
    return rgb.reshape(h,w,3)


def previews(cloud, mesh, frames, atlas, uvs, scene, out):
    ids = [i for i in [0, 12, 20, 28, 44, 52, 60] if i < len(frames)]
    h, w = frames[0]['depth'].shape
    sheet = Image.new('RGB', (w*4, (h+26)*len(ids)))
    metrics = []
    for row, i in enumerate(ids):
        f = frames[i]
        tex, gray, depth, good = render_mesh(scene, mesh, atlas, uvs, f)
        raw = (f['rgb']*255).clip(0,255).astype('uint8')
        points = render_points(np.asarray(cloud.points), np.asarray(cloud.colors), f)
        for col, (name, rgb) in enumerate(zip(['Captured RGB','Filtered points','Poisson surface','Photo textured'], [raw, points, gray, tex])):
            im = Image.fromarray(rgb)
            im.save(out/f'kf_{i:03d}_{col}.jpg', quality=95)
            sheet.paste(im, (col*w, row*(h+26)+26))
            ImageDraw.Draw(sheet).text((col*w+4, row*(h+26)+4), f'KF {i}: {name}', fill='white')
        sensor = good & (f['source'] == 1) & np.isfinite(f['depth']) & (f['depth'] > .25)
        error = np.abs(depth[sensor]-f['depth'][sensor])
        metrics.append(dict(id=i, rendered_coverage=float(good.mean()), depth_absolute_median_m=float(np.median(error)) if len(error) else None,
                            depth_absolute_p90_m=float(np.quantile(error,.9)) if len(error) else None))
    sheet.save(out/'comparison.jpg', quality=93)
    return metrics


def build(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    source = Path(args.session).resolve()
    out = Path(args.output).resolve()
    if source == out or source in out.parents:
        raise ValueError('Output must be outside the source map directory')
    before = manifest_hash(source)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frames = frames_numpy(source)
    stats = dict(source=str(source), keyframes=len(frames), method='Open3D screened Poisson + depth/mesh-visible camera selection + gain-compensated photo atlas',
                 voxel_m=args.voxel, poisson_depth=args.poisson_depth, source_sha256=before,
                 pose_optimization=False, image_size=list(frames[0]['rgb'].shape), used_nonkeyframes=False)
    log(f'Loaded {len(frames)} saved RGB-D keyframes')
    stats['normal_radius_m'] = args.normal_radius
    cloud = point_cloud(frames, args.voxel, out, stats, args.normal_radius)
    mesh = poisson_mesh(cloud, args, out, stats)
    gains, pairs = gain_compensation(cloud, frames)
    stats['gain_pairs'] = pairs
    stats['gains_range'] = [float(gains.min()), float(gains.max())]
    scene = ray_scene(mesh)
    labels = assign_photos(mesh, frames, scene)
    stats['textured_faces'] = int((labels >= 0).sum())
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=1)*.5
    stats['textured_area_fraction'] = float(area[labels >= 0].sum()/area.sum())
    atlas, uvs = texture_export(mesh, frames, gains, labels, out)
    stats['preview_metrics'] = previews(cloud, mesh, frames, atlas, uvs, scene, out)
    stats['source_unchanged'] = before == manifest_hash(source)
    if not stats['source_unchanged']:
        raise RuntimeError('Source changed during the run')
    stats['seconds'] = time.monotonic()-started
    stats['limitations'] = ['Saved poses fixed; no full-history bundle adjustment.',
                            'Same-source RGB/depth preview checks; no independent ground truth accuracy claim.',
                            'Normals, depth and pose errors can produce doubled or bent surfaces.',
                            'Photo label smoothing and exposure gains used; AliceVision multiband blending not implemented.',
                            'Unassigned faces are gray; no generated textures. Poisson interpolation can bridge gaps between samples.',
                            'Input RGB is processed 512x384; moving objects are not semantically removed.']
    (out/'report.json').write_text(json.dumps(stats,indent=2))
    log(json.dumps(stats,indent=2))


def view(args):
    import viser
    root = Path(args.result)
    server = viser.ViserServer(host='127.0.0.1',port=args.port)
    server.scene.set_up_direction('-y')
    data = np.load(root/'mesh_data.npz')
    photo = server.scene.add_glb('/photo',glb_data=(root/'textured_mesh.glb').read_bytes(),cast_shadow=False,receive_shadow=False)
    gray = server.scene.add_mesh_simple('/surface', data['vertices'], data['faces'], color=(175,175,175), side='double', visible=False)
    p = o3d.io.read_point_cloud(str(root/'filtered_cloud.ply'))
    points = server.scene.add_point_cloud('/points', points=np.asarray(p.points).astype('f4'), colors=(np.asarray(p.colors)*255).astype('uint8'), point_size=.012, visible=False)
    mode = server.gui.add_dropdown('Display', options=['Photo texture','Surface','Filtered points'])
    camera = server.gui.add_slider('Camera keyframe', min=0,max=len(data['poses'])-1,step=1,initial_value=min(20,len(data['poses'])-1))
    go = server.gui.add_button('Move to saved camera')
    server.gui.add_markdown('Saved-map postprocessing preview. Gray areas have no accepted photo. Original SLAM data is unchanged.')
    @mode.on_update
    def update(event):
        photo.visible = mode.value == 'Photo texture'
        gray.visible = mode.value == 'Surface'
        points.visible = mode.value == 'Filtered points'
    def place(client):
        pose=data['poses'][camera.value]
        client.camera.position=pose[:3,3]
        client.camera.up_direction=-pose[:3,1]
        client.camera.look_at=pose[:3,3]+pose[:3,2]
        client.camera.fov=1.1
    @server.on_client_connect
    def connect(client): place(client)
    @go.on_click
    def move(event):
        if event.client is not None: place(event.client)
    log(f'Viewer http://127.0.0.1:{args.port}')
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: server.stop()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('session'); b.add_argument('--output',required=True)
    b.add_argument('--voxel',type=float,default=.015)
    b.add_argument('--poisson-depth',type=int,default=9)
    b.add_argument('--max-faces',type=int,default=200000)
    b.add_argument('--normal-radius',type=float,default=.08)
    v=sub.add_parser('view'); v.add_argument('result'); v.add_argument('--port',type=int,default=8091)
    args=parser.parse_args()
    build(args) if args.command=='build' else view(args)
