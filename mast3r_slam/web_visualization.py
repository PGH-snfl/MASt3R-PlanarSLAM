"""Bounded live snapshots for a browser; no OpenGL, disk paging or inference here.

Call publish from the tracking thread. Viser only receives detached CPU arrays;
its network threads never access the tracker or CUDA tensors.
"""
import math
import time

import numpy as np


def sample_indices(length, limit):
    return np.linspace(0, length - 1, min(length, limit), dtype=np.int64)


def frame_cloud(frame, limit):
    """Sample before the GPU -> CPU copy. Preserve the complete Sim3 transform."""
    import torch
    if frame.X_canon is None or limit <= 0:
        return np.empty((0, 6), np.float32)
    indices = sample_indices(len(frame.X_canon), limit)
    idx = torch.as_tensor(indices, device=frame.X_canon.device)
    xyz = frame.X_canon.detach()[idx]
    valid = torch.isfinite(xyz).all(-1) & (xyz[:, 2] > 0)
    if frame.C is not None:
        confidence = frame.C.detach().reshape(-1)[idx]
        valid &= torch.isfinite(confidence) & (confidence > 0)
    world = (frame.T_WC * xyz).detach().cpu().numpy().reshape(-1, 3)
    colors = frame.uimg.detach().reshape(-1, 3)[indices].cpu().numpy()
    valid = valid.cpu().numpy() & np.isfinite(world).all(-1) & np.isfinite(colors).all(-1)
    return np.concatenate((world[valid], colors[valid]), axis=1).astype(np.float32)


def map_cloud(store, limit):
    """Use only already-resident keyframes and the bounded historical overview."""
    recent = list(store.recent.values())
    history = store.preview
    history_budget = limit // 2 if recent else limit
    history = history[sample_indices(len(history), history_budget)]
    remaining = limit - len(history)
    clouds = [history]
    for i, frame in enumerate(recent):
        budget = remaining // (len(recent) - i)
        clouds.append(frame_cloud(frame, budget))
        remaining -= budget
    cloud = np.concatenate(clouds).astype(np.float32)
    return cloud[np.isfinite(cloud).all(-1)]


def cloud_arrays(cloud):
    return (np.ascontiguousarray(cloud[:, :3], dtype=np.float32),
            np.ascontiguousarray(np.clip(cloud[:, 3:6] * 255, 0, 255), dtype=np.uint8))


class WebViewer:
    def __init__(self, port=8080, fps=2., max_points=50000):
        if not 1 <= port <= 65535 or not math.isfinite(fps) or fps <= 0 or max_points < 100:
            raise ValueError('Invalid browser viewer port, FPS or point budget')
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError('Browser viewer requires: python -m pip install -r requirements-web.txt') from exc
        # Do not publish an unauthenticated viewer on a public interface.
        self.server = viser.ViserServer(host='127.0.0.1', port=port, label='MASt3R-PlanarSLAM', verbose=False)
        self.period, self.next_update = 1. / fps, 0.
        self.max_points = max_points
        self.map_revision = None
        self.map_count = 0
        self.updates = 0
        try:
            if self.server.get_port() != port:
                raise RuntimeError(f'Viewer port {port} is busy. Choose another --web-port and SSH -L port.')
            scene, gui = self.server.scene, self.server.gui
            # First camera coordinates: +X right, +Y down, +Z forward.
            scene.set_up_direction('-y')
            self.server.initial_camera.position = (3., -3., -3.)
            self.server.initial_camera.look_at = (0., 0., 2.)
            self.server.initial_camera.up_direction = (0., -1., 0.)
            self.status = gui.add_markdown('**Starting SLAM…** Waiting for a processed frame.')
            gui.add_markdown('Drag: rotate · Right drag: pan · Wheel: zoom.\n\nClosing this tab leaves SLAM running. Stop with Ctrl+C in the server terminal.')
            self.image = gui.add_image(np.zeros((240, 320, 3), np.uint8), label='RGB', format='jpeg', jpeg_quality=75)
            self.point_size = gui.add_slider('Point size (m)', min=.002, max=.06, step=.002, initial_value=.012)
            self.show_map = gui.add_checkbox('Map', True)
            self.show_current = gui.add_checkbox('Current frame', True)
            empty = np.empty((0, 3), np.float32)
            colors = np.empty((0, 3), np.uint8)
            self.map = scene.add_point_cloud('/map', empty, colors, point_size=.012, precision='float32')
            self.current = scene.add_point_cloud('/current', empty, colors, point_size=.012, precision='float32')
            self.path = scene.add_line_segments('/keyframe_path', np.empty((0, 2, 3), np.float32), (255, 170, 40), thickness=.008)
            self.camera = scene.add_camera_frustum('/camera', fov=1., aspect=4/3, scale=.15, color=(30, 210, 255))
            scene.add_frame('/origin', axes_length=.25, axes_radius=.004)
            reset = gui.add_button('Reset view')

            @reset.on_click
            def reset_view(event):
                if event.client is not None:
                    event.client.camera.position = (3., -3., -3.)
                    event.client.camera.look_at = (0., 0., 2.)
                    event.client.camera.up_direction = (0., -1., 0.)

            @self.point_size.on_update
            def change_size(_):
                self.map.point_size = self.current.point_size = self.point_size.value

            @self.show_map.on_update
            def show_map(_):
                self.map.visible = self.show_map.value

            @self.show_current.on_update
            def show_current(_):
                self.current.visible = self.show_current.value
        except BaseException:
            self.server.stop()
            raise
        print(f'Browser viewer: http://127.0.0.1:{port} (forward this port with SSH -L)', flush=True)

    def ready(self):
        now = time.monotonic()
        if not self.server.get_clients() or now < self.next_update:
            return False
        self.next_update = now + self.period
        return True

    def publish(self, frame, store, mode):
        import cv2
        revision = (len(store), store.revision, store.pose_revision)
        with self.server.atomic():
            if revision != self.map_revision:
                cloud = map_cloud(store, self.max_points)
                self.map.points, self.map.colors = cloud_arrays(cloud)
                self.map_count = len(cloud)
                ids = sample_indices(len(store.poses), 4096)
                positions = np.asarray([store.poses[i].reshape(-1)[:3].numpy() for i in ids], dtype=np.float32).reshape(-1, 3)
                self.path.points = np.stack((positions[:-1], positions[1:]), axis=1)
                self.map_revision = revision
            # An unlocalized frame is not reliable world geometry.
            cloud = frame_cloud(frame, min(4000, self.max_points)) if mode == 'TRACKING' else np.empty((0, 6), np.float32)
            self.current.points, self.current.colors = cloud_arrays(cloud)
            pose = frame.T_WC.data.detach().cpu().numpy().reshape(-1)
            self.camera.position = pose[:3]
            self.camera.wxyz = pose[[6, 3, 4, 5]]
            self.camera.visible = mode == 'TRACKING'
            rgb = frame.uimg.detach().cpu().numpy()
            h, w = rgb.shape[:2]
            K = frame.K.detach().cpu().numpy().reshape(3, 3)
            self.camera.fov = float(2 * np.arctan(h / (2 * K[1, 1])))
            self.camera.aspect = w / h
            self.image.image = cv2.resize(np.clip(rgb * 255, 0, 255).astype(np.uint8), (320, max(1, round(320*h/w))))
            self.status.content = (f'**{mode}** · Frame {frame.frame_id} · Keyframes {len(store)}\n\n'
                                   f'Map: {self.map_count:,} points · Current: {len(cloud):,} points\n\n'
                                   f'Updated {time.strftime("%H:%M:%S")} (server time)')
        self.updates += 1

    def close(self):
        self.server.stop()
