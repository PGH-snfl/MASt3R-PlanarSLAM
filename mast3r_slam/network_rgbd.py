"""Network RGB-D adapter with the same geometry/depth contract as USB input."""
import numpy as np

from mast3r_slam.config import config
from mast3r_slam.dataloader import Intrinsics, MonocularDataset
from mast3r_slam.realsense_rgbd import resize_depth
from mast3r_slam.rgbd_transport import RGBDClient, StreamError


class NetworkRGBDDataset(MonocularDataset):
    def __init__(self, host, port=9877, timeout=10.):
        super().__init__()
        self.img_size = config.get('dataset', {}).get('image_size', 512)
        if self.img_size not in (512, 640):
            raise ValueError('Network RGB-D image_size must be 512 or 640')
        if not config.get('use_calib') or config['dataset'].get('img_downsample', 1) != 1:
            raise ValueError('Network RGB-D requires use_calib: true and img_downsample: 1')
        self.dataset_path = None
        self.save_results = False
        self.last_depth = None
        self.last_valid_fraction = 0.
        self.last_network_stats = {}
        self.cfg = config.get('realsense', {})
        self.client = RGBDClient(host, port, timeout)
        try:
            meta, rgb, depth = self.client.next()
            self.metadata = meta
            K = np.asarray(meta['K'], dtype=np.float64)
            h, w = depth.shape
            v, u = np.mgrid[:h, :w].astype(np.float32)
            # Sender has already aligned depth and rectified RGB/depth together.
            # Do not undistort or change the principal point a second time.
            self.camera_intrinsics = Intrinsics(self.img_size, w, h, K, K.copy(), np.zeros(5), u, v)
            self.raw_shape = (h, w)
            self._pending = self._convert(meta, rgb, depth)
            self.shape = self._pending[2].shape
            print(f"Network RGB-D: {host}:{port}, {meta.get('camera', 'RealSense')}; "
                  f"{w}x{h}, processing={self.shape[1]}x{self.shape[0]}, "
                  f"RGB={meta['rgb_codec']}, depth=lossless", flush=True)
        except BaseException:
            self.close()
            raise

    def _convert(self, meta, rgb, depth):
        z = depth.astype(np.float32) * meta['depth_scale']
        good = np.isfinite(z) & (z >= self.cfg.get('min_depth', .25)) & (z <= self.cfg.get('max_depth', 5.))
        z[~good] = 0
        self.last_network_stats = dict(self.client.stats)
        return meta['timestamp'], rgb.astype(np.float32)/255., resize_depth(z, self.img_size)

    def __getitem__(self, idx):
        try:
            sample = self._pending if self._pending is not None else self._convert(*self.client.next())
            self._pending = None
            timestamp, rgb, self.last_depth = sample
            self.last_valid_fraction = float((self.last_depth > 0).mean())
            # Live streams have no need to retain an unbounded timestamp list.
            self.timestamps[:] = [timestamp]
            return timestamp, rgb
        except (OSError, StreamError) as exc:
            self.close()
            raise RuntimeError(f'Network RGB-D stopped: {exc}. Check send_rgbd.py and Tailscale; '
                               'restart SLAM to begin a new session.') from exc

    def __len__(self):
        return 2**31-1

    def get_img_shape(self):
        return self.shape, self.raw_shape

    def close(self):
        self.client.close()
