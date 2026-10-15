import dataclasses
import weakref
import time
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Window(WindowEvents):
    title = "MASt3R-PlanarSLAM"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog
        self.feature_prog = self.load_program("programs/features.glsl")
        self.feature_buffer = self.ctx.buffer(reserve=512*6*4, dynamic=True)
        self.feature_vao = self.ctx.vertex_array(self.feature_prog, [(self.feature_buffer, "3f 3f", "in_position", "in_color")])
        self.show_features = True
        self.feature_size = 7.
        self.feature_count = 0
        self.feature_reported = False

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main
        self.disk_map = None
        self.disk_map_count = 0
        self.disk_map_stats = {}
        self.disk_map_next = 0.
        self.disk_surfel_textures = {}
        self.disk_surfels = []
        if getattr(keyframes, 'map_view_only', False) and hasattr(keyframes, 'map_directory'):
            from mast3r_slam.disk_map import DiskMapReader
            reader_cls = DiskMapReader
            if config.get('map_storage', {}).get('view_render_mode') == 'surfels':
                from mast3r_slam.surfel_map import SurfelMapReader
                reader_cls = SurfelMapReader
            self.disk_map = reader_cls(keyframes.map_directory, config.get('map_storage', {}))
        if getattr(keyframes, 'map_view_only', False):
            self.show_curr_pointmap = False
            self.show_features = False
            self.reset_saved_view()

    def reset_saved_view(self):
        """Start at the recorded camera, not two metres behind it inside a wall."""
        pose = as_SE3(self.states.get_frame().T_WC).cpu().matrix().numpy().reshape(4, 4)
        self.camera.unfollow_cam()
        self.camera.follow_cam(np.linalg.inv(pose))
        self.camera.unfollow_cam()
        self.follow_cam = False
        self.show_all = True
        self.disk_map_next = 0.

    def render(self, t: float, frametime: float):
        # The live window may replace CPU slots while this process renders.
        with self.keyframes.lock:
            self._render_locked(t, frametime)

    def _render_locked(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        else:
            self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.follow_cam:
            offset = 0. if getattr(self.keyframes, 'map_view_only', False) else -2.
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, offset], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        alive = {self.keyframes[i].frame_id for i in range(N_keyframes)} | {"curr"}
        for old in list(self.textures):
            if old not in alive:
                for texture in self.textures.pop(old):
                    texture.release()

        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)

            color = [1, 0, 0, 1]
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            if self.show_all and self.disk_map is None:
                self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        if self.show_all and self.disk_map is not None:
            self.render_disk_map()
        elif self.show_all and hasattr(self.keyframes, "history"):
            self.render_history()
        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.feature_count = 0 if curr_frame.feature_points is None else len(curr_frame.feature_points)
        if self.show_features and self.feature_count:
            self.render_features(curr_frame)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)
        if self.disk_map is not None:
            s = self.disk_map_stats
            if imgui.button('Reset saved view'):
                self.reset_saved_view()
            if not s:
                imgui.text('Loading saved geometry...')
            imgui.text(f"Saved map: {s.get('keyframes', 0)} keyframes / {s.get('nonkeyframes', 0)} other frames")
            imgui.text(f"Detailed points: {s.get('detailed_points', 0):,} / overview: {s.get('overview_points', 0):,}")
            if config.get('map_storage', {}).get('view_render_mode') == 'surfels':
                imgui.text(f"Surfel keyframes: {s.get('surfel_keyframes', 0)} / {self.disk_map.surfel_limit}")
                imgui.text(f"Surfel samples: {s.get('surfel_points', 0):,}")
            imgui.text("Uncheck follow cam to explore saved regions")

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.show_features = imgui.checkbox("show tracking features", self.show_features)
        _, self.feature_size = imgui.slider_float("feature size", self.feature_size, 3., 12.)
        imgui.text(f"Current tracking samples: {self.feature_count} / 512")
        imgui.text("Cyan: sensor depth | Orange: model depth")
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_disk_map(self):
        result = self.disk_map.poll()
        if result is not None:
            first_load = not self.disk_map_stats
            points, self.disk_map_stats = result
            if first_load:
                print(f"Saved map loaded: {self.disk_map_stats.get('surfel_keyframes', 0)} surfel keyframes, "
                      f"{len(points):,} auxiliary/display points", flush=True)
            self.disk_surfels = self.disk_map_stats.pop('surfels', [])
            alive = {s['frame_id'] for s in self.disk_surfels}
            for fid in list(self.disk_surfel_textures):
                if fid not in alive:
                    for texture in self.disk_surfel_textures.pop(fid):
                        texture.release()
            for surface in self.disk_surfels:
                fid = surface['frame_id']
                if fid not in self.disk_surfel_textures:
                    h, w = surface['shape']
                    self.disk_surfel_textures[fid] = tuple(
                        self.ctx.texture((w, h), channels, data=surface[name].tobytes(), dtype='f4', alignment=4)
                        for name, channels in [('xyz', 3), ('confidence', 1), ('rgb', 3)])
            if not hasattr(self, 'disk_map_buffer'):
                budget = self.disk_map.point_budget+int(self.disk_map.cfg.get('view_overview_points', 100000))
                self.disk_map_buffer = self.ctx.buffer(reserve=max(1, budget)*24, dynamic=True)
                self.disk_map_vao = self.ctx.vertex_array(self.feature_prog,
                    [(self.disk_map_buffer, '3f 3f', 'in_position', 'in_color')])
            self.disk_map_count = len(points)
            if len(points):
                self.disk_map_buffer.write(points.astype(np.float32, copy=False).tobytes())
        now = time.monotonic()
        if now >= self.disk_map_next:
            eye = np.linalg.inv(self.camera.T_CW)[:3, 3]
            self.disk_map.request(eye, self.state.C_conf_threshold,
                                  self.camera.proj_mat.matrix @ self.camera.T_CW)
            self.disk_map_next = now+config.get('map_storage', {}).get('view_refresh_s', .5)
        if self.disk_map_count:
            self.feature_prog['m_model'].write(np.eye(4, dtype=np.float32).tobytes())
            self.feature_prog['m_camera'].write(self.camera.gl_matrix())
            self.feature_prog['m_proj'].write(self.camera.proj_mat.gl_matrix())
            self.feature_prog['point_size'].value = 2.*self.scale
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.disk_map_vao.render(mode=moderngl.POINTS, vertices=self.disk_map_count)
            self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)
        for surface in self.disk_surfels:
            h, w = surface['shape']
            self.render_pointmap(None, w, h, *self.disk_surfel_textures[surface['frame_id']],
                                 model_matrix=surface['matrix'])

    def render_history(self):
        count = self.keyframes.history_count.value
        if not count:
            return
        revision = self.keyframes.history_revision.value
        if not hasattr(self, 'history_buffer'):
            self.history_buffer = self.ctx.buffer(reserve=self.keyframes.history.numel()*4, dynamic=True)
            self.history_vao = self.ctx.vertex_array(self.feature_prog, [(self.history_buffer, '3f 3f', 'in_position', 'in_color')])
            self.history_revision = -1
        if revision != self.history_revision:
            self.history_buffer.write(self.keyframes.history[:count].numpy().tobytes())
            self.history_revision = revision
        self.feature_prog['m_model'].write(np.eye(4, dtype=np.float32).tobytes())
        self.feature_prog['m_camera'].write(self.camera.gl_matrix())
        self.feature_prog['m_proj'].write(self.camera.proj_mat.gl_matrix())
        self.feature_prog['point_size'].value = 2.*self.scale
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.history_vao.render(mode=moderngl.POINTS, vertices=count)
        self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)

    def render_features(self, frame):
        self.feature_buffer.write(frame.feature_points.numpy().astype(np.float32).tobytes())
        self.feature_prog["m_model"].write(frame.T_WC.matrix().detach().cpu().numpy().reshape(4, 4).astype(np.float32).T.copy())
        self.feature_prog["m_camera"].write(self.camera.gl_matrix())
        self.feature_prog["m_proj"].write(self.camera.proj_mat.gl_matrix())
        self.feature_prog["point_size"].value = self.feature_size*self.scale
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.feature_vao.render(mode=moderngl.POINTS, vertices=self.feature_count)
        self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)
        if not self.feature_reported:
            print(f"Tracking feature overlay rendered: {self.feature_count} points", flush=True)
            self.feature_reported = True

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0, model_matrix=None):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = np.ascontiguousarray(
            (T_WC.matrix().numpy() if model_matrix is None else model_matrix).astype(np.float32).T)

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(cfg, states, keyframes, main2viz, viz2main, ready_event=None) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing and states.get_mode() != Mode.TERMINATED:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if ready_event is not None and not ready_event.is_set():
            ready_event.set()
            print("Viewer ready: MASt3R-PlanarSLAM (RGB, point map, camera poses)", flush=True)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    if window_config.disk_map is not None:
        window_config.disk_map.close()
    for textures in window_config.disk_surfel_textures.values():
        for texture in textures:
            texture.release()
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
    # Release imported CUDA IPC storages while the producer is still alive.
    # Window/renderer reference cycles can otherwise retain them to teardown.
    if ready_event is not None:
        import gc
        for owner in (states, keyframes):
            for name, value in list(vars(owner).items()):
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    setattr(owner, name, None)
        del value
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
