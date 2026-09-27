"""The R1's head camera: what the real robot sees, emulated in MuJoCo.

Published specs for the R1 Basic/EDU head camera (MyBotShop's R1 docs; Unitree's
own product page only says "binocular camera"): a binocular depth camera, up to
150° horizontal × 124° vertical field of view, 1280×1088 RGB, 544×448 depth,
global shutter. That field of view is far wider than a pinhole lens can cover,
so this treats it as an equidistant fisheye.

MuJoCo only renders pinhole images, so `HeadCamera.capture` renders one wide
pinhole view and remaps it into fisheye pixels. `FisheyeModel` (pixel ⇄ ray) is
the part that carries over to the real robot, with calibrated numbers in place
of these nominal ones.

Unverified, measure on the robot: where the camera sits in the head and how far
it tilts down (`world.HEAD_CAMERA_POS`, `world.HEAD_CAMERA_PITCH`), whether the
real depth stream is radial range or z-depth, and its noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

H_FOV = math.radians(150)
V_FOV = math.radians(124)
RGB_SIZE = (1280, 1088)  # the camera's native RGB resolution (width, height)
DEPTH_SIZE = (544, 448)
MODEL_SCALE = 0.75  # images go to the model at 960×816, under vision models' ~1.15 MP limit
MIN_RANGE, MAX_RANGE = 0.1, 6.0  # m; outside this depth is reported as missing


@dataclass(frozen=True)
class FisheyeModel:
    """Equidistant fisheye: a pixel's distance from centre is proportional to its ray's angle
    from the optical axis. Camera frame: x right, y down, z forward (OpenCV)."""
    width: int
    height: int
    h_fov: float = H_FOV
    v_fov: float = V_FOV

    @property
    def fx(self) -> float:
        return self.width / self.h_fov  # px per radian

    @property
    def fy(self) -> float:
        return self.height / self.v_fov

    def rays(self, u, v) -> np.ndarray:
        """Unit rays for pixel coordinates (arrays), shape (..., 3)."""
        a = (np.asarray(u, float) - self.width / 2) / self.fx
        b = (np.asarray(v, float) - self.height / 2) / self.fy
        theta = np.hypot(a, b)
        phi = np.arctan2(b, a)
        s = np.sin(theta)
        return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)], axis=-1)

    def project(self, points_cam) -> tuple[np.ndarray, np.ndarray]:
        """Pixel (u, v) of camera-frame points, shape (..., 3)."""
        p = np.asarray(points_cam, float)
        theta = np.arctan2(np.hypot(p[..., 0], p[..., 1]), p[..., 2])
        phi = np.arctan2(p[..., 1], p[..., 0])
        return (self.width / 2 + self.fx * theta * np.cos(phi),
                self.height / 2 + self.fy * theta * np.sin(phi))

    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        v, u = np.mgrid[0:self.height, 0:self.width]
        return u + 0.5, v + 0.5


@dataclass
class Capture:
    """One frame, as the real robot would have it."""
    rgb: np.ndarray  # (h, w, 3) uint8, fisheye, at the model's resolution
    depth: np.ndarray  # (h, w) float32 range along each pixel's ray in m; NaN where missing
    self_mask: np.ndarray  # (h, w) bool: the robot's own body (known from its joint angles)
    T_robot_cam: np.ndarray  # 4x4: camera (OpenCV axes) in the robot frame at capture time
    rgb_model: FisheyeModel
    depth_model: FisheyeModel

    def point(self, u: float, v: float, window: int = 2) -> np.ndarray | None:
        """Robot-frame point seen at RGB pixel (u, v): the median depth in a small window."""
        du = u * self.depth_model.width / self.rgb_model.width
        dv = v * self.depth_model.height / self.rgb_model.height
        i, j = int(dv), int(du)
        patch = self.depth[max(0, i - window):i + window + 1, max(0, j - window):j + window + 1]
        patch = patch[np.isfinite(patch)]
        if patch.size == 0:
            return None
        ray = self.depth_model.rays(du, dv)
        p_cam = ray * float(np.median(patch))
        return (self.T_robot_cam @ np.r_[p_cam, 1.0])[:3]

    def cloud(self, stride: int = 2) -> np.ndarray:
        """Robot-frame points for valid depth pixels, excluding the robot's own body."""
        u, v = self.depth_model.pixel_grid()
        sel = (slice(None, None, stride), slice(None, None, stride))
        d, keep = self.depth[sel], np.isfinite(self.depth[sel]) & ~self.self_mask_depth()[sel]
        rays = self.depth_model.rays(u[sel][keep], v[sel][keep])
        p_cam = rays * d[keep][:, None]
        return p_cam @ self.T_robot_cam[:3, :3].T + self.T_robot_cam[:3, 3]

    def self_mask_depth(self) -> np.ndarray:
        h, w = self.depth.shape
        rows = (np.arange(h) * self.self_mask.shape[0] / h).astype(int)
        cols = (np.arange(w) * self.self_mask.shape[1] / w).astype(int)
        return self.self_mask[np.ix_(rows, cols)]


class HeadCamera:
    """Renders the head camera in MuJoCo and converts it to what the real camera delivers."""

    def __init__(self, model: mujoco.MjModel, camera: str = "head", noise: bool = True, seed: int = 0):
        self.model = model
        self.cam_id = model.camera(camera).id
        self.rgb_model = FisheyeModel(round(RGB_SIZE[0] * MODEL_SCALE), round(RGB_SIZE[1] * MODEL_SCALE))
        self.depth_model = FisheyeModel(*DEPTH_SIZE)
        # Wide pinhole render with the same centre resolution as the fisheye output.
        self.tan_h, self.tan_v = math.tan(H_FOV / 2), math.tan(V_FOV / 2)
        f = self.rgb_model.fx
        self.pw, self.ph = int(2 * f * self.tan_h), int(2 * f * self.tan_v)
        self.f_pin = self.ph / (2 * self.tan_v)
        self.renderer = mujoco.Renderer(model, self.ph, self.pw)
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self._maps = {name: self._remap(m) for name, m in
                      (("rgb", self.rgb_model), ("depth", self.depth_model))}
        pelvis_root = model.body("pelvis").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == pelvis_root}

    def _remap(self, fm: FisheyeModel):
        """For each fisheye pixel: its source pinhole pixel, the range factor, and validity."""
        rays = fm.rays(*fm.pixel_grid())
        z = rays[..., 2]
        valid = z > 1e-3
        xn = np.where(valid, rays[..., 0] / np.maximum(z, 1e-3), 0)
        yn = np.where(valid, rays[..., 1] / np.maximum(z, 1e-3), 0)
        col = xn * self.f_pin + self.pw / 2
        row = yn * self.f_pin + self.ph / 2
        valid &= (col >= 0) & (col < self.pw - 1) & (row >= 0) & (row < self.ph - 1)
        # MuJoCo depth is distance along the optical axis; range along the ray is that / cos(theta).
        return (np.clip(row, 0, self.ph - 1).astype(np.int32), np.clip(col, 0, self.pw - 1).astype(np.int32),
                1.0 / np.maximum(z, 1e-3), valid)

    def capture(self, data: mujoco.MjData, T_robot_world: np.ndarray) -> Capture:
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # collision geoms stay invisible, as in the viewer
        r = self.renderer
        r.update_scene(data, self.cam_id, opt)
        pin_rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, self.cam_id, opt)
        pin_depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, self.cam_id, opt)
        seg = r.render().copy()
        r.disable_segmentation_rendering()

        rows, cols, _, valid = self._maps["rgb"]
        rgb = np.where(valid[..., None], pin_rgb[rows, cols], 0).astype(np.uint8)
        geom_ids = seg[..., 0][rows, cols]
        is_geom = seg[..., 1][rows, cols] == int(mujoco.mjtObj.mjOBJ_GEOM)
        bodies = np.where(is_geom & (geom_ids >= 0), self.model.geom_bodyid[np.maximum(geom_ids, 0)], -1)
        self_mask = np.isin(bodies, list(self.robot_bodies)) & valid

        drows, dcols, factor, dvalid = self._maps["depth"]
        depth = pin_depth[drows, dcols] * factor
        if self.noise:  # stereo-like: error grows with the square of range
            depth = depth + self.rng.normal(0, 1, depth.shape) * (0.002 + 0.004 * depth ** 2)
        depth = np.where(dvalid & (depth > MIN_RANGE) & (depth < MAX_RANGE), depth, np.nan).astype(np.float32)

        # Camera pose: MuJoCo camera axes are x right, y up, z backward; OpenCV's are x right, y down, z forward.
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = data.cam_xmat[self.cam_id].reshape(3, 3) @ np.diag([1, -1, -1])
        T_world_cam[:3, 3] = data.cam_xpos[self.cam_id]
        return Capture(rgb, depth, self_mask, T_robot_world @ T_world_cam, self.rgb_model, self.depth_model)

    def close(self) -> None:
        self.renderer.close()


def annotate(rgb: np.ndarray, step: int = 80) -> np.ndarray:
    """Pixel-coordinate ticks and labels along the image edges, so a model can name pixels."""
    import cv2
    img = rgb.copy()
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, 8), (255, 255, 0), 1)
        cv2.line(img, (x, h - 9), (x, h - 1), (255, 255, 0), 1)
        if x:
            cv2.putText(img, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (8, y), (255, 255, 0), 1)
        cv2.line(img, (w - 9, y), (w - 1, y), (255, 255, 0), 1)
        if y:
            cv2.putText(img, str(y), (11, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return img
