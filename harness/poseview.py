"""A picture of the robot's own configuration for the model. No SDK, no physics.

The fixed-base MuJoCo model is posed from the measured joints (arms and waist) and rendered from a fixed
three-quarter viewpoint with the hand tip (cyan sphere), where the last move aimed (yellow sphere), the
reachable workspace box (white edges) and the table height (translucent plane). It goes to the model as
ROBOT POSE VIEW next to the camera images, so the model can locate its hand when no camera shows it and
check that the last move went where it aimed: a second reading of the same state the joint numbers give.
Offscreen rendering needs MUJOCO_GL=cgl on macOS (set here if unset); without a GL context render()
returns None and the packet simply has no pose view.
"""
import io
import math
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .kinematics import ARM_JOINTS, OTHER_JOINTS, ROOT

CAM = {"pos": [1.25, -1.05, 1.35], "lookat": [0.3, 0.0, 0.85]}     # front right, a little above: the upper body, the hand, the box
TIP_RGBA = (0.0, 1.0, 1.0, 1.0)
AIM_RGBA = (1.0, 0.85, 0.1, 0.9)
BOX_RGBA = (1.0, 1.0, 1.0, 0.7)
TABLE_RGBA = (0.6, 0.6, 0.7, 0.25)


def _free_camera(pos, lookat):
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    d = lookat - pos; dist = float(np.linalg.norm(d)); f = d / dist
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist
    cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0]))); cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))
    return cam


def _line(scn, p0, p1, rgba, width=3.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def _sphere(scn, p, rgba, r=0.025):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def _box(scn, center, half, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float), np.asarray(center, float),
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


class PoseView:
    def __init__(self, cfg, arm, table_z=None, width=640, height=360, props=False):
        """props: keep the sim scene's table and cube visible (sim); False hides them (the real robot's room is not modelled)."""
        path = cfg["robot"]["model"]
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.arm, self.w, self.h = arm, int(width), int(height)
        names = ARM_JOINTS["left"] + ARM_JOINTS["right"] + OTHER_JOINTS
        self.adr = {}
        for n in names:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                self.adr[n] = int(self.model.jnt_qposadr[j])
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_hand_preview")
        if not props:
            for g in range(self.model.ngeom):
                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or ""
                if gname.startswith("pickup") or bname.startswith("pickup"):
                    self.model.geom_rgba[g, 3] = 0.0
        ws = cfg["workspace"]
        self.box_min, self.box_max = np.asarray(ws["box_min_m"], float), np.asarray(ws["box_max_m"], float)
        self.table_z = None if table_z is None else float(table_z)
        self.cam = _free_camera(CAM["pos"], CAM["lookat"])
        self.renderer = None
        self.error = ""
        self.font = ImageFont.load_default(size=14)

    def tip(self, joints):
        """Hand tip of the configured arm for these joints (robot frame), via the model."""
        self._pose(joints)
        return self.data.site_xpos[self.site].copy()

    def _pose(self, joints):
        self.data.qpos[:] = 0.0
        for n, a in self.adr.items():
            if n in joints:
                self.data.qpos[a] = float(joints[n])
        mujoco.mj_forward(self.model, self.data)

    def render(self, joints, last_target=None):
        """-> PIL image, or None when no GL context is available. joints: name -> rad; last_target: robot-frame point."""
        try:
            if self.renderer is None:
                os.environ.setdefault("MUJOCO_GL", "cgl")
                self.renderer = mujoco.Renderer(self.model, height=self.h, width=self.w)
            self._pose(joints)
            tip = self.data.site_xpos[self.site].copy()
            self.renderer.update_scene(self.data, camera=self.cam)
            scn = self.renderer.scene
            lo, hi = self.box_min, self.box_max
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            for i in range(8):
                for j in range(i + 1, 8):
                    if np.sum(corners[i] != corners[j]) == 1:          # an edge: one coordinate differs
                        _line(scn, corners[i], corners[j], BOX_RGBA)
            if self.table_z is not None:
                _box(scn, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, self.table_z], [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, 0.003], TABLE_RGBA)
            if last_target is not None:
                _sphere(scn, last_target, AIM_RGBA, 0.02)
                _line(scn, last_target, tip, AIM_RGBA, 2.0)
            _sphere(scn, tip, TIP_RGBA)
            im = Image.fromarray(self.renderer.render().copy())
            d = ImageDraw.Draw(im)
            t = tip * 100.0
            cap = f"ROBOT POSE VIEW (not a camera)   {self.arm} hand tip x={t[0]:.0f} y={t[1]:.0f} z={t[2]:.0f} cm"
            if self.table_z is not None:
                cap += f", {t[2] - self.table_z * 100:.0f} cm above the table"
            d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0)); d.text((4, 3), cap, fill=(255, 255, 255), font=self.font)
            d.rectangle([0, im.height - 20, im.width, im.height], fill=(0, 0, 0))
            d.text((4, im.height - 17), "cyan: hand tip   yellow: where the last move aimed   white box: workspace   plane: table height",
                   fill=(220, 220, 220), font=self.font)
            return im
        except Exception as e:                                       # no GL: the loop runs without the pose view
            self.error = str(e)
            return None
