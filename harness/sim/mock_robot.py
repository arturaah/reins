"""Kinematic mock of the R1 arms on the MuJoCo scene: no physics, no hardware, no SDK.

Joints jump through the streamed frames (optionally in real time). Hand type "virtual" grasps the
scene's pickup_cube when the hand tip is within hand.grasp_reach_m and carries it until RELEASE.
Images: MuJoCo offscreen renders of a head-mounted context camera and a wrist camera, when an
OpenGL context is available (MUJOCO_GL=cgl on macOS); otherwise flat placeholder images.
"""
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from ..executor import Backend, StreamError
from ..kinematics import ARM_JOINTS, OTHER_JOINTS, ROOT

CUBE_START = np.array([0.30, 0.16, 0.685])       # y is mirrored for the right arm (the scene XML places it for the left)
PLATE_POS = np.array([0.34, 0.28, 0.665])        # the white plate on the table (geom pickup_plate): in the head camera's view, in reach
CONTEXT_CAM = {"pos": [0.13, 0.0, 1.22], "lookat": [0.34, 0.0, 0.70]}   # head camera stand-in: in front of the face, looking forward-down


def free_camera(pos, lookat):
    """MjvCamera placed at pos looking at lookat (MuJoCo free cameras are defined by lookat, distance, azimuth, elevation)."""
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    d = lookat - pos; dist = float(np.linalg.norm(d)); f = d / dist
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist
    cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0]))); cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))
    return cam


class MockBackend(Backend):
    name = "mock"

    def __init__(self, cfg, realtime=False, render=True, start="raised"):
        """start: 'rest' (the controller's standing pose) or 'raised' (cfg robot.start_pose_rad)."""
        self.cfg = cfg
        path = Path(cfg["robot"]["model"]); path = path if path.is_absolute() else ROOT / path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime
        self.names = [n for side in ("left", "right") for n in ARM_JOINTS[side]] + OTHER_JOINTS
        self.adr = {n: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in self.names}
        self.q = {n: 0.0 for n in self.names}
        # the built-in controller's standing arm pose (measured 2026-09-26, tools/plans/cup_grab_right.json t=0)
        rest = {"shoulder_pitch": 0.16, "shoulder_roll": -0.02, "shoulder_yaw": 0.53, "elbow": 1.45, "wrist_roll": -0.05}
        for side, sgn in (("left", -1), ("right", 1)):
            for j, v in rest.items():
                self.q[f"{side}_{j}_joint"] = v * (sgn if j in ("shoulder_roll", "shoulder_yaw", "wrist_roll") else 1)
        if start == "raised":
            for side in ("left", "right"):
                for n, v in zip(ARM_JOINTS[side], cfg["robot"]["start_pose_rad"][side]):
                    self.q[n] = float(v)
        self.vel = {n: 0.0 for n in self.names}
        self.sites = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand_preview") for s in ("left", "right")}
        cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pickup_cube")
        self.cube_mocap = int(self.model.body_mocapid[cube]) if cube >= 0 else -1
        self.side = 1.0 if cfg["robot"]["arm"] == "left" else -1.0        # mirror the props for the working arm
        self.cube_start = CUBE_START * [1, self.side, 1]
        self.plate = PLATE_POS * [1, self.side, 1]
        if self.cube_mocap >= 0:
            self.data.mocap_pos[self.cube_mocap] = self.cube_start
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pickup_table")
        if table >= 0:
            self.model.geom_pos[table] = [0.28, 0.20 * self.side, 0.3325]; self.model.geom_size[table] = [0.16, 0.16, 0.3325]
        plate = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pickup_plate")
        if plate >= 0:                                                     # drawn where the success checks expect it
            self.model.geom_pos[plate] = self.plate + [0, 0, 0.004]
        self.hand_closed = {"left": False, "right": False}
        self.holding = {"left": None, "right": None}           # offset of the cube from the hand tip while held
        self.engaged = False
        self.frames_sent = 0
        self._renderer = None
        self._viewer = None
        self._viewer_estop = None
        self._render_ok = render
        self._forward()

    # -- state ------------------------------------------------------------------------------------
    def _forward(self):
        for n, a in self.adr.items():
            self.data.qpos[a] = self.q[n]
        mujoco.mj_kinematics(self.model, self.data)
        for side, off in self.holding.items():
            if off is not None and self.cube_mocap >= 0:
                self.data.mocap_pos[self.cube_mocap] = self.tip(side) + off
        mujoco.mj_kinematics(self.model, self.data)
        self.sync_viewer()

    # -- optional live simulation window --------------------------------------------------------------
    def open_viewer(self, estop):
        import mujoco.viewer
        self._viewer_estop = estop
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
            key_callback=lambda key: estop.set() if key in (88, 120, 256) else None)
        with self._viewer.lock():
            self._viewer.cam.lookat[:] = [0.25, 0.12 * self.side, 0.85]
            self._viewer.cam.distance = 1.5
            self._viewer.cam.azimuth = 135 if self.side < 0 else -135
            self._viewer.cam.elevation = -25
        self._viewer.sync()

    def sync_viewer(self):
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            self._viewer_estop.set()
            return
        self._viewer.sync()

    def wait_viewer(self):
        while self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(0.03)

    def close_viewer(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def tip(self, side):
        return self.data.site_xpos[self.sites[side]].copy()

    def cube_pos(self):
        return self.data.mocap_pos[self.cube_mocap].copy() if self.cube_mocap >= 0 else None

    def joints(self):
        self.sync_viewer()
        return dict(self.q)

    def velocities(self):
        return dict(self.vel)

    # -- Backend --------------------------------------------------------------------------------------
    def engage(self): self.engaged = True
    def release(self): self.engaged = False
    def freeze(self): pass

    def stream(self, arm, frames, dt):
        names = ARM_JOINTS[arm]
        prev = np.array([self.q[n] for n in names])
        for f in frames:
            self.sync_viewer()
            if self._viewer_estop is not None and self._viewer_estop.is_set():
                raise StreamError("simulation stopped from the viewer")
            f = np.asarray(f, float)
            for n, v in zip(names, f):
                self.q[n] = float(v)
            for n, v in zip(names, (f - prev) / dt):
                self.vel[n] = float(v)
            prev = f
            self.frames_sent += 1
            self._forward()
            if self.realtime:
                time.sleep(dt)
        for n in names:
            self.vel[n] = 0.0

    def hand(self, arm, closed):
        kind = self.cfg["hand"]["type"]
        if kind == "none":
            time.sleep(0.0 if not self.realtime else float(self.cfg["hand"]["pause_s"]))
            return "this robot has no hand: nothing to grasp with, the arm paused"
        self.hand_closed[arm] = closed
        if closed:
            reach = float(self.cfg["hand"]["grasp_reach_m"])
            if self.cube_mocap >= 0 and np.linalg.norm(self.cube_pos() - self.tip(arm)) <= reach:
                self.holding[arm] = self.cube_pos() - self.tip(arm)
                return "hand closed on the object"
            return "EMPTY grasp: the hand closed on nothing"
        self.holding[arm] = None
        if self.cube_mocap >= 0:
            c = self.cube_pos(); c[2] = self.cfg["workspace"]["sim_table_z_m"] + 0.02   # drop to the table
            self.data.mocap_pos[self.cube_mocap] = c
        self._forward()
        return "hand opened"

    def hand_state(self, arm):
        return None if self.cfg["hand"]["type"] == "none" else self.hand_closed[arm]

    # -- images ---------------------------------------------------------------------------------------
    def render(self, view, arm="right", width=640, height=360):
        """view: 'context' or 'wrist'. Returns an RGB uint8 array, or None when no GL context is available."""
        if not self._render_ok:
            return None
        try:
            if self._renderer is None:
                os.environ.setdefault("MUJOCO_GL", "cgl")
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._forward()
            mujoco.mj_forward(self.model, self.data)
            if view == "context":
                cam = free_camera(CONTEXT_CAM["pos"], CONTEXT_CAM["lookat"])
            else:
                # a camera on top of the wrist looking along the forearm and down at the hand tip
                tip = self.tip(arm)
                R = self.data.site_xmat[self.sites[arm]].reshape(3, 3)
                wrist = tip - R @ np.array([0.13, 0.0, 0.0])
                cam = free_camera(wrist - R @ np.array([0.04, 0.0, 0.0]) + [0.0, 0.0, 0.11],
                                  tip + R @ np.array([0.12, 0.0, 0.0]) + [0.0, 0.0, -0.12])
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render().copy()
        except Exception as e:                    # no GL: the loop still runs with placeholder images
            self._render_ok = False
            self.render_error = str(e)
            return None

    def context_camera(self, width=640, height=360):
        """Pinhole model of the context render, for the hand-marker overlay (robot frame)."""
        pos = np.asarray(CONTEXT_CAM["pos"], float); fwd = np.asarray(CONTEXT_CAM["lookat"], float) - pos; fwd /= np.linalg.norm(fwd)
        fovy = float(self.model.vis.global_.fovy)
        fy = height / 2 / np.tan(np.radians(fovy) / 2); fx = fy
        return {"pos": pos, "forward": fwd, "up": np.array([0, 0, 1.0]), "fx": fx, "fy": fy, "cx": width / 2, "cy": height / 2}
