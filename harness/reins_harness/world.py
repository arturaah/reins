"""The simulated world an LLM drives: the R1 walking, moving its arms, and
picking things up.

Kinematic throughout, like `reins_loco.sim.SimLoco`, which it builds on. The
base follows velocity commands, the arms follow joint trajectories exactly, and
a held object rides along with the hand. Nothing is stepped through physics.

This file is split in two, and the split is the point:

- **The sim** (`true_pose`, `object_pos`, `surfaces`, `ground_truth`, the grasp
  check) is how the world really is. Only the simulator itself, the human's
  preview drawings and tests use it. No model-facing tool reads it.
- **The robot's senses** (`odom_pose`, `proprioception`, `capture`, `obstacles`)
  are what the real R1 could know: its joint angles, dead-reckoned odometry
  that drifts, the head camera's image and depth, and an obstacle map built
  only from that depth.

Frames: `odom` is fixed where the robot started, estimated by dead reckoning,
and is the frame contract walk steps call `map`. `robot` has its origin on the
floor under the pelvis, x forward, y left, z up (contract/README.md).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from reins_loco.base import OK, Pose2, wrap
from reins_loco.sim import SimLoco, walk_preview

from .camera import V_FOV, Capture, HeadCamera

REPO = Path(__file__).resolve().parents[2]

SCENE = REPO / "sim" / "models" / "r1" / "scene_harness.xml"
HANDS = ("left_hand", "right_hand")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
# Relaxed arms, the same pose walk_preview.Poser uses.
HOME = {"left_shoulder_roll_joint": 0.15, "right_shoulder_roll_joint": -0.15,
        "left_elbow_joint": 1.0, "right_elbow_joint": 1.0}
GRASP_TOLERANCE = 0.03  # m from the hand site to the object's centre
HAND_SITE_OFFSET = (0.13, 0.0, 0.0)  # on the wrist roll link, as in R1_fixed_base.xml
# Unverified: measure on the robot. Front of the head, in the torso link's frame.
HEAD_CAMERA_POS = (0.118, 0.0, 0.37)
HEAD_CAMERA_PITCH = math.radians(20)  # down from horizontal
# Dead reckoning error: odometry integrates velocity with these scale errors.
ODOM_LINEAR_SCALE = 1.03
ODOM_ANGULAR_SCALE = 0.97
MAP_CELL = 0.05  # m, obstacle map resolution
# Radius kept clear around the pelvis. The R1's body front is about 0.15 m out; the loco
# sim's 0.25 m default keeps it too far from a table to reach anything on it.
FOOTPRINT = 0.20
MAP_MIN_Z, MAP_MAX_Z = 0.08, 1.8  # m: depth points in this height band are obstacles


def side(hand: str) -> str:
    if hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
    return hand.split("_")[0]


def arm_joint_names(hand: str) -> list[str]:
    return [f"{side(hand)}_{j}_joint" for j in ARM_JOINTS]


def pose_matrix(p: Pose2, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (p.x, p.y, z)
    return T


@dataclass
class Surface:
    """The top of a box obstacle (sim side: used to settle released objects)."""
    name: str
    x: float
    y: float
    yaw: float
    hx: float
    hy: float
    top: float

    def contains(self, x: float, y: float, inset: float = 0.0) -> bool:
        lx, ly = self.local(x, y)
        return abs(lx) <= self.hx - inset and abs(ly) <= self.hy - inset

    def local(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * (x - self.x) + s * (y - self.y), -s * (x - self.x) + c * (y - self.y)

    def nearest_point(self, x: float, y: float, inset: float) -> tuple[float, float]:
        lx, ly = self.local(x, y)
        lx = min(max(lx, -self.hx + inset), self.hx - inset)
        ly = min(max(ly, -self.hy + inset), self.hy - inset)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * lx - s * ly, self.y + s * lx + c * ly


@dataclass
class Grip:
    obj: str
    local_pos: np.ndarray  # object position in the hand site's frame
    local_rot: np.ndarray  # object orientation in the hand site's frame, 3x3


class ObstacleMap:
    """Occupied floor cells in the odom frame, from depth only. Unseen space counts as free,
    as it would on the real robot: walking into something unseen is caught by bumping into it."""

    def __init__(self):
        self.counts: dict[tuple[int, int], int] = {}
        self._tree: cKDTree | None = None

    def add(self, points_odom: np.ndarray) -> None:
        keep = (points_odom[:, 2] > MAP_MIN_Z) & (points_odom[:, 2] < MAP_MAX_Z)
        for cell in map(tuple, np.floor(points_odom[keep, :2] / MAP_CELL).astype(int)):
            self.counts[cell] = self.counts.get(cell, 0) + 1
        occupied = [c for c, n in self.counts.items() if n >= 2]  # a single noisy return isn't a wall
        self._tree = cKDTree((np.array(occupied) + 0.5) * MAP_CELL) if occupied else None

    def clearance(self, x: float, y: float) -> float:
        """Distance from the robot's footprint edge to the nearest seen obstacle (inf if none)."""
        if self._tree is None:
            return math.inf
        return float(self._tree.query([x, y])[0]) - MAP_CELL / 2 - FOOTPRINT

    def cells(self) -> np.ndarray:
        return (np.array([c for c, n in self.counts.items() if n >= 2]).reshape(-1, 2) + 0.5) * MAP_CELL


class _WorldPoser(walk_preview.Poser):
    """Poses the base and gait like walk_preview.Poser, then the arms and held objects."""

    def __init__(self, model: mujoco.MjModel, world: "SimWorld"):
        super().__init__(model)
        self.world = world

    def set(self, data, x, y, yaw, gait_phase=None) -> None:
        super().set(data, x, y, yaw, gait_phase)
        self.world._apply_arms_and_grips(data)


class OdometryLoco:
    """The `Loco` the walking controller gets: commands go to the sim, but the pose it
    steers by is dead-reckoned odometry, as on the real robot."""

    def __init__(self, world: "SimWorld"):
        self.world = world
        self.limits = world.loco.limits

    def __getattr__(self, name):  # damp, stance, start, set_velocity, stop, fsm_id
        return getattr(self.world.loco, name)

    def pose(self) -> Pose2:
        return self.world.odom_pose()

    def pose_is_estimated(self) -> bool:
        return True


class SimWorld:
    def __init__(self, scene: Path = SCENE, start: Pose2 = Pose2(0.0, 0.0, 0.0),
                 odom_drift: bool = True, depth_noise: bool = True):
        self.model = _build_model(scene)
        self.arm_q = {name: HOME.get(name, 0.0) for hand in HANDS for name in arm_joint_names(hand)}
        self.grips: dict[str, Grip] = {}
        self.grip_command = {hand: "open" for hand in HANDS}
        self._qadr = {name: int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in self.arm_q}
        self._sites = {hand: self.model.site(hand).id for hand in HANDS}
        self.objects = {self.model.body(b).name.removeprefix("object_"): b
                        for b in range(self.model.nbody) if self.model.body(b).name.startswith("object_")}
        self.loco = SimLoco(model=self.model, poser=_WorldPoser(self.model, self), start=start,
                            footprint=FOOTPRINT)
        self.data = self.loco.data
        self.surfaces = _surfaces(self.model, self.data)
        self.drift = (ODOM_LINEAR_SCALE, ODOM_ANGULAR_SCALE) if odom_drift else (1.0, 1.0)
        self._start = start
        self._odom = Pose2(0.0, 0.0, 0.0)
        self.walker = OdometryLoco(self)
        self.camera = HeadCamera(self.model, noise=depth_noise)
        self.obstacles = ObstacleMap()
        self.last_capture: Capture | None = None
        self.last_capture_odom: Pose2 | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    # --- time -----------------------------------------------------------------

    def advance(self, dt: float) -> None:
        self.loco.advance(dt)
        vx, vy, wz = self.loco.velocity()  # what the legs report doing, integrated with an error
        lin, ang = self.drift
        p = self._odom
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        self._odom = Pose2(p.x + lin * (c * vx - s * vy) * dt, p.y + lin * (s * vx + c * vy) * dt,
                           wrap(p.yaw + ang * wz * dt))

    def refresh(self) -> None:
        """Re-pose the model after changing arms or grips, without advancing time."""
        self.loco._update_model()

    # --- the robot's senses ------------------------------------------------------

    def odom_pose(self) -> Pose2:
        """Where the robot thinks it is, relative to where it started."""
        return self._odom

    def to_robot(self, xyz_odom, pose: Pose2 | None = None) -> np.ndarray:
        p = pose or self._odom
        bx, by = p.to_body(float(xyz_odom[0]), float(xyz_odom[1]))
        return np.array([bx, by, float(xyz_odom[2])])

    def to_odom(self, xyz_robot, pose: Pose2 | None = None) -> np.ndarray:
        p = pose or self._odom
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        x, y, z = (float(v) for v in xyz_robot)
        return np.array([p.x + c * x - s * y, p.y + s * x + c * y, z])

    def hand_in_robot(self, hand: str) -> np.ndarray:
        """Forward kinematics from the joint angles: exact on the real robot too."""
        return self._true_to_robot(self.data.site_xpos[self._sites[hand]])

    def proprioception(self) -> dict:
        joints = {self.model.joint(j).name: round(float(self.data.qpos[self.model.jnt_qposadr[j]]), 3)
                  for j in range(self.model.njnt)
                  if self.model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_HINGE)
                  and self.model.jnt_range[j][0] < self.model.jnt_range[j][1]}  # skip fixed dummies
        p = self._odom
        return {
            "odometry": {"x": round(p.x, 3), "y": round(p.y, 3), "yaw": round(p.yaw, 3),
                         "note": "dead reckoning from where the robot started; drifts as it walks"},
            "imu": {"roll": 0.0, "pitch": 0.0},
            "walking": self.loco.fsm_id() == 811 and max(abs(v) for v in self.loco.velocity()) > 0.01,
            "hands": {h: {"position_robot_frame": _r(self.hand_in_robot(h)), "grip": self.grip_command[h]}
                      for h in HANDS},
            "joint_positions_rad": joints,
        }

    def capture(self) -> Capture:
        """A head camera frame. Also updates the obstacle map from its depth."""
        cap = self.camera.capture(self.data, np.linalg.inv(self._T_world_robot()))
        self.last_capture, self.last_capture_odom = cap, self._odom
        cloud = cap.cloud()
        # The robot can't know what it's holding, but it knows where its hands are.
        for hand in HANDS:
            near = np.linalg.norm(cloud - self.hand_in_robot(hand), axis=1) < 0.15
            cloud = cloud[~near]
        T = pose_matrix(self._odom)
        self.obstacles.add(cloud @ T[:3, :3].T + T[:3, 3])
        return cap

    # --- arms and grips ------------------------------------------------------------

    def set_arm(self, q: dict[str, float]) -> None:
        unknown = set(q) - set(self.arm_q)
        if unknown:
            raise ValueError(f"not arm joints: {sorted(unknown)}")
        self.arm_q.update({k: float(v) for k, v in q.items()})
        self.refresh()

    def grip(self, hand: str, action: str) -> str:
        """Close or open a hand. Returns only what the robot would know: that it did.

        In the sim, closing attaches the nearest object if it's within
        GRASP_TOLERANCE of the hand, and opening drops what it held onto
        whatever is below. Whether that happened, the model has to see.
        """
        if action == "close":
            self.grip_command[hand] = "closed"
            if hand not in self.grips:
                free = [n for n in self.objects if n not in {g.obj for g in self.grips.values()}]
                hand_xyz = self.data.site_xpos[self._sites[hand]].copy()
                near = [n for n in free if np.linalg.norm(self.object_pos(n) - hand_xyz) <= GRASP_TOLERANCE]
                if near:
                    obj = min(near, key=lambda n: np.linalg.norm(self.object_pos(n) - hand_xyz))
                    rot = self.data.site_xmat[self._sites[hand]].reshape(3, 3)
                    obj_rot = self.data.xmat[self.objects[obj]].reshape(3, 3)
                    self.grips[hand] = Grip(obj, rot.T @ (self.object_pos(obj) - hand_xyz), rot.T @ obj_rot)
            return f"{hand} closed"
        if action == "open":
            self.grip_command[hand] = "open"
            grip = self.grips.pop(hand, None)
            if grip is not None:
                self._settle(grip.obj)
            return f"{hand} opened"
        raise ValueError(f"grip action must be close or open, got {action!r}")

    # --- the sim: never shown to the model ------------------------------------

    def true_pose(self) -> Pose2:
        return self.loco.pose()

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.mocap_pos[self._mocap(name)].copy()

    def object_half_height(self, name: str) -> float:
        geom = int(self.model.body_geomadr[self.objects[name]])
        size = self.model.geom_size[geom]
        kind = int(self.model.geom_type[geom])
        return float(size[1] if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
                     else size[0] if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE) else size[2])

    def held_by(self, hand: str) -> str | None:
        grip = self.grips.get(hand)
        return grip.obj if grip else None

    def surface_under(self, x: float, y: float, below_z: float = math.inf) -> tuple[str, float]:
        best = ("floor", 0.0)
        for s in self.surfaces.values():
            if s.contains(x, y) and best[1] < s.top <= below_z + 1e-6:
                best = (s.name, s.top)
        return best

    def ground_truth(self) -> dict:
        """The true state, for tests and evaluation only."""
        p = self.true_pose()
        return {"robot": {"x": round(p.x, 3), "y": round(p.y, 3), "yaw": round(p.yaw, 3)},
                "objects": {n: {"map": _r(self.object_pos(n)), "held_by": next(
                    (h for h, g in self.grips.items() if g.obj == n), None),
                    "on": self.surface_under(*self.object_pos(n)[:2], self.object_pos(n)[2])[0]}
                    for n in self.objects}}

    def true_to_odom(self, xyz_world) -> np.ndarray:
        """World point in the odom frame, as the robot would place it given its drift (for previews)."""
        return self.to_odom(self._true_to_robot(xyz_world))

    def odom_to_true(self, xyz_odom) -> np.ndarray:
        """Where an odom-frame point really is (for drawing plans in the scene)."""
        r = self.to_robot(xyz_odom)
        return (self._T_world_robot() @ np.r_[r, 1.0])[:3]

    def _T_world_robot(self) -> np.ndarray:
        return pose_matrix(self.true_pose())

    def _true_to_robot(self, xyz_world) -> np.ndarray:
        return (np.linalg.inv(self._T_world_robot()) @ np.r_[np.asarray(xyz_world, float), 1.0])[:3]

    def _settle(self, name: str) -> None:
        mocap = self._mocap(name)
        p = self.data.mocap_pos[mocap]
        half = self.object_half_height(name)
        _, top = self.surface_under(p[0], p[1], p[2] - half)
        p[2] = top + half
        m = self.data.xmat[self.objects[name]].reshape(3, 3)
        self.data.mocap_quat[mocap] = walk_preview.yaw_quat(math.atan2(m[1, 0], m[0, 0]))
        self.refresh()

    def _apply_arms_and_grips(self, data: mujoco.MjData) -> None:
        for name, q in self.arm_q.items():
            data.qpos[self._qadr[name]] = q
        mujoco.mj_kinematics(self.model, data)
        for hand, grip in self.grips.items():
            site = self._sites[hand]
            rot = data.site_xmat[site].reshape(3, 3)
            mocap = self._mocap(grip.obj)
            data.mocap_pos[mocap] = data.site_xpos[site] + rot @ grip.local_pos
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rot @ grip.local_rot).ravel())
            data.mocap_quat[mocap] = quat
        mujoco.mj_forward(self.model, data)

    def _mocap(self, name: str) -> int:
        return int(self.model.body_mocapid[self.objects[name]])

    # --- rendering for the human -----------------------------------------------------

    def render(self, camera: str | mujoco.MjvCamera, width: int = 640, height: int = 480,
               decorate=None) -> np.ndarray:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        renderer = self._renderers[key]
        opt = mujoco.MjvOption()
        opt.geomgroup[0] = 0  # hide collision geoms
        renderer.update_scene(self.data, camera, opt)
        if decorate:
            decorate(renderer.scene)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self.camera.close()


def png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def _r(values) -> list[float]:
    return [round(float(v), 3) for v in values]


def _build_model(scene: Path) -> mujoco.MjModel:
    """The scene plus what the R1 file lacks: hand sites and the head camera."""
    spec = mujoco.MjSpec.from_file(str(scene))
    for hand, rgba in (("left_hand", [0, 1, 1, 1]), ("right_hand", [1, 0.4, 0, 1])):
        spec.body(f"{side(hand)}_wrist_roll_link").add_site(
            name=hand, pos=list(HAND_SITE_OFFSET), size=[0.01, 0, 0], rgba=rgba, group=4)
    s, c = math.sin(HEAD_CAMERA_PITCH), math.cos(HEAD_CAMERA_PITCH)
    # Vertical FOV of the wide pinhole render that HeadCamera remaps into the fisheye.
    cam = spec.body("torso_link").add_camera(name="head", pos=list(HEAD_CAMERA_POS),
                                             fovy=math.degrees(V_FOV))
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [0, -1, 0, s, 0, c]  # image right is the robot's right; tilted down
    # The fisheye emulation renders one wide pinhole frame; make room for it.
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 2800, 1440
    return spec.compile()


def _surfaces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Surface]:
    found = {}
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("obstacle") or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        x, y, z = (float(v) for v in data.geom_xpos[g])
        m = data.geom_xmat[g].reshape(3, 3)
        hx, hy, hz = (float(v) for v in model.geom_size[g])
        short = name.removeprefix("obstacle_")
        found[short] = Surface(short, x, y, math.atan2(m[1, 0], m[0, 0]), hx, hy, z + hz)
    return found


__all__ = ["SimWorld", "HANDS", "HOME", "arm_joint_names", "png_bytes", "OK"]
