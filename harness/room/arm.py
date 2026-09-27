"""Artur's arm policy (harness.loop.Episode) driving the room sim's arm.

The room layer walks the robot to the work; this hands the arm to the
closed-loop VLM policy in the rest of `harness`: it sees the head and wrist
cameras, picks one small hand move per step, and every move goes through
`harness.safety.SafetyGate`, the only producer of arm joint targets. Nothing
here computes joint targets itself.

`RoomArmBackend` is a `harness.executor.Backend` over `SimWorld`, the same
interface as `MockBackend` (sim), `DryRunBackend` and `ArmClientBackend` (the
real robot), so on hardware this module is the one to swap.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from ..executor import ArmExecutor, Backend
from ..kinematics import ARM_JOINTS, ArmKinematics
from ..loop import Episode
from ..perception import Perception
from ..recorder import Recorder
from ..safety import SafetyGate
from ..sim.mock_robot import free_camera
from .world import SimWorld


class RoomArmBackend(Backend):
    name = "room"
    dry_run = False

    def __init__(self, world: SimWorld, tick: Callable[[float], None], should_stop: Callable[[], bool]):
        self.world, self.tick, self.should_stop = world, tick, should_stop
        self.gate: SafetyGate | None = None
        self.vel = {n: 0.0 for n in world.arm_q}

    def joints(self) -> dict:
        # The room robot keeps its waist straight; the policy's kinematics poses it from these.
        return {**self.world.arm_q, "waist_roll_joint": 0.0, "waist_yaw_joint": 0.0}

    def velocities(self) -> dict:
        return dict(self.vel)

    def stream(self, arm, frames, dt):
        names = ARM_JOINTS[arm]
        prev = np.array([self.world.arm_q[n] for n in names])
        for f in frames:
            if self.should_stop():
                if self.gate is not None:
                    self.gate.estop.set()  # the operator stopped it: the episode ends after this step
                break
            f = np.asarray(f, float)
            self.world.arm_q.update({n: float(v) for n, v in zip(names, f)})
            self.vel.update({n: float(v) for n, v in zip(names, (f - prev) / dt)})
            prev = f
            self.tick(dt)
        for n in names:
            self.vel[n] = 0.0

    def hand(self, arm, closed) -> str:
        hand = f"{arm}_hand"
        if not closed:
            self.world.grip(hand, "open")
            return "hand opened"
        self.world.grip(hand, "close")
        # A fitted gripper reports whether it closed all the way; the sim's stand-in for that sensor.
        return "hand closed on the object" if self.world.held_by(hand) else "EMPTY grasp: the hand closed on nothing"

    def hand_state(self, arm):
        return self.world.grip_command[f"{arm}_hand"] == "closed"


class RoomCameras:
    """The policy's two views: the head camera (the room's fisheye) and a wrist camera."""

    def __init__(self, world: SimWorld, arm: str, width: int = 640, height: int = 360):
        self.world, self.arm, self.w, self.h = world, arm, width, height

    def frames(self) -> dict:
        context = Image.fromarray(self.world.capture().rgb)
        # On top of the wrist, looking along the forearm and down at the hand tip, as in MockBackend.
        site = self.world.model.site(f"{self.arm}_hand").id
        tip = self.world.data.site_xpos[site].copy()
        R = self.world.data.site_xmat[site].reshape(3, 3)
        wrist = tip - R @ np.array([0.13, 0.0, 0.0])
        cam = free_camera(wrist - R @ np.array([0.04, 0.0, 0.0]) + [0.0, 0.0, 0.11],
                          tip + R @ np.array([0.12, 0.0, 0.0]) + [0.0, 0.0, -0.12])
        wrist_img = Image.fromarray(self.world.render(cam, self.w, self.h))
        return {"CONTEXT VIEW": context, f"{self.arm.upper()} WRIST VIEW": wrist_img}


def run_episode(world: SimWorld, cfg, vlm, hand: str, task: str, surface_z: float,
                tick: Callable[[float], None], should_stop: Callable[[], bool] = lambda: False,
                confirm: Callable[[str], bool] | None = None, record: bool = True, log=print) -> dict:
    """One manipulation episode with Artur's loop. Returns its summary dict."""
    arm = hand.split("_")[0]
    kin = ArmKinematics(cfg["robot"]["model"], arm, float(cfg["limits"]["joint_margin_rad"]))
    gate = SafetyGate(cfg, kin, surface_z, live=False)
    backend = RoomArmBackend(world, tick, should_stop)
    backend.gate = gate
    ex = ArmExecutor(cfg, kin, gate, backend, arm)
    ex.confirm = confirm
    width = int(cfg["perception"]["width_px"])
    per = Perception(cfg, arm, RoomCameras(world, arm, width, width * 9 // 16), None)
    start = ex.go_to_joints(cfg["robot"]["start_pose_rad"][arm], "start pose")
    if not start.ok:
        return {"success": False, "reason": start.feedback, "steps": 0}
    rec = Recorder(cfg, "room", task) if record else None
    return Episode(cfg, vlm, ex, per, rec, log).run(task)
