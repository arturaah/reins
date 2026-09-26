"""Loading the Unitree R1 and posing it kinematically."""

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "unitree_r1"
SCENE_XML = ASSETS / "scene.xml"

# Pelvis height with all joints at zero and the foot contact spheres touching the floor.
STAND_HEIGHT = 0.743

# Nominal stride used by the cosmetic gait, in metres per full left+right cycle.
STRIDE = 0.6


def load(path: Path = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Poser:
    """Sets the R1 to a base pose plus an optional cosmetic walking cycle.

    This is kinematic only: it writes qpos and runs forward kinematics. There is
    no balance or locomotion controller, so it shows *where* the robot will go,
    not whether it can physically get there.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._qadr = {
            name: model.jnt_qposadr[model.joint(name).id]
            for name in (
                "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
                "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_elbow_joint", "right_elbow_joint",
            )
        }
        base = model.joint("floating_base_joint").id
        self._base = model.jnt_qposadr[base]

    def set(self, data: mujoco.MjData, x: float, y: float, yaw: float,
            gait_phase: float | None = None) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        q = data.qpos
        a = self._qadr

        # Relaxed arms: slightly out from the body, forearms angled down.
        # (Elbow zero on the R1 holds the forearm horizontal.)
        q[a["left_shoulder_roll_joint"]] = 0.15
        q[a["right_shoulder_roll_joint"]] = -0.15
        q[a["left_elbow_joint"]] = 1.0
        q[a["right_elbow_joint"]] = 1.0

        z = STAND_HEIGHT
        if gait_phase is not None:
            s = np.sin(2 * np.pi * gait_phase)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                fwd = sign * s  # >0 while this leg swings forward
                # Hip pitch is negative for forward flexion on the R1.
                hip = -0.2 - 0.35 * fwd
                knee = 0.4 + 0.5 * max(fwd, 0.0)
                q[a[f"{side}_hip_pitch_joint"]] = hip
                q[a[f"{side}_knee_joint"]] = knee
                q[a[f"{side}_ankle_pitch_joint"]] = -(hip + knee)  # keep the sole level
                # Arms counter-swing against the legs.
                q[a[f"{side}_shoulder_pitch_joint"]] = 0.4 * fwd
            z -= 0.05

        q[self._base:self._base + 3] = (x, y, z)
        q[self._base + 3:self._base + 7] = yaw_quat(yaw)
        mujoco.mj_forward(self.model, data)
