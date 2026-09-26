"""Solve a fixed-base R1 left-arm pickup trajectory in MuJoCo coordinates."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(ROOT / "models/r1/scene_fixed_base.xml"))
data = mujoco.MjData(model)
names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
         "left_shoulder_yaw_joint", "left_elbow_joint"]
joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
qadrs = [int(model.jnt_qposadr[j]) for j in joints]
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_preview")
limits = np.array([model.jnt_range[j] for j in joints])


def position(q):
    data.qpos[qadrs] = q
    mujoco.mj_forward(model, data)
    return data.site_xpos[site].copy()


def solve(target, seed):
    result = least_squares(lambda q: np.r_[30 * (position(q) - target),
                                            0.035 * (q - seed)], seed,
                           bounds=(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5),
                           max_nfev=400)
    error = np.linalg.norm(position(result.x) - target)
    if error > 0.025:
        raise RuntimeError(f"IK missed target by {error:.3f} m: {target}")
    return result.x, error


if __name__ == "__main__":
    waypoints = [(0.0, None), (1.5, [0.39, 0.16, 0.79]),
                 (2.5, [0.39, 0.16, 0.69]), (3.1, [0.39, 0.16, 0.69]),
                 (4.3, [0.34, 0.16, 0.83]), (5.0, [0.34, 0.16, 0.83])]
    q = np.zeros(len(names))
    frames = []
    for t, target in waypoints:
        if target is not None:
            q, error = solve(np.array(target), q)
            print(f"IK {t:.1f}s: {error:.4f} m")
        frames.append({"time_s": t, "joint_targets_rad":
                       {name: round(float(angle), 6) for name, angle in zip(names, q)}})
    plan = {"schema_version": 1, "name": "left hand pickup proxy",
            "duration_s": 5.0, "grasp_time_s": 3.1,
            "keyframes": frames}
    path = ROOT / "plans/pick_cube.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(path)
