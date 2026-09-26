"""Preview a named-joint R1 motor plan before optional simulated execution.

This module never opens the Unitree SDK or sends commands to physical hardware.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "r1" / "scene_fixed_base.xml"
DEFAULT_PLAN = ROOT / "plans" / "left_reach.json"
DEFAULT_OUTPUT = ROOT / "preview.json"
SITE_NAME = "left_hand_preview"
SAMPLE_PERIOD = 0.04


def load_plan(model: mujoco.MjModel, path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError("Plan schema_version must be 1")
    frames = plan.get("keyframes")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Plan needs at least two keyframes")
    times = [float(f["time_s"]) for f in frames]
    if times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Keyframes must start at 0 and have increasing times")
    if abs(times[-1] - float(plan["duration_s"])) > 1e-9:
        raise ValueError("Last keyframe must equal duration_s")
    names = set(frames[0]["joint_targets_rad"])
    if not names or any(set(f["joint_targets_rad"]) != names for f in frames):
        raise ValueError("Each keyframe must name the same joints")
    actuated = {int(model.actuator_trnid[i, 0]) for i in range(model.nu)}
    joint_ids = {}
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or joint not in actuated:
            raise ValueError(f"Unknown or unactuated joint: {name}")
        for frame in frames:
            value = float(frame["joint_targets_rad"][name])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite target for {name}")
            if model.jnt_limited[joint] and not (model.jnt_range[joint, 0] <= value <= model.jnt_range[joint, 1]):
                raise ValueError(f"Target outside joint range: {name}={value}")
        joint_ids[name] = joint
    plan["_times"] = times
    plan["_joint_ids"] = joint_ids
    return plan


def target_at(model: mujoco.MjModel, plan: dict, t: float) -> np.ndarray:
    target = np.zeros(model.njnt)
    for name, joint in plan["_joint_ids"].items():
        values = [float(f["joint_targets_rad"][name]) for f in plan["keyframes"]]
        target[joint] = np.interp(t, plan["_times"], values)
    return target


def apply_pd(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0])
        qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        torque = 40.0 * (target[joint] - data.qpos[qadr]) - 3.0 * data.qvel[vadr]
        if model.actuator_ctrllimited[i]:
            torque = np.clip(torque, *model.actuator_ctrlrange[i])
        data.ctrl[i] = torque


def hand_xyz(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site < 0:
        raise ValueError(f"Missing site: {SITE_NAME}")
    return data.site_xpos[site].copy()


def draw_path(viewer, points: np.ndarray) -> None:
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for start, end in zip(points, points[1:]):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([0.05, 0.9, 1.0, 0.9], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 4.0, start, end)
            scene.ngeom += 1
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([0.025, 0, 0]), points[-1], np.eye(3).ravel(),
                               np.array([0.1, 1, 0.2, 0.9], dtype=np.float32))
            scene.ngeom += 1


def run(plan_path: Path, output_path: Path, headless: bool, execute: bool) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    plan = load_plan(model, plan_path)
    live = mujoco.MjData(model)
    predicted = mujoco.MjData(model)
    mujoco.mj_forward(model, live)
    mujoco.mj_copyData(predicted, model, live)
    steps = round(float(plan["duration_s"]) / model.opt.timestep)
    sample_every = max(1, round(SAMPLE_PERIOD / model.opt.timestep))
    samples = []
    for step in range(steps + 1):
        if step % sample_every == 0 or step == steps:
            target = target_at(model, plan, predicted.time)
            samples.append({
                "time_s": round(float(predicted.time), 4),
                "hand_xyz_m": hand_xyz(model, predicted).round(5).tolist(),
                "joint_targets_rad": {name: round(float(target[joint]), 5)
                                      for name, joint in sorted(plan["_joint_ids"].items())},
            })
        if step < steps:
            apply_pd(model, predicted, target_at(model, plan, predicted.time))
            mujoco.mj_step(model, predicted)
    export = {
        "schema_version": 1,
        "source_plan": plan_path.name,
        "robot": "Unitree R1 fixed-base simulation",
        "frame": "mujoco_world",
        "position_units": "m",
        "joint_units": "rad",
        "tracked_site": SITE_NAME,
        "samples": samples,
    }
    output_path.write_text(json.dumps(export, indent=2) + "\n")
    points = np.array([s["hand_xyz_m"] for s in samples])
    print(f"Preview: {len(samples)} path points, {np.linalg.norm(np.diff(points, axis=0), axis=1).sum():.3f} m of hand travel")
    print(f"Saved: {output_path}")
    viewer = None
    if not headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, live)
        draw_path(viewer, points)
        viewer.sync()
        print("Cyan path and green endpoint show the predicted hand movement.")
    try:
        if not execute:
            print("Preview only. Pass --execute to run this plan in simulation.")
            if viewer:
                while viewer.is_running():
                    time.sleep(0.1)
            return
        if viewer:
            print("Executing in 3 seconds...")
            time.sleep(3)
        for _ in range(steps):
            apply_pd(model, live, target_at(model, plan, live.time))
            mujoco.mj_step(model, live)
            if viewer:
                viewer.sync()
                time.sleep(model.opt.timestep)
        error = np.linalg.norm(hand_xyz(model, live) - hand_xyz(model, predicted))
        print(f"Simulated execution complete; final prediction error: {error:.6f} m")
        if viewer:
            while viewer.is_running():
                time.sleep(0.1)
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute in MuJoCo only")
    args = parser.parse_args()
    run(args.plan, args.output, args.headless, args.execute)
