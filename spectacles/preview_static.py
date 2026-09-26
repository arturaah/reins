#!/usr/bin/env python3
"""Render the exact fixed trajectories sent by trajectory_server --static."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trajectory_server import trajectory


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim/models/r1/R1_fixed_base.xml"
COLORS = {"left": (0.0, 0.95, 1.0, 1.0), "right": (1.0, 0.48, 0.05, 1.0)}


def add_sphere(scene, point, rgba, radius=0.014):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]), np.asarray(point), np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_path(scene, points, rgba):
    for start, end in zip(points[:-1], points[1:]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 7.0, start, end)
        scene.ngeom += 1
    add_sphere(scene, points[0], rgba)
    add_sphere(scene, points[-1], rgba, 0.010)


def render(output, azimuth, elevation):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    paths = trajectory(0.0)["hands"]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.17, 0, 0.9]
    camera.distance = 1.20
    camera.azimuth = azimuth
    camera.elevation = elevation
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera=camera)
        for side, coordinates in paths.items():
            add_path(renderer.scene, np.asarray(coordinates), COLORS[side])
        image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 34), fill=(15, 18, 24))
    draw.text((12, 9), "STATIC AR FEED  |  cyan: robot left  |  orange: robot right", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    for side, points in paths.items():
        print(f"{side}: {points[0]} -> {points[-1]}")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/static-trajectory-mujoco.png")
    parser.add_argument("--azimuth", type=float, default=155)
    parser.add_argument("--elevation", type=float, default=-25)
    args = parser.parse_args()
    render(args.output, args.azimuth, args.elevation)
