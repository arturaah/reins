"""Preview a planned trajectory for the Unitree R1.

Interactive viewer:

    uv run reins-preview examples/around_the_table.json

Offscreen, for handing to a reviewer or a UI:

    uv run reins-preview examples/around_the_table.json --out out/plan.png
    uv run reins-preview examples/around_the_table.json --out out/plan.gif
"""

import argparse
import os
import sys
import sysconfig
import time
from pathlib import Path

import mujoco
import numpy as np

from . import model as r1
from .overlay import draw_plan
from .trajectory import Trajectory

WALK_SPEED = 0.5  # m/s, for playback only
END_PAUSE = 1.0  # seconds to hold at the goal before looping


class Preview:
    def __init__(self, traj: Trajectory):
        self.traj = traj
        self.model, self.data = r1.load()
        self.ghost = mujoco.MjData(self.model)
        self.poser = r1.Poser(self.model)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0  # hide collision geoms

    def pose_robot(self, s: float) -> None:
        moving = 0.0 < s < self.traj.length
        phase = s / r1.STRIDE if moving else None
        self.poser.set(self.data, *self.traj.pose_at(s), gait_phase=phase)

    def decorate(self, scn: mujoco.MjvScene, s: float | None) -> None:
        draw_plan(scn, self.model, self.poser, self.ghost, self.traj, progress=s)

    def overview_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        lo, hi = self.traj.points.min(0), self.traj.points.max(0)
        cam.lookat[:] = (*((lo + hi) / 2), 0.4)
        cam.distance = max(3.0, 1.4 * float(np.linalg.norm(hi - lo)) + 1.5)
        # Look along the overall direction of travel, from behind and above the start.
        d = self.traj.points[-1] - self.traj.points[0]
        heading = np.degrees(np.arctan2(d[1], d[0])) if np.any(d) else 0.0
        cam.azimuth = heading + 25
        cam.elevation = -40
        return cam

    def timeline(self, fps: float):
        """Arc-length values for one playback pass, including the pause at the goal."""
        n = max(2, int(self.traj.length / WALK_SPEED * fps))
        hold = int(END_PAUSE * fps)
        return list(np.linspace(0, self.traj.length, n)) + [self.traj.length] * hold

    # --- offscreen ---------------------------------------------------------

    def render(self, s: float, width: int, height: int,
               renderer: mujoco.Renderer) -> np.ndarray:
        self.pose_robot(s)
        renderer.update_scene(self.data, self.overview_camera(), self.opt)
        self.decorate(renderer.scene, s)
        return renderer.render()

    def save(self, out: Path, width: int, height: int, fps: float) -> None:
        from PIL import Image

        out.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(self.model, height, width) as renderer:
            if out.suffix.lower() == ".gif":
                frames = [Image.fromarray(self.render(s, width, height, renderer))
                          for s in self.timeline(fps)]
                frames[0].save(out, save_all=True, append_images=frames[1:],
                               duration=int(1000 / fps), loop=0)
            else:
                # Still image: robot at the start, full plan ahead of it.
                Image.fromarray(self.render(0.0, width, height, renderer)).save(out)

    # --- interactive -------------------------------------------------------

    def view(self) -> None:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(self.model, self.data)

        fps = 60.0
        with viewer:
            viewer.opt.geomgroup[0] = 0
            cam = self.overview_camera()
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance = cam.distance
            viewer.cam.azimuth = cam.azimuth
            viewer.cam.elevation = cam.elevation

            frames = self.timeline(fps)
            i = 0
            while viewer.is_running():
                tick = time.perf_counter()
                s = frames[i % len(frames)]
                with viewer.lock():
                    self.pose_robot(s)
                    viewer.user_scn.ngeom = 0
                    self.decorate(viewer.user_scn, s)
                viewer.sync()
                i += 1
                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - tick)))


def _reexec_under_mjpython() -> None:
    """On macOS the passive viewer must run under mjpython; hop over to it.

    mjpython from a uv-managed venv can't find libpython on its own, so point
    the dynamic loader at the interpreter's lib directory.
    """
    if sys.platform != "darwin" or os.environ.get("REINS_UNDER_MJPYTHON"):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    env = dict(os.environ, REINS_UNDER_MJPYTHON="1")
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (libdir, env.get("DYLD_LIBRARY_PATH")) if p)
    os.execve(mjpython, [str(mjpython), "-m", "reins_sim.preview", *sys.argv[1:]], env)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", type=Path, help="trajectory JSON")
    p.add_argument("--out", type=Path,
                   help="render offscreen to .png (still) or .gif (animation) instead of opening the viewer")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=float, default=20)
    args = p.parse_args(argv)

    if not args.out:
        _reexec_under_mjpython()
    preview = Preview(Trajectory.load(args.plan))
    if args.out:
        preview.save(args.out, args.width, args.height, args.fps)
        print(f"wrote {args.out}")
    else:
        preview.view()


if __name__ == "__main__":
    main()
