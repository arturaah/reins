"""The algorithm layer must import without the robot SDK, so it runs and tests on the Mac."""
import subprocess
import sys

ALGORITHM_MODULES = ["harness.actions", "harness.interpreter", "harness.kinematics", "harness.safety",
                     "harness.executor", "harness.prompts", "harness.perception", "harness.loop",
                     "harness.recorder", "harness.config", "harness.sim.mock_robot", "harness.vlm.base",
                     "harness.vlm.scripted", "harness.vlm.chat", "harness.vlm.claude_cli", "harness.robot.arm_client",
                     "harness.demos", "harness.preview", "harness.stats", "harness.feedback", "harness.poseview", "tools.framelog"]


def test_algorithm_layer_has_no_sdk_import():
    code = ("import sys, importlib\n"
            + "".join(f"importlib.import_module({m!r})\n" for m in ALGORITHM_MODULES)
            + "bad = [m for m in sys.modules if m.startswith('unitree_sdk2py') or m.startswith('cyclonedds')]\n"
            "print('BAD', bad)\nsys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
