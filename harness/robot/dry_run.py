"""Dry-run backend: real joint state from rt/lowstate, never publishes. Prints what would be sent."""
import numpy as np

from ..executor import Backend
from ..kinematics import ARM_JOINTS
from .lowstate import LowStateReader, query_fsm


class DryRunBackend(Backend):
    name = "dry-run"
    dry_run = True

    def __init__(self, iface, log=print):
        self.reader = LowStateReader(iface)
        self.log = log
        if not self.reader.wait():
            raise RuntimeError(f"no rt/lowstate on {iface}")
        self.fsm, self.fsm_name = query_fsm()
        self.log(f"dry run: rt/lowstate ok, FSM {self.fsm} = {self.fsm_name}; nothing will be published")
        self.pretend = {}          # joints the dry run pretends to have moved (so a whole episode can be walked through)

    def joints(self):
        j = self.reader.joints()
        j.update(self.pretend)
        return j

    def velocities(self):
        return {n: 0.0 for n in self.reader.velocities()}

    def stream(self, arm, frames, dt):
        q = np.asarray(frames[-1], float)
        names = ARM_JOINTS[arm]
        self.log(f"   would stream {len(frames)} frames over {len(frames) * dt:.1f} s to rt/arm_sdk: "
                 + ", ".join(f"{n.replace('_joint', '')}={v:+.3f}" for n, v in zip(names, q)))
        for n, v in zip(names, q):
            self.pretend[n] = float(v)

    def hand(self, arm, closed):
        return "dry run: no hand on this robot" if True else ""

    def close(self):
        self.reader.close()
