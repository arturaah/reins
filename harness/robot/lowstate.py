"""Subscribe-only reader of rt/lowstate plus the read-only FSM query. Imports the SDK."""
import json
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# controller slot -> MuJoCo joint name (unitree_sdk2/include/unitree/dds_wrapper/robots/r1/defines.h)
SLOT_TO_JOINT = {
    12: "waist_roll_joint", 13: "waist_yaw_joint",
    15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint", 17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint", 19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint", 24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint", 26: "right_wrist_roll_joint",
}
JOINT_TO_SLOT = {v: k for k, v in SLOT_TO_JOINT.items()}
HEAD_SLOTS = {29: "head_pitch", 30: "head_yaw"}
# Unitree's SDK names 0, 1, 4 and 811 (r1_loco_client.hpp). The others are the R1 table of legion1581/unitree_webrtc_connect
# (constants.py, read out of the app's protocol), marked (wrtc). 816 verified here on 2026-09-27: the robot reports it from the
# moment rt/arm_sdk carries a weight > 0 and is back in 811 the moment the weight is 0 again (harness/README.md, Locomotion).
FSM_NAMES = {0: "ZeroTorque", 1: "Damp", 4: "StandUp", 5: "Keep (wrtc)", 6: "MoveTo (wrtc)", 7: "SitDown (wrtc)",
             601: "Dance1 (wrtc)", 602: "Dance2 (wrtc)", 603: "Dance3 (wrtc)", 604: "Twist (wrtc)", 607: "KungFu (wrtc)",
             608: "JeetKuneDo (wrtc)", 701: "StandUp from the ground (wrtc)", 702: "LieDown (wrtc)", 800: "Motion (wrtc)",
             811: "Start (balance control)", 812: "AmpMotion (wrtc)", 813: "WalkStraightKnee (wrtc)", 814: "Walk (wrtc)",
             815: "AmpLocomotion (wrtc)",
             816: "ArmSdkLoco (wrtc): balance control with the arm topic active; the R1 enters it while rt/arm_sdk carries a weight > 0 and leaves it at weight 0",
             830: "Loco20Dof (wrtc)", 831: "LocoArmSdk (wrtc)"}
FSM_ARM_OK = {4, 811}
_initialized = {}


def init_dds(iface, domain=0):
    """ChannelFactoryInitialize once per process."""
    key = (domain, iface)
    if key not in _initialized:
        ChannelFactoryInitialize(domain, iface)
        _initialized[key] = True


class LowStateReader:
    def __init__(self, iface, domain=0):
        init_dds(iface, domain)
        self.lock = threading.Lock()
        self.msg, self.count, self.t_last = None, 0, 0.0
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_msg, 10)

    def _on_msg(self, m):
        with self.lock:
            self.msg, self.count, self.t_last = m, self.count + 1, time.time()

    def wait(self, seconds=3.0, n=10):
        t0 = time.time()
        while self.count < n and time.time() - t0 < seconds:
            time.sleep(0.05)
        return self.msg is not None

    def joints(self):
        with self.lock:
            m = self.msg
        return {n: float(m.motor_state[s].q) for s, n in SLOT_TO_JOINT.items()} if m else {}

    def velocities(self):
        with self.lock:
            m = self.msg
        return {n: float(m.motor_state[s].dq) for s, n in SLOT_TO_JOINT.items()} if m else {}

    def head(self):
        with self.lock:
            m = self.msg
        return {n: float(m.motor_state[s].q) for s, n in HEAD_SLOTS.items()} if m else {}

    def age(self):
        return time.time() - self.t_last

    def close(self):
        self.sub.Close()


def query_fsm(timeout=3.0):
    """(fsm_id, name) via the loco client's read-only GET_FSM_ID; (None, 'unknown') on failure."""
    try:
        from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
        from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
        lc = LocoClient(); lc.SetTimeout(timeout); lc.Init()
        code, data = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "")
        fsm = json.loads(data)["data"] if code == 0 and data else None
        return fsm, FSM_NAMES.get(fsm, "unknown")
    except Exception as e:
        return None, f"query failed: {e}"
