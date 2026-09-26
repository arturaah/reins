"""Subscribe-only check that this machine sees the R1 over DDS.

Reads rt/lowstate for a few seconds and prints rate, mode, IMU and a few joints.
Publishes nothing, calls no service, cannot move the robot.

    .venv/bin/python tools/lowstate_peek.py en6      # Mac adapter
    python3 tools/lowstate_peek.py eth10             # on the Jetson
"""
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

iface = sys.argv[1] if len(sys.argv) > 1 else "en6"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
count, last = 0, None

def on_msg(msg: LowState_):
    global count, last
    count += 1
    last = msg

ChannelFactoryInitialize(0, iface)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_msg, 10)
t0 = time.time()
while time.time() - t0 < seconds:
    time.sleep(0.1)
sub.Close()

if last is None:
    print(f"no rt/lowstate messages on {iface} in {seconds:.0f} s")
    sys.exit(1)
m = last
# Slots follow the controller's 35-slot layout (unitree_sdk2/include/unitree/dds_wrapper/robots/r1/defines.h):
# legs 0-11, waist roll 12, waist yaw 13, L arm 15-19, R arm 22-26, head 29-30. Slots 14, 20, 21, 27, 28 are unused on the A5.
q = [round(s.q, 3) for s in m.motor_state[:31]]
print(f"rt/lowstate on {iface}: {count} msgs in {seconds:.0f} s = {count/seconds:.0f} Hz")
print(f"mode_machine={m.mode_machine}  tick={m.tick}")
print(f"imu rpy={[round(v, 3) for v in m.imu_state.rpy]}  quat={[round(v, 3) for v in m.imu_state.quaternion]}")
print(f"L leg  q[0:6]   ={q[0:6]}")
print(f"R leg  q[6:12]  ={q[6:12]}")
print(f"waist  q[12:14] ={q[12:14]}  (roll, yaw)")
print(f"L arm  q[15:20] ={q[15:20]}  (sh pitch, sh roll, sh yaw, elbow, wrist roll)")
print(f"R arm  q[22:27] ={q[22:27]}")
print(f"head   q[29:31] ={q[29:31]}  (pitch, yaw)")
print(f"motor temps (first 6)  ={[s.temperature for s in m.motor_state[:6]]}")
