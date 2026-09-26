# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reins is a VLM-agnostic harness for robot control: the model's plan is surfaced for human review before any actuator fires. See README.md. Status: the first simulation-only trajectory preview lives in `sim/`; no hardware harness exists yet.

Target hardware: Unitree R1 EDU humanoid. 26 DoF on the A5 arm variant (7-DoF arms on A7), Jetson Orin NX onboard at 192.168.123.164, DDS over the 192.168.123.x subnet.

## Layout

- `unitree_sdk2/` is a plain vendored copy of https://github.com/unitreerobotics/unitree_sdk2 at upstream commit 63096d0, minus its `.github/` workflows. Edit in place and commit here. It is deliberately not a submodule and is never synced with upstream. R1 code lives in `unitree_sdk2/example/r1/` (`high_level/`, `low_level/`, `audio/`).
- `contract/` defines the messages between the harness parts (VLM harness, core/IK, review surfaces such as the MuJoCo preview and Spectacles, and the `rt/arm_sdk` streamer): `README.md` is the spec, `reins.schema.json` the source of truth, `examples/` full sessions, `reins_contract.py` a validator. Change all four together and run `python3 -m pytest contract`.
- In this workspace, project knowledge lives one level up at `../.knowledge/reins/` (index.md, concepts/, worklog/). The R1 ecosystem survey is `concepts/r1-edu-ecosystem.md`; read it before researching R1 repos again.

## Building the SDK

Linux only (Ubuntu 20.04, x86_64 or aarch64). The prebuilt libraries in `unitree_sdk2/lib/` and `unitree_sdk2/thirdparty/lib/` exist only for those two targets, so it does not build on macOS. Build on the R1's Jetson or a Linux box.

```
apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev
cd unitree_sdk2 && mkdir -p build && cd build && cmake .. && make
```

Binaries land in `unitree_sdk2/build/bin/`. Every R1 example takes the network interface connected to the robot (e.g. `eth0`) as an argument; the exact flags are in each file's header comment. To use the SDK from a separate CMake project, `make install` it and copy `unitree_sdk2/example/cmake_sample`.

The only tests are the contract's (`python3 -m pytest contract`, needs `jsonschema` and `pytest`). No linters.

## Chosen control method (decision, 2026-09-26)

Reins drives the R1 through the **`rt/arm_sdk` DDS topic**: stream arm and head joint targets at 250 Hz with the blend weight ramped 0 to 1, while the robot's onboard controller keeps balance. Everything the harness does for manipulation maps onto this one path:

- **Plan**: the VLM proposes end-effector goals; IK on the R1 URDF (Pinocchio, as Unitree's xr_teleoperate does) turns them into a joint trajectory `q(t)`.
- **Preview**: replay `q(t)` kinematically in MuJoCo on the official R1 model (set joint positions, render). No physics; balance is the robot's job. This is the human review step.
- **Execute**: on approval, publish `q(t)` to `rt/arm_sdk` at 250 Hz. Ramp the weight down to hand the arms back.
- **Teleop and record**: Unitree's xr_teleoperate publishes the same topic and records episodes; replay is the execute step.

The **high-level loco client** is used only as the session's safety wrapper: Stance before streaming, Damp on abort, FSM checks. The **low-level `rt/lowcmd`** surface is not used; it takes balance away from the robot for no manipulation benefit.

Live twin: DDS is many-to-many, so a subscriber-only MuJoCo viewer can read `rt/arm_sdk` (commanded) and `rt/lowstate` (actual) alongside the robot. Do not run Unitree's MuJoCo bridge on the robot's DDS domain (0): it publishes `rt/lowstate` and would collide. Unitree's own config puts sim on domain 1.

Machine roles: the Mac hosts the harness, VLM calls, IK and the MuJoCo preview. The DDS streamer runs on the R1's Jetson first (guaranteed path), then optionally on the Mac. The Python SDK is pure Python with a pure-Python checksum fallback off Linux, and CycloneDDS ships Apple Silicon wheels for Python 3.8 to 3.10, so a native Mac streamer is feasible but untested by Unitree.

## R1 control surfaces (which one to reach for)

1. **High-level loco client** (`high_level/r1_loco_client_example.cpp`): Move, SetVelocity, Stance, Damp, Lie2StandUp, SetFsmId, WaveHand, ShakeHand. The onboard policy keeps balance. Used only as the safety wrapper (see the decision above).
2. **Arm action service and the `rt/arm_sdk` topic** (`high_level/r1_arm_action_example.cpp`, `r1_arm_sdk_dds_example.cpp`): run preset or teach-recorded arm actions, or stream upper-body joint targets with a 0..1 blend weight while the robot balances itself. Do not substitute G1's arm action client: same service name and API IDs, different action IDs and error codes.
3. **Low-level `rt/lowcmd` / `rt/lowstate`** (`low_level/`): all 26 motors under PD control on a 2 ms loop. You own balance.

Sim: `unitreerobotics/unitree_mujoco` ships an R1 model that consumes the same DDS topics, so a plan can be dry-run before touching hardware. Its MJCF has 29 actuators (3-DoF waist and wrists) while the A5 low-level joint map has 26. Check the variant before trusting joint indices.

## Connecting a Mac to the robot

Port facts from Unitree's docs: the **RJ45 Gigabit Ethernet** port on the R1's upper body is the PC link. The R1's **USB-C port is the internal link to the EDU Jetson**; plugging a Mac into it gives no network connection.

Robot wired network, 192.168.123.0/24:
- `192.168.123.164` EDU Jetson (hostname `ubuntu`, Ubuntu aarch64, robot-side interface `eth10`). SSH as `unitree`, default password `123`. Change it on first login. Host key fingerprint on our unit: `SHA256:b49bi+OYx/3BYWPsTlMZF1psSs5FW8FnpmFfHpfoDrk`.
- `192.168.123.161` motion controller (MAC `7e:1d:75:60:f5:89`, confirmed from the Jetson's neighbour table). It does not answer ping or ARP from the Mac; whether it answers the Jetson is not yet checked.

Topology, confirmed from the Jetson on 2026-09-26: the Jetson module has one NIC on the robot network, `eth10`, behind a small internal 100 Mb/s switch with two external sockets. The body cable is in one socket, the Mac in the other, so body, Jetson and Mac share one segment. The Jetson's `eth0` is an internal USB 10/100 chip that carries nothing; ignore it. Its CycloneDDS config file in `~/cyclonedds_ws` names `eth0` and is not in use; always pass `eth10` explicitly.

Safety, from a community R1 project: entering locomotion FSM 811 can start leg and balance motion even at zero velocity. Never use an FSM change as a connectivity test. Use ping, `ip neigh`, or a subscribe-only DDS read.

Steps:
1. Power the robot on (short press the battery button, then hold it for more than 2 s). The Ethernet link only comes up with the robot powered.
2. USB-C Ethernet adapter in the Mac, Ethernet cable from the adapter to the robot's RJ45. Find the adapter's interface and service name with `networksetup -listallhardwareports`. On the lab MacBook Air the UGREEN adapter is interface `en6`, service `USB 10/100/1000 LAN`.
3. Confirm link: `ifconfig en6 | grep status` must say `active`. If it says `inactive`, the problem is the cable or the robot side, not the Mac.
4. Give the Mac a static address on the robot subnet (needs the admin password, so a human runs it):
   ```
   sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.99 255.255.255.0
   ```
   No router. Wi-Fi stays as is and keeps internet on `en0`.
5. `ping -c 3 192.168.123.164`, then `ssh unitree@192.168.123.164`.
6. Python SDK on the Mac (verified, Apple Silicon, Python 3.10):
   ```
   uv python install 3.10 && uv venv --python 3.10 .venv
   uv pip install --python .venv/bin/python "cyclonedds==0.10.2" numpy
   uv pip install --python .venv/bin/python --no-deps "git+https://github.com/unitreerobotics/unitree_sdk2_python"
   .venv/bin/python tools/lowstate_peek.py en6
   ```
   Pass `en6` (or whatever `networksetup -listallhardwareports` shows) as the interface. `tools/lowstate_peek.py` is subscribe-only and is the standard "can this machine see the robot" check. On the Jetson the interface is `eth10`.

Status 2026-09-26, evening, all verified: body Ethernet cable straight into the Mac's USB-C adapter, 1000BASE-T, controller .161 pings in 0.6 ms, and `tools/lowstate_peek.py en6` receives rt/lowstate at about 1 kHz from the Mac with no Jetson involved. The Jetson module's internal switch is faulty (100 Mb/s, one-way: controller frames arrive, nothing reaches the controller; capture evidence in the knowledge worklog). With the body cable on the Mac, the Jetson at .164 is off the robot network; to use both, put a small gigabit switch between body cable, Jetson RJ45 and Mac, and report the module switch to Unitree. Next unverified step: first `rt/arm_sdk` stream, which moves the robot and needs Artur's explicit go.

Rule for agents: any command that sends anything to the robot (ping, SSH, DDS subscribe or publish) is proposed as a question and run only after Artur approves it. Mac-local checks need no approval.
