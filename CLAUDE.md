# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reins is a VLM-agnostic harness for robot control: the model's plan is surfaced for human review before any actuator fires. See README.md. Status: the first simulation-only trajectory preview lives in `sim/`; no hardware harness exists yet.

Target hardware: Unitree R1 EDU humanoid. 26 DoF on the A5 arm variant (7-DoF arms on A7), Jetson Orin NX onboard at 192.168.123.164, DDS over the 192.168.123.x subnet.

## Layout

- `unitree_sdk2/` is a plain vendored copy of https://github.com/unitreerobotics/unitree_sdk2 at upstream commit 63096d0, minus its `.github/` workflows. Edit in place and commit here. It is deliberately not a submodule and is never synced with upstream. R1 code lives in `unitree_sdk2/example/r1/` (`high_level/`, `low_level/`, `audio/`).
- In this workspace, project knowledge lives one level up at `../.knowledge/reins/` (index.md, concepts/, worklog/). The R1 ecosystem survey is `concepts/r1-edu-ecosystem.md`; read it before researching R1 repos again.

## Building the SDK

Linux only (Ubuntu 20.04, x86_64 or aarch64). The prebuilt libraries in `unitree_sdk2/lib/` and `unitree_sdk2/thirdparty/lib/` exist only for those two targets, so it does not build on macOS. Build on the R1's Jetson or a Linux box.

```
apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev
cd unitree_sdk2 && mkdir -p build && cd build && cmake .. && make
```

Binaries land in `unitree_sdk2/build/bin/`. Every R1 example takes the network interface connected to the robot (e.g. `eth0`) as an argument; the exact flags are in each file's header comment. To use the SDK from a separate CMake project, `make install` it and copy `unitree_sdk2/example/cmake_sample`.

No tests or linters exist yet.

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
- `192.168.123.164` EDU Jetson (Ubuntu, aarch64). SSH as `unitree`, default password `123`. Change it on first login.
- `192.168.123.161` is the controller address in community R1 projects, but it does not answer ping on our robot. The R1 controller's address is still unknown; find it from the Jetson's neighbour table.

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
6. DDS from the Mac: pass `en6` as the network interface to the SDK's channel factory. On the Jetson the interface is typically `eth0`; check with `ip link`.

Status 2026-09-26, verified on the lab MacBook Air: link active at 100BASE-TX, static IP 192.168.123.99 set, Jetson 192.168.123.164 answers ping in about 1 ms. Not yet verified: SSH to the Jetson, the controller's address, DDS from the Mac. Update this line as steps are confirmed.

Rule for agents: any command that sends anything to the robot (ping, SSH, DDS subscribe or publish) is proposed as a question and run only after Artur approves it. Mac-local checks need no approval.
