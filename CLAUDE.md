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

## R1 control surfaces (which one to reach for)

1. **High-level loco client** (`high_level/r1_loco_client_example.cpp`): Move, SetVelocity, Stance, Damp, Lie2StandUp, SetFsmId, WaveHand, ShakeHand. The onboard policy keeps balance. Default surface for VLM-issued commands.
2. **Arm action service and the `rt/arm_sdk` topic** (`high_level/r1_arm_action_example.cpp`, `r1_arm_sdk_dds_example.cpp`): run preset or teach-recorded arm actions, or stream upper-body joint targets with a 0..1 blend weight while the robot balances itself. Do not substitute G1's arm action client: same service name and API IDs, different action IDs and error codes.
3. **Low-level `rt/lowcmd` / `rt/lowstate`** (`low_level/`): all 26 motors under PD control on a 2 ms loop. You own balance.

Sim: `unitreerobotics/unitree_mujoco` ships an R1 model that consumes the same DDS topics, so a plan can be dry-run before touching hardware. Its MJCF has 29 actuators (3-DoF waist and wrists) while the A5 low-level joint map has 26. Check the variant before trusting joint indices.
