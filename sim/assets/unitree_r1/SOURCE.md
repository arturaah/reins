# Unitree R1 model

Vendored from [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
at commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`, path `unitree_robots/r1/`.
BSD 3-Clause, see `LICENSE`.

Changes from upstream:

- `R1_C++.xml` renamed to `r1.xml`. Contents unchanged.
- `scene.xml` rewritten for Reins: floor moved to geom group 2, higher ambient light,
  larger offscreen buffer for rendering previews.

`r1.xml` keeps the five dummy joints upstream adds (`waist_pitch`, `*_wrist_pitch`,
`*_wrist_yaw`, parked at z=20) so actuator indices line up with `unitree_sdk2`.
