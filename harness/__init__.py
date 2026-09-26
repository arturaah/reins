"""Reins VLM end-effector harness: the model is the policy, one discrete hand action per step.

Layers (see DESIGN.md):
  algorithm  actions, interpreter, kinematics, safety, executor, prompts, perception, loop, recorder
             -> no unitree_sdk2py import anywhere; runs and tests on the Mac without a robot
  robot      harness.robot.*  -> DDS (lowstate reader, arm_sdk streamer process)
  vlm        harness.vlm.*    -> API adapters
"""
