<p align="center"><img src="reins.png" alt="Reins logo" width="480"></p>

# Reins

**Any VLM. Your hands on the reins.**

A VLM-agnostic harness for robot control that surfaces the model's plan for human review before a single actuator fires. Swap in any vision-language model, and it gets smarter every time the models do.

## Intention

The novelty here is not the model. It is the harness around it.

Reins optimizes the harness for VLM-driven robot control. Making it model-agnostic means its value compounds passively: every improvement from Anthropic, OpenAI, or anyone else lands in Reins for free. We don't have to win the model race. We just have to be the best place to plug the winner in.

Where we add value is the interaction between the VLM and the human. Before the model instructs the robot, the harness surfaces what the model intends to do, and the human gets feedback on that intention first. Reviewing the plan before acting on it, rather than watching the robot and reacting after, is the core idea. That loop is what turns a capable model into a controllable one.

## Status

A first simulation-only preview is available in [`sim/`](sim/README.md). It loads Unitree R1, predicts a named-joint plan, draws the planned hand path, and exports world-frame points for future AR rendering. It does not command physical hardware.
