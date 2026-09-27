<p align="center"><img src="reins.png" alt="Reins logo" width="480"></p>

# Reins

**Any VLM. Your hands on the reins.**

A VLM-agnostic harness for robot control that surfaces the model's plan for human review before a single actuator fires. Swap in any vision-language model, and it gets smarter every time the models do.

## Intention

The novelty here is not the model. It is the harness around it.

Reins optimizes the harness for VLM-driven robot control. Making it model-agnostic means its value compounds passively: every improvement from Anthropic, OpenAI, or anyone else lands in Reins for free. We don't have to win the model race. We just have to be the best place to plug the winner in.

Where we add value is the interaction between the VLM and the human. Before the model instructs the robot, the harness surfaces what the model intends to do, and the human gets feedback on that intention first. Reviewing the plan before acting on it, rather than watching the robot and reacting after, is the core idea. That loop is what turns a capable model into a controllable one.

## Status

The dashboard connects new trajectory generation, the visual action harness,
human review, Spectacles and the R1 arm streamer through a shared control pipeline.
The core trajectory planner runs first. When it needs more context, the visual
harness proposes camera-guided steps through the same validator and review gate.

## Run

```sh
python3 -m pip --python .venv/bin/python install -r requirements.txt
.venv/bin/python tools/dashboard.py --iface YOUR_ROBOT_INTERFACE
```

Start in simulation, describe a motion or prepare a nudge, and review the checked
path. Explicitly connect robot control to approve physical execution. Dashboard
and paired glasses review the same proposal; Stop releases trajectory control.

See the [dashboard guide](tools/dashboard/README.md) for robot connection,
camera services, glasses pairing, model providers, operating limits and tests.
The dashboard keeps firmware preset buttons but has no recording/replay library.
Session logs retain review and execution evidence.

The simulation is a fixed-base kinematic preview, not a balance or contact model.
Physical commissioning and Lens tracking still require validation on the actual setup.

## Voice in simulation

The optional [voice lab](voice/README.md) uses GPT-Live-1 for listening and
streaming speech, with client delegation to an independent robot-LLM backend.
Voice Focus 2.2 enhances input and Tyto 1.1 requests clearer audio when needed.
The included GPT-5-mini adapter is a conversation-only test backend with no
motion tools. Start it with `python -m voice.live`; the project `.env` supplies
`OPENAI_KEY` and `AIC_KEY`. [Spectacles and R1 speaker setup](spectacles/VOICE.md)
connects the glasses microphone to GPT-Live and streams replies to the robot.
Run the dashboard with `--sim --voice-url http://127.0.0.1:8770/` beside the
MuJoCo preview. The UI uses a neutral theme with the colourful Reins logo.
