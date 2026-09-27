# Quick laptop microphone demo

From the updated repository, with `OPENAI_KEY` and `AIC_KEY` in its `.env`:

```sh
bash tools/voice_demo.sh
```

Open **http://127.0.0.1:8770/**, click **Start conversation**, and allow the laptop
microphone. GPT-Live handles speech in and out, with Voice Focus and Tyto enabled.
This mode makes no separate reasoning-model call and never queues robot tasks.

The launcher uses the R1 speaker if it finds the robot's body Ethernet link;
otherwise it uses laptop speakers. The robot, Spectacles and desktop planner can
all be disconnected for a laptop-only demo. To choose explicitly:

```sh
bash tools/voice_demo.sh --laptop   # completely standalone
bash tools/voice_demo.sh en8        # R1 speaker on this actual body interface
```

The launcher installs voice dependencies in `.venv-voice` if needed, using Python
3.12 (or `uv` to obtain it). R1 output uses the existing `.venv/bin/python` with the
Unitree SDK; set `ROBOT_PYTHON=/path/to/python` if that environment is elsewhere.
Keys stay on the computer. Initial ai-coustics model downloads need internet.

If port 8770 is already used, stop the old voice service or run
`DEMO_PORT=8773 bash tools/voice_demo.sh`. Reload the browser after changing the
service. **Stop** or Escape ends the conversation; Ctrl-C shuts down the service.
Tyto needs five seconds of microphone input before its first analysis. Barge-in
is disabled while speech plays. The robot speaker's existing volume is preserved.
