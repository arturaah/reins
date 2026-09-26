"""Deterministic stand-in for a VLM: a fixed plan and a queue of decisions. For tests and --sim smoke runs."""
import json

from .base import VLM, VLMResponse


class ScriptedVLM(VLM):
    name = "scripted"

    def __init__(self, plan=None, decisions=None, on_act=None):
        self.plan_obj = plan or {"subgoals": [
            {"id": "hover_block", "target": "the block", "affordance": "block top", "motion": "REACH",
             "description": "bring the hand tip above the block", "completion": "hand tip directly above the block"}]}
        self.decisions = list(decisions or [])
        self.on_act = on_act                 # optional callable(prompt, images) -> decision text, when the queue is empty
        self.calls = []

    def plan(self, prompt, images, schema=None):
        self.calls.append(("plan", prompt))
        return VLMResponse(json.dumps(self.plan_obj), "scripted", 0.0)

    def act(self, prompt, images, schema=None, retry_note=None):
        self.calls.append(("act", prompt, retry_note))
        if self.decisions:
            d = self.decisions.pop(0)
        elif self.on_act:
            d = self.on_act(prompt, images)
        else:
            d = {"decision": "DONE", "reasoning": "WRIST: NO. nothing queued"}
        text = d if isinstance(d, str) else json.dumps(d)
        return VLMResponse(text, "scripted", 0.0)
