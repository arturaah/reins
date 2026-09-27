"""Decider interface: a System One model answers typed questions about a text state. No images.

ask(state, questions, facts) -> (answers, VLMResponse). questions map an id you choose to a Choice or a Noul;
answers come back under the same ids. `facts` are the numbers the state was written from; a real model never sees
them (Jev reads the words in `state`), the scripted stand-in decides from them. The VLMResponse carries latency,
tokens and errors so harness.stats and harness.recorder log decider calls exactly like VLM calls.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Choice:
    instructions: object                  # str, or a dict/list (TypeSafe's structured instructions)
    criteria: dict                        # option -> description (or None)
    type = "choice"


@dataclass
class Noul:
    instructions: object
    criteria: Optional[dict] = None       # {"true": what a yes means, "false": what a no means}
    type = "noul"


@dataclass
class Answer:
    type: str
    choice: Optional[str] = None
    probabilities: dict = field(default_factory=dict)
    confidence: Optional[float] = None    # Choice only: TypeSafe's summary of how peaked the probabilities are
    noul: Optional[float] = None          # Noul only: probability of yes, 0..1


def question_json(q):
    out = {"type": q.type, "instructions": q.instructions}
    if q.criteria is not None:
        out["criteria"] = q.criteria
    return out


class Decider:
    name = "decider"

    def ask(self, state, questions, facts=None):
        """-> (dict id -> Answer, VLMResponse). On error the dict is empty and resp.error says why."""
        raise NotImplementedError


def make(cfg, name=None):
    name = name or cfg["executor"]["decider"]
    if name == "jev":
        from .jev import JevDecider
        return JevDecider(cfg)
    if name == "scripted":
        from .scripted import ScriptedDecider
        return ScriptedDecider()
    raise ValueError(f"unknown decider {name!r}")
