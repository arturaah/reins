"""Deterministic stand-in for Jev, for tests and --executor scripted sim runs. Decides from the numeric facts.

Moves along the largest remaining gap; at the goal it GRASPs in a GRASP stage, RELEASEs in a RELEASE stage while
holding, otherwise says DONE. Queued answers (dicts id -> Answer) are returned first, which is how tests inject low
confidence, LOOK or a wrong DONE.
"""
import json

from ..vlm.base import VLMResponse
from .base import Answer, Decider


def geometric_token(f):
    """The move that closes the largest remaining gap; at the goal GRASP / RELEASE when the stage calls for it, else DONE.
    f: the facts harness.split.decision_state computed (largest, motion, holding, hand)."""
    if f.get("largest"):
        return f["largest"]
    motion, holding, hand = f.get("motion", ""), f.get("holding", False), f.get("hand", "none")
    if hand != "none" and "GRASP" in motion and not holding:
        return "GRASP"
    if hand != "none" and "RELEASE" in motion and holding:
        return "RELEASE"
    return "DONE"


class ScriptedDecider(Decider):
    name = "scripted"

    def __init__(self, queue=None, confidence=0.9):
        self.queue = list(queue or [])
        self.confidence = confidence
        self.calls = []

    def ask(self, state, questions, facts=None):
        self.calls.append((state, sorted(questions), facts))
        if self.queue:
            out = {k: v for k, v in self.queue.pop(0).items() if k in questions}
        else:
            out = self.policy(questions, facts or {})
        return out, VLMResponse(json.dumps({k: vars(v) for k, v in out.items()}), "scripted", 0.0)

    def policy(self, questions, f):
        token = geometric_token(f)
        out = {}
        for k, q in questions.items():
            if q.type == "choice":
                pick = token if token in q.criteria else ("DONE" if "DONE" in q.criteria else next(iter(q.criteria)))
                rest = (1.0 - self.confidence) / max(1, len(q.criteria) - 1)
                out[k] = Answer("choice", choice=pick, confidence=self.confidence,
                                probabilities={o: (self.confidence if o == pick else rest) for o in q.criteria})
            elif k == "stage_done":
                out[k] = Answer("noul", noul=0.9 if token == "DONE" else 0.05)
            else:
                out[k] = Answer("noul", noul=0.05)
        return out
