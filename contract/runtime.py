"""Authoritative motion payload and approval boundary used by coordinator and transports.

An approval travels only over the private coordinator-to-actuator connection. A digest
identifies content; it is not a credential. Transports must authenticate that connection
and consume each proposal ID once, in addition to these shared checks.
"""
import hashlib
import json
import math
import time

TERMINAL = frozenset({"executed", "declined", "expired", "cancelled", "blocked", "failed"})
MAX_DURATION_S = 180.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_motion(payload):
    if not isinstance(payload, dict):
        raise ValueError("Motion must be an object")
    kind = payload.get("kind")
    required = {"arm": {"kind", "arm", "plan"},
                "walk": {"kind", "vx", "vy", "vyaw", "duration_s"},
                "hand": {"kind", "arm", "closed"}}.get(kind)
    if required is None or set(payload) != required:
        raise ValueError("Unknown motion kind or fields")
    if kind in ("arm", "hand") and payload["arm"] not in ("left", "right"):
        raise ValueError("Choose one arm")
    if kind == "hand":
        if type(payload["closed"]) is not bool:
            raise ValueError("closed must be a boolean")
    elif kind == "walk":
        duration = finite(payload["duration_s"], "duration_s")
        values = [finite(payload[k], k) for k in ("vx", "vy", "vyaw")]
        if not 0 < duration <= 15 or math.hypot(*values[:2]) > .3 or abs(values[2]) > .5:
            raise ValueError("Walking exceeds the runtime duration/speed limits")
        if math.hypot(*values[:2])*duration > .6 + 1e-9 or abs(values[2])*duration > math.pi/4 + 1e-9:
            raise ValueError("Walking exceeds the runtime distance/turn limits")
        if not any(values):
            raise ValueError("Walking motion is empty")
    else:
        plan = payload["plan"]
        if not isinstance(plan, dict) or plan.get("schema_version") != 1:
            raise ValueError("Expected a resolved schema_version 1 arm plan")
        duration = finite(plan.get("duration_s"), "duration_s")
        keys = plan.get("keyframes")
        if not 0 < duration <= MAX_DURATION_S or not isinstance(keys, list) or not 2 <= len(keys) <= 36001:
            raise ValueError("Arm plan exceeds the duration/sample limits")
        if any(not isinstance(k, dict) for k in keys):
            raise ValueError("Each keyframe must be an object")
        times = [finite(k.get("time_s"), "keyframe time") for k in keys]
        if times[0] != 0 or any(b <= a for a,b in zip(times,times[1:])) or abs(times[-1]-duration)>1e-6:
            raise ValueError("Plan times must increase from zero to duration_s")
    # Recursively rejects non-finite JSON numbers, including nested plan metadata.
    digest(payload)
    return payload


def validate_approval(approval, payload_digest):
    if not isinstance(approval, dict) or set(approval) != {"proposal_id", "revision", "digest", "expires_at"}:
        raise ValueError("A coordinator approval is required")
    ident = approval["proposal_id"]
    if not isinstance(ident, str) or not 8 <= len(ident) <= 128:
        raise ValueError("Invalid proposal ID")
    if type(approval["revision"]) is not int or approval["revision"] < 1:
        raise ValueError("Invalid proposal revision")
    if approval["digest"] != payload_digest:
        raise ValueError("Approved motion changed")
    expires = finite(approval["expires_at"], "Approval expiry")
    if not time.time() < expires <= time.time()+180:
        raise ValueError("Approval expired or expiry is invalid")
    return approval
