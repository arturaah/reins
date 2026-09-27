"""Declarative skill requirements and conservative, text-only intent routing."""
from dataclasses import dataclass
import re
import numpy as np
from core.motion_validation import slow_acceleration


@dataclass(frozen=True)
class Skill:
    name: str
    requirements: tuple


SKILLS = {
    'wave': Skill('wave', ('joint_pose', 'workspace_clearance')),
    'raise_arm': Skill('raise_arm', ('joint_pose', 'workspace_clearance')),
    'point': Skill('point', ('joint_pose', 'workspace_clearance', 'object_position')),
    'touch': Skill('touch', ('joint_pose', 'workspace_clearance', 'object_position')),
    'approach': Skill('approach', ('joint_pose', 'workspace_clearance', 'object_position')),
}


def route_intent(prompt):
    """Only exact standalone gestures bypass visual grounding.

    Targeted/compound commands never degrade to a generic wave. Unknown actions
    are rejected rather than letting a classifier's guess dispatch motion.
    """
    text = re.sub(r'\s+', ' ', prompt.strip().lower()).rstrip('.!?')
    text = re.sub(r'^(?:can you |could you |please )+', '', text)
    text = re.sub(r',? please$', '', text)
    wave = re.fullmatch(r'wave(?: (?:your |the )?(?:(left|right) )?(?:hand|arm))?(?: with (?:your |the )?(left|right)(?: (?:hand|arm))?)?', text)
    if wave:
        sides = [x for x in wave.groups() if x]
        if len(set(sides)) > 1: raise ValueError('Conflicting arm choices; choose left or right.')
        return {'skill':'wave', 'arm':next(iter(sides),'right'), 'selector':None}
    raised = re.fullmatch(r'(?:raise|lift)(?: up)? (?:your |the )?(?:(left|right) )?(?:hand|arm)(?: up)?', text)
    if raised:
        return {'skill':'raise_arm', 'arm':raised[1] or 'right', 'selector':None}
    obj = re.fullmatch(r'(point(?: at| to)|touch|approach|reach for) (.+)', text)
    if obj:
        selector = obj[2]
        if re.search(r'\b(?:then|and|after|before)\b', selector):
            raise ValueError('Submit one action at a time; compound actions are not supported.')
        arm = 'auto'
        suffix = re.search(r' with (?:your |the )?(left|right)(?: (?:hand|arm))?$', selector)
        if suffix: arm, selector = suffix[1], selector[:suffix.start()]
        selector = re.sub(r'^(?:the|a|an) ', '', selector).strip()
        if not selector: raise ValueError('Specify the object to act on.')
        name = 'point' if obj[1].startswith('point') else 'touch' if obj[1]=='touch' else 'approach'
        return {'skill':name, 'arm':arm, 'selector':selector}
    if text.startswith('wave at ') or text.startswith('wave to '):
        raise ValueError('Waving at a specific person requires directed-gesture support. Use “point at the person” or a standalone “wave”.')
    raise ValueError('No built-in path for this request. Author a new single-arm trajectory from the current robot geometry, then validate it for automatic preview.')


def joint_skill_plan(ik, side, poses, pose, name):
    """Slow cosine transitions from the actual supplied seed; dense linear samples."""
    names=ik.joint_names(side)
    current=np.array([pose.get(n,0.) for n in names],float)
    frames=[{'time_s':0.,'joint_targets_rad':dict(zip(names,current.tolist()))}]
    t=0.
    for target in poses:
        target=np.asarray(target,float)
        duration=max(.8,float(np.max(np.abs(target-current)))*np.pi/(2*.28))
        n=max(2,int(np.ceil(duration/.075)))
        for k in range(1,n+1):
            r=k/n; q=current+(target-current)*(.5-.5*np.cos(np.pi*r))
            frames.append({'time_s':round(t+duration*r,6),'joint_targets_rad':dict(zip(names,q.tolist()))})
        t+=duration; current=target
    plan={'schema_version':1,'name':name,'duration_s':frames[-1]['time_s'],'keyframes':frames,
          'held_joints_rad':{k:v for k,v in pose.items() if k not in names}}
    slow_acceleration(plan)
    return plan


def gesture_plan(ik, skill, side, pose):
    sign=1 if side=='left' else -1
    raised=np.array([-1.5,sign*.8,0.,.8])
    start=np.array([pose.get(n,0.) for n in ik.joint_names(side)])
    # Outward shoulder-roll oscillation keeps a visible wave clear of the head.
    poses=[raised]
    if skill=='wave':
        for roll in (1.0,.65,1.0,.65,.8):
            q=raised.copy(); q[1]=sign*roll; poses.append(q)
        poses.append(start)
    return joint_skill_plan(ik,side,poses,pose,f'{side.title()} '+('wave' if skill=='wave' else 'arm raise'))


def pointing_plan(ik, side, surface, pose):
    """Point toward a distant target from comfortable hand positions, not at it.

    A5 uses the forearm/site axis, not an articulated index finger. Direction is
    soft in ArmIK, so independently reject final angular error above 10 degrees.
    """
    sign=1 if side=='left' else -1
    solutions=[]
    for x in (.22,.30,.38):
        for y in (.16,.25,.34):
            for z in (.86,.98,1.08):
                position=np.array([x,sign*y,z]); direction=np.asarray(surface)-position
                if np.linalg.norm(direction)<.15: continue
                direction/=np.linalg.norm(direction)
                sol=ik.solve(side,position,direction,pose=pose,restarts=2)
                error=np.degrees(np.arccos(np.clip(sol.direction@direction,-1,1)))
                if sol.ok and error<=10:
                    q=np.array([sol.q[n] for n in ik.joint_names(side)])
                    start=np.array([pose.get(n,0.) for n in ik.joint_names(side)])
                    solutions.append((float(np.linalg.norm(q-start)),error,q,sol.position))
    if not solutions: raise ValueError('No reachable pointing pose within the 10° forearm-direction tolerance.')
    # Caller checks each candidate's complete trajectory for collisions.
    for _,error,q,position in sorted(solutions,key=lambda item:item[0]):
        plan=joint_skill_plan(ik,side,[q],pose,f'{side.title()} arm point')
        yield plan,position,float(error)
