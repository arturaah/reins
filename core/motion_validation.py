"""Conservative arm-volume checks for preview plans, not hardware certification."""
import math
import numpy as np
import mujoco


class MotionRejected(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def segment_samples(a,b,step=.01):
    return np.linspace(a,b,max(2,math.ceil(np.linalg.norm(b-a)/step)+1))


class MotionValidator:
    # Approximate envelopes must be measured against the actual A5 hardware.
    RADIUS = .05
    MARGIN = .015
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)

    def arm(self, side, trim_mount=False):
        d,m = self.data,self.model
        shoulder = d.xpos[m.body(f'{side}_shoulder_roll_link').id]
        elbow = d.xpos[m.body(f'{side}_elbow_link').id]
        tip = d.site_xpos[m.site(f'{side}_hand_preview').id]
        # The shoulder mount is adjacent to the torso by construction.
        start = shoulder + .4*(elbow-shoulder) if trim_mount else shoulder
        return np.vstack([segment_samples(start,elbow),segment_samples(elbow,tip)])

    @staticmethod
    def box_distance(points, box):
        lo,hi = np.asarray(box['min'],float),np.asarray(box['max'],float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):
            raise MotionRejected('Invalid collision box')
        return np.linalg.norm(np.maximum(np.maximum(lo-points,points-hi),0),axis=1)

    def check(self, plan, side, obstacles, observation=None):
        if side not in ('left', 'right') or not 2 <= len(plan.get('keyframes', [])) <= 36001:
            raise MotionRejected('Invalid arm or trajectory sample count')
        names=list(plan['keyframes'][0]['joint_targets_rad'])
        if any(set(f['joint_targets_rad']) != set(names) for f in plan['keyframes']):
            raise MotionRejected('Trajectory joints must match at every sample')
        for n, v in plan.get('held_joints_rad', {}).items():
            joint = self.model.joint(n)
            if n in names or not math.isfinite(v) or (joint.limited and not joint.range[0] <= v <= joint.range[1]):
                raise MotionRejected('Invalid held joint: ' + n)
        allowed={f'{side}_{j}_joint' for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')}
        if set(names) not in (allowed, allowed | {f'{side}_wrist_roll_joint'}):
            raise MotionRejected('Generated trajectory must move only the selected A5 arm')
        ts=np.array([f['time_s'] for f in plan['keyframes']],float)
        qs=np.array([[f['joint_targets_rad'][n] for n in names] for f in plan['keyframes']],float)
        if not np.isfinite(ts).all() or not np.isfinite(qs).all() or ts[0] != 0 or np.any(np.diff(ts)<=0):
            raise MotionRejected('Invalid trajectory samples')
        for col,name in enumerate(names):
            lo,hi=self.model.joint(name).range
            if np.any(qs[:,col]<lo+.049) or np.any(qs[:,col]>hi-.049):
                raise MotionRejected(f'Joint limit margin violated: {name}')
        velocities=np.diff(qs,axis=0)/np.diff(ts)[:,None]
        peak=float(np.max(np.abs(velocities)))
        if peak>.401:
            raise MotionRejected('Joint velocity exceeds 0.4 rad/s')
        accel=float(np.max(np.abs(np.diff(velocities,axis=0))/((np.diff(ts)[1:]+np.diff(ts)[:-1])/2)[:,None])) if len(velocities)>1 else 0
        if accel>1.51:
            raise MotionRejected('Sampled joint acceleration exceeds 1.5 rad/s²')
        def collision(message):
            return MotionRejected(message, {
                'keyframe_index': i,
                'hand_position_m': self.data.site_xpos[self.model.site(f'{side}_hand_preview').id].round(4).tolist(),
                'joint_targets_rad': dict(zip(names, q.tolist())),
            })
        count=0
        # Model-driven torso box anchored to the waist-yaw body.
        for i in range(len(qs)-1):
            steps=max(1,math.ceil(float(np.max(np.abs(qs[i+1]-qs[i])))/.01))
            for q in np.linspace(qs[i],qs[i+1],steps+1):
                self.data.qpos[:]=0
                for n,v in plan.get('held_joints_rad',{}).items():
                    self.data.qpos[self.model.joint(n).qposadr[0]]=v
                for n,v in zip(names,q):
                    self.data.qpos[self.model.joint(n).qposadr[0]]=v
                mujoco.mj_forward(self.model,self.data)
                points=self.arm(side)
                other=self.arm('right' if side=='left' else 'left')
                radius=self.RADIUS+self.MARGIN
                if np.min(np.linalg.norm(points[:,None]-other[None,:],axis=2))<2*radius:
                    raise collision('Self-collision: opposite arm envelope')
                # Validate against all active native collision contacts involving this arm.
                for c in self.data.contact:
                    a=self.model.body(self.model.geom_bodyid[c.geom1]).name or ''
                    b=self.model.body(self.model.geom_bodyid[c.geom2]).name or ''
                    arm=lambda n:n.startswith(side+'_') and any(k in n for k in ('shoulder','elbow','wrist'))
                    if c.dist<-.001 and (arm(a) or arm(b)):
                        raise collision(f'Self-collision: {a} / {b}')
                torso_id=self.model.body('torso_link').id
                local=(self.arm(side,trim_mount=True)-self.data.xpos[torso_id]) @ self.data.xmat[torso_id].reshape(3,3)
                torso={'min':[-.085,-.075,-.13],'max':[.075,.075,.16]}
                if np.min(self.box_distance(local,torso))<radius:
                    raise collision('Self-collision: torso envelope')
                head=self.data.xpos[torso_id]+self.data.xmat[torso_id].reshape(3,3)@np.array([0.,0.,.35])
                if np.min(np.linalg.norm(points-head,axis=1))<radius+.10:
                    raise collision('Self-collision: head envelope')
                if np.min(points[:,2])<radius:
                    raise collision('Floor collision')
                for box in obstacles:
                    if np.min(self.box_distance(points,box))<radius:
                        raise collision(f'Collision with {box.get("name","obstacle")}')
                if observation:
                    observation.require_free(points,radius)
                count+=1
        return {'samples':count,'max_velocity_rad_s':round(peak,3),'max_sampled_acceleration_rad_s2':round(accel,3),
                'joint_margin_rad':.05,'collision_margin_m':self.MARGIN,'coverage':'depth-observed volume' if observation else 'simulated scene'}


def slow_acceleration(plan, limit=1.4):
    ts=np.array([f['time_s'] for f in plan['keyframes']]); q=np.array([list(f['joint_targets_rad'].values()) for f in plan['keyframes']])
    dt=np.diff(ts); v=np.diff(q,axis=0)/dt[:,None]
    peak=np.max(np.abs(np.diff(v,axis=0))/((dt[1:]+dt[:-1])/2)[:,None]) if len(v)>1 else 0
    scale=max(1,math.sqrt(peak/limit)*1.01)
    for frame in plan['keyframes']: frame['time_s']=round(frame['time_s']*scale,6)
    plan['duration_s']=plan['keyframes'][-1]['time_s']
