"""Run the Lens's real protocol/gesture functions with a fake native scene."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("REINS_NODE") or shutil.which("node")
LENS = Path(__file__).resolve().parents[1]/"Assets/R1Trajectory.js"


@unittest.skipUnless(NODE, "Node is needed for Lens JavaScript protocol tests")
class LensProtocolTests(unittest.TestCase):
    def test_review_priority_tracking_expiry_and_draft_gates(self):
        program = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
function part(start,end){const a=source.indexOf(start); return source.slice(a,source.indexOf(end,a));}
let now=10, sent=[], voiceStarts=0, voiceStops=0;
const c={getTime:()=>now, print:()=>{}, socketReady:true, socketSession:'connection',
  pendingReview:null, reviewExpiresAt:0, tagAnchored:true,
  observedTags:{left:{time:10},right:{time:10}}, reviewSent:false,
  reviewMessage:'', latestTrajectory:null, voiceListening:false, voicePinchAt:-1000, voiceCancelPinchAt:-1000,
  voiceGestureLockUntil:0, socket:{send:x=>sent.push(JSON.parse(x))},
  leftVisual:{},rightVisual:{},drawPoint:()=>{},drawTube:()=>{}, isTrajectoryPath:Array.isArray,
  startVoice:()=>{voiceStarts++;c.voiceListening=true}, stopVoice:()=>{voiceStops++;c.voiceListening=false}, voiceNotice:()=>{}};
vm.createContext(c);
vm.runInContext(part('function trackingState()', '\ntry {\n    var gestureModule'),c);
vm.runInContext(part('function voicePinch(side)', '\ntry {\n    if (script.tagDetectedSound)'),c);
vm.runInContext(part('function applyTrajectory(message)', 'function localMock'),c);
const proposal={id:'a'.repeat(32),digest:'b'.repeat(64),revision:1,session:'connection',expires_in_s:60,mode:'sim'};
const message={type:'trajectory',version:1,phase:'draft',frame:'robot_base',units:'m',hands:{left:[],right:[]},review:null};
c.applyTrajectory(message); c.reviewPinch('approve'); assert.equal(sent.length,0); assert.equal(c.pendingReview,null);
c.voiceListening=true;
c.applyTrajectory({...message,phase:'review',review:proposal}); assert.equal(voiceStops,1);
c.voicePinch('right');
assert.equal(voiceStarts,0); assert.equal(sent.length,1); assert.equal(sent[0].session,'connection');
assert.equal(sent[0].digest,proposal.digest); assert.equal(sent[0].revision,1);
assert.equal(sent[0].decision,'approve');
// A single pinch accepts once; repeated events cannot send another decision.
c.voicePinch('right');now+=.5;c.voicePinch('right');assert.equal(sent.length,1);
// A new revision needs its own pinch and sends the exact new identity.
c.applyTrajectory({...message,phase:'review',review:{...proposal,revision:2}});
assert.equal(sent.length,1);
c.applyTrajectory({...message,phase:'review',review:{...proposal,revision:3}});
c.voicePinch('right');assert.equal(sent.length,2);assert.equal(sent[1].revision,3);
// Stale tags block approval, but never prevent rejection.
c.applyTrajectory({...message,phase:'review',review:{...proposal,revision:4}});
now=20;c.reviewPinch('approve');assert.equal(sent.length,2);
c.voicePinch('left');assert.equal(sent.length,3);assert.equal(sent[2].decision,'decline');
c.applyTrajectory({...message,phase:'review',review:{...proposal,revision:5,expires_in_s:.1}});
now+=.2;c.reviewPinch('approve');assert.equal(sent.length,3);
// Data from a former socket session never offers review.
c.applyTrajectory({...message,phase:'review',review:{...proposal,session:'old'}});assert.equal(c.pendingReview,null);
c.reviewPinch('approve');assert.equal(sent.length,3);
'''
        result = subprocess.run([NODE, "-e", program, str(LENS)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
