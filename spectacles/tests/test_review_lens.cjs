'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../Assets/R1Trajectory.js'),'utf8');
function reviewFixture() {
  let now=0;const sent=[];
  const c=vm.createContext({getTime:()=>now,print(){},script:{},JSON});
  vm.runInContext('var pendingReview=null, tagAnchored=false, socketReady=false, reviewSent=false, reviewChoice="", reviewChoiceAt=-1000, reviewMessage="";'+
    source.slice(source.indexOf('function reviewPinch('),source.indexOf('try {\n    var gestureModule')),c);
  c.socket={send:message=>sent.push(JSON.parse(message))};
  return {c,sent,time:value=>now=value};
}
test('review sends only on the second matching pinch with calibration and connection',()=>{
  const f=reviewFixture();f.c.pendingReview={id:'proposal-123456789'};
  f.c.reviewPinch('approve');assert.equal(f.sent.length,0);
  f.c.tagAnchored=true;f.c.socketReady=true;
  f.c.reviewPinch('approve');assert.equal(f.sent.length,0);
  f.time(.5);f.c.reviewPinch('approve');
  assert.deepEqual(f.sent,[{type:'review_decision',version:1,id:'proposal-123456789',decision:'approve'}]);
  f.time(.7);f.c.reviewPinch('approve');assert.equal(f.sent.length,1);
});
test('left pinch rejects independently of right pinch',()=>{
  const f=reviewFixture();f.c.pendingReview={id:'proposal-123456789'};f.c.tagAnchored=true;f.c.socketReady=true;
  f.c.reviewPinch('approve');f.time(.5);f.c.reviewPinch('decline');
  assert.equal(f.sent.length,0);f.time(1);f.c.reviewPinch('decline');
  assert.equal(f.sent[0].decision,'decline');
});
test('Lens has no speech UI, microphone, or voice messages',()=>{
  assert.doesNotMatch(source,/AsrModule|microphoneAudio|voice_start|voice_command|GPT-LIVE|LISTENING|SPEAK NOW/);
  assert.match(source,/reviewPinch\("approve"\)/);
  assert.match(source,/reviewPinch\("decline"\)/);
});
