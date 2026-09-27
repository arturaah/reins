'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../Assets/R1Trajectory.js'),'utf8');
function fixture() {
  let now=0,started=0,stopped=0;const sent=[],reviews=[];
  const control={sampleRate:0,maxFrameSize:640,start(){started++},stop(){stopped++},
    getAudioFramePCM16(samples){samples.fill(1234);return {x:640,y:1,z:1}}};
  const c=vm.createContext({script:{useLiveVoice:true,microphoneAudio:{control}},getTime:()=>now,print(){},Int16Array,Uint8Array,Date,Math});
  vm.runInContext(source.slice(source.indexOf('var socket ='),source.indexOf('\ntry {\n    if (script.tagDetectedSound)')),c);
  c.socketReady=true;c.socket={send:value=>sent.push(value)};c.reviewPinch=value=>reviews.push(value);
  return {c,sent,reviews,control,time:v=>now=v,started:()=>started,stopped:()=>stopped};
}
test('glasses wait for GPT-Live readiness then stream mono PCM16, without ASR text commands',()=>{
  const f=fixture();f.c.startVoice();
  assert.equal(JSON.parse(f.sent[0]).type,'voice_start');assert.equal(f.started(),0);
  f.c.streamLiveMicrophone();assert.equal(f.sent.length,1);
  f.c.liveVoiceEvent({type:'listening'});f.c.streamLiveMicrophone();
  assert.equal(f.started(),1);assert.equal(f.control.sampleRate,16000);
  assert.equal(f.sent.length,3);assert.equal(f.sent[1].length,640);
  assert.deepEqual(Array.from(f.sent[1].slice(0,2)),[210,4]);
  f.c.liveVoiceEvent({type:'log',stage:'playback',status:'started'});f.c.streamLiveMicrophone();
  assert.ok(f.sent[3].every(v=>v===0));
  f.c.stopVoice(true);assert.equal(JSON.parse(f.sent.at(-1)).type,'voice_stop');assert.equal(f.stopped(),1);
  f.c.liveVoiceEvent({type:'listening'});assert.equal(f.started(),1);
});
test('review gestures stop audio and retain the existing review route',()=>{
  const f=fixture();f.c.startVoice();f.c.liveVoiceEvent({type:'listening'});
  f.c.pendingReview={id:'proposal'};f.c.voicePinch('right');
  assert.deepEqual(f.reviews,['approve']);assert.equal(f.c.voiceListening,false);
  assert.equal(JSON.parse(f.sent.at(-1)).type,'voice_stop');
});
test('failed connection or microphone does not permanently block a later retry',()=>{
  const f=fixture();f.c.startVoice();f.time(21);f.c.streamLiveMicrophone();
  assert.equal(f.c.voiceListening,false);
  f.c.startVoice();f.control.start=()=>{throw new Error('denied')};f.c.liveVoiceEvent({type:'listening'});
  assert.equal(f.c.voiceListening,false);assert.equal(f.c.microphoneBlocked,false);
  f.control.start=()=>{};f.c.startVoice();f.c.liveVoiceEvent({type:'listening'});
  assert.equal(f.c.liveReady,true);
});
test('legacy ASR retains main\'s transient-error retries and permission diagnostics',()=>{
  const f=fixture();let options,starts=0;
  f.c.script.useLiveVoice=false;
  f.c.global={deviceInfoSystem:{isInternetAvailable:()=>true}};
  f.c.AsrModule={AsrMode:{HighAccuracy:1},AsrStatusCode:{Unauthenticated:401,NoInternet:503},
    AsrTranscriptionOptions:{create(){return {onTranscriptionUpdateEvent:{add(){}},onTranscriptionErrorEvent:{add(cb){this.callback=cb}}}}}};
  f.c.asrModule={startTranscribing(o){options=o;starts++},stopTranscribing(){}};
  f.c.startVoice();options.onTranscriptionErrorEvent.callback(503);
  assert.equal(f.c.microphoneBlocked,false);assert.equal(f.c.voiceMessage,'NO INTERNET FOR SPEECH');
  f.c.startVoice();assert.equal(starts,2);options.onTranscriptionErrorEvent.callback(401);
  assert.equal(f.c.microphoneBlocked,true);assert.equal(f.c.voiceMessage,'SPEECH PERMISSION DENIED');
});

function connectionFixture(token) {
  const f=fixture();let closed=0;
  const socket={send:value=>f.sent.push(value),close(){closed++}};
  f.c.script.pairingToken=token;
  f.c.script.websocketUrl='ws://192.168.1.5:8765';
  f.c.script.internetModule={createWebSocket:()=>socket};
  vm.runInContext(source.slice(source.indexOf('function connect() {'),source.indexOf('script.createEvent("UpdateEvent")')),f.c);
  f.c.connect();socket.onopen();
  return {...f,socket,closed:()=>closed,message:m=>socket.onmessage({data:JSON.stringify(m)})};
}
test('wireless microphone waits for pairing and each reconnect authenticates again',()=>{
  const f=connectionFixture('test-token');
  assert.equal(f.c.socketReady,false);
  f.c.startVoice();assert.equal(f.c.voiceListening,false);assert.equal(f.started(),0);
  f.message({type:'voice_event',id:f.c.voiceCommandId,event:{type:'listening'}});
  assert.equal(f.started(),0);
  f.message({type:'pairing_required',version:1});
  assert.deepEqual(JSON.parse(f.sent[0]),{type:'pair',version:1,token:'test-token'});
  f.message({type:'pairing_result',version:1,accepted:true});
  assert.equal(f.c.socketReady,true);
  f.c.startVoice();f.message({type:'voice_event',id:f.c.voiceCommandId,event:{type:'listening'}});
  assert.equal(f.started(),1);
  f.socket.onclose({code:1000});assert.equal(f.stopped(),1);assert.equal(f.c.socketReady,false);
  f.c.connect();f.socket.onopen();assert.equal(f.c.socketReady,false);
  f.message({type:'pairing_required',version:1});
  assert.equal(JSON.parse(f.sent.at(-1)).type,'pair');
});
test('missing or incorrect pairing token never starts capture and shows the fix',()=>{
  for (const token of ['', 'incorrect']) {
    const f=connectionFixture(token);
    f.message({type:'pairing_required',version:1});
    f.message({type:'pairing_result',version:1,accepted:false});
    f.c.startVoice();
    assert.equal(f.c.socketReady,false);assert.equal(f.started(),0);assert.equal(f.closed(),1);
    assert.equal(f.sent.filter(v=>JSON.parse(v).type==='voice_start').length,0);
  }
});
test('USB works without a token and an old unpaired feed times out when a token is configured',()=>{
  const usb=connectionFixture('');usb.c.startVoice();
  assert.equal(usb.c.socketReady,true);assert.equal(JSON.parse(usb.sent[0]).type,'voice_start');
  const wifi=connectionFixture('test-token');wifi.time(6);wifi.c.checkPairingTimeout();
  assert.equal(wifi.closed(),1);assert.match(wifi.c.voiceMessage,/PAIRING TIMED OUT/);
});
