// Exercise the actual page script with deterministic media, socket and timer boundaries.
// No browser automation, microphone capture, network calls or provider keys are used.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');

async function fixture() {
  const elements=new Map(), streams=[], sockets=[], sounds=[], timers=new Map(), listeners={};
  let now=0, nextTimer=0, pendingMic=null;
  class Element {
    constructor() { this.children=[]; this.textContent=''; this.checked=false; this.disabled=false; this.value=''; this.scrollHeight=0; this.scrollTop=0; this.clientHeight=0; }
    append(...children) { this.children.push(...children); children.forEach(c=>{if(typeof c==='object') c.parent=this;}); }
    replaceChildren() { this.children=[]; }
    get childElementCount() { return this.children.length; }
    get firstElementChild() { return this.children[0]; }
    remove() { this.parent.children=this.parent.children.filter(c=>c!==this); }
    closest() { return this; }
    click() { if(!this.disabled) return this.onclick?.(); }
  }
  const $=id=>{if(!elements.has(id)) elements.set(id,new Element());return elements.get(id);};
  $('autoEnd').checked=true;
  const document={hidden:false,getElementById:$,createElement:()=>new Element(),createTextNode:text=>({textContent:text}),addEventListener:(event,callback)=>{listeners[event]=callback;}};
  const window={addEventListener:(event,callback)=>{listeners[event]=callback;}}; window.parent=window;
  class Socket {
    static OPEN=1;
    constructor() { this.readyState=1; this.sent=[]; this.bufferedAmount=0; sockets.push(this); }
    send(data) { this.sent.push(typeof data==='string'?JSON.parse(data):data); }
    close() { this.readyState=3; }
  }
  class Audio {
    constructor(options) { this.sampleRate=options?.sampleRate || 24000; this.destination={}; this.audioWorklet={addModule:async()=>{}}; }
    async resume() {}
    async close() {}
    createMediaStreamSource() { return {connect(){}}; }
    createGain() { return {gain:{},connect(){}}; }
    createBuffer(channels,length,rate) { return {duration:length/rate,getChannelData:()=>new Float32Array(length)}; }
    createBufferSource() {
      const sound={connect(){},disconnect(){},start(){this.started=true;},stop(){this.stopped=true;}};
      sounds.push(sound); return sound;
    }
  }
  class Capture {
    constructor() { this.port={postMessage:()=>this.port.onmessage?.({data:'flushed'})}; }
    connect() {}
    disconnect() {}
  }
  function newStream() {
    const track={label:'Test microphone',stopped:false,stop(){this.stopped=true;}};
    const stream={track,getTracks:()=>[track],getAudioTracks:()=>[track]}; streams.push(stream); return stream;
  }
  const context=vm.createContext({document,window,location:{host:'127.0.0.1:8770'},WebSocket:Socket,AudioContext:Audio,AudioWorkletNode:Capture,
    navigator:{mediaDevices:{getUserMedia:async()=>{if(pendingMic) return pendingMic.promise;return newStream();}}},
    fetch:async()=>({ok:true,json:async()=>({provider:'audio',token:'test-token',chat_model:'gpt-5-mini',stt_streaming:true,voice_focus:true,tyto_nudge:true})}),
    setTimeout:(callback,delay)=>{const id=++nextTimer;timers.set(id,{at:now+delay,callback});return id;},clearTimeout:id=>timers.delete(id),
    ArrayBuffer,Int16Array,Float32Array,console});
  vm.runInContext(source,context);
  async function settle() { for(let i=0;i<20;i++) await Promise.resolve(); }
  await settle();
  async function tick(ms) {
    const target=now+ms;
    while(true) {
      const next=[...timers.entries()].filter(([,v])=>v.at<=target).sort((a,b)=>a[1].at-b[1].at)[0];
      if(!next) break;
      now=next[1].at;timers.delete(next[0]);await next[1].callback();await settle();
    }
    now=target;await settle();
  }
  async function event(data,socket=sockets.at(-1)) { await socket.onmessage({data:JSON.stringify(data)});await settle(); }
  async function connect() { await $('connect').click();const socket=sockets.at(-1);socket.onopen();await event({type:'ready'});return socket; }
  async function start() { await $('talk').click();await event({type:'recording'}); }
  async function end() { await event({type:'endpoint'});await settle();await event({type:'thinking'}); }
  async function audio() {
    await event({type:'audio',sample_rate:24000});await sockets.at(-1).onmessage({data:new ArrayBuffer(4800)});await settle();
  }
  async function finished() { sounds.at(-1).onended();await settle();await event({type:'ready'}); }
  return {$,document,listeners,streams,sockets,sounds,event,connect,start,end,audio,finished,tick,settle,
    liveMics:()=>streams.filter(s=>!s.track.stopped).length,
    deferMic:()=>{let resolve;const promise=new Promise(r=>resolve=r);pendingMic={promise};return ()=>{pendingMic=null;const stream=newStream();resolve(stream);return stream;};}};
}

test('conversation loops through replies only after playback acknowledgement, with no barge-in',async()=>{
  const f=await fixture(), socket=await f.connect();
  assert.equal(f.$('conversationLoop').checked,true);
  assert.equal(f.liveMics(),0,'Connect alone must not start the mic');
  await f.start();
  for(let turn=0;turn<3;turn++) {
    assert.equal(f.liveMics(),1);
    await f.end();assert.equal(f.liveMics(),0,'mic is released before inference');
    await f.audio();await f.tick(1000);
    assert.equal(f.liveMics(),0,'no capture during speaker playback');
    f.sounds.at(-1).onended();await f.tick(500);
    assert.equal(f.liveMics(),0,'wait for server acknowledgement, not only audio end');
    assert.equal(socket.sent.at(-1).type,'played');
    await f.event({type:'ready'});await f.tick(299);assert.equal(f.liveMics(),0);
    await f.tick(1);assert.equal(f.liveMics(),1);await f.event({type:'recording'});
  }
  assert.equal(socket.sent.filter(e=>e.type==='start').length,4);
  await f.$('stop').click();assert.equal(f.liveMics(),0);
});

test('Stop cancels a pending restart and ignores stale Ready events',async()=>{
  const f=await fixture(),socket=await f.connect();await f.start();await f.end();await f.audio();await f.finished();
  await f.$('stop').click();await f.event({type:'ready'},socket);await f.tick(2000);
  assert.equal(f.streams.length,1);assert.equal(f.liveMics(),0);
});

test('Stop also disposes a microphone permission result that arrives after stopping',async()=>{
  const f=await fixture();await f.connect();await f.start();await f.end();await f.audio();await f.finished();
  const resolve=f.deferMic();await f.tick(300);await f.$('stop').click();
  const acquired=resolve();await f.settle();await f.tick(1000);
  assert.equal(acquired.track.stopped,true);assert.equal(f.liveMics(),0);
  assert.equal(f.sockets[0].sent.filter(e=>e.type==='start').length,1);
});

test('single-turn mode and greeting tests never start an unintended microphone loop',async()=>{
  const f=await fixture();await f.connect();
  await f.$('hello').click();await f.audio();await f.finished();await f.tick(500);
  assert.equal(f.liveMics(),0,'a test greeting is not consent to start capture');
  f.$('conversationLoop').checked=false;f.$('conversationLoop').onchange();
  await f.start();await f.end();await f.audio();await f.finished();await f.tick(1000);
  assert.equal(f.streams.length,1);assert.equal(f.liveMics(),0);
});

test('Tyto clarification and recoverable no-speech results rearm the active loop',async()=>{
  const f=await fixture();await f.connect();await f.start();
  await f.event({type:'nudge'});assert.equal(f.liveMics(),0);
  await f.audio();await f.finished();await f.tick(300);assert.equal(f.liveMics(),1);
  await f.event({type:'recording'});await f.end();
  await f.event({type:'result',blocked:'empty_transcript',notice:'No speech found.'});
  await f.event({type:'ready'});await f.tick(300);assert.equal(f.liveMics(),1);
  assert.equal(f.streams.length,3);
  await f.$('stop').click();
});

test('turning off the loop or hiding the page cancels scheduled capture',async()=>{
  const f=await fixture();await f.connect();await f.start();await f.end();await f.audio();await f.finished();
  f.$('conversationLoop').checked=false;f.$('conversationLoop').onchange();await f.tick(1000);
  assert.equal(f.streams.length,1);
  f.$('conversationLoop').checked=true;f.$('conversationLoop').onchange();await f.start();
  f.document.hidden=true;await f.listeners.visibilitychange();await f.settle();
  assert.equal(f.liveMics(),0);await f.tick(1000);assert.equal(f.streams.length,2);
});
