const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/live.js'),'utf8');
async function fixture(){
 const elements=new Map(),sockets=[],streams=[],sounds=[],captures=[],contexts=[],intervals=new Map(),listeners={};let pendingMic;
 class Element{
  constructor(){this.children=[];this.textContent='';this.disabled=false;this.scrollHeight=this.scrollTop=this.clientHeight=0;}
  append(...kids){this.children.push(...kids);kids.forEach(k=>k.parent=this);}replaceChildren(){this.children=[];}
  get childElementCount(){return this.children.length;}get firstElementChild(){return this.children[0];}
  remove(){this.parent.children=this.parent.children.filter(c=>c!==this);}click(){if(!this.disabled)return this.onclick?.();}
 }
 const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 const document={hidden:false,getElementById:$,createElement:()=>new Element(),addEventListener:(k,v)=>listeners[k]=v};
 class Socket{static OPEN=1;constructor(){this.readyState=1;this.sent=[];this.bufferedAmount=0;sockets.push(this);}send(x){this.sent.push(typeof x==='string'?JSON.parse(x):x);}close(){this.readyState=3;}}
 class Audio{
  constructor(options){this.sampleRate=options?.sampleRate||48000;this.currentTime=0;this.destination={};this.audioWorklet={addModule:async()=>{}};contexts.push(this);}
  async resume(){}async close(){this.closed=true;}createMediaStreamSource(){return{connect(){}};}createGain(){return{gain:{},connect(){}};}
  createBuffer(c,n,r){return{getChannelData:()=>new Float32Array(n)};}
  createBufferSource(){const s={connect(){},disconnect(){},start(at){this.at=at;},stop(){this.stopped=true;}};sounds.push(s);return s;}
 }
 class Capture{constructor(){this.port={};captures.push(this);}connect(){}disconnect(){this.disconnected=true;}}
 function newStream(){const track={label:'Test mic',stop(){this.stopped=true;}};const s={track,getTracks:()=>[track],getAudioTracks:()=>[track]};streams.push(s);return s;}
 const context=vm.createContext({document,window:{addEventListener:(k,v)=>listeners[k]=v},location:{host:'127.0.0.1:8771'},WebSocket:Socket,AudioContext:Audio,AudioWorkletNode:Capture,
  navigator:{mediaDevices:{getUserMedia:async()=>pendingMic?pendingMic.promise:newStream()}},
  fetch:async()=>({ok:true,json:async()=>({token:'test',live_model:'gpt-live-1',voice:'cedar',backend_model:'gpt-5-mini',voice_focus:true,enhancement_level:.8,tyto:true})}),
  setInterval:fn=>{const id=intervals.size+1;intervals.set(id,fn);return id;},clearInterval:id=>intervals.delete(id),ArrayBuffer,Int16Array,Float32Array,console});
 vm.runInContext(source,context);
 async function settle(){for(let i=0;i<20;i++)await Promise.resolve();}await settle();
 async function event(e,socket=sockets.at(-1)){await socket.onmessage({data:JSON.stringify(e)});await settle();}
 async function start(){await $('start').click();sockets.at(-1).onopen();await event({type:'ready'});await event({type:'listening'});}
 async function audio(value){const data=new Int16Array(1600).fill(value).buffer;await sockets.at(-1).onmessage({data});await settle();}
 function mic(){captures.at(-1).port.onmessage({data:new Int16Array(320).fill(1000).buffer});}
 return{$,sockets,streams,sounds,captures,contexts,intervals,document,listeners,settle,event,start,audio,mic,
  tick:seconds=>{contexts.forEach(c=>c.currentTime+=seconds);for(const fn of intervals.values())fn();},
  deferMic:()=>{let resolve;const promise=new Promise(r=>resolve=r);pendingMic={promise};return()=>{pendingMic=null;const s=newStream();resolve(s);return s;};}};
}
test('live audio plays before any done event and suppresses mic during audible playback',async()=>{
 const f=await fixture();await f.start();f.mic();assert.equal(new Int16Array(f.sockets[0].sent.at(-1))[0],1000);
 await f.audio(0);assert.equal(f.sounds.length,0,'silent output must not mute the caller');
 await f.audio(2000);assert.equal(f.sounds.length,1);f.mic();assert.equal(new Int16Array(f.sockets[0].sent.at(-1))[0],0);
 f.tick(1);f.mic();assert.equal(new Int16Array(f.sockets[0].sent.at(-1))[0],1000);
 await f.$('stop').click();assert.ok(f.streams.every(s=>s.track.stopped));assert.equal(f.intervals.size,0);assert.ok(f.sounds[0].stopped);
});
test('Stop releases delayed microphone permission and never opens the websocket',async()=>{
 const f=await fixture(),resolve=f.deferMic();const pending=f.$('start').click();await f.settle();await f.$('stop').click();
 const acquired=resolve();await pending;assert.equal(acquired.track.stopped,true);assert.equal(f.sockets.length,0);
});
test('stale audio is ignored after Stop and hidden pages release the session',async()=>{
 const f=await fixture();await f.start();const socket=f.sockets[0];f.document.hidden=true;await f.listeners.visibilitychange();await f.settle();
 await socket.onmessage({data:new Int16Array(1600).fill(2000).buffer});assert.equal(f.sounds.length,0);assert.ok(f.streams[0].track.stopped);
});
test('Tyto clears queued speech and excessive playback backlog stops visibly',async()=>{
 const f=await fixture();await f.start();await f.audio(2000);await f.event({type:'clear_audio'});assert.ok(f.sounds[0].stopped);
 assert.deepEqual(f.sockets[0].sent.at(-1),{type:'speaker',busy:false});
 for(let n=0;n<30&&f.sockets[0].readyState===1;n++)await f.audio(2000);
 assert.match(f.$('error').textContent,/Playback fell behind/);assert.ok(f.streams[0].track.stopped);
});
