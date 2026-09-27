'use strict';
const $=id=>document.getElementById(id);
let config,ws,input,output,stream,capture,gateTimer;
let epoch=0,starting=false,listening=false,speaking=false,nextPlay=0,speechUntil=0;
let lastRole=null,lastLine=null;
const sources=new Set();
function log(event) {
  const row=document.createElement('div');row.className='log-entry';
  const title=document.createElement('div');title.className='log-title';
  title.textContent=`${event.stage} · ${event.status}`;
  const meta=document.createElement('div');meta.className='log-meta';
  meta.textContent=new Date((event.timestamp || Date.now()/1000)*1000).toLocaleTimeString();
  const detail=document.createElement('div');
  const parts=[];
  if(event.model)parts.push(event.model);
  if(typeof event.elapsed_s==='number')parts.push(`${event.elapsed_s.toFixed(2)}s`);
  if(typeof event.enhancement_level==='number')parts.push(`Enhancement ${Math.round(event.enhancement_level*100)}%`);
  if(event.text)parts.push(event.text);
  if(event.scores)parts.push(Object.entries(event.scores).map(([k,v])=>`${k.replaceAll('_',' ')} ${Number(v).toFixed(3)}`).join(' · '));
  if(event.smoothed)parts.push(`Smoothed risk ${event.smoothed.risk_score.toFixed(3)}`);
  detail.textContent=parts.join(' · ');row.append(meta,title,detail);
  const panel=$('pipelineLog'),follow=panel.scrollHeight-panel.scrollTop-panel.clientHeight<50;
  panel.append(row);while(panel.childElementCount>200)panel.firstElementChild.remove();
  if(follow)panel.scrollTop=panel.scrollHeight;
}
function controls() {
  $('start').disabled=!config || starting || !!ws;
  $('stop').disabled=!starting && !ws && !stream;
}
function send(event) {if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(event));}
function updateGate() {
  if(config?.output==='r1') {
    if(listening)$('status').textContent=speaking?'R1 is speaking · microphone suppressed':'Listening · R1 speaker connected';
    return;
  }
  const busy=!!output && output.currentTime<speechUntil;
  if(busy!==speaking) {
    speaking=busy;
    send({type:'speaker',busy});
    log({stage:'Playback',status:busy?'started':'completed'});
  }
  if(listening)$('status').textContent=speaking?'Reins is speaking · microphone suppressed':'Listening · conversation active';
}
function clearAudio() {
  for(const source of sources){source.onended=null;source.stop();source.disconnect();}
  sources.clear();nextPlay=output?.currentTime || 0;speechUntil=0;
  // End the old playback gate before a Tyto clarification begins a new one.
  updateGate();
}
async function stop() {
  epoch++;starting=listening=false;clearInterval(gateTimer);
  const socket=ws;ws=null;
  if(capture){capture.port.onmessage=null;capture.disconnect();capture=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  clearAudio();speaking=false;
  const contexts=[input,output];input=output=null;
  if(socket){if(socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({type:'stop'}));socket.close();}
  await Promise.all(contexts.filter(Boolean).map(c=>c.close()));
  $('status').textContent='Microphone off · stopped';$('mic').textContent='Microphone off';controls();
}
async function fail(error) {
  $('error').textContent=error?.message || String(error);
  log({stage:'Session',status:'error',text:$('error').textContent});await stop();
}
function transcript(role,text) {
  if(role!==lastRole || !lastLine) {
    const row=document.createElement('div');row.className='line';
    const label=document.createElement('span');label.className='label';label.textContent=role==='user'?'You':'Reins';
    lastLine=document.createElement('span');row.append(label,lastLine);$('transcript').append(row);lastRole=role;
    while($('transcript').childElementCount>80)$('transcript').firstElementChild.remove();
  }
  lastLine.textContent=(lastLine.textContent+text).slice(-8000);
  $('transcript').scrollTop=$('transcript').scrollHeight;
}
function play(data) {
  if(!output || !data.byteLength || data.byteLength%2 || data.byteLength>64000)throw new Error('Invalid live audio');
  const samples=new Int16Array(data);
  const at=Math.max(nextPlay,output.currentTime+.035);
  if(at-output.currentTime>2)throw new Error('Playback fell behind. Reconnect to avoid delayed speech.');
  nextPlay=at+samples.length/16000;
  let audible=false;
  for(let i=0;i<samples.length;i++)if(Math.abs(samples[i])>100){audible=true;break;}
  if(!audible)return; // GPT-Live sends silence too; silence is not a spoken response.
  speechUntil=nextPlay+.4;updateGate();
  const buffer=output.createBuffer(1,samples.length,16000),channel=buffer.getChannelData(0);
  for(let i=0;i<samples.length;i++)channel[i]=samples[i]/32768;
  const source=output.createBufferSource();source.buffer=buffer;source.connect(output.destination);sources.add(source);
  source.onended=()=>{sources.delete(source);source.disconnect();};source.start(at);
}
$('start').onclick=async()=>{
  if(starting || ws || !config)return;
  const generation=++epoch;starting=true;$('error').textContent='';controls();
  try {
    output=new AudioContext();await output.resume();
    if(generation!==epoch)return;
    const acquired=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:!config.voice_focus,autoGainControl:!config.voice_focus},video:false});
    if(generation!==epoch){acquired.getTracks().forEach(t=>t.stop());return;}
    stream=acquired;input=new AudioContext({sampleRate:16000});const context=input;
    await context.resume();if(generation!==epoch)return;
    if(context.sampleRate!==16000)throw new Error('This browser cannot record at 16 kHz. Try Chrome.');
    await context.audioWorklet.addModule('/live-mic.js');if(generation!==epoch)return;
    const source=context.createMediaStreamSource(stream),mute=context.createGain();mute.gain.value=0;
    capture=new AudioWorkletNode(context,'live-microphone');
    const socket=new WebSocket(`ws://${location.host}/voice`);ws=socket;socket.binaryType='arraybuffer';
    $('status').textContent='Connecting GPT-Live…';controls();
    socket.onopen=()=>socket.send(JSON.stringify({token:config.token}));
    socket.onmessage=async({data})=>{
      if(ws!==socket)return;
      try {
        if(data instanceof ArrayBuffer){play(data);return;}
        const event=JSON.parse(data);
        if(event.type==='ready') {
          send({type:'listen',sample_rate:16000});
          capture.port.onmessage=({data})=>{
            if(ws!==socket || socket.readyState!==WebSocket.OPEN)return;
            if(socket.bufferedAmount>32768){fail('Audio connection fell behind. Reconnect.');return;}
            updateGate();
            const samples=new Int16Array(data);
            if(speaking)samples.fill(0);
            socket.send(data);
          };
          source.connect(capture);capture.connect(mute);mute.connect(context.destination);
          gateTimer=setInterval(updateGate,50);
        }
        if(event.type==='listening') {
          starting=false;listening=true;
          $('mic').textContent=`${stream.getAudioTracks()[0]?.label || 'Default microphone'} · 20 ms audio frames`;
          log({stage:'Voice',status:'streaming',model:config.live_model});updateGate();controls();
        }
        if(event.type==='transcript_delta')transcript(event.role,event.text);
        if(event.type==='log') {
          log(event);
          if(config.output==='r1' && event.stage==='playback') {
            speaking=event.status==='started';updateGate();
          }
        }
        if(event.type==='clear_audio')clearAudio();
        if(event.type==='error')await fail(event.text);
      }catch(error){await fail(error);}
    };
    socket.onerror=()=>{if(ws===socket)fail('Voice connection failed. Reconnect to try again.');};
    socket.onclose=()=>{if(ws===socket)stop();};
  }catch(error){if(generation===epoch)await fail(error.name==='NotAllowedError'?'Microphone access was not granted.':error);}
};
$('stop').onclick=stop;
$('clearLog').onclick=()=>$('pipelineLog').replaceChildren();
window.addEventListener('pagehide',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
document.addEventListener('keydown',event=>{if(event.key==='Escape')stop();});
fetch('/config').then(r=>{if(!r.ok)throw new Error('Voice configuration unavailable');return r.json();}).then(c=>{
  config=c;$('voice').textContent=`${c.live_model} · ${c.voice}${c.metallic?' · metallic':''}`;$('backend').textContent=c.backend_model;
  $('outputMode').textContent=c.output==='r1'?'R1 SPEAKER':'COMPUTER AUDIO';
  $('focus').textContent=c.voice_focus?`On · ${Math.round(c.enhancement_level*100)}%`:'Off';
  $('tyto').textContent=c.tyto?'On · nudges after 5s':'Off';controls();
}).catch(fail);
