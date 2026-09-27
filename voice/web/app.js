'use strict';
const $ = id => document.getElementById(id);
let config, ws, input, stream, capture, output, playback, timer, flush;
let ready = false, busy = false, recording = false, epoch = 0, expectingAudio = false;
let loopActive = false, resumeTimer;
const stageNames = {stt:'STT',tts:'TTS',llm:'LLM',voice_focus:'Voice Focus 2.2',tyto:'Tyto 1.1',vad:'VAD',playback:'Playback',session:'Session'};
function pipelineLog(event) {
  if(event.stage==='stt' && (event.status==='partial' || event.status==='completed')) {
    $('liveTranscript').hidden=false; $('partialText').textContent=event.text || '';
    $('liveLabel').textContent=event.status==='partial' ? 'Transcribing · provisional' : 'Transcript';
    if(event.status==='partial') return;
  }
  if(event.stage==='stt' && ['discarded','no_speech','error'].includes(event.status)) $('liveTranscript').hidden=true;
  const row = document.createElement('div'); row.className = 'log-entry';
  const meta = document.createElement('div'); meta.className = 'log-meta';
  const date = new Date((event.timestamp || Date.now()/1000)*1000);
  meta.textContent = `${date.toLocaleTimeString([], {hour12:false})}.${String(date.getMilliseconds()).padStart(3,'0')}${event.turn ? ` · turn ${event.turn}` : ''}`;
  const title = document.createElement('div'); title.className = `log-title${event.status === 'error' ? ' log-error' : ''}`;
  title.textContent = `${stageNames[event.stage] || event.stage} · ${event.status.replaceAll('_',' ')}`;
  const detail = [];
  if (event.model) detail.push(event.model);
  if (typeof event.enabled === 'boolean') detail.push(event.enabled ? 'on' : 'off');
  if (typeof event.enhancement_level === 'number') detail.push(`enhancement ${Math.round(event.enhancement_level*100)}%`);
  if (typeof event.elapsed_s === 'number') detail.push(`${event.elapsed_s.toFixed(2)}s processing`);
  if (typeof event.audio_s === 'number') detail.push(`${event.audio_s.toFixed(2)}s audio`);
  if (event.cause) detail.push(`cause: ${event.cause.replaceAll('_',' ')}`);
  if (event.code) detail.push(`code: ${event.code}`);
  if (event.error_kind) detail.push(`type: ${event.error_kind}`);
  if (typeof event.input_rms_dbfs === 'number') detail.push(`mic ${event.input_rms_dbfs.toFixed(1)} dBFS`);
  if (typeof event.output_rms_dbfs === 'number') detail.push(`enhanced ${event.output_rms_dbfs.toFixed(1)} dBFS`);
  if (typeof event.risk_score === 'number') detail.push(`smoothed risk ${event.risk_score.toFixed(3)}`);
  const description = document.createElement('div'); description.textContent = detail.join(' · ');
  row.append(meta,title,description);
  for (const [label,scores] of [['Raw',event.scores],['Smoothed',event.smoothed]]) {
    if (!scores) continue;
    const values = document.createElement('div'); values.className = 'log-scores';
    values.textContent = `${label}: ` + Object.entries(scores).map(([key,value])=>`${key.replaceAll('_',' ')} ${Number(value).toFixed(3)}`).join(' · ');
    row.append(values);
  }
  if (event.text) { const text=document.createElement('div'); text.className='log-text'; text.textContent=event.text; row.append(text); }
  const panel=$('pipelineLog'), follow=panel.scrollHeight-panel.scrollTop-panel.clientHeight<50;
  panel.append(row); $('logEmpty').hidden=true;
  while(panel.childElementCount>200) panel.firstElementChild.remove();
  if(follow) panel.scrollTop=panel.scrollHeight;
  if(event.stage==='tyto' && config?.tyto) {
    $('tytoState').textContent=event.status==='reading' ? `On · risk ${event.scores.risk_score.toFixed(2)}` : event.status.replaceAll('_',' ');
  }
}
$('clearLog').onclick=()=>{ $('pipelineLog').replaceChildren(); $('logEmpty').hidden=false; };
function notifyParent(event) {
  if (config?.dashboard_origin && window.parent !== window) window.parent.postMessage(event,config.dashboard_origin);
}
function controls() {
  $('connect').disabled = !!ws || !config || busy;
  $('talk').disabled = !ready || busy;
  $('talk').textContent = recording ? 'Send' : $('conversationLoop').checked ? 'Start conversation' : 'Talk';
  $('hello').disabled = $('sendText').disabled = !ready || busy || recording;
  $('stop').disabled = !ws && !output;
  notifyParent({type:'reins-voice-ready',ready:ready && !busy && !recording && !loopActive});
}
function updateHint() {
  $('hint').textContent = $('conversationLoop').checked
    ? 'Connect, then Start conversation. Listening resumes after each reply. The microphone stays off while Reins thinks and speaks. Stop ends the loop.'
    : 'Connect, then click Talk. Send manually or let VAD finish after a pause.';
  if(config?.tyto_nudge) $('hint').textContent += ' Tyto checks audio after a 5-second warm-up.';
}
function cancelResume() { clearTimeout(resumeTimer); resumeTimer = null; }
function resumeListening() {
  cancelResume();
  if(!loopActive || !$('conversationLoop').checked || document.hidden) return;
  const generation=epoch, socket=ws;
  $('status').textContent='Conversation active · resuming listening…';
  // Wait for the server's Ready acknowledgement and a brief speaker tail; never capture during playback.
  resumeTimer=setTimeout(()=>{
    resumeTimer=null;
    if(generation!==epoch || ws!==socket || !loopActive || !$('conversationLoop').checked
      || document.hidden || !ready || busy || recording || playback || expectingAudio
      || socket?.readyState!==WebSocket.OPEN) return;
    startCapture();
  },300);
}
$('conversationLoop').onchange=()=>{
  if(!$('conversationLoop').checked) {
    loopActive=false; cancelResume();
    if(ready && !busy && !recording) $('status').textContent='Ready · microphone off';
  }
  else if(recording) loopActive=true;
  updateHint(); controls();
};
async function releaseMic() {
  recording = false; clearTimeout(timer);
  $('micLevel').textContent='Microphone off';
  if (capture) { capture.port.onmessage = null; capture.disconnect(); capture = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (input) { const old = input; input = null; await old.close(); }
  if (flush) { flush(); flush = null; }
}
async function stop() {
  loopActive=false; cancelResume();
  if (ws) pipelineLog({stage:'session',status:'stopped'});
  epoch++; ready = busy = expectingAudio = false;
  const old = ws; ws = null;
  if (playback) { playback.onended = null; playback.stop(); playback.disconnect(); playback = null; }
  if (output) { const oldOutput = output; output = null; await oldOutput.close(); }
  if (old) { if (old.readyState === WebSocket.OPEN) old.send(JSON.stringify({type:'stop'})); old.close(); }
  await releaseMic(); $('status').textContent = 'Microphone off · stopped'; controls();
}
async function fail(error) { $('error').textContent = error.message || String(error); pipelineLog({stage:'session',status:'error',text:$('error').textContent}); await stop(); }
function transcript(role, text) {
  const row = document.createElement('div'); row.className = 'line';
  const label = document.createElement('span'); label.className = 'label'; label.textContent = role;
  row.append(label, document.createTextNode(text)); $('transcript').append(row);
  while ($('transcript').childElementCount > 80) $('transcript').firstElementChild.remove();
  $('transcript').scrollTop = $('transcript').scrollHeight;
}
$('connect').onclick = async () => {
  const generation = ++epoch; busy = true; $('error').textContent = '';
  try {
    // Created inside the user gesture, so subsequent replies can play without autoplay prompts.
    output = new AudioContext(); controls(); await output.resume();
    if (generation !== epoch) return;
    const socket = new WebSocket(`ws://${location.host}/voice`); ws = socket; socket.binaryType = 'arraybuffer';
    $('status').textContent = 'Connecting…'; controls();
    socket.onopen = () => socket.send(JSON.stringify({token:config.token}));
    socket.onmessage = async ({data}) => {
      if (ws !== socket) return;
      try {
        if (data instanceof ArrayBuffer) {
          if (!expectingAudio || !data.byteLength || data.byteLength % 2 || data.byteLength > 30 * 24000 * 2) throw new Error('Invalid reply audio');
          expectingAudio = false;
          const samples = new Int16Array(data), buffer = output.createBuffer(1, samples.length, 24000);
          const channel = buffer.getChannelData(0); for (let i=0;i<samples.length;i++) channel[i]=samples[i]/32768;
          playback = output.createBufferSource(); playback.buffer = buffer; playback.connect(output.destination);
          playback.onended = () => { pipelineLog({stage:'playback',status:'completed'}); playback?.disconnect(); playback = null; if (ws === socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'played'})); };
          pipelineLog({stage:'playback',status:'started',audio_s:buffer.duration});
          $('status').textContent = config.provider === 'demo' ? 'Playing offline test tone' : 'Reins is speaking'; playback.start(); return;
        }
        const event = JSON.parse(data);
        if (event.type === 'log') { pipelineLog(event); return; }
        if (event.type === 'nudge' && recording && !busy) {
          busy=true; controls();
          await releaseMic(); // Discard the capture tail; provisional STT must never reach the LLM.
          if(ws===socket && socket.readyState===WebSocket.OPEN) socket.send(JSON.stringify({type:'end'}));
          $('status').textContent='Audio paused · preparing Tyto clarification';
          return;
        }
        if (event.type === 'ready') {
          ready = true; busy = false; $('status').textContent = 'Ready · microphone off';
          resumeListening();
        }
        if (event.type === 'connecting') $('status').textContent = 'Connecting audio providers…';
        if (event.type === 'recording') {
          $('status').textContent = loopActive ? 'Conversation active · listening' : 'Listening · pause or click Send';
          pipelineLog({stage:'session',status:'listening'});
        }
        if (event.type === 'endpoint' && recording && !busy) $('talk').click();
        if (event.type === 'thinking') {
          busy = true; $('result').hidden = true;
          $('status').textContent = config.provider === 'demo' ? 'Checking local audio…' : 'Processing the turn…';
        }
        if (event.type === 'result') {
          $('result').hidden = false;
          const timing = Object.entries(event.timings || {}).map(([k,v])=>`${k.replace('_s','')} ${v.toFixed(2)}s`).join(' · ');
          const quality = event.acoustics?.tyto === 'scored' ? ` · interference ${event.acoustics.interfering_speech.toFixed(2)} · noise ${event.acoustics.noise.toFixed(2)}` : event.acoustics?.tyto === 'needs_5_seconds' ? ' · Tyto needs ≥5s audio' : '';
          $('metrics').textContent = `${event.llm_model || event.stt_model || event.tts_model || event.blocked || ''} · ${timing}${quality}`;
          $('qualityNote').textContent = event.notice || (event.blocked ? 'Audio held for clarification. No text inserted into the prompt.' : event.llm_model ? `Conversation reply · ${event.llm_model}` : event.stt_model ? 'Transcript ready. Review it in the existing prompt box.' : 'Text-to-speech playback.');
        }
        if (event.type === 'audio') { if (event.sample_rate !== 24000) throw new Error('Unsupported reply format'); expectingAudio = true; }
        if (event.type === 'transcript') {
          transcript(event.role, event.text);
          if (event.role === 'you' && config.provider !== 'demo' && !config.chat_model) notifyParent({type:'reins-voice-transcript',text:event.text});
        }
        if (event.type === 'error') { await fail(event.text); return; }
        controls();
      } catch(error) { await fail(error); }
    };
    socket.onclose = () => { if (ws === socket) stop(); };
    socket.onerror = () => { if (ws === socket) fail('Voice service disconnected. Reconnect to try again.'); };
  } catch(error) { await fail(error); }
};
async function startCapture() {
  if(!ready || busy || recording || playback || expectingAudio || !ws || ws.readyState!==WebSocket.OPEN) return;
  const generation = epoch, socket = ws;
  try {
    busy = true; controls();
    $('liveTranscript').hidden=true; $('partialText').textContent='';
    const acquired = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:!config.voice_focus,autoGainControl:!config.voice_focus},video:false});
    if (generation !== epoch || ws !== socket || socket.readyState !== WebSocket.OPEN) { acquired.getTracks().forEach(t=>t.stop()); return; }
    stream = acquired; const context = new AudioContext({sampleRate:16000}); input = context;
    const micName=stream.getAudioTracks()[0]?.label || 'Default microphone';
    pipelineLog({stage:'session',status:'microphone',text:micName});
    await context.resume(); if (generation !== epoch) return;
    if (context.sampleRate !== 16000) throw new Error('This browser cannot record at 16 kHz. Try Chrome.');
    await context.audioWorklet.addModule('/mic.js'); if (generation !== epoch) return;
    const source = context.createMediaStreamSource(stream), mute = context.createGain(); mute.gain.value = 0;
    capture = new AudioWorkletNode(context, 'microphone-capture');
    socket.send(JSON.stringify({type:'start',sample_rate:16000,auto_end:$('autoEnd').checked})); recording = true; busy = false;
    capture.port.onmessage = ({data}) => {
      if (data === 'flushed') { flush?.(); return; }
      if (!recording || ws !== socket || socket.readyState !== WebSocket.OPEN) return;
      const samples=new Int16Array(data);
      let energy=0; for(const s of samples) energy+=(s/32768)**2;
      const db=20*Math.log10(Math.max(Math.sqrt(energy/Math.max(samples.length,1)),1e-6));
      $('micLevel').textContent=`${micName} · ${db.toFixed(0)} dBFS${db < -55 ? ' · very quiet' : ''}`;
      if (socket.bufferedAmount > 65536) { fail('Audio connection is too slow; recording stopped.'); return; }
      socket.send(data);
    };
    source.connect(capture); capture.connect(mute); mute.connect(context.destination);
    timer = setTimeout(() => { if (recording) $('talk').click(); }, 29000); controls();
  } catch(error) { await fail(error.name === 'NotAllowedError' ? 'Microphone permission was not granted. Allow it in your browser or use typed messages.' : error); }
}
$('talk').onclick = async () => {
  cancelResume();
  if(!ready || busy) return;
  const generation=epoch, socket=ws;
  if(recording) {
    try {
      busy=true; controls();
      await new Promise(resolve=>{ flush=resolve; capture.port.postMessage('flush'); setTimeout(resolve,250); });
      await releaseMic();
      if(generation!==epoch || ws!==socket) return;
      socket.send(JSON.stringify({type:'end'})); controls();
    } catch(error) { await fail(error); }
    return;
  }
  loopActive=$('conversationLoop').checked;
  await startCapture();
};
$('hello').onclick = () => { busy = true; controls(); ws.send(JSON.stringify({type:'hello'})); };
$('textForm').onsubmit = event => { event.preventDefault(); if (!ready || busy || recording) return; busy = true; controls(); const text=$('text').value; ws.send(JSON.stringify({type:config.chat_model ? 'chat' : 'text',text})); $('text').value=''; };
$('stop').onclick = stop;
window.addEventListener('message', event => {
  if (!config || event.source !== window.parent || event.origin !== config.dashboard_origin
      || event.data?.type !== 'reins-voice-speak' || !ready || busy || recording || loopActive) return;
  const text = event.data.text;
  if (typeof text !== 'string' || !text.trim() || text.length > 1000) return;
  busy = true; controls(); ws.send(JSON.stringify({type:'text',text}));
});
window.addEventListener('pagehide',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden) stop();});
document.addEventListener('keydown',event=>{if(event.key==='Escape') stop();});
fetch('/config').then(r=>{if(!r.ok) throw new Error('Voice service unavailable');return r.json();}).then(c=>{
  config=c;
  $('voiceName').textContent = c.provider === 'demo' ? 'Offline test tone' : c.voice;
  if (window.parent === window) $('description').textContent = 'Transcribe speech and test text-to-speech here. Use the dashboard voice panel to dictate into its existing prompt box.';
  $('acoustics').textContent = `STT: ${c.stt_model || 'off'} · TTS: ${c.tts_model || 'offline tone'} · VAD: ${c.vad || 'off'}`;
  $('focusState').textContent=c.voice_focus ? 'On · before STT' : 'Off';
  $('focusLevel').textContent=c.voice_focus && typeof c.enhancement_level==='number' ? `${Math.round(c.enhancement_level*100)}% · SDK setting` : 'Not active';
  $('tytoState').textContent=c.tyto ? 'On · original input' : 'Off';
  $('nudgeState').textContent=c.tyto_nudge ? 'On · 5s warm-up' : 'Off';
  $('conversationLoop').checked=!!c.chat_model;
  updateHint();
  $('autoEnd').closest('label').hidden = c.provider === 'demo';
  if (c.provider === 'demo') { $('description').textContent = 'Offline microphone and speaker transport test.'; $('hint').textContent='Connect, click Talk, then Send when you finish.'; }
  $('privacy').textContent = c.provider === 'demo' ? 'Offline demo: no AI or cloud calls. Audio stays on this computer. The response is a test tone.' : 'On Send or a VAD pause, OpenAI transcribes your audio. Dictation goes into the existing prompt box. The TTS provider receives only the text to read. ai-coustics runs locally with license/usage telemetry. This app saves no audio or transcripts. Stop ends capture and playback.';
  if (c.chat_model) {
    $('heading').textContent = 'Talk to Reins';
    $('description').textContent = `Conversation test with ${c.chat_model}. Speak or type a question and hear the answer.`;
    $('textLabel').textContent = `Message to ${c.chat_model}`;
    $('text').placeholder = 'Hello, Reins. What can we test?';
    $('sendText').textContent = 'Send message';
    $('privacy').textContent = `OpenAI transcribes your audio, then ${c.chat_model} answers using the recent conversation. The TTS provider reads the answer. ai-coustics runs locally with license/usage telemetry. This app saves no audio or transcripts. Stop clears the session and stops playback.`;
  }
  if(c.stt_streaming) {
    $('description').textContent = `${c.chat_model ? `Conversation test with ${c.chat_model}. ` : ''}Streaming STT transcribes while you speak.`;
    $('privacy').textContent = `Enhanced microphone audio streams to OpenAI while you speak. Only the accepted final transcript goes to ${c.chat_model || 'the prompt box'}. Tyto nudges discard provisional text. TTS reads the answer. ai-coustics runs locally with license/usage telemetry. Stop ends capture and playback.`;
  }
  $('privacy').textContent += ' Logs stay in this page until Clear or reload; transcripts stay until reload. They are not written to disk.';
  controls();
}).catch(fail);
controls();
