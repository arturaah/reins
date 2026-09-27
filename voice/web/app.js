'use strict';
const $ = id => document.getElementById(id);
let config, ws, input, stream, capture, output, playback, timer, flush;
let ready = false, busy = false, recording = false, epoch = 0, expectingAudio = false;
function notifyParent(event) {
  if (config?.dashboard_origin && window.parent !== window) window.parent.postMessage(event,config.dashboard_origin);
}
function controls() {
  $('connect').disabled = !!ws || !config || busy;
  $('talk').disabled = !ready || busy;
  $('talk').textContent = recording ? 'Send' : 'Talk';
  $('hello').disabled = $('sendText').disabled = !ready || busy || recording;
  $('stop').disabled = !ws && !output;
  notifyParent({type:'reins-voice-ready',ready:ready && !busy && !recording});
}
async function releaseMic() {
  recording = false; clearTimeout(timer);
  if (capture) { capture.port.onmessage = null; capture.disconnect(); capture = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (input) { const old = input; input = null; await old.close(); }
  if (flush) { flush(); flush = null; }
}
async function stop() {
  epoch++; ready = busy = expectingAudio = false;
  const old = ws; ws = null;
  if (playback) { playback.onended = null; playback.stop(); playback.disconnect(); playback = null; }
  if (output) { const oldOutput = output; output = null; await oldOutput.close(); }
  if (old) { if (old.readyState === WebSocket.OPEN) old.send(JSON.stringify({type:'stop'})); old.close(); }
  await releaseMic(); $('status').textContent = 'Microphone off · stopped'; controls();
}
async function fail(error) { $('error').textContent = error.message || String(error); await stop(); }
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
          playback.onended = () => { playback?.disconnect(); playback = null; if (ws === socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'played'})); };
          $('status').textContent = config.provider === 'demo' ? 'Playing offline test tone' : 'Reins is speaking'; playback.start(); return;
        }
        const event = JSON.parse(data);
        if (event.type === 'ready') { ready = true; busy = false; $('status').textContent = 'Ready · microphone off'; }
        if (event.type === 'connecting') $('status').textContent = 'Connecting audio providers…';
        if (event.type === 'recording') $('status').textContent = 'Listening · pause or click Send';
        if (event.type === 'endpoint' && recording && !busy) $('talk').click();
        if (event.type === 'thinking') {
          busy = true; $('result').hidden = true;
          $('status').textContent = config.provider === 'demo' ? 'Checking local audio…' : 'Processing the turn…';
        }
        if (event.type === 'result') {
          $('result').hidden = false;
          const timing = Object.entries(event.timings || {}).map(([k,v])=>`${k.replace('_s','')} ${v.toFixed(2)}s`).join(' · ');
          const quality = event.acoustics?.tyto === 'scored' ? ` · interference ${event.acoustics.interfering_speech.toFixed(2)} · noise ${event.acoustics.noise.toFixed(2)}` : event.acoustics?.tyto === 'needs_5_seconds' ? ' · Tyto needs ≥5s audio' : '';
          $('metrics').textContent = `${event.stt_model || event.tts_model || event.blocked || ''} · ${timing}${quality}`;
          $('qualityNote').textContent = event.blocked ? 'Audio held for clarification. No text inserted into the prompt.' : event.stt_model ? 'Transcript ready. Review it in the existing prompt box.' : 'Text-to-speech playback.';
        }
        if (event.type === 'audio') { if (event.sample_rate !== 24000) throw new Error('Unsupported reply format'); expectingAudio = true; }
        if (event.type === 'transcript') {
          transcript(event.role, event.text);
          if (event.role === 'you' && config.provider !== 'demo') notifyParent({type:'reins-voice-transcript',text:event.text});
        }
        if (event.type === 'error') { await fail(event.text); return; }
        controls();
      } catch(error) { await fail(error); }
    };
    socket.onclose = () => { if (ws === socket) stop(); };
    socket.onerror = () => { if (ws === socket) fail('Voice service disconnected. Reconnect to try again.'); };
  } catch(error) { await fail(error); }
};
$('talk').onclick = async () => {
  const generation = epoch, socket = ws;
  try {
    if (recording) {
      busy = true; controls();
      await new Promise(resolve => { flush = resolve; capture.port.postMessage('flush'); setTimeout(resolve, 250); });
      await releaseMic();
      if (generation !== epoch || ws !== socket) return;
      socket.send(JSON.stringify({type:'end'})); controls(); return;
    }
    busy = true; controls();
    const acquired = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:!config.voice_focus,autoGainControl:!config.voice_focus},video:false});
    if (generation !== epoch || ws !== socket || socket.readyState !== WebSocket.OPEN) { acquired.getTracks().forEach(t=>t.stop()); return; }
    stream = acquired; const context = new AudioContext({sampleRate:16000}); input = context;
    await context.resume(); if (generation !== epoch) return;
    if (context.sampleRate !== 16000) throw new Error('This browser cannot record at 16 kHz. Try Chrome.');
    await context.audioWorklet.addModule('/mic.js'); if (generation !== epoch) return;
    const source = context.createMediaStreamSource(stream), mute = context.createGain(); mute.gain.value = 0;
    capture = new AudioWorkletNode(context, 'microphone-capture');
    socket.send(JSON.stringify({type:'start',sample_rate:16000,auto_end:$('autoEnd').checked})); recording = true; busy = false;
    capture.port.onmessage = ({data}) => {
      if (data === 'flushed') { flush?.(); return; }
      if (!recording || ws !== socket || socket.readyState !== WebSocket.OPEN) return;
      if (socket.bufferedAmount > 65536) { fail('Audio connection is too slow; recording stopped.'); return; }
      socket.send(data);
    };
    source.connect(capture); capture.connect(mute); mute.connect(context.destination);
    timer = setTimeout(() => { if (recording) $('talk').click(); }, 29000); controls();
  } catch(error) { await fail(error.name === 'NotAllowedError' ? 'Microphone permission was not granted. Allow it in your browser or use typed messages.' : error); }
};
$('hello').onclick = () => { busy = true; controls(); ws.send(JSON.stringify({type:'hello'})); };
$('textForm').onsubmit = event => { event.preventDefault(); if (!ready || busy || recording) return; busy = true; controls(); const text=$('text').value; ws.send(JSON.stringify({type:'text',text})); $('text').value=''; };
$('stop').onclick = stop;
window.addEventListener('message', event => {
  if (!config || event.source !== window.parent || event.origin !== config.dashboard_origin
      || event.data?.type !== 'reins-voice-speak' || !ready || busy || recording) return;
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
  $('acoustics').textContent = `Voice Focus ${c.voice_focus ? 'on' : 'off'} · VAD ${c.vad || 'off'} · Tyto 1.1 ${c.tyto ? 'on (experimental)' : 'off'}`;
  $('autoEnd').closest('label').hidden = c.provider === 'demo';
  if (c.provider === 'demo') { $('description').textContent = 'Offline microphone and speaker transport test.'; $('hint').textContent='Connect, click Talk, then Send when you finish.'; }
  $('privacy').textContent = c.provider === 'demo' ? 'Offline demo: no AI or cloud calls. Audio stays on this computer. The response is a test tone.' : 'On Send or a VAD pause, OpenAI transcribes your audio. Dictation goes into the existing prompt box. The TTS provider receives only the text to read. ai-coustics runs locally with license/usage telemetry. This app saves no audio or transcripts. Stop ends capture and playback.';
  controls();
}).catch(fail);
controls();
