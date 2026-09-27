'use strict';
const $ = id => document.getElementById(id);
let token = '', plans = [], state = null, camera = 'head', simSource = 'simulation', sharing = null;
let scrubbing = false, toastTimer, statusBusy = false;
let runSeq = 0, runStarted = null, plansLibrary = 0, runBusy = false, voiceReady = false;
const DRYRUN_PLAN = 'sim/plans/arm_lift_dryrun.json';
const GRID = {head:'gridHead', left:'gridLeft', right:'gridRight'};
const frameBusy = new Set(), frameUrls = {};
const toast = text => { $('toast').textContent = text; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5500); };
const badge = (id, text, good) => { $(id).textContent = text; $(id).classList.toggle('good', good); };
const clock = value => Number(value).toFixed(1).padStart(4, '0');
async function control(command) {
  try {
    const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json','X-Reins-Token':token}, body:JSON.stringify(command)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Unable to update preview');
    if (state) state.simulation = body;
    updatePlayback(body);
    return true;
  } catch (error) { toast(error.message); return false; }
}
function updatePlayback(sim) {
  $('play').innerHTML = sim.playing ? '<span style="font-size:14px;font-weight:700">Ⅱ</span>' : '<svg><use href="#i-play"/></svg>';
  $('play').setAttribute('aria-label', sim.playing ? 'Pause trajectory' : 'Play trajectory');
  $('duration').textContent = clock(sim.duration);
  if (!scrubbing) { $('time').textContent = clock(sim.time); $('scrubber').max = sim.duration; $('scrubber').value = sim.time; }
  $('speed').value = sim.speed;
  $('paths').checked = sim.paths;
  const plan = plans.find(p => p.id === sim.plan);
  if (plan) { $('planName').textContent = plan.name; $('jointCount').textContent = plan.joints; }
}
async function frame(key, imgId, visible) {
  const img = $(imgId);
  if (!visible) { img.classList.remove('visible'); return; }
  if (frameBusy.has(imgId)) return;
  frameBusy.add(imgId);
  try {
    const response = await fetch(`/frame/${key}`, {signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('No frame');
    const url = URL.createObjectURL(await response.blob());
    // A source can change while the frame request is in flight.
    if ((imgId === 'robotImage' && key !== camera) || (Object.values(GRID).includes(imgId) && camera !== 'all') || (imgId === 'simImage' && key !== simSource) || (imgId === 'glassesImage' && sharing)) { URL.revokeObjectURL(url); return; }
    const old = frameUrls[imgId];
    img.src = url; frameUrls[imgId] = url; img.classList.add('visible');
    if (old) URL.revokeObjectURL(old);
  } catch { img.classList.remove('visible'); }
  finally { frameBusy.delete(imgId); }
}
function renderStatus() {
  if (state?.voice_url && $('voiceFrame').getAttribute('src') !== state.voice_url) {
    $('voicePanel').hidden = false;
    $('voiceFrame').src = state.voice_url; $('voiceOpen').href = state.voice_url;
    $('speakPrompt').hidden = false;
  }
  if (state?.mode === 'sim') {
    $('controlPanel').hidden = true;
    $('controlJump').hidden = $('heroControl').hidden = true;
    $('simSource').querySelector('[value="twin"]').disabled = true;
    $('promptSource').querySelector('[value="camera"]').disabled = true;
    document.querySelector('.local-pill').textContent = 'SIMULATION ONLY';
  }
  if (!state) return;
  const sim = state.simulation, feeds = state.feeds;
  updatePlayback(sim);
  const simReady = simSource === 'simulation' ? sim.ready : feeds.twin.online;
  badge('simBadge', simReady ? (simSource === 'simulation' ? 'PREVIEW' : 'LIVE') : (simSource === 'simulation' ? 'UNAVAILABLE' : 'OFFLINE'), simReady);
  $('simEmpty').hidden = simReady;
  $('simDetail').textContent = simSource === 'simulation' ? (sim.error || 'Loading the R1 model…') : (feeds.twin.configured ? 'Waiting for the live twin feed.' : 'Add --twin http://127.0.0.1:8082/twin when starting the dashboard.');
  $('simEmpty').querySelector('strong').textContent = simSource === 'simulation' ? 'MuJoCo preview' : 'Live robot twin';
  $('renderLabel').textContent = simSource === 'simulation' ? 'KINEMATIC PREVIEW' : 'LIVE ROBOT STATE';
  const liveCams = Object.keys(GRID).filter(k => feeds[k].online).length;
  const robot = camera === 'all' ? {online: liveCams > 0} : feeds[camera];
  badge('robotBadge', robot.online ? (camera === 'all' ? `${liveCams} / 3 LIVE` : 'LIVE') : 'OFFLINE', robot.online);
  $('robotEmpty').hidden = robot.online;
  $('camGrid').hidden = camera !== 'all' || !robot.online;
  $('robotPanel').classList.toggle('grid-mode', camera === 'all' && robot.online);
  $('robotAge').textContent = robot.online ? 'LIVE FEED' : 'NO SIGNAL';
  const glassLive = !!sharing || feeds.glasses.online;
  badge('glassesBadge', sharing ? 'WINDOW SHARE' : glassLive ? 'LIVE' : 'NOT CONNECTED', glassLive);
  $('glassesEmpty').hidden = glassLive;
  $('glassesMode').textContent = sharing ? 'SHARED WINDOW' : glassLive ? 'VIDEO STREAM' : 'AWAITING SOURCE';
  $('connectionCount').textContent = `${Number(robot.online) + Number(glassLive)} / 2`;
  $('glassesAction').textContent = sharing ? 'Stop sharing ×' : 'Connect source ↗';
  // The transport always previews a plan; make that distinction explicit in live mode.
  $('play').disabled = simSource !== 'simulation';
  $('reset').disabled = simSource !== 'simulation';
  $('scrubber').disabled = simSource !== 'simulation';
  $('speed').disabled = simSource !== 'simulation';
  if (!simReady) $('simImage').classList.remove('visible');
  if (!robot.online || camera === 'all') $('robotImage').classList.remove('visible');
  for (const [k, id] of Object.entries(GRID)) if (!feeds[k].online) $(id).classList.remove('visible');
  renderControl();
  renderPrompt();
  renderOverview();
  if (!feeds.glasses.online || sharing) $('glassesImage').classList.remove('visible');
}

// Text-only bridge: dictation edits the existing field; the user submits it normally.
// The harness can read any reply through window.ReinsVoice.speak(text).
window.ReinsVoice = {
  speak(text) {
    if (!voiceReady || !state?.voice_url || typeof text !== 'string' || !text.trim() || text.length > 1000) return false;
    $('voiceFrame').contentWindow.postMessage({type:'reins-voice-speak',text},new URL(state.voice_url).origin);
    voiceReady = false; $('speakPrompt').disabled = true;
    return true;
  }
};
window.addEventListener('message', event => {
  if (!state?.voice_url || event.source !== $('voiceFrame').contentWindow
      || event.origin !== new URL(state.voice_url).origin) return;
  if (event.data?.type === 'reins-voice-ready') {
    voiceReady = event.data.ready === true;
    $('speakPrompt').disabled = !voiceReady;
  }
  if (event.data?.type === 'reins-voice-transcript') {
    const text = event.data.text, field = $('actionPrompt');
    if (typeof text !== 'string' || !text.trim() || text.length > 1000) return;
    const size = field.value.length - (field.selectionEnd - field.selectionStart) + text.length;
    if (size > field.maxLength) { toast('Dictation is too long for the prompt. Copy it from the voice transcript.'); return; }
    field.setRangeText(text, field.selectionStart, field.selectionEnd, 'end');
    field.dispatchEvent(new Event('input', {bubbles:true}));
    toast('Dictation added to the prompt. Review it, then Generate plan.');
  }
});
$('speakPrompt').onclick = () => {
  if (!window.ReinsVoice.speak($('promptMessage').textContent)) toast('Connect voice and wait for playback to finish. Replies must be under 1,000 characters.');
};
async function poll() {
  if (statusBusy) return;
  statusBusy = true;
  try {
    const response = await fetch(`/api/status?since=${runSeq}`, {signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('Server offline');
    state = await response.json(); consumeRun(state.run);
    if (state.simulation.library !== plansLibrary) loadPlans();
    renderStatus();
  } catch {
    badge('simBadge','SERVER OFFLINE',false); badge('robotBadge','OFFLINE',false);
    $('simImage').classList.remove('visible'); $('robotImage').classList.remove('visible'); $('glassesImage').classList.remove('visible');
    $('simEmpty').hidden = false; $('simDetail').textContent = 'Dashboard disconnected. Waiting for the local server…';
    $('robotEmpty').hidden = false;
    if (!sharing) { $('glassesEmpty').hidden = false; badge('glassesBadge','OFFLINE',false); }
    state = null;
  } finally { statusBusy = false; }
}
function refreshFrames() {
  if (!state || document.hidden) return;
  frame(simSource,'simImage',simSource === 'simulation' ? state.simulation.ready : state.feeds.twin.online);
  if (camera === 'all') for (const [k, id] of Object.entries(GRID)) frame(k, id, state.feeds[k].online);
  else frame(camera,'robotImage',state.feeds[camera].online);
  if (!sharing) frame('glasses','glassesImage',state.feeds.glasses.online);
}
function showLibrary() { drawPlans(); $('library').showModal(); $('planSearch').focus(); }
function drawPlans() {
  const query = $('planSearch').value.toLowerCase(); $('planList').replaceChildren();
  const filtered = plans.filter(p => `${p.name} ${p.id}`.toLowerCase().includes(query));
  for (const plan of filtered) {
    const row = document.createElement('button'); row.className = 'plan-row' + (state?.simulation.plan === plan.id ? ' current' : '');
    const icon = document.createElementNS('http://www.w3.org/2000/svg','svg'); icon.innerHTML = '<use href="#i-list"/>';
    const info = document.createElement('div'), title = document.createElement('strong'), detail = document.createElement('small'), duration = document.createElement('span');
    title.textContent = plan.name; detail.textContent = `${plan.kind} · ${plan.joints} joints`; duration.textContent = `${plan.duration.toFixed(1)} s`;
    info.append(title,detail); row.append(icon,info,duration);
    row.onclick = async () => { if (await control({action:'plan',id:plan.id})) { $('library').close(); toast('Trajectory loaded into simulation'); } };
    $('planList').append(row);
  }
  if (!filtered.length) { const p = document.createElement('p'); p.textContent = 'No matching trajectories.'; $('planList').append(p); }
}
function stopShare() {
  const stream = sharing; sharing = null;
  stream?.getTracks().forEach(track => track.stop());
  $('glassesVideo').srcObject = null; $('glassesVideo').hidden = true; renderStatus();
}
async function share() {
  if (!navigator.mediaDevices?.getDisplayMedia) { toast('Window sharing requires a desktop browser on localhost. You can also supply an MJPEG URL.'); return; }
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({video:{frameRate:20},audio:false});
    if (sharing) stopShare(); sharing = stream;
    $('glassesVideo').srcObject = stream; $('glassesVideo').hidden = false;
    stream.getVideoTracks()[0].addEventListener('ended',stopShare,{once:true});
    if ($('connections').open) $('connections').close(); renderStatus();
  } catch (error) { toast(error.name === 'NotAllowedError' ? 'Window sharing cancelled or not permitted. You can try again.' : `Could not share the window: ${error.message}`); }
}
$('play').onclick = () => state && control({action:state.simulation.playing ? 'pause' : 'play'});
$('reset').onclick = async () => { await control({action:'pause'}); await control({action:'seek',time:0}); };
$('scrubber').addEventListener('input', () => { scrubbing = true; $('time').textContent = clock($('scrubber').value); });
$('scrubber').addEventListener('change', async () => { await control({action:'seek',time:Number($('scrubber').value)}); scrubbing = false; });
$('speed').onchange = () => control({action:'speed',value:Number($('speed').value)});
$('paths').onchange = () => control({action:'paths',value:$('paths').checked});
$('simSource').onchange = () => { simSource = $('simSource').value; if (simSource === 'twin') control({action:'pause'}); $('simImage').classList.remove('visible'); document.querySelectorAll('[data-view]').forEach(b => b.disabled = simSource !== 'simulation'); $('paths').disabled = simSource !== 'simulation'; renderStatus(); };
document.querySelectorAll('[data-view]').forEach(button => button.onclick = async () => { if (await control({action:'view',value:button.dataset.view})) { document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('selected',b === button)); } });
document.querySelectorAll('[data-camera]').forEach(button => button.onclick = () => { camera = button.dataset.camera; $('robotImage').classList.remove('visible'); $('cameraLabel').textContent = camera === 'all' ? 'ALL CAMERAS' : `${button.textContent.toUpperCase()} CAMERA`; document.querySelectorAll('[data-camera]').forEach(b => b.classList.toggle('selected',b === button)); renderStatus(); });
document.querySelectorAll('.expand').forEach(button => button.onclick = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await button.closest('.panel').requestFullscreen(); } catch { toast('Fullscreen is unavailable in this browser.'); } });
['settings','connectButton'].forEach(id => $(id).onclick = () => $('connections').showModal());
document.querySelectorAll('.open-connections').forEach(b => b.onclick = () => $('connections').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => $(b.dataset.close).close());
['planPicker','libraryToggle'].forEach(id => $(id).onclick = showLibrary);
$('planSearch').oninput = drawPlans;
$('shareGlasses').onclick = share; $('shareFromSettings').onclick = share;
$('glassesAction').onclick = () => sharing ? stopShare() : $('connections').showModal();
document.addEventListener('keydown', event => { if (event.code === 'Space' && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) && !document.querySelector('dialog[open]')) { event.preventDefault(); $('play').click(); } });
window.addEventListener('beforeunload', () => sharing?.getTracks().forEach(t => t.stop()));
async function init() {
  try { await loadPlans(); await poll(); }
  catch (e) { toast(e.message + '. Refresh after starting the server.'); }
  setInterval(poll,300); setInterval(refreshFrames,80);
  setInterval(() => $('clock').textContent = new Date().toLocaleTimeString([], {hour12:false}) + ' · LOCAL',1000);
}
async function loadPlans() {
  const response = await fetch('/api/plans'); if (!response.ok) throw new Error('Cannot load plans');
  const data = await response.json(); token = data.token; plans = data.plans; plansLibrary = data.library;
  if ($('library').open) drawPlans();
  $('planCount').textContent = plans.length;
}

/* ---- Trajectory control: dry run, execute, abort (tools/arm_lift.py) ---- */
async function runCommand(command) {
  if (runBusy) return false;
  runBusy = true;
  try {
    const response = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json','X-Reins-Token':token}, body:JSON.stringify(command)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Run request failed');
    await poll();
    return true;
  } catch (error) { toast(error.message); return false; }
  finally { runBusy = false; }
}
function lineClass(line) {
  if (line.startsWith('$ ')) return 'cmd';
  if (/^\[abort|^ABORT|interrupted/.test(line)) return 'abort';
  if (/^\[exit 0\]/.test(line)) return 'exit-ok';
  if (/^\[exit/.test(line)) return 'exit-bad';
  if (/^NOTE|^note:|DRY RUN/.test(line)) return 'note';
  return '';
}
function consumeRun(run) {
  if (!run) return;
  const started = run.job?.started ?? null;
  const pre = $('console');
  if (started !== runStarted) { runStarted = started; pre.replaceChildren(); }
  if (run.lines.length) {
    const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    pre.querySelector('.console-empty')?.remove();
    for (const line of run.lines) {
      const span = document.createElement('span'); span.textContent = line + '\n';
      const cls = lineClass(line); if (cls) span.className = cls;
      pre.append(span);
    }
    while (pre.childElementCount > 4000) pre.firstElementChild.remove();
    if (stick) pre.scrollTop = pre.scrollHeight;
  }
  runSeq = run.seq;
}
function planLabel(id) { const p = plans.find(x => x.id === id); return p ? p.name : (id || '—'); }
function runTarget() {
  // The dry-run result stands for the plan it was resolved from.
  const current = state?.simulation.plan;
  if (current === DRYRUN_PLAN) return state?.run.source || null;
  return current;
}
function chip(id, text, cls) { const el = $(id); el.querySelector('span').textContent = text; el.classList.remove('good','bad'); if (cls) el.classList.add(cls); el.title = text; }
function renderControl() {
  if (!state) return;
  const run = state.run, target = runTarget(), cleared = run.cleared;
  const speed = Number($('runSpeed').value), kp = Number($('runKp').value);
  $('runTarget').textContent = target ? planLabel(target) : 'Choose a trajectory';
  $('runTarget').classList.toggle('plan', !!target);
  const kind = plans.find(p => p.id === target)?.kind;
  $('runTargetKind').textContent = state.simulation.plan === DRYRUN_PLAN ? 'Previewing its dry-run result' : (kind ? `${kind} · ${target}` : '');
  $('ifaceChip').textContent = `IFACE ${run.iface.toUpperCase()}`;
  const robotText = state.robot || '';
  chip('robotLink', robotText ? `Robot · ${robotText}` : 'Twin server not reachable', /no rt\/lowstate|^$/.test(robotText) ? 'bad' : 'good');
  chip('fsmChip', run.fsm ? `FSM ${run.fsm.id} · ${run.fsm.name}` : 'FSM · dry run to read', run.fsm ? (run.fsm.ok ? 'good' : 'bad') : '');
  const matches = cleared && cleared.plan === target && Math.abs(cleared.speed - speed) < 1e-9 && Math.abs(cleared.kp_scale - kp) < 1e-9;
  const running = run.running, kindRunning = running ? run.job.kind : null;
  // gate: dry run
  const dry = $('gateDry'); dry.className = '';
  let dryText = 'Reads the robot, checks limits and speed. Publishes nothing.';
  if (kindRunning === 'dry') { dry.className = 'active'; dryText = 'Running…'; }
  else if (matches) { dry.className = 'done'; const m = Math.floor(cleared.expires_in / 60), sec = String(cleared.expires_in % 60).padStart(2, '0'); dryText = `Passed · valid for ${m}:${sec}`; }
  else if (cleared) { dryText = 'Settings or target changed since the dry run. Run it again.'; }
  else if (run.job?.kind === 'dry' && run.exit !== null && run.exit !== 0) { dry.className = 'failed'; dryText = `Failed (exit ${run.exit}). See the output.`; }
  $('gateDryText').textContent = dryText;
  dry.querySelector('.gate-dot').innerHTML = dry.className === 'done' ? '<svg><use href="#i-check"/></svg>' : '2';
  // gate: execute
  const ex = $('gateExec'); ex.className = '';
  let exText = 'Unlocks after a successful dry run';
  if (kindRunning === 'execute') { ex.className = 'active'; exText = 'Robot moving. Abort stands ready.'; }
  else if (run.job?.kind === 'execute' && run.exit !== null) { ex.className = run.exit === 0 ? 'done' : 'failed'; exText = run.exit === 0 ? 'Completed and recorded. Dry run again to repeat.' : `Stopped (exit ${run.exit}). See the output.`; }
  else if (matches) { ex.className = 'ready'; exText = run.fsm && !run.fsm.ok ? `Controller in FSM ${run.fsm.id}; the tool will refuse` : 'Ready. Asks for confirmation.'; }
  $('gateExecText').textContent = exText;
  ex.querySelector('.gate-dot').innerHTML = `<svg><use href="#i-${ex.className === 'done' ? 'check' : ex.className === 'ready' || ex.className === 'active' ? 'play' : 'lock'}"/></svg>`;
  const previewOnly = !!plans.find(p => p.id === target)?.preview_only;
  $('dryRun').disabled = state.mode === 'sim' || running || !target || previewOnly;
  $('execute').disabled = running || !matches || previewOnly;
  if (previewOnly) { $('gateDryText').textContent = 'Prompt-generated preview. Physical execution is locked.'; $('gateExecText').textContent = 'Requires verified calibration, contact control and execution validation.'; }
  $('abort').disabled = !running;
  $('runSpeed').disabled = $('runKp').disabled = running;
  const b = $('runBadge');
  b.classList.remove('good','warn','bad');
  if (running) { b.textContent = kindRunning === 'execute' ? 'EXECUTING' : 'DRY RUN'; b.classList.add(kindRunning === 'execute' ? 'warn' : 'good'); }
  else if (run.exit !== null) { b.textContent = `EXIT ${run.exit}`; b.classList.add(run.exit === 0 ? 'good' : 'bad'); }
  else b.textContent = 'IDLE';
}
function validInputs() {
  const speed = Number($('runSpeed').value), kp = Number($('runKp').value);
  if (!(speed >= 0.1 && speed <= 2)) { toast('Speed must be between 0.1 and 2.'); return null; }
  if (!(kp >= 0.5 && kp <= 2)) { toast('kp scale must be between 0.5 and 2.'); return null; }
  return {speed, kp};
}
$('dryRun').onclick = () => { const v = validInputs(), target = runTarget(); if (v && target) runCommand({action:'dry', plan:target, speed:v.speed, kp_scale:v.kp}); };
$('execute').onclick = () => {
  const run = state?.run, c = run?.cleared; if (!c) return;
  $('execPlan').textContent = planLabel(c.plan); $('execSpeed').textContent = `${c.speed}×`; $('execKp').textContent = `${c.kp_scale}×`;
  $('execFsm').textContent = run.fsm ? `${run.fsm.id} · ${run.fsm.name}` : 'unknown'; $('execIface').textContent = run.iface;
  $('executeDialog').showModal(); $('executeDialog').querySelector('.subtle').focus();
};
$('confirmExecute').onclick = async () => { $('executeDialog').close(); await runCommand({action:'execute', confirm:true}); };
$('abort').onclick = () => { fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json','X-Reins-Token':token}, body:JSON.stringify({action:'abort'})}).then(poll).catch(() => toast('Abort request failed. Use the remote.')); };
['runSpeed','runKp'].forEach(id => $(id).addEventListener('input', renderControl));
$('changeTarget').onclick = showLibrary;
document.addEventListener('keydown', event => { if (event.key === 'Escape' && state?.run.running && !document.querySelector('dialog[open]')) { event.preventDefault(); $('abort').click(); toast('Abort sent'); } });
// Closing or reloading the tab aborts a run, as closing the Tk window does. sendBeacon cannot carry headers.
window.addEventListener('pagehide', () => { if (state?.run.running) navigator.sendBeacon('/api/abort-beacon', new Blob([JSON.stringify({token})], {type:'text/plain'})); });
/* ---- Overview: hero, status cards, sidebar ---- */
function delta(id, text, cls) { const el = $(id); el.textContent = text; el.classList.remove('good','bad'); if (cls) el.classList.add(cls); }
function renderOverview() {
  if (!state) return;
  const sim = state.simulation, feeds = state.feeds, run = state.run;
  const plan = plans.find(p => p.id === sim.plan);
  $('heroPlan').textContent = plan ? plan.name : sim.plan;
  $('heroPlan').title = sim.plan;
  $('heroDuration').textContent = `${Number(sim.duration).toFixed(1)} s`;
  $('heroKind').textContent = plan ? `${plan.kind} · ${plan.id}` : '';
  $('heroJoints').textContent = plan ? `${plan.joints} planned joints` : '';
  // robot link, from the twin server's rt/lowstate status line
  const robotText = state.robot || '';
  const live = /msgs/.test(robotText) && !/no rt\/lowstate/.test(robotText);
  const age = robotText.match(/last ([\d.]+)s ago/);
  const fresh = live && age && Number(age[1]) < 1;
  $('statRobot').textContent = live ? (fresh ? 'Live' : 'Stale') : robotText ? 'No data' : 'Offline';
  delta('statRobotDelta', live ? (age ? `${age[1]} s ago` : 'rt/lowstate') : 'port 8082', live ? (fresh ? 'good' : 'bad') : '');
  $('statRobotSub').textContent = robotText || 'twin server (tools/cockpit.py) not reachable';
  // controller FSM from the last dry run
  $('statFsm').textContent = run.fsm ? `FSM ${run.fsm.id}` : '—';
  delta('statFsmDelta', run.fsm ? (run.fsm.ok ? 'arm topic OK' : 'arm topic off') : 'unknown', run.fsm ? (run.fsm.ok ? 'good' : 'bad') : '');
  $('statFsmSub').textContent = run.fsm ? run.fsm.name : 'dry run to read the FSM';
  // cameras
  const cams = ['head', 'left', 'right'].filter(k => feeds[k].online).length;
  $('statCams').textContent = `${cams}/3`;
  delta('statCamsDelta', cams === 3 ? 'all live' : cams ? 'partial' : 'offline', cams === 3 ? 'good' : cams ? '' : 'bad');
  $('camCount').textContent = `${cams}/3`;
  // glasses
  const glasses = !!sharing || feeds.glasses.online;
  $('statGlasses').textContent = sharing ? 'Window' : feeds.glasses.online ? 'Stream' : '—';
  delta('statGlassesDelta', glasses ? 'live' : 'no source', glasses ? 'good' : '');
  $('statGlassesSub').textContent = sharing ? 'mirrored glasses window' : feeds.glasses.online ? 'MJPEG video stream' : 'share a window or stream';
  $('railStatus').textContent = run.running ? (run.job.kind === 'execute' ? 'Executing a trajectory' : 'Dry run in progress') : live ? robotText : 'Robot state unknown';
  if (state.mode === 'sim') {
    $('railStatus').textContent = 'Simulation only';
    $('statRobot').textContent = 'Simulated'; delta('statRobotDelta','local','good');
    $('statRobotSub').textContent = 'Hardware connection disabled';
    $('statFsmSub').textContent = 'Hardware controls disabled';
  }
}
function navTo(buttonId, target) {
  document.querySelectorAll('.rail-button').forEach(b => b.classList.toggle('active', b.id === buttonId));
  (target === 'top' ? document.body : $(target)).scrollIntoView({behavior:'smooth', block:'start'});
}
$('navObservatory').onclick = () => navTo('navObservatory', 'top');
$('navCameras').onclick = () => navTo('navCameras', 'robotPanel');
$('navGlasses').onclick = () => navTo('navGlasses', 'glassesPanel');
$('controlJump').onclick = () => navTo('controlJump', 'controlPanel');
$('heroControl').onclick = () => navTo('controlJump', 'controlPanel');
['heroLibrary', 'searchButton'].forEach(id => $(id).onclick = showLibrary);
$('headerConnections').onclick = () => $('connections').showModal();
if (/Mac|iPhone|iPad/.test(navigator.platform)) $('searchKey').textContent = '⌘';
document.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k' && !document.querySelector('dialog[open]')) { event.preventDefault(); showLibrary(); } });
init();

// Prompt orchestration is asynchronous; the server owns status and proposal IDs.
let promptVersion = '', promptRequestBusy = false, lastPromptId = null;
function renderPrompt() {
  const p = state?.prompt;
  if (!p) return;
  badge('promptBadge', p.state.toUpperCase(), ['proposed','previewed'].includes(p.state));
  $('sendPrompt').disabled = p.state === 'planning' || promptRequestBusy || !!state.run?.running;
  $('promptConfig').textContent = p.configured.observation && p.configured.vision ? `Vision: ${p.model}` : 'Gestures need no vision · object tasks need calibrated context';
  const version = JSON.stringify([p.id,p.state,p.message,p.events?.length]);
  if (version === promptVersion) return;
  promptVersion = version;
  if (p.id && p.id !== lastPromptId) { lastPromptId = p.id; $('promptSource').value = p.source; }
  $('promptSummary').textContent = p.prompt ? `${p.source === 'demo' ? 'Simulation demo · ' : ''}“${p.prompt}”` : 'Plan from a physical intention.';
  $('promptMessage').textContent = p.message;
  $('promptEvents').replaceChildren();
  for (const event of p.events || []) { const li = document.createElement('li'); const tag = document.createElement('span'); tag.textContent = event.stage; const text = document.createElement('span'); text.textContent = event.message; li.append(tag,text); $('promptEvents').append(li); }
  $('cancelPrompt').hidden = !['planning','proposed','previewed'].includes(p.state);
  const ready = ['proposed','previewed'].includes(p.state);
  $('promptReview').hidden = !ready;
  $('promptMetrics').replaceChildren();
  if (ready) {
    const metrics = [['Skill',p.context?.skill || 'approach'],['Vision',p.context?.vision || 'demo'],['Target',p.target.label],['Arm',p.target.arm],['Surface · robot frame',p.target.surface_m ? p.target.surface_m.map(x=>x.toFixed(3)).join(', ')+' m' : 'Not required'],['Stand-off',p.target.standoff_m == null ? 'Not applicable' : (p.target.standoff_m*100).toFixed(0)+' cm'],['Validation',p.validation.samples+' sampled poses'],['Execution','Preview only · no contact']];
    for (const [label,value] of metrics) { const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent=label; dd.textContent=value; row.append(dt,dd); $('promptMetrics').append(row); }
    $('groundingImage').hidden = !p.has_image;
    if (p.has_image) $('groundingImage').src = '/frame/grounding?id='+encodeURIComponent(p.id);
    $('previewPrompt').disabled = !!state.run?.running;
  }
}
async function promptCommand(command) {
  promptRequestBusy = true; $('sendPrompt').disabled = true;
  try {
    const response = await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify(command)});
    const body = await response.json(); if (!response.ok) throw new Error(body.error || 'Prompt request failed');
    if (state) state.prompt=body;
    renderPrompt(); return true;
  } catch(error) { toast(error.message); return false; }
  finally { promptRequestBusy=false; if (state) renderPrompt(); }
}
$('promptForm').onsubmit = async event => { event.preventDefault(); await promptCommand({action:'submit',prompt:$('actionPrompt').value,source:$('promptSource').value}); };
$('cancelPrompt').onclick = () => promptCommand({action:'cancel'});
$('previewPrompt').onclick = async () => {
  if (!state?.prompt?.id) return;
  if (await promptCommand({action:'preview',id:state.prompt.id})) {
    await loadPlans(); await poll(); simSource='simulation'; $('simSource').value='simulation'; $('simSource').dispatchEvent(new Event('change'));
    $('simulationPanel').scrollIntoView({behavior:'smooth',block:'center'});
  }
};
$('actionPrompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();$('promptForm').requestSubmit();}});
