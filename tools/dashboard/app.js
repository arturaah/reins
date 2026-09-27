'use strict';
const $ = id => document.getElementById(id);
let token = '', state = null, sharing = null;
let toastTimer, statusBusy = false, gestureRequestBusy = false;
const SOURCES = {simulation:'MuJoCo simulation', head:'Head camera', left:'Left wrist camera', right:'Right wrist camera', glasses:'Glasses feed', twin:'Live robot twin'};
const viewers = {main:'simulation', secondary:'head'}, viewerRevision = {main:0, secondary:0};
try { const saved=JSON.parse(localStorage.getItem('reins.viewers') || '{}'); for (const slot of Object.keys(viewers)) if (Object.hasOwn(SOURCES,saved[slot])) viewers[slot]=saved[slot]; } catch {}
for (const slot of Object.keys(viewers)) $(slot+'Source').value=viewers[slot];
const frameBusy = new Set(), frameUrls = {};
const toast = text => { $('toast').textContent = text; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5500); };
const badge = (id, text, good) => { $(id).textContent = text; $(id).classList.toggle('good', good); };
async function control(command) {
  try {
    const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json','X-Reins-Token':token}, body:JSON.stringify(command)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Unable to update preview');
    if (state) state.simulation = body;
    renderViewers();
    return true;
  } catch (error) { toast(error.message); return false; }
}
function viewerReady(key) {
  if (key === 'glasses' && sharing) return true;
  return !!state && (key === 'simulation' ? state.simulation.ready : state.feeds[key]?.online);
}
function setViewer(slot, source) {
  if (!Object.hasOwn(viewers,slot) || !Object.hasOwn(SOURCES,source)) return;
  viewers[slot]=source;viewerRevision[slot]++;
  $(slot+'Source').value=source;
  const img=$(slot+'Image');img.classList.remove('visible');img.removeAttribute('src');
  if (frameUrls[slot]) { URL.revokeObjectURL(frameUrls[slot]);delete frameUrls[slot]; }
  try { localStorage.setItem('reins.viewers',JSON.stringify(viewers)); } catch {}
  renderViewers();refreshFrames();
}
async function viewerFrame(slot) {
  const key=viewers[slot], revision=viewerRevision[slot], img=$(slot+'Image');
  if (!viewerReady(key) || (key==='glasses' && sharing)) { img.classList.remove('visible');return; }
  if (frameBusy.has(slot)) return;
  frameBusy.add(slot);
  try {
    const response=await fetch('/frame/'+key,{signal:AbortSignal.timeout(3000)});
    if(!response.ok)throw new Error('No frame');
    const blob=await response.blob();
    if(revision!==viewerRevision[slot] || !viewerReady(key) || (key==='glasses' && sharing))return;
    const url=URL.createObjectURL(blob),old=frameUrls[slot];
    img.src=url;img.classList.add('visible');frameUrls[slot]=url;
    if(old)URL.revokeObjectURL(old);
  } catch { if(revision===viewerRevision[slot])img.classList.remove('visible'); }
  finally {frameBusy.delete(slot);}
}
function renderViewers() {
  for(const [slot,key] of Object.entries(viewers)) {
    const ready=viewerReady(key),windowShare=key==='glasses' && !!sharing;
    const video=$(slot+'Video');
    if(video.srcObject!==(windowShare?sharing:null))video.srcObject=windowShare?sharing:null;
    video.hidden=!windowShare;
    if(!ready || windowShare)$(slot+'Image').classList.remove('visible');
    $(slot+'Image').alt=SOURCES[key];
    badge(slot+'Badge', ready ? (key==='simulation'?'PREVIEW':windowShare?'SHARED':'LIVE') : state?'OFFLINE':'CONNECTING',ready);
    $(slot+'Empty').hidden=ready;
    $(slot+'EmptyTitle').textContent=key==='simulation'?'Preparing simulation':'Waiting for '+SOURCES[key].toLowerCase();
    $(slot+'Detail').textContent=!state?'Waiting for the dashboard…':key==='simulation'?(state.simulation.error||'Loading the R1 model…'):key==='glasses'?'Connect a glasses stream or share a window.':'The selected feed will appear when it is online.';
    $(slot+'Caption').textContent=key==='simulation'?'MUJOCO · PLAN PREVIEW':key==='twin'?'LIVE ROBOT STATE':SOURCES[key].toUpperCase();
  }
  const simulationVisible=Object.values(viewers).includes('simulation');
  $('simulationStatus').hidden=!simulationVisible;
  const sim=state?.simulation;
  $('simulationMessage').textContent=sim?.playing ? 'Showing '+sim.name+'…' : sim?.plan ? (sim.time < sim.duration ? 'Preview stopped · ' : 'Preview finished · ')+sim.name : 'Ask in chat to create a new motion preview.';
  if(sim?.description)$('simulationMessage').textContent+=' · '+sim.description;
  $('stopPreview').hidden=!sim?.playing;
  $('stopGlassesShare').hidden=!sharing;
}
function renderStatus() {
  renderVoice();
  if (!state) return;
  renderViewers();renderGestures();renderPrompt();renderChat();renderDetection();renderOverview();renderPipeline();
}
async function poll() {
  if (statusBusy) return;
  statusBusy = true;
  try {
    const response = await fetch('/api/status', {signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('Server offline');
    state = await response.json();
    renderStatus();
  } catch {
    state=null;renderViewers();renderChat();renderDetection();renderGestures();renderPipeline();
    for(const id of ['statusRobot','statusGestures','statusCameras','statusGlasses'])$(id).classList.remove('online');
    for(const id of ['statRobot','statGestures','statCams','statGlasses'])$(id).textContent='—';
  } finally { statusBusy = false; }
}
function refreshFrames() {
  if (document.hidden) return;
  for(const slot of Object.keys(viewers))viewerFrame(slot);
}
function stopShare() {
  const stream = sharing; sharing = null;
  stream?.getTracks().forEach(track => track.stop());
  renderViewers();renderOverview();
}
async function share() {
  if (!navigator.mediaDevices?.getDisplayMedia) { toast('Window sharing requires a desktop browser on localhost. You can also supply an MJPEG URL.'); return; }
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({video:{frameRate:20},audio:false});
    if (sharing) stopShare(); sharing = stream;
    if (!Object.values(viewers).includes('glasses')) setViewer('secondary','glasses');
    stream.getVideoTracks()[0].addEventListener('ended',stopShare,{once:true});
    if ($('connections').open) $('connections').close(); renderStatus();
  } catch (error) { toast(error.name === 'NotAllowedError' ? 'Window sharing cancelled or not permitted. You can try again.' : `Could not share the window: ${error.message}`); }
}
$('stopPreview').onclick=()=>control({action:'stop'});
for (const slot of Object.keys(viewers)) $(slot+'Source').onchange=()=>setViewer(slot,$(slot+'Source').value);
$('swapViews').onclick=()=>{const main=viewers.main,side=viewers.secondary;setViewer('main',side);setViewer('secondary',main);};
$('viewConnections').onclick=()=>$('connections').showModal();
document.querySelectorAll('.expand').forEach(button => button.onclick = async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await button.closest('.panel').requestFullscreen(); } catch { toast('Fullscreen is unavailable in this browser.'); } });
['settings'].forEach(id => $(id).onclick = () => $('connections').showModal());
document.querySelectorAll('.open-connections').forEach(b => b.onclick = () => $('connections').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => $(b.dataset.close).close());
$('shareFromSettings').onclick=share; $('stopGlassesShare').onclick=stopShare;
window.addEventListener('beforeunload', () => sharing?.getTracks().forEach(t => t.stop()));
async function init() {
  try { const response=await fetch('/api/session'); if(!response.ok)throw new Error('Cannot connect'); token=(await response.json()).token; await poll(); }
  catch (e) { toast(e.message + '. Refresh after starting the server.'); }
  setInterval(poll,300); setInterval(refreshFrames,80);
  setInterval(() => $('clock').textContent = new Date().toLocaleTimeString([], {hour12:false}) + ' · LOCAL',1000);
}
function renderGestures() {
  const g=state?.gestures;
  const busy=gestureRequestBusy || !!g?.busy;
  badge('gestureBadge', !g ? 'OFFLINE' : busy ? 'WORKING' : g.connected ? 'READY' : 'OFFLINE', !!g?.connected && !busy);
  $('refreshGestures').disabled=busy || !state || state.mode==='sim';
  $('refreshGestures').textContent=g?.connected ? 'Refresh gestures' : 'Connect gestures';
  $('gestureMessage').textContent=state?.mode==='sim'?'Firmware gestures are disabled in simulation mode.':g?.message || 'Dashboard disconnected.';
  $('gestureMessage').classList.toggle('error',!!g?.error);
  $('gestureConnection').textContent=g ? 'Robot connection: '+g.iface : '';
  const list=$('gestureButtons');
  const version=JSON.stringify([g?.actions,busy,g?.connected,state?.pipeline?.connected,state?.pipeline?.busy,state?.pipeline?.proposal?.id,state?.mode]);
  if(list.dataset.version===version)return;
  list.dataset.version=version;list.replaceChildren();
  for(const action of g?.actions || []) {
    const button=document.createElement('button');
    button.type='button';button.className='button gesture-button'+(action.id===99?' release-gesture':'');
    button.textContent=action.label;button.disabled=state?.mode==='sim' || busy || !g.connected || !!state?.pipeline?.connected || !!state?.pipeline?.busy || !!state?.pipeline?.proposal;
    button.onclick=()=>gestureCommand({action:'gesture',id:action.id});
    list.append(button);
  }
}
async function gestureCommand(command) {
  if(gestureRequestBusy || state?.gestures?.busy)return;
  gestureRequestBusy=true;renderGestures();
  try {
    const response=await fetch('/api/gestures',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify(command)});
    const body=await response.json();
    if(!response.ok)throw new Error(body.error || 'Gesture request failed');
    if(state)state.gestures=body;
  } catch(error){toast(error.message);}
  finally{gestureRequestBusy=false;renderGestures();}
}
$('refreshGestures').onclick=()=>gestureCommand({action:'refresh'});

/* Compact status strip; detailed state remains in Robot controls. */
function renderOverview() {
  if (!state) return;
  const feeds=state.feeds,gestures=state.gestures,robotText=state.robot||'';
  const live=/msgs/.test(robotText)&&!/no rt\/lowstate/.test(robotText),age=robotText.match(/last ([\d.]+)s ago/);
  const fresh=!!state?.pipeline?.connected || live&&age&&Number(age[1])<1;
  $('statRobot').textContent=fresh?'Live':live?'Stale':'Offline';
  $('statusRobot').title=robotText||'Robot state unavailable';$('statusRobot').classList.toggle('online',!!fresh);
  $('statGestures').textContent=gestures.busy?'Working':gestures.connected?'Ready':'Offline';$('statusGestures').classList.toggle('online',gestures.connected);
  const cams=['head','left','right'].filter(k=>feeds[k].online).length;
  $('statCams').textContent=`${cams}/3`;$('camCount').textContent=`${cams}/3`;$('statusCameras').classList.toggle('online',cams>0);
  const paired=Number(state.pipeline?.glasses?.connected||0)>0,glasses=paired||!!sharing||feeds.glasses.online;
  $('statGlasses').textContent=paired?'Paired':sharing?'Shared':glasses?'Video':'Offline';$('statusGlasses').classList.toggle('online',glasses);
  $('connectionCount').textContent=`${Number(cams>0)+Number(glasses)}/2`;
  $('railStatus').textContent=gestures.busy?'R1 request active':gestures.connected?'R1 gestures connected':live?robotText:'Robot state unknown';
}
function navTo(buttonId, target) {
  document.querySelectorAll('.rail-button').forEach(b => b.classList.toggle('active', b.id === buttonId));
  (target === 'top' ? document.body : $(target)).scrollIntoView({behavior:'smooth', block:'start'});
}
$('navObservatory').onclick = () => navTo('navObservatory', 'top');
$('navCameras').onclick=()=>{setViewer('secondary','head');navTo('navCameras','viewerWorkspace');};
$('navGlasses').onclick=()=>{setViewer('secondary','glasses');navTo('navGlasses','viewerWorkspace');};
$('controlJump').onclick=()=>navTo('controlJump','robotPanel');
$('headerConnections').onclick = () => $('connections').showModal();
init();

// Prompt orchestration is asynchronous; the server owns status and proposal IDs.
let promptVersion = '', promptRequestBusy = false;
// Planning and preview run automatically; Accept is the only execution step.
function renderFlow() {
  const p=state?.pipeline, planning=state?.prompt?.state==='planning'||state?.chat?.busy;
  const set=(id,cls,note)=>{const li=$(id);li.className=cls;li.querySelector('small').textContent=note;};
  set('flowPrompt',state?.chat?.messages?.length||p?.draft||p?.proposal?'done':'active','Ask the assistant for a motion');
  set('flowPreview',p?.draft||p?.proposal?'done':planning?'active':p?.state==='blocked'?'failed':'',
    p?.proposal?.kind==='firmware'?'Onboard path unavailable for preview':p?.proposal?'Playing the checked motion':p?.draft?'Checking the complete motion':planning?'Observing, planning and checking…':'The checked motion plays automatically');
  set('flowShow',p?.state==='completed'?'done':p?.state==='review'?'ready':p?.state==='executing'?'active':'',
    p?.state==='completed'?'Motion completed':p?.state==='review'?'Accept here or pinch in paired glasses':p?.state==='executing'?'Accepted motion in progress':'One Accept runs this motion');
}
function renderPrompt() {
  const p=state?.prompt;
  renderFlow();
  if(!p)return;
  $('promptBadge').hidden=p.state!=='planning';
  badge('promptBadge',p.stage==='revise'&&p.state==='planning'?'RECALCULATING':p.state.toUpperCase(),['proposed','previewed'].includes(p.state));
  const version=JSON.stringify([p.id,p.state,p.message,p.events?.length,state?.pipeline?.state,state?.pipeline?.draft?.id,state?.pipeline?.proposal?.id]);
  if(version===promptVersion)return;
  promptVersion=version;
  $('promptSummary').textContent=p.prompt?'“'+p.prompt+'”':'No motion prepared yet.';
  $('promptMessage').textContent=state?.pipeline?.proposal?.kind==='firmware'?'Onboard preset ready. Accept to send it to the robot.':state?.pipeline?.proposal?'Preview ready. Accept to run this exact motion.':state?.pipeline?.draft?'The assistant is finishing the checked motion.':state?.pipeline?.last_result?'The outcome is shown above. Ask for a new motion to continue.':p.message;
  $('planningDetails').hidden=!p.events?.length;
  $('promptEvents').replaceChildren();
  for(const event of p.events||[]){const li=document.createElement('li'),tag=document.createElement('span'),text=document.createElement('span');tag.textContent=event.stage;text.textContent=event.message;li.append(tag,text);$('promptEvents').append(li);}
  $('cancelPrompt').hidden=!!state?.pipeline?.proposal || (p.state!=='planning'&&!state?.pipeline?.draft);
  $('cancelPrompt').textContent='Cancel planning';
  $('promptMetrics').replaceChildren();
  for(const [label,value] of [['Arm',p.target?.arm],['Validation',p.validation?.samples!=null?p.validation.samples+' sampled poses':null]]) {
    if(value==null)continue;
    const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;row.append(dt,dd);$('promptMetrics').append(row);
  }
  $('groundingImage').hidden=!p.has_image;
  if(p.has_image)$('groundingImage').src='/frame/grounding?id='+encodeURIComponent(p.id);
}
async function promptCommand(command) {
  if (promptRequestBusy) return false;
  promptRequestBusy = true; renderPrompt(); renderChat();
  try {
    const response = await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify(command)});
    const body = await response.json(); if (!response.ok) throw new Error(body.error || 'Prompt request failed');
    if (state) state.prompt=body;
    renderPrompt(); return true;
  } catch(error) { toast(error.message); return false; }
  finally { promptRequestBusy=false; if (state) { renderPrompt(); renderChat(); } }
}
$('cancelPrompt').onclick = () => promptCommand({action:'cancel'});
// The endpoint only serves this exact result ID; late source changes invalidate requests.
let detectionBusy = false, detectionFrame = '', detectionUrl = null, detectionRequest = '';
async function loadDetectionImage(id) {
  detectionRequest = id;
  try {
    const response = await fetch('/frame/detection?id='+encodeURIComponent(id), {signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error('Detection expired');
    const url = URL.createObjectURL(await response.blob());
    if (state?.detection?.result?.id !== id || detectionRequest !== id) { URL.revokeObjectURL(url); return; }
    const image = new Image();
    image.onload = () => {
      if (state?.detection?.result?.id !== id || detectionRequest !== id) { URL.revokeObjectURL(url); return; }
      if (detectionUrl) URL.revokeObjectURL(detectionUrl);
      detectionUrl = url; $('detectionImage').src = url; $('detectionImage').hidden = false;
      $('detectionEmpty').hidden = true;
    };
    image.onerror = () => URL.revokeObjectURL(url);
    image.src = url;
  } catch { if (detectionRequest === id) { detectionFrame = ''; $('detectionImage').hidden = true; $('detectionEmpty').hidden = false; } }
}
function renderDetection() {
  const d = state?.detection;
  const ready = !!d?.ready, result = ready ? d.result : null;
  badge('detectionBadge', !d ? 'OFFLINE' : !d.enabled ? 'PAUSED' : ready ? 'DETECTING' : 'WAITING', ready);
  $('detectionToggle').textContent = d?.enabled ? 'Pause detection' : 'Start detection';
  $('detectionToggle').setAttribute('aria-pressed', String(!!d?.enabled));
  $('detectionToggle').disabled = detectionBusy || !d;
  $('detectionSource').disabled = detectionBusy || !d;
  if (d) {
    if (d.sources.includes('observation') && !$('detectionSource').querySelector('[value="observation"]')) {
      const option = document.createElement('option'); option.value='observation'; option.textContent='Calibrated RGB + depth'; $('detectionSource').append(option);
    }
    if (!detectionBusy) $('detectionSource').value = d.source;
    $('detectionModel').textContent = d.model+' · '+Math.round(d.confidence*100)+'% minimum confidence';
  }
  $('detectionMessage').textContent = d?.message || 'Dashboard disconnected. Waiting for the local server…';
  $('detectionCount').textContent = result ? result.objects.length : '—';
  $('detectionStamp').hidden = !ready;
  if (result) $('detectionStamp').textContent = `${d.source.toUpperCase()} · ${result.inference_ms} ms · ${result.age_s.toFixed(1)} s ${result.captured_at ? 'since capture' : 'since receipt'}`;
  if (!ready) {
    detectionFrame = ''; detectionRequest = '';
    $('detectionImage').hidden = true; $('detectionEmpty').hidden = false;
    if (detectionUrl) { URL.revokeObjectURL(detectionUrl); detectionUrl=null; }
  }
  if (result && detectionFrame !== result.id) { detectionFrame=result.id; $('detectionImage').hidden=true; $('detectionEmpty').hidden=false; loadDetectionImage(result.id); }
  $('detectionObjects').replaceChildren();
  if (result?.objects.length) {
    for (const object of result.objects) {
      const row=document.createElement('div'), label=document.createElement('strong'), confidence=document.createElement('span'), depth=document.createElement('small');
      row.className='detection-object'; label.textContent=object.label; confidence.textContent=Math.round(object.confidence*100)+'%';
      depth.textContent=object.surface_m ? 'Surface · '+object.surface_m.map(x=>x.toFixed(3)).join(', ')+' m · robot frame' : object.depth_detail;
      row.append(label,confidence,depth); $('detectionObjects').append(row);
    }
  } else {
    const text=document.createElement('p');text.className='detection-hint';
    text.textContent=ready ? 'No objects above the confidence threshold.' : 'Detections will appear with their confidence and available depth.';
    $('detectionObjects').append(text);
  }
}
async function detectionCommand(enabled, source) {
  if (detectionBusy) return;
  detectionBusy=true;renderDetection();
  try {
    const response=await fetch('/api/detection',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify({enabled,source})});
    const body=await response.json();if(!response.ok)throw new Error(body.error||'Could not change detection');
    if(state)state.detection=body;
  } catch(error){toast(error.message);}
  finally{detectionBusy=false;renderDetection();}
}
$('detectionToggle').onclick=()=>detectionCommand(!state?.detection?.enabled,$('detectionSource').value);
$('detectionSource').onchange=()=>detectionCommand(!!state?.detection?.enabled,$('detectionSource').value);

// Motion replies are submitted by the server, including replies from glasses voice.
function displayMessage(message) {
  const result=message?.runtime_result;
  if(!result)return message?.text||'';
  return (result.name?result.name+': ':'')+(result.message||result.outcome||'Result recorded.')+
    (typeof result.tracking_error==='number'?' Tracking error '+result.tracking_error.toFixed(3)+' rad.':'');
}
let chatBusy = false, chatVersion = '';
function renderChat() {
  const chat = state?.chat;
  badge('chatBadge', !chat ? 'OFFLINE' : !chat.configured ? 'SETUP NEEDED' : chat.busy ? 'THINKING' : 'READY', !!chat?.configured && !chat.busy);
  $('sendChat').disabled = chatBusy || !chat?.configured || chat.busy;
  $('clearChat').disabled = chatBusy || !chat || !chat.messages.length;
  $('retryChat').disabled = chatBusy || !chat || chat.busy;
  $('chatConfig').textContent = chat?.configured ? chat.provider_label+' · '+chat.model : (chat?.setup || 'Connecting to the dashboard…');
  $('stopChat').hidden = !chat?.busy;
  $('stopChat').disabled = chatBusy;
  const cli = chat?.provider === 'codex' || chat?.provider === 'claude';
  $('chatTransport').textContent = chat ? chat.provider_label+(cli ? ' · existing sign-in' : '')+' · separate conversation' : '';
  $('chatTitle').textContent = chat?.provider_label?.replace(/ (CLI|API)$/, '') || 'Assistant';
  $('chatSubtitle').textContent = chat ? (cli ? 'CLI' : 'API')+' · '+(chat.model || '').toUpperCase() : 'CONNECTING';
  renderChatBackends(chat);
  renderFlow();
  renderToolCalls(state?.tools || [], chat);
  $('chatInput').placeholder = 'Message '+($('chatTitle').textContent || 'the assistant')+'…';
  $('chatStatus').textContent = chat?.busy ? (chat.messages.length ? 'Reins is thinking…' : 'Finishing the previous request. Its reply will be discarded.') : chat?.trimmed ? 'Earlier messages have left the conversation context.' : '';
  $('chatError').hidden = !chat?.error;
  $('chatErrorText').textContent = chat?.error || '';
  if (!chat) return;
  const version = JSON.stringify([chat.session_id,chat.version,state.prompt?.state,promptRequestBusy,state.pipeline?.busy,state.pipeline?.proposal?.id]);
  if (version === chatVersion) return;
  chatVersion = version;
  const log = $('chatTranscript'), nearBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 80;
  log.replaceChildren();
  if (!chat.messages.length) {
    const welcome=document.createElement('p');welcome.className='cli-welcome';
    welcome.textContent=chat.configured?'Ready. Ask a question or describe a task.':'Connect '+chat.provider_label+' to start a conversation, or pick another assistant above.';log.append(welcome);
  }
  for (const message of chat.messages) {
    const row=document.createElement('article'), name=document.createElement('span'), text=document.createElement('div');
    row.className='chat-message '+message.role;name.className='chat-speaker';name.textContent=message.runtime_result?'robot':message.role==='user'?'you':(message.provider||'assistant').replace(/ (CLI|API)$/,'').toLowerCase();
    text.className='chat-text';text.textContent=displayMessage(message);row.append(name,text);
    if (message.robot_request) {
      const suggestion=document.createElement('div');
      suggestion.className='chat-suggestion';suggestion.textContent=message.robot_request+(message.trajectory ? ' · '+message.trajectory.waypoints.length+' authored waypoints' : '');
      row.append(suggestion);
    }
    log.append(row);
  }
  if (nearBottom) log.scrollTop=log.scrollHeight;
}
async function chatCommand(command) {
  if (chatBusy) return false;
  chatBusy=true;renderChat();
  try {
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify(command),signal:AbortSignal.timeout(10000)});
    const body=await response.json();if(!response.ok)throw new Error(body.error||'Could not send message');
    if(state)state.chat=body;renderChat();return true;
  } catch(error){toast(error.message);return false;}
  finally{chatBusy=false;renderChat();}
}
$('chatForm').onsubmit=async event=>{event.preventDefault();const text=$('chatInput').value;if(await chatCommand({action:'send',message:text})){if($('chatInput').value===text)$('chatInput').value='';$('chatInput').focus();}};
$('chatInput').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('sendChat').disabled)$('chatForm').requestSubmit();}});
$('clearChat').onclick=()=>chatCommand({action:'clear'});
$('retryChat').onclick=()=>chatCommand({action:'retry'});

$('stopChat').onclick=()=>chatCommand({action:'cancel'});
// Reins tool calls made by the assistant (detection, planning, preview): shown so they can be reviewed.
function renderToolCalls(calls, chat) {
  const box = $('chatTools');
  const recent = calls.filter(c => Date.now() / 1000 - c.at < 900).slice(-6);
  box.hidden = !recent.length;
  const key = JSON.stringify(recent.map(c => [c.at, c.ok]));
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  const title = document.createElement('span'); title.className = 'chat-tools-title';
  title.textContent = chat?.tools ? 'Tools used' : 'Tools';
  box.replaceChildren(title, ...recent.map(c => {
    const chip = document.createElement('span'); chip.className = 'tool-chip ' + (c.ok ? 'ok' : 'failed');
    const arg = c.arguments?.camera || c.arguments?.object || c.arguments?.label || c.arguments?.name || '';
    chip.textContent = (c.ok ? '✓ ' : '✗ ') + c.tool.replaceAll('_', ' ') + (arg ? ' · ' + arg : '');
    chip.title = c.summary + ' (' + c.duration_s + ' s)';
    return chip;
  }));
}
// Assistant picker: Claude CLI, Codex CLI or OpenAI API; the conversation carries over when switching.
function renderChatBackends(chat) {
  const select = $('chatBackend');
  const options = chat?.backends || [];
  const key = JSON.stringify(options.map(o => [o.backend, o.configured]));
  if (select.dataset.key !== key) {
    select.replaceChildren(...options.map(o => { const opt = document.createElement('option'); opt.value = o.backend; opt.textContent = o.provider_label + (o.configured ? '' : ' · setup needed'); return opt; }));
    select.dataset.key = key;
  }
  if (chat && document.activeElement !== select) select.value = chat.backend;
  select.disabled = chatBusy || !chat || chat.busy || options.length < 2;
  select.title = chat?.busy ? 'Wait for the reply, or stop it, before switching.' : 'Choose which assistant answers';
}
$('chatBackend').onchange = async () => {
  const wanted = $('chatBackend').value;
  if (await chatCommand({action:'backend', backend:wanted})) toast('Now chatting with '+(state.chat.backends.find(o => o.backend === wanted)?.provider_label || wanted)+'. The conversation carries over.');
  else renderChat();
};

$('chatInput').addEventListener('input',()=>{const input=$('chatInput');input.style.height='auto';input.style.height=Math.min(input.scrollHeight,130)+'px';});

/* One reviewed motion pipeline, shared with the paired glasses. */
let robotRequestBusy=false, displayedProposalId=null;
function renderPipeline() {
  renderFlow();
  const p=state?.pipeline,proposal=p?.proposal,draft=p?.draft,reviewing=p?.state==='review'&&!!proposal&&!proposal.expired;
  if(proposal&&proposal.id!==displayedProposalId) {
    displayedProposalId=proposal.id;
    if(proposal.kind!=='firmware'&&viewers.main!=='simulation')setViewer('main','simulation');
  }
  const item=proposal||draft,kind=item?.kind||item?.motion_kind||'arm';
  $('robotModeLabel').textContent=state?.mode==='sim'?'SIMULATION · HARDWARE DISABLED':kind==='firmware'?'PHYSICAL ROBOT · ONBOARD PRESET':p?.mode==='live'?'PHYSICAL ROBOT · '+(p.connected?'STATE CONNECTED':'DISCONNECTED'):'SIMULATION';
  badge('pipelineBadge',p?.state?.toUpperCase()||'OFFLINE',reviewing||p?.state==='completed');
  $('pipelineSummary').textContent=item?.name||'No motion awaiting review.';
  $('pipelineMessage').textContent=p?.message||'Dashboard disconnected.';
  const target=kind==='firmware'?'onboard preset':kind==='walk'?'base path':kind==='hand'?(item?.arm||'selected')+' hand action':(item?.arm||'selected')+' arm';
  const duration=kind==='firmware'?'Duration controlled by robot':Number(item?.duration_s||0).toFixed(1)+' s';
  $('pipelineDetails').textContent=item?(item.mode==='live'?'Physical robot':'Simulation')+' · '+target+' · '+duration+(proposal?' · revision '+proposal.revision+(proposal.expired?' · Expired':' · '+Math.max(0,Math.ceil(proposal.expires_at-Date.now()/1000))+' s to accept'):' · Preparing proposal')+(item.description?' · '+item.description:''):'';
  $('pipelineActions').hidden=!reviewing;
  $('approveMotion').textContent=proposal?.mode==='live'?'Accept & run on robot':'Accept & run in simulation';
  $('approveMotion').disabled=robotRequestBusy||!reviewing;
  const busy=robotRequestBusy||!!p?.busy||!!proposal||state?.prompt?.state==='planning';
  $('robotConnect').disabled=!p||busy||p.connected||state?.mode==='sim'||p?.capabilities?.hardware_allowed===false;
  $('robotConnect').textContent=p?.connected?'Robot state connected':'Connect robot';
  $('robotHome').disabled=!p||busy;
  $('robotArm').disabled=busy;
  $('tableHeight').disabled=busy||!!p?.connected;
  document.querySelectorAll('[data-jog],[data-roll]').forEach(b=>b.disabled=!p||busy);
  const walking=p?.capabilities?.walking===true,hands=p?.capabilities?.hands||{},hand=hands[$('robotArm').value]===true;
  document.querySelectorAll('[data-walk]').forEach(b=>b.disabled=!p||busy||!walking);
  document.querySelectorAll('[data-hand]').forEach(b=>b.disabled=!p||busy||!hand);
  $('extraControlStatus').textContent=(walking?'Walking enabled':'Walking disabled')+' · '+(hand?'Selected hand available':'Selected hand unavailable')+(p?.capabilities?.walking_note?' · '+p.capabilities.walking_note:'');
  $('robotStop').disabled=!p;
  if(p?.table_z_m!=null&&document.activeElement!==$('tableHeight'))$('tableHeight').value=p.table_z_m;
  const robot=p?.robot_state;
  $('robotStateStatus').textContent=robot?(p.mode==='live'?'Measured state':'Simulated state')+(robot.lowstate_age_s!=null?' · '+Number(robot.lowstate_age_s).toFixed(2)+' s old':''):'Connect to read the current pose.';
  $('robotStatePose').textContent=robot?JSON.stringify(robot,null,2):'';
  const result=p?.last_result;
  $('motionOutcome').hidden=!result;
  if(result) {
    $('outcomeSummary').textContent='Last motion: '+(result.outcome||'finished');
    $('outcomeDetails').textContent=result.message||(result.tracking_error!=null?'Tracking error: '+(typeof result.tracking_error==='number'?result.tracking_error.toFixed(3)+' rad':JSON.stringify(result.tracking_error)):'Result recorded.');
    $('outcomePose').textContent=JSON.stringify(result.measured_end_pose||{},null,2);
  }
}
async function robotCommand(command) {
  const stopping=command.action==='stop';
  if(robotRequestBusy&&!stopping)return;
  if(!stopping)robotRequestBusy=true;
  renderPipeline();
  try {
    const response=await fetch('/api/robot',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},
      body:JSON.stringify(command),signal:AbortSignal.timeout(stopping?20000:15000)});
    const body=await response.json();if(!response.ok)throw new Error(body.error||'Robot request failed');
    if(state)state.pipeline=body;
    if(['jog','home','roll','walk','hand','preview','propose'].includes(command.action)) {
      setViewer('main','simulation');$('motionDetails').scrollIntoView({behavior:'smooth',block:'center'});
    }
  } catch(error){toast(error.message);}
  finally{if(!stopping)robotRequestBusy=false;renderPipeline();renderGestures();}
}
$('robotConnect').onclick=()=>robotCommand({action:'connect',table_z_m:$('tableHeight').value===''?null:Number($('tableHeight').value)});
$('robotStop').onclick=()=>robotCommand({action:'stop'});
$('robotHome').onclick=()=>robotCommand({action:'home',arm:$('robotArm').value});
document.querySelectorAll('[data-jog]').forEach(b=>b.onclick=()=>robotCommand({action:'jog',arm:$('robotArm').value,direction:b.dataset.jog}));
$('robotArm').onchange=()=>{renderPipeline();robotCommand({action:'settings',arm:$('robotArm').value});};
const walkCommands={forward:[.1,0,0],back:[-.1,0,0],left:[0,.1,0],right:[0,-.1,0],turn_left:[0,0,.1],turn_right:[0,0,-.1]};
document.querySelectorAll('[data-walk]').forEach(button=>button.onclick=()=>{const [dx,dy,dyaw]=walkCommands[button.dataset.walk];robotCommand({action:'walk',dx,dy,dyaw});});
document.querySelectorAll('[data-hand]').forEach(button=>button.onclick=()=>robotCommand({action:'hand',arm:$('robotArm').value,closed:button.dataset.hand==='close'}));
$('approveMotion').onclick=()=>{
  const p=state?.pipeline?.proposal;if(!p)return;
  robotCommand({action:'decision',id:p.id,digest:p.digest,decision:'approve'});
};
document.addEventListener('keydown',event=>{if(event.key==='Escape' && state?.pipeline?.connected){event.preventDefault();robotCommand({action:'stop'});}});
async function loadGlassesPairing() {
  try {
    const response=await fetch('/api/glasses');if(!response.ok)throw new Error('Pairing information unavailable');
    const g=await response.json();
    $('glassesPairingStatus').textContent=g.error?'AR server: '+g.error:'Lens WebSocket: ws://THIS_COMPUTER_IP:'+g.port+' · '+g.connected+' connected';
    $('glassesDevices').replaceChildren();
    for(const device of g.paired_devices||[]) {
      const li=document.createElement('li'),label=document.createElement('span');
      label.textContent=device.label+' · '+device.device_id+(device.revoked_at?' · Revoked':'');li.append(label);
      if(!device.revoked_at){const button=document.createElement('button');button.className='text-button';button.textContent='Revoke';button.onclick=()=>pairingCommand({action:'revoke_device',device_id:device.device_id});li.append(button);}
      $('glassesDevices').append(li);
    }
  } catch(error){$('glassesPairingStatus').textContent=error.message;}
}
async function pairingCommand(command) {
  $('pairGlasses').disabled=true;
  try {
    const response=await fetch('/api/glasses',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},body:JSON.stringify(command)});
    const body=await response.json();if(!response.ok)throw new Error(body.error||'Pairing failed');
    if(body.device){const device=body.device;$('glassesPairing').hidden=false;$('glassesPairing').textContent='Save in the Lens now; this token is shown only once.\ndeviceId: '+device.device_id+'\nreviewToken: '+device.token;}
    if(command.action==='revoke_device'){$('glassesPairing').textContent='';$('glassesPairing').hidden=true;}
    await loadGlassesPairing();
  } catch(error){toast(error.message);}
  finally{$('pairGlasses').disabled=false;}
}
$('pairGlassesForm').onsubmit=event=>{event.preventDefault();pairingCommand({action:'create_device',label:$('glassesDeviceName').value});};
$('connections').addEventListener('toggle',()=>{if($('connections').open)loadGlassesPairing();});
$('connections').addEventListener('close',()=>{$('glassesPairing').textContent='';$('glassesPairing').hidden=true;});

setInterval(()=>{
  if(token && (state?.pipeline?.connected || state?.pipeline?.state==='connecting')) {
    fetch('/api/robot',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},
      body:JSON.stringify({action:'heartbeat'}),signal:AbortSignal.timeout(2000)}).catch(()=>{});
  }
},1000);

document.querySelectorAll('[data-roll]').forEach(b=>b.onclick=()=>robotCommand({action:'roll',arm:$('robotArm').value,sign:Number(b.dataset.roll)}));

// The optional voice panel exchanges text only; dictation never submits a motion.
let voiceReady=false;
function renderVoice() {
  const url=state?.voice_url;
  $('voicePanel').hidden=!url;
  $('speakPrompt').hidden=!url;
  if(url && $('voiceFrame').getAttribute('src')!==url) {
    $('voiceFrame').src=url; $('voiceOpen').href=url; voiceReady=false;
  }
  $('speakPrompt').disabled=!voiceReady;
}
window.ReinsVoice={speak(text) {
  if(!voiceReady || !state?.voice_url || typeof text!=='string' || !text.trim() || text.length>1000)return false;
  $('voiceFrame').contentWindow.postMessage({type:'reins-voice-speak',text},new URL(state.voice_url).origin);
  voiceReady=false; $('speakPrompt').disabled=true; return true;
}};
window.addEventListener('message',event=>{
  if(!state?.voice_url || event.source!==$('voiceFrame').contentWindow || event.origin!==new URL(state.voice_url).origin)return;
  if(event.data?.type==='reins-voice-ready') {
    voiceReady=event.data.ready===true; $('speakPrompt').disabled=!voiceReady;
  }
  if(event.data?.type==='reins-voice-transcript') {
    const text=event.data.text,field=$('chatInput');
    if(typeof text!=='string' || !text.trim() || text.length>1000)return;
    if(field.value.length-(field.selectionEnd-field.selectionStart)+text.length>field.maxLength) {
      toast('Dictation is too long for the prompt. Copy it from the voice transcript.'); return;
    }
    field.setRangeText(text,field.selectionStart,field.selectionEnd,'end');
    field.dispatchEvent(new Event('input',{bubbles:true}));
    toast('Dictation added. Review the prompt, then send it.');
  }
});
$('speakPrompt').onclick=()=>{
  const reply=[...(state?.chat?.messages||[])].reverse().find(message=>message.role==='assistant');
  if(!window.ReinsVoice.speak(displayMessage(reply)||$('pipelineMessage').textContent))toast('Connect voice and wait for playback to finish. Replies must be under 1,000 characters.');
};

// Historical examples inform planning; they are never a replay library.
async function loadExperience() {
  try {
    const response=await fetch('/api/experience');
    if(!response.ok)throw new Error('Experience status unavailable');
    renderExperience(await response.json());
  } catch(error) {$('experienceStatus').textContent=error.message;}
}
function renderExperience(memory) {
  $('experienceStatus').textContent=memory.enabled?
    `${memory.count} saved proposal${memory.count===1?'':'s'} · up to ${memory.max_entries} retained. `+
    'Historical cards and selected camera images accompany planning context.' : 'Historical experience is disabled.';
  if(memory.error)$('experienceStatus').textContent+=' Storage issue: '+memory.error;
  $('forgetExperience').disabled=!memory.count;
}
$('connections').addEventListener('toggle',()=>{if($('connections').open)loadExperience();});
$('forgetExperience').onclick=async()=>{
  $('forgetExperience').disabled=true;
  try {
    const response=await fetch('/api/experience',{method:'POST',headers:{'Content-Type':'application/json','X-Reins-Token':token},
      body:JSON.stringify({action:'forget'})});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not forget experience');
    renderExperience(result);toast('Historical proposal memory cleared. New completed reviews can create new examples.');
  } catch(error) {toast(error.message);loadExperience();}
};
