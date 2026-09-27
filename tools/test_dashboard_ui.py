"""Exercise the browser's real motion controls without a robot or model."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest


NODE = os.environ.get("REINS_NODE") or shutil.which("node")
DASHBOARD = Path(__file__).resolve().parent / "dashboard"


@unittest.skipUnless(NODE, "Node is needed for dashboard JavaScript tests")
class AutomaticPreviewUiTests(unittest.TestCase):
    def test_motion_reply_needs_only_one_mode_specific_accept(self):
        program = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const source=fs.readFileSync(process.argv[1]+'/app.js','utf8');
const html=fs.readFileSync(process.argv[1]+'/index.html','utf8');
function element(tag='div') {return {tagName:tag,textContent:'',value:'right',children:[],dataset:{},
  scrollTop:0,clientHeight:500,scrollHeight:500,hidden:false,disabled:false,
  append(...children){this.children.push(...children)},replaceChildren(...children){this.children=children}};}
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
const commands=[],switches=[];
const c={Date,JSON,Number,Math,Object,document:{activeElement:null,createElement:element,querySelectorAll:()=>[]},
  $:id=>{assert.ok(elements[id],'Unknown element '+id);return elements[id]},
  badge(){},renderFlow(){},renderChatBackends(){},renderToolCalls(){},promptRequestBusy:false,
  viewers:{main:'head'},setViewer:(slot,source)=>{switches.push([slot,source]);c.viewers[slot]=source},
  robotCommand:command=>commands.push(command)};
vm.createContext(c);
function section(start,end){const a=source.indexOf(start);assert.ok(a>=0);return source.slice(a,source.indexOf(end,a));}
vm.runInContext(section('/* One reviewed motion pipeline', 'async function robotCommand'),c);
vm.runInContext(section("$('approveMotion').onclick=", 'document.addEventListener'),c);
const proposal={id:'proposal-1',digest:'exact-digest',mode:'sim',arm:'right',revision:1,duration_s:2,expires_at:Date.now()/1000+60,name:'Blow a kiss'};
c.state={mode:'sim',prompt:{state:'proposed'},chat:{busy:false},pipeline:{state:'review',mode:'sim',proposal,capabilities:{}}};
c.renderPipeline();
assert.equal(elements.pipelineActions.hidden,false);
assert.equal(elements.approveMotion.textContent,'Accept & run in simulation');
assert.equal(elements.approveMotion.disabled,false);
assert.deepEqual(switches,[['main','simulation']]);assert.equal(commands.length,0);
c.renderPipeline();assert.equal(switches.length,1);assert.equal(commands.length,0);
elements.approveMotion.onclick();
assert.equal(JSON.stringify(commands[0]),JSON.stringify({action:'decision',id:proposal.id,digest:proposal.digest,decision:'approve'}));
c.state.pipeline.proposal={...proposal,id:'proposal-2',mode:'live'};c.state.mode='live';c.state.pipeline.mode='live';
c.renderPipeline();assert.equal(elements.approveMotion.textContent,'Accept & run on robot');
c.state.pipeline.proposal.expired=true;c.renderPipeline();
assert.equal(elements.pipelineActions.hidden,true);assert.equal(elements.approveMotion.disabled,true);
c.viewers.main='head';
c.state.pipeline.proposal={...proposal,id:'firmware-1',kind:'firmware',mode:'live',duration_s:null};
c.renderPipeline();
assert.equal(c.viewers.main,'head');assert.equal(switches.length,1);
assert.ok(elements.pipelineDetails.textContent.includes('onboard preset'));
assert.ok(elements.pipelineDetails.textContent.includes('Duration controlled by robot'));
c.state.pipeline={state:'draft',draft:proposal,capabilities:{}};c.renderPipeline();
assert.equal(elements.pipelineActions.hidden,true);assert.equal(commands.length,1);
for(const id of ['draftActions','previewDraft','proposeMotion','rejectMotion','promptDraft'])assert.ok(!elements[id]);
const review=html.match(/<div id="pipelineActions"[\s\S]*?<\/div>/)[0];
assert.equal((review.match(/<button/g)||[]).length,1);
vm.runInContext(section('function displayMessage(message)', 'async function chatCommand'),c);
c.state.chat={configured:true,provider:'codex',provider_label:'Codex CLI',model:'test',version:1,
  messages:[{id:'motion-message',role:'assistant',text:'Preparing your motion.',robot_request:'Blow a kiss',trajectory:{waypoints:[{},{}]}}]};
c.renderChat();
function descendants(node){return [node,...node.children.flatMap(descendants)];}
const rendered=descendants(elements.chatTranscript);
assert.equal(rendered.filter(n=>n.tagName==='button').length,0);
assert.ok(rendered.some(n=>n.textContent.includes('Blow a kiss')));
assert.equal(commands.length,1);
'''
        result = subprocess.run([NODE, "-e", program, str(DASHBOARD)], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
