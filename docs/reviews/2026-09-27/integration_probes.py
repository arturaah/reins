import asyncio,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as real_serve
from spectacles.plan_feed import serve_feed
from spectacles.review import ReviewMailbox
from core.prompt_planner import PromptPlanner
from core.reins_tools import ReinsTools
from core.test_trajectory_revision import UNREACHABLE
from spectacles.make_walk_plan import combine

out={}
planner=PromptPlanner()
tools=ReinsTools(SimpleNamespace(available=False,name="fake"),{},planner,lambda *a:None,lambda:{})
d=UNREACHABLE
result=tools.plan_hand_path(d["name"],d["arm"],d["waypoints"],d["return_to_start"])
out["mcp_replanning"]={"state":result["state"],"retryable":result.get("retryable"),"attempts":planner.status().get("attempt"),"max_attempts":planner.status().get("max_attempts")}
planner.cancel()
out["walking_wrapper"]={"input_preview_only":True,"output_preview_only":combine(
    {"schema_version":1,"preview_only":True,"keyframes":[{"time_s":0,"joint_targets_rad":{"right_elbow_joint":0}},{"time_s":1,"joint_targets_rad":{"right_elbow_joint":.1}}]},
    .3,2).get("preview_only")}

async def ar_probe():
    ready=asyncio.Event()
    servers=[]
    def wrapped(handler,host,port,*args,**kwargs):
        server=real_serve(handler,host,port,*args,**kwargs)
        class Context:
            async def __aenter__(self):
                active=await server.__aenter__();servers.append(active);ready.set();return active
            async def __aexit__(self,*exc):
                return await server.__aexit__(*exc)
        return Context()
    with tempfile.TemporaryDirectory() as tmp:
        plan=Path(tmp)/"plan.json";plan.write_text("{}")
        mailbox=ReviewMailbox(Path(tmp)/"review.json")
        proposal=mailbox.propose(plan,"Offline test proposal","live")
        class Feed:
            path=plan
            text=json.dumps({"type":"trajectory","version":1,"hands":{}})
            def current(self,*args): return self.text
        with patch("websockets.asyncio.server.serve",wrapped):
            task=asyncio.create_task(serve_feed(Feed(),"127.0.0.1",0,.02,review=mailbox))
            await asyncio.wait_for(ready.wait(),2)
            port=servers[0].sockets[0].getsockname()[1]
            try:
                # A plain, unauthenticated local client, not the Spectacles app.
                async with connect("ws://127.0.0.1:"+str(port)) as ws:
                    payload=json.loads(await ws.recv())
                    await ws.send(json.dumps({"type":"review_decision","version":1,"id":payload["review"]["id"],"decision":"approve"}))
                    while True:
                        message=json.loads(await ws.recv())
                        if message.get("type")=="review_ack": break
                    out["ar_approval"]={"credential_supplied":False,"mode":payload["review"]["mode"],"ack_accepted":message["accepted"],"harness_decision":mailbox.take(proposal["id"],plan)}
            finally:
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
asyncio.run(ar_probe())
print(json.dumps(out,indent=2))
Path("/tmp/reins-review-integration-results.json").write_text(json.dumps(out,indent=2)+"\n")
