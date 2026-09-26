import json
import threading
import time

from harness.vlm.chat import ChatVLM


def test_chat_provider_round_trip(cfg, tmp_path):
    cfg["vlm"]["chat_inbox"] = str(tmp_path / "inbox"); cfg["vlm"]["chat_timeout_s"] = 5
    v = ChatVLM(cfg, log=lambda *_: None)

    def answer():
        d = tmp_path / "inbox" / "001_act"
        while not (d / "request.json").exists():
            time.sleep(0.05)
        req = json.loads((d / "request.json").read_text())
        assert req["images"] == ["context.jpg", "right.jpg"] and (d / "prompt.txt").read_text() == "P"
        (d / "answer.json").write_text(json.dumps({"decision": "MV_UP", "reasoning": "WRIST: NO"}))
    threading.Thread(target=answer, daemon=True).start()
    r = v.act("P", [("CONTEXT VIEW", b"\xff\xd8"), ("RIGHT WRIST VIEW", b"\xff\xd8")])
    assert not r.error and json.loads(r.text)["decision"] == "MV_UP"
    cfg["vlm"]["chat_timeout_s"] = 0.3
    v2 = ChatVLM(cfg, log=lambda *_: None)
    assert "no answer" in v2.plan("P", []).error
