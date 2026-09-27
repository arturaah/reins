"""harness/stats.py: the per-call inference log and the inference-time-vs-context plot."""
import io

from PIL import Image

from harness.stats import InferenceLog, context_estimate
from harness.vlm.base import VLMResponse


def jpeg(w, h):
    b = io.BytesIO(); Image.new("RGB", (w, h), (9, 9, 9)).save(b, "JPEG"); return b.getvalue()


def test_record_estimates_context_and_replots(tmp_path):
    log = InferenceLog(tmp_path / "log.jsonl", tmp_path / "plot.png", session="s1", mode="sim", task="reach the block")
    e = log.record("plan", VLMResponse("{}", "model-x", 1.5, 10, 20, raw={"cache_read_input_tokens": 5000, "total_cost_usd": 0.01}),
                   "x" * 400, [("CONTEXT VIEW", jpeg(64, 32))])
    assert e["context_est_tokens"] == context_estimate(400, 64 * 32) == 102
    assert e["cache_read_tokens"] == 5000 and e["cost_usd"] == 0.01 and e["n_images"] == 1 and e["image_px"] == 2048
    log.record("act", VLMResponse("{}", "model-x", 0.7), "y" * 40, [])
    rows = InferenceLog.load(tmp_path / "log.jsonl")
    assert [r["kind"] for r in rows] == ["plan", "act"] and rows[1]["latency_s"] == 0.7 and log.count == 2
    png = tmp_path / "plot.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(png) as im:
        assert im.width == 640 and im.height == 200
    first = png.stat().st_mtime_ns
    log2 = InferenceLog(tmp_path / "log.jsonl", tmp_path / "plot.png", session="s2")      # a later session: earlier points stay
    log2.record("act", VLMResponse("", "model-x", 0.9, error="boom"), "z", [])              # errors are logged but not plotted
    log2.record("act", VLMResponse("{}", "model-x", 0.9), "z", [])
    assert len(InferenceLog.load(tmp_path / "log.jsonl")) == 4 and png.stat().st_mtime_ns >= first


def test_plot_without_calls(tmp_path):
    log = InferenceLog(tmp_path / "none.jsonl", tmp_path / "p.png", session="s")
    assert log.plot().exists()
