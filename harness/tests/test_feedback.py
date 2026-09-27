"""harness/feedback.py: accepted and rejected moves kept across sessions and turned into a prompt block."""
from harness.feedback import FeedbackStore


def test_block_orders_same_task_first_and_counts_bare_accepts(tmp_path):
    a = FeedbackStore(tmp_path / "fb.jsonl", session="A", max_in_prompt=10)
    a.add("task one", "reach", "MV_LEFT", False, "tape is to the left", hand_tip=[0.3, 0.1, 0.8], height_cm=28)
    a.add("task one", "reach", "MV_FWD", True)
    a.add("task one", "reach", "MV_UP", True, "good, keep this height", height_cm=30)
    a.add("other task", "push", "MV_DOWN", False)
    assert a.block("task one") == ""                                    # a session never sees its own entries
    assert len(a.entries()) == 4 and a.entries()[0]["hand_tip_m"] == [0.3, 0.1, 0.8]
    b = FeedbackStore(tmp_path / "fb.jsonl", session="B")
    lines = b.block("task one").splitlines()
    assert lines[0].startswith("OPERATOR FEEDBACK")
    assert lines[1] == '- this task, stage reach, hand 30 cm above the floor: ✓ MV_UP — "good, keep this height"'   # newest same-task first
    assert lines[2] == '- this task, stage reach, hand 28 cm above the floor: ✗ MV_LEFT — "tape is to the left"'
    assert lines[3] == '- task "other task", stage push: ✗ MV_DOWN'
    assert lines[4] == "- plus 1 accepted move(s) without comment" and len(lines) == 5
    c = FeedbackStore(tmp_path / "fb.jsonl", session="C", max_in_prompt=1)
    assert len(c.block("task one").splitlines()) == 3                     # head, one detailed entry, the count


def test_empty_store(tmp_path):
    assert FeedbackStore(tmp_path / "x.jsonl", session="A").block("t") == ""
