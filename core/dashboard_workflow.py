"""Bind chat, model tools and the planner to one automatic review workflow."""
import time


def bind_motion_workflow(chat, registry, planner, pipeline):
    registry.pipeline = pipeline
    registry.show_proposal = pipeline.show_primary
    chat.before_turn = registry.begin_turn
    chat.on_cancel = registry.cancel
    chat.on_invalid_motion = registry.reject_final_motion
    pipeline.on_result = chat.record_motion_result
    pipeline.on_stop = chat.cancel
    planner.reviser_factory = chat.motion_reviser

    def planner_notice(status):
        with pipeline.lock:
            if status.get('pipeline_generation') != pipeline.generation or pipeline.cancelled.is_set():
                return
        chat.record_planning_event(status)

    def tool_notice(status):
        with registry.lock:
            if status.get('generation') != registry.generation or registry.cancelled:
                return
        chat.record_planning_event(status)

    planner.on_event = planner_notice
    pipeline.on_planning_blocked = planner_notice
    registry.on_planning_event = tool_notice

    def complete_reply(answer, turn):
        has_motion = answer.get('trajectory') is not None or bool(answer.get('robot_request'))
        result = registry.finish_turn(turn, has_motion=has_motion)
        if result['state'] == 'blocked':
            chat.record_planning_event(result)
            return {'retry': bool(result.get('retryable')),
                    'message': result.get('message', 'Planning failed. Observe or revise the path.')}
        if result['state'] != 'none' or not has_motion:
            return None
        # Serialize the last check and submission against Stop/clear/new turns.
        # Only preparation happens here; decide() remains the execution boundary.
        try:
            with registry.lock:
                if turn != registry.generation or registry.cancelled:
                    return None
                remaining = registry.MAX_PLANS - registry.plans
                if remaining <= 0:
                    raise ValueError('Planning revision budget exhausted. No motion was submitted.')
                planner.submit(answer.get('robot_request') or answer['trajectory']['name'],
                    trajectory=answer.get('trajectory'), reviser=chat.motion_reviser(),
                    attempt_limit=min(planner.MAX_TRAJECTORY_ATTEMPTS, remaining))
                worker = planner.worker
        except (ValueError, RuntimeError) as exc:
            chat.record_planning_event({'state': 'blocked', 'message': str(exc)})
            return None
        # Keep this model turn open for planner feedback. The HTTP send already
        # returned; Stop/clear remain available while compilation runs.
        while worker.is_alive():
            worker.join(.05)
            with registry.lock:
                if turn != registry.generation or registry.cancelled:
                    return None
                if time.monotonic() - registry.started > registry.TURN_SECONDS:
                    registry.cancel()
                    chat.record_planning_event({'state': 'blocked', 'message': 'Automatic planning reached its time limit.'})
                    return None
        with registry.lock:
            if turn != registry.generation or registry.cancelled:
                return None
            registry.plans += max(1, planner.status().get('attempt', 1))
            remaining = registry.MAX_PLANS - registry.plans
            status = pipeline.status()
        if status['state'] == 'blocked' and chat.tools and remaining > 0:
            # A tool-capable model can gather fresh context after the fallback
            # compiler failed. Continue the same token and budget, never a new turn.
            return {'retry': True, 'message': status['message']}
        return None

    chat.on_reply = complete_reply
