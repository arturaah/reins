"""Delegate to the existing Spectacles dry-run inbox, retaining human review."""
from uuid import uuid4

from spectacles.voice_inbox import VoiceInbox


class InboxBackend:
    model = 'Spectacles dry-run inbox'

    def __init__(self, path):
        self.inbox = VoiceInbox(path)
        self.last_request = None

    async def respond(self, messages):
        text = next((m['content'].strip() for m in reversed(messages) if m['role'] == 'user'), '')
        if not text or len(text) > 500:
            return 'Ask the caller to repeat one short, complete robot task. Nothing was queued.'
        if text == self.last_request:
            return 'That same task was already queued. Nothing additional was queued or executed.'
        if not self.inbox.enqueue('live-' + uuid4().hex, text):
            return 'The robot task inbox is busy. Ask the caller to wait. Nothing additional was queued.'
        self.last_request = text
        return ('The request was queued for the desktop dry-run planner. The desktop UI must be running. '
                'No movement was executed. The operator still reviews proposals separately.')

    async def close(self):
        pass
