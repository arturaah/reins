"""Voice submits tasks to the dashboard agent; it holds no motion authority."""
import asyncio
from urllib.parse import urlsplit

import httpx


def dashboard_url(value):
    url = urlsplit(value)
    if (url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1')
            or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment):
        raise ValueError('Dashboard URL must be a loopback HTTP origin')
    try:
        port = url.port
    except ValueError:
        raise ValueError('Invalid dashboard port') from None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError('Invalid dashboard port')
    return value.rstrip('/')


class DashboardBackend:
    model = 'Reins dashboard agent'
    description = 'Planning request only; operator review is still required.'
    failure_message = ('The dashboard did not confirm the request. Check its chat before retrying. '
                       'Voice has not authorized any motion.')

    def __init__(self, url='http://127.0.0.1:8090', *, client=None):
        self.url = dashboard_url(url)
        self.client = client or httpx.AsyncClient(timeout=4, trust_env=False, follow_redirects=False)
        self.last_request = None
        self.lock = asyncio.Lock()

    async def _get(self, path):
        response = await self.client.get(self.url + path)
        response.raise_for_status()
        return response.json()

    async def respond(self, messages):
        items = messages if isinstance(messages, list) else []
        text = next((m['content'].strip() for m in reversed(items)
                     if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), str)), '')
        if not text or len(text) > 4000:
            return 'Ask for one complete robot task of at most 4000 characters. No task was submitted.'
        async with self.lock:
            if text == self.last_request:
                return 'That same request was already submitted to the dashboard. Check its existing draft or review; no duplicate task was submitted.'
            try:
                state = await self._get('/api/status')
                if not isinstance(state, dict):
                    raise ValueError('Invalid dashboard status')
                chat, pipeline = state.get('chat', {}), state.get('pipeline', {})
                if not isinstance(chat, dict) or not isinstance(pipeline, dict):
                    raise ValueError('Invalid dashboard status')
                if chat.get('busy'):
                    return 'The dashboard agent is still working. Wait for its reply; no new task was submitted.'
                if pipeline.get('proposal'):
                    return 'A motion is awaiting review in the dashboard or glasses. Finish that review first; no new task was submitted.'
                token = (await self._get('/api/session'))['token']
                if not isinstance(token, str) or not token:
                    raise ValueError('Invalid dashboard session')
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                return 'The local dashboard is unavailable. Start it and check the voice dashboard URL. No task was submitted.'
            # Never retry an uncertain POST: it may already have reached the agent.
            try:
                response = await self.client.post(self.url + '/api/chat',
                    json={'action': 'send', 'message': text}, headers={'X-Reins-Token': token})
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or not isinstance(result.get('messages'), list):
                    raise ValueError('Invalid dashboard response')
            except (httpx.HTTPError, ValueError, TypeError):
                return 'The dashboard did not confirm this request. Check its chat before trying again; voice did not authorize any motion.'
            self.last_request = text
            return ('The task was submitted to the Reins dashboard agent for planning. '
                    'Follow its answer and preview in the dashboard or glasses. '
                    'Any proposed motion still needs operator approval; voice has not authorized movement.')

    async def close(self):
        await self.client.aclose()
