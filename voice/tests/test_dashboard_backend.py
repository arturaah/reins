"""Scripted loopback HTTP only: no model, robot or speaker is started."""
import asyncio
import json

import httpx
import pytest

from voice.dashboard_backend import DashboardBackend, dashboard_url


@pytest.mark.parametrize('url', ['https://127.0.0.1', 'http://robot.local', 'http://localhost/path',
    'http://localhost@evil.test', 'http://127.0.0.1/?x=1', 'http://localhost:99999'])
def test_adapter_cannot_target_remote_or_redirected_dashboard(url):
    with pytest.raises(ValueError): dashboard_url(url)


def fixture(state=None, post_code=200):
    calls = []
    def serve(request):
        calls.append(request)
        if request.url.path == '/api/status':
            return httpx.Response(200, json=state or {'chat': {'busy': False}, 'pipeline': {'proposal': None}})
        if request.url.path == '/api/session':
            return httpx.Response(200, json={'token': 'test-private-token'})
        assert request.url.path == '/api/chat'
        assert request.method == 'POST'
        assert request.headers['X-Reins-Token'] == 'test-private-token'
        assert json.loads(request.content) == {'action': 'send', 'message': 'Blow a kiss'}
        return httpx.Response(post_code, json={'messages': [{'role': 'user', 'text': 'Blow a kiss'}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(serve), follow_redirects=False)
    return DashboardBackend('http://127.0.0.1:8090', client=client), calls


def test_submission_enters_only_chat_and_duplicate_is_not_repeated():
    async def run():
        backend, calls = fixture()
        messages = [{'role': 'user', 'content': 'Blow a kiss'}]
        reply = await backend.respond(messages)
        assert 'submitted' in reply and 'operator approval' in reply
        assert 'private' not in reply
        assert 'already submitted' in await backend.respond(messages)
        assert [call.url.path for call in calls] == ['/api/status', '/api/session', '/api/chat']
        await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('state', [{'chat': {'busy': True}}, {'pipeline': {'proposal': {'id': 'pending'}}}])
def test_existing_work_or_review_prevents_voice_submission(state):
    async def run():
        backend, calls = fixture(state)
        assert 'no new task was submitted' in (await backend.respond([{'role': 'user', 'content': 'Blow a kiss'}])).lower()
        assert len(calls) == 1
        await backend.close()
    asyncio.run(run())


def test_uncertain_submission_is_never_retried_or_claimed_successful():
    async def run():
        backend, calls = fixture(post_code=502)
        reply = await backend.respond([{'role': 'user', 'content': 'Blow a kiss'}])
        assert 'did not confirm' in reply
        assert len([call for call in calls if call.method == 'POST']) == 1
        await backend.close()
    asyncio.run(run())

@pytest.mark.parametrize('state', [[], 'invalid', {'chat': []}, {'pipeline': 'invalid'}])
def test_malformed_status_never_submits_a_task(state):
    async def run():
        calls = []
        def serve(request):
            calls.append(request)
            return httpx.Response(200, json=state)
        backend = DashboardBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(serve)))
        assert 'unavailable' in await backend.respond([{'role': 'user', 'content': 'Blow a kiss'}])
        assert all(call.method == 'GET' for call in calls)
        await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('messages', [None, {}, [], [{'role': 'user', 'content': 42}], ['hello']])
def test_invalid_transcript_is_not_submitted(messages):
    async def run():
        backend, calls = fixture()
        assert 'No task was submitted' in await backend.respond(messages)
        assert calls == []
        await backend.close()
    asyncio.run(run())
