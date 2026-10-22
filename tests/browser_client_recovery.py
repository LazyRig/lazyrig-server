"""Lost receipts and reloads between client-owned tool stages, over real HTTP."""
import copy
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from lazyrig import __version__
from lazyrig.server import handler
from lazyrig.worker import Cancelled
from support import DEFAULT, Service, FixtureWorker
from test_chat_image_tools import rewritten


def main():
    evidence = {'scope': 'Real HTTP/Chromium with gated deterministic workers', 'checks': []}
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT); config['data_dir'] = root
        service = Service(config)
        gates = {name: (threading.Event(), threading.Event()) for name in ('enhance', 'image', 'continue')}
        calls = []
        def factory(*args):
            worker = FixtureWorker(*args); original = worker.run
            def run(payload, timeout, cancel):
                if payload['action'] == 'image':
                    stage = 'image'
                elif payload.get('response_format', {}).get('type') == 'json_schema':
                    stage = 'enhance'
                elif payload['prepared']['messages'][-1]['role'] == 'tool':
                    stage = 'continue'
                else:
                    stage = 'chat'
                calls.append(stage)
                if stage in gates:
                    started, release = gates[stage]; started.set()
                    while not release.wait(.01):
                        if cancel.is_set(): raise Cancelled()
                if stage == 'chat':
                    text = '<tool_call>{"name":"generate_image","arguments":{"prompt":"A train"}}</tool_call>'
                elif stage == 'enhance':
                    text = rewritten()
                elif stage == 'continue':
                    text = 'The image is ready.'
                else:
                    yield from original(payload, timeout, cancel); return
                yield {'kind': 'result', 'text': text, 'generated_tokens': 100, 'accuracy': 'estimated-test'}
            worker.run = run; return worker
        service.scheduler.factory = factory
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(); context = browser.new_context()
                failures = []; seen_keys = set(); requests = []
                def lost_receipt(route):
                    key = route.request.headers['idempotency-key']; requests.append(key)
                    if key in seen_keys: route.continue_(); return
                    seen_keys.add(key)
                    response = route.fetch()
                    assert response.status == 202, response.text()
                    route.abort('failed')  # Accepted on server, acknowledgement lost.
                context.route('**/v1/**', lost_receipt)
                page = context.new_page(); page.on('pageerror', lambda e: failures.append(str(e)))
                base = 'http://127.0.0.1:' + str(server.server_port)
                page.goto(base); expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click(); page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-options [name="image_model"]').select_option('qwen-image')
                page.locator('#chat-options [name="enhance_prompt"]').check(); page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-prompt').fill('Draw a train.'); page.locator('#chat-send').click()
                expect(page.locator('.job-progress')).to_contain_text('Enhancing', timeout=15000)
                # Let browser HTTP callbacks run; a synchronous Event.wait would
                # block the Playwright route handler that admits this request.
                page.wait_for_function("document.querySelector('.job-progress').textContent.includes('Enhancing')")
                page.wait_for_timeout(700)
                assert gates['enhance'][0].is_set()
                page.reload(); expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                expect(page.locator('.job-progress')).to_contain_text('Enhancing')
                gates['enhance'][1].set()
                expect(page.locator('.job-progress')).to_contain_text('Generating image', timeout=15000)
                assert gates['image'][0].is_set()
                old_page = page; page = context.new_page(); page.on('pageerror', lambda e: failures.append(str(e)))
                page.goto(base); expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                expect(page.locator('#chat-stop')).to_be_visible()
                gates['image'][1].set()
                expect(page.locator('.message-images img')).to_have_count(1, timeout=15000)
                page.wait_for_timeout(700)
                assert gates['continue'][0].is_set()
                assert calls == ['chat', 'enhance', 'image', 'continue'], calls
                old_page.close()
                page.reload(); expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click(); gates['continue'][1].set()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                expect(page.locator('.message.assistant .message-body')).to_contain_text('The image is ready.')
                assert not page.locator('.job-error').count(), page.locator('.job-error').all_text_contents()
                assert calls == ['chat', 'enhance', 'image', 'continue'], calls
                assert len(service.store.jobs()) == 4
                assert len(seen_keys) == 4 and len(requests) >= 8, requests
                assert not failures, failures
                evidence['checks'].append('Lost acknowledgements for all four stages reuse their request IDs, yielding exactly four jobs and one image')
                evidence['checks'].append('Reload during enhancement, two tabs watching image completion, and reload during continuation resume without repeated work')
                evidence['browser'] = browser.version
                browser.close()
        finally:
            for _, release in gates.values(): release.set()
            server.shutdown(); server.server_close(); service.close()
    out = Path('test-results'); out.mkdir(exist_ok=True)
    (out/('browser-client-recovery-'+__version__+'.json')).write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()
