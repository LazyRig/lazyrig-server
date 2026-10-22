"""Visible one-shot image recovery, reload continuity and failed JSON diagnostics."""
import json
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from test_gpu_recovery import GPURecoveryTests


def main():
    test = GPURecoveryTests('test_generation_recovers_once_preserves_seed_and_retains_healthy_worker')
    test.setUp()
    test.workers.reload_gate = threading.Event()
    evidence = {'scope': 'Real HTTP/Chromium with injected native errors; no physical GPU', 'checks': []}
    try:
        with sync_playwright() as p:
            launch = {'headless': True}
            if os.environ.get('CHROMIUM_EXECUTABLE'):
                launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
            browser = p.chromium.launch(**launch)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors, posts, responses = [], [], []
            page.on('response', lambda response: responses.append((response.status, response.url)))
            page.route('**/*', lambda route: route.continue_() if route.request.url.startswith(test.base + '/') else route.abort())
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: posts.append(request.url)
                    if request.method == 'POST' and '/v1/images/' in request.url else None)
            page.goto(test.base)
            expect(page.locator('#connection')).to_have_text('Server online')
            page.locator('nav [data-view="chat"]').click()
            page.locator('#chat-task').select_option('image')
            page.locator('#chat-prompt').fill('Draw a lake.')
            page.locator('#chat-send').click()
            try:
                expect(page.locator('.image-recovery')).to_have_text('GPU recovery · automatic retry 1/1', timeout=15000)
            except Exception:
                print(page.locator('body').inner_text()[-4000:])
                print(json.dumps(responses, indent=2))
                raise
            expect(page.locator('.job-progress progress')).to_have_count(0)
            assert test.workers.reloading.is_set()
            assert len(posts) == 1 and len(test.workers.attempts) == 1
            page.reload()
            expect(page.locator('#connection')).to_have_text('Server online')
            page.locator('nav [data-view="chat"]').click()
            expect(page.locator('.image-recovery')).to_have_text('GPU recovery · automatic retry 1/1')
            expect(page.locator('.job-progress progress')).to_have_count(0)
            test.workers.reload_gate.set()
            expect(page.locator('.image-recovery')).to_have_text('Recovered from a GPU resource error · retried once', timeout=15000)
            expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
            expect(page.locator('.message.assistant .message-images img')).to_have_count(1)
            expect(page.locator('.job-error')).to_have_count(0)
            assert len(posts) == 1 and len(test.workers.attempts) == 2
            assert test.workers.attempts[0] == test.workers.attempts[1]
            evidence['checks'].append('One HTTP submission; recovery notice and reset progress survive reload; one final image after two identical worker attempts')
            page.reload()
            expect(page.locator('#connection')).to_have_text('Server online')
            page.locator('nav [data-view="chat"]').click()
            expect(page.locator('.image-recovery')).to_contain_text('Recovered from a GPU resource error')
            evidence['checks'].append('Successful recovery remains visible in saved conversation history')
            test.workers.failures = 100
            page.locator('#chat-prompt').fill('Another lake.')
            page.locator('#chat-send').click()
            expect(page.locator('.job-error')).to_contain_text('CL_OUT_OF_RESOURCES', timeout=15000)
            expect(page.locator('.image-recovery').last).to_have_text('Automatic GPU recovery did not complete (1/1)')
            expect(page.locator('#chat-stop')).not_to_be_visible()
            assert len(posts) == 2 and len(test.workers.attempts) == 4
            with page.expect_download() as download:
                page.locator('[data-result]').last.click()
            payload = json.loads(Path(download.value.path()).read_text())
            assert payload['status'] == 'failed'
            assert 'CL_OUT_OF_RESOURCES' in payload['error']
            assert payload['image_recovery']['status'] == 'failed'
            assert payload['image_recovery']['retry_count'] == 1
            assert not payload['artifacts']
            evidence['checks'].append('A repeated failure ends after two attempts; Response JSON retains status, native error and recovery diagnostic')
            assert not errors, errors
            evidence['browser'] = browser.version
            evidence['javascript_errors'] = errors
            browser.close()
    except Exception:
        print(json.dumps({'errors': errors, 'posts': posts, 'jobs': test.service.store.jobs(), 'attempts': len(test.workers.attempts)}, default=str, indent=2))
        raise
    finally:
        test.tearDown()
    out = Path('test-results'); out.mkdir(exist_ok=True)
    (out / 'browser-gpu-recovery.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
