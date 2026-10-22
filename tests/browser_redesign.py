"""Exercise redesigned dialogs, saved settings and chat against the real HTTP API.

Hardware and inference are deterministic fixtures, never a native GPU validation.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from browser_device_placement import fixture_config
from lazyrig.server import handler
from support import Service


def scenario(browser, width, height, output):
    with tempfile.TemporaryDirectory() as directory:
        service = Service(fixture_config(Path(directory)))
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        context = browser.new_context(viewport={'width': width, 'height': height}, reduced_motion='reduce')
        context.route('**/*', lambda r: r.continue_() if urlsplit(r.request.url).hostname in {'127.0.0.1', 'localhost', '::1'} else r.abort())
        page = context.new_page()
        errors, posts, checks = [], [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
        page.on('request', lambda request: posts.append(request.post_data_json) if urlsplit(request.url).path == '/api/settings' and request.method == 'POST' else None)

        def shot(name):
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Horizontal overflow'
            page.evaluate('hideToast()')
            page.screenshot(path=str(output / f'{name}-{width}.png'))

        def edit(mid='qwen-chat', tab='general'):
            page.locator(f'.model-name[data-edit="{mid}"]').click()
            expect(page.locator('#model-editor')).to_be_visible()
            page.locator(f'[data-settings-tab="{tab}"]').click()
            return page.locator('#model-form')

        def save(selector, endpoint='/api/settings'):
            with page.expect_response(lambda r: urlsplit(r.url).path == endpoint and r.request.method == 'POST') as response:
                page.locator(selector).click()
            assert response.value.ok, response.value.text()
            page.wait_for_function('()=>!pendingRefresh')

        try:
            page.goto(f'http://127.0.0.1:{http.server_port}')
            expect(page.locator('#connection')).to_have_text('Server online')
            expect(page.locator('#models-table tr:visible')).to_have_count(3)
            page.locator('[data-model-filter="text-to-image"]').click()
            expect(page.locator('#models-table tr:visible')).to_have_count(1)
            page.locator('#model-search').fill('missing model')
            expect(page.locator('#model-no-matches')).to_be_visible()
            page.locator('#model-search').fill('')
            page.locator('[data-model-filter="all"]').click()
            shot('models')
            if width > 600:
                page.locator('[data-inspect-model="qwen-chat"]').click()
                expect(page.locator('#model-inspection-content')).to_contain_text('Export structure checked')
                page.keyboard.press('Escape')
                assert service.model('qwen-chat')['name'] == 'Qwen3.8 · 27B'
            checks.append('Model filters and search preserve the registered models')

            form = edit()
            assert page.locator('#model-editor').evaluate("el=>el.matches(':modal')")
            assert page.locator('#model-editor').evaluate("el=>getComputedStyle(el,'::backdrop').backdropFilter") == 'blur(9px)'
            form.locator('[name="name"]').fill('Unsaved name')
            count = len(posts)
            page.locator('#close-editor').click()
            assert len(posts) == count
            assert service.model('qwen-chat')['name'] != 'Unsaved name'
            form = edit()
            shot('model-settings')
            # Native validation must reveal a field on a non-selected tab.
            page.locator('[data-settings-tab="devices"]').click()
            form.locator('[name="memory_gib"]').fill('')
            page.locator('[data-settings-tab="general"]').click()
            form.locator('button[type="submit"]').click()
            expect(form.locator('[name="memory_gib"]')).to_be_visible()
            expect(form.locator('[name="memory_gib"]')).to_be_focused()
            assert len(posts) == count
            form.locator('[name="memory_gib"]').fill('20')
            # JSON validation likewise exposes the relevant field without closing.
            page.locator('[data-settings-tab="advanced"]').click()
            form.locator('[name="plugin_config"]').fill('{bad json')
            page.locator('[data-settings-tab="general"]').click()
            form.locator('button[type="submit"]').click()
            assert form.locator('[name="plugin_config"]').evaluate('el=>!!el.validationMessage')
            expect(form.locator('[name="plugin_config"]')).to_be_focused()
            assert len(posts) == count
            form.locator('[name="plugin_config"]').fill('{"PERFORMANCE_HINT":"LATENCY"}')
            form.locator('[name="template_context"]').fill('{bad json')
            page.locator('[data-settings-tab="general"]').click()
            form.locator('button[type="submit"]').click()
            expect(page.locator('#model-editor #toast')).to_contain_text('Invalid JSON')
            expect(form.locator('[name="template_context"]')).to_be_focused()
            form.locator('[name="template_context"]').fill('{}')
            page.evaluate('hideToast()')
            page.locator('[data-settings-tab="generation"]').click()
            form.locator('[name="gen_max_tokens"]').fill('4096')
            page.locator('[data-settings-tab="runtime"]').click()
            form.locator('[name="runtime_EXECUTION_MODE_HINT"]').select_option('ACCURACY')
            page.locator('[data-settings-tab="general"]').click()
            form.locator('[name="name"]').fill('Saved model name')
            # Nested folder picker must keep this dialog and its draft intact.
            form.locator('[data-browse="edit-path"]').click()
            expect(page.locator('#folder-dialog')).to_be_visible()
            page.locator('#folder-select').click()
            expect(page.locator('#model-editor')).to_be_visible()
            expect(form.locator('[name="name"]')).to_have_value('Saved model name')
            save('#model-form button[type="submit"]')
            expect(page.locator('#model-editor')).not_to_be_visible()
            saved = service.model('qwen-chat')
            assert saved['name'] == 'Saved model name'
            assert saved['generation']['max_tokens'] == 4096
            assert saved['plugin_config']['EXECUTION_MODE_HINT'] == 'ACCURACY'
            checks.append('Modal cancel, hidden-field validation, JSON errors, nested folder picker and real settings save')

            form = edit(tab='runtime')
            expect(form.locator('[name="runtime_EXECUTION_MODE_HINT"]')).to_have_value('ACCURACY')
            page.keyboard.press('Escape')
            expect(page.locator('#model-editor')).not_to_be_visible()
            page.locator('nav [data-view="devices"]').click()
            shot('devices')
            page.locator('[data-edit-device="CPU"]').click()
            form = page.locator('#device-settings-dialog form')
            form.locator('[name="inference_threads"]').fill('4')
            save('#device-settings-dialog button.primary')
            expect(page.locator('#device-settings-dialog')).not_to_be_visible()
            assert next(d for d in service.config['devices'] if d['id'] == 'CPU')['inference_threads'] == 4
            checks.append('Device dialog saves CPU thread settings through the server API')

            page.locator('nav [data-view="server"]').click()
            page.locator('#server-settings-open').click()
            expect(page.locator('#server-settings-dialog')).to_be_visible()
            page.locator('#server-api-key').fill('unsaved-demo-key')
            page.locator('#server-settings-dialog .close-dialog').first.click()
            page.locator('#server-settings-open').click()
            expect(page.locator('#server-api-key')).to_have_value('')
            page.locator('#server-form [name="request_timeout_seconds"]').fill('2400')
            save('#server-form button[type="submit"]', '/api/server')
            expect(page.locator('#server-settings-dialog')).not_to_be_visible()
            assert service.config['request_timeout_seconds'] == 2400
            checks.append('Server settings dialog saves and closes')

            page.locator('nav [data-view="chat"]').click()
            page.locator('#chat-settings-toggle').click()
            expect(page.locator('#chat-settings-dialog')).to_be_visible()
            page.locator('#chat-options [name="temperature"]').fill('0.7')
            page.locator('#chat-options [name="system_prompt"]').fill('Be concise')
            page.locator('#done-chat-settings').click()
            page.locator('#chat-settings-toggle').click()
            expect(page.locator('#chat-options [name="system_prompt"]')).to_have_value('Be concise')
            shot('conversation-settings')
            page.keyboard.press('Escape')
            expect(page.locator('#chat-options')).to_be_hidden()
            expect(page.locator('#chat-settings-toggle')).to_have_attribute('aria-expanded', 'false')
            page.locator('#chat-image-tool').click()
            expect(page.locator('#chat-image-tool')).to_have_attribute('aria-pressed', 'false')
            page.locator('#chat-image-tool').click()
            expect(page.locator('#chat-image-tool')).to_have_attribute('aria-pressed', 'true')
            page.locator('#chat-prompt').fill('Hello, please answer briefly.')
            page.locator('#chat-send').click()
            expect(page.locator('.message.assistant .message-body')).to_contain_text('Test worker response', timeout=15000)
            expect(page.locator('#chat-stop')).to_be_hidden()
            shot('chat')
            page.reload()
            expect(page.locator('#chat-view')).to_be_visible()
            expect(page.locator('.message.assistant .message-body')).to_contain_text('Test worker response')
            page.locator('#chat-settings-toggle').click()
            expect(page.locator('#chat-options [name="system_prompt"]')).to_have_value('Be concise')
            checks.append('Conversation settings autosave, image tool toggle, actual fixture generation and refresh recovery')
            assert not errors, errors
            return {'viewport': [width, height], 'status': 'PASS', 'checks': checks, 'browser_errors': errors}
        except Exception:
            page.screenshot(path=str(output / f'failure-{width}.png'), full_page=True)
            raise
        finally:
            context.close(); http.shutdown(); http.server_close(); thread.join(3); service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('test-results/redesign'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            scenarios = [scenario(browser, w, h, args.output_dir) for w, h in ((1440, 1000), (390, 844))]
        finally:
            browser.close()
    result = {'scope': 'Real HTTP/dashboard/Chromium with fixture hardware and inference', 'scenarios': scenarios}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
