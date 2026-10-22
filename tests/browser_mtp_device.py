"""Real dashboard/API MTP settings check with fixture hardware, no inference."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright
from browser_device_placement import fixture_config
from lazyrig.server import handler
from support import Service


def scenario(browser, width, output):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = fixture_config(root)
        for suffix in ('xml', 'bin'):
            (root / 'export' / ('openvino_mtp_model.' + suffix)).write_text('fixture')
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        context = browser.new_context(viewport={'width': width, 'height': 900})
        context.route('**/*', lambda route: route.continue_() if urlsplit(route.request.url).hostname in {'127.0.0.1', 'localhost', '::1'} else route.abort())
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            page.goto(f'http://127.0.0.1:{http.server_port}')
            expect(page.locator('#connection')).to_have_text('Server online')
            for selected in ('CPU', 'GPU.0', 'GPU.2', ''):
                page.locator('.model-name[data-edit="qwen-chat"]').click()
                form = page.locator('#model-form')
                expect(form).to_be_visible()
                page.locator('[data-settings-tab="generation"]').click()
                choice = form.locator('[name="mtp_device"]')
                options = choice.locator('option').evaluate_all('(items)=>items.map(item=>item.value)')
                assert set(options) == {'', 'GPU.0', 'GPU.1', 'GPU.2', 'CPU'}
                expect(choice.locator('option[value="GPU.2"]')).to_contain_text('Experimental')
                expect(choice.locator('option[value="GPU.2"]')).not_to_be_disabled()
                form.locator('[name="mtp_enabled"]').select_option('true')
                choice.select_option(selected)
                choice.scroll_into_view_if_needed()
                page.evaluate('hideToast()')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                if selected == 'CPU':
                    page.screenshot(path=str(output / f'mtp-device-{width}.png'))
                with page.expect_response(lambda response: urlsplit(response.url).path == '/api/settings' and response.request.method == 'POST') as response:
                    form.locator('button[type="submit"]').click()
                assert response.value.ok, response.value.text()
                expect(page.locator('#toast')).to_contain_text('Settings saved')
                page.wait_for_function('()=>!pendingRefresh')
                saved = json.loads(service.settings_path.read_text())['models'][0]
                assert saved['mtp'].get('device', '') == selected
                assert saved['mtp']['enabled'] is True
                assert saved['device'] == 'GPU.1'
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('.model-name[data-edit="qwen-chat"]').click()
                expect(page.locator('[name="mtp_device"]')).to_have_value(selected)
                page.locator('#close-editor').click()
            assert errors == [], errors
            return {'width': width, 'status': 'PASS', 'saved_and_reopened': ['CPU', 'GPU.0', 'GPU.2 (Experimental)', 'Same as model'], 'javascript_errors': errors}
        finally:
            context.close()
            http.shutdown()
            http.server_close()
            thread.join(3)
            service.close()


if __name__ == '__main__':
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            results = [scenario(browser, width, output) for width in (1440, 390)]
        finally:
            browser.close()
    report = {'scope': 'HTTP/Chromium with fixture hardware; no native GPU inference', 'scenarios': results}
    (output / 'mtp-device-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
