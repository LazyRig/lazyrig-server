"""Check readable chat device names and selection-row alignment in Chromium."""
import copy
import json
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service


def main():
    evidence = {'scope': 'Real Chromium and HTTP with deterministic workers; no native GPU test', 'checks': []}
    out = Path(os.environ.get('DEVICE_QA_OUTPUT', 'test-results/device-label'))
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = temp
        config['devices'][0].update(id='GPU.1', name='Configured GPU')
        for model in config['models']:
            model['device'] = 'GPU.1'
        config['models'][0]['id'] = 'Qwen3.8-27B-int4-ov'
        config['models'].append({**copy.deepcopy(config['models'][0]), 'id': 'CPU-chat', 'device': 'TEST.CPU'})
        service = Service(config)
        service.detected = copy.deepcopy(service.detected)
        service.detected['devices'][0]['name'] = 'Intel(R) Arc(TM) A770 Graphics'
        service.detected['devices'] = service.detected['devices'][:1]
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                options = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    options['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = p.chromium.launch(**options)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto('http://127.0.0.1:' + str(server.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                state = page.locator('#chat-model-state')
                expect(state).to_have_text('unloaded · GPU.1 · Intel(R) Arc(TM) A770 Graphics')
                a, b = state.bounding_box(), page.locator('#chat-model').bounding_box()
                assert abs(a['y'] + a['height']/2 - b['y'] - b['height']/2) < 1, (a, b)
                evidence['checks'].append('Detected full hardware name and device ID align with the selected-model value')
                page.locator('#chat-model').select_option('CPU-chat')
                assert state.inner_text() == 'unloaded · TEST.CPU · Test CPU'
                evidence['checks'].append('Changing models updates device identity; configured name is a fallback')
                page.locator('#chat-model').select_option('Qwen3.8-27B-int4-ov')
                page.locator('#chat-prompt').fill('A new outfit concept')
                page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                expect(page.locator('.message.assistant')).to_be_visible()
                expect(state).to_have_text('loaded · GPU.1 · Intel(R) Arc(TM) A770 Graphics')
                page.locator('#chat-history-toggle').click()
                page.screenshot(path=str(out/'desktop-history.png'))
                page.locator('#chat-folder-new').click()
                page.locator('#chat-folder-form [name="folder_name"]').fill('Outfit ideas')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                page.locator('[data-thread-menu]').first.click()
                page.locator('#chat-move').click()
                folder = page.locator('#chat-folder-form [name="folder_id"] option').nth(1).get_attribute('value')
                page.locator('#chat-folder-form [name="folder_id"]').select_option(folder)
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                page.locator('[data-thread-menu]').first.click()
                page.screenshot(path=str(out/'desktop-folder-menu.png'))
                page.keyboard.press('Escape')
                page.locator('#chat-history-toggle').click()
                for width in (1100, 820, 390):
                    page.set_viewport_size({'width': width, 'height': 844})
                    expect(state).to_be_visible()
                    box = state.bounding_box()
                    assert box['x'] >= 0 and box['x'] + box['width'] <= width, box
                    assert state.evaluate('e => e.scrollWidth <= e.clientWidth'), width
                    assert page.locator('.chat-toolbar').evaluate('e => e.scrollWidth <= e.clientWidth'), width
                    page.screenshot(path=str(out/f'chat-device-{width}.png'))
                evidence['checks'].append('Full device label stays visible and contained at 1100, 820 and 390px widths')
                assert not errors, errors
                evidence['checks'].append('No JavaScript errors during selection, generation and folder actions')
                browser.close()
        finally:
            server.shutdown()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
