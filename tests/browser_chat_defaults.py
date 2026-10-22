"""Dashboard defaults and preference persistence through real HTTP and Chromium."""
import copy
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from lazyrig.server import handler
from support import DEFAULT, Service


def main():
    with tempfile.TemporaryDirectory() as directory:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = directory
        config['models'].append({**copy.deepcopy(config['models'][1]), 'id': 'other-image'})
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        checks = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto('http://127.0.0.1:' + str(http.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('[data-edit="qwen-chat"]').first.click()
                expect(page.locator('[name="gen_max_tokens"]')).to_have_value('8192')
                expect(page.locator('[name="default_image_tool"]')).to_have_count(0)
                page.locator('#close-editor').click()

                def close_settings():
                    if page.locator('#chat-settings-dialog').is_visible():
                        page.locator('#done-chat-settings').click()
                    if page.locator('#model-editor').is_visible():
                        page.locator('#close-editor').click()

                def chat(new=False):
                    close_settings()
                    page.locator('nav [data-view="chat"]').click()
                    if new:
                        page.locator('#chat-new').click()
                    if page.locator('#chat-options').is_hidden():
                        page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    return page.locator('#chat-options [name="image_model"]')

                def model_settings(mid):
                    close_settings()
                    page.locator('nav [data-view="models"]').click()
                    if not page.locator('#model-editor').is_visible() or not page.locator('#model-editor h2').first.inner_text().startswith(mid+' ·'):
                        page.locator('[data-edit="'+mid+'"]').first.click()
                    expect(page.locator('#model-editor')).to_be_visible()

                def save_settings():
                    with page.expect_response(lambda r: r.url.endswith('/api/settings') and r.request.method == 'POST') as response:
                        page.locator('#model-form button[type="submit"]').click()
                    assert response.value.ok, response.value.text()
                    expected = [{'id': m['id'], 'enabled': m.get('enabled', True),
                                 'preferred': m.get('default_image_tool', False)} for m in response.value.json()['models']]
                    page.wait_for_function('''models => models.every(expected => {
                        const actual = state.models.find(m => m.id === expected.id);
                        return actual && (actual.enabled !== false) === expected.enabled &&
                            !!actual.default_image_tool === expected.preferred;
                    })''', arg=expected)

                tool = chat()
                expect(tool).to_have_value('qwen-image')
                expect(page.locator('#chat-options [name="max_tokens"]')).to_have_attribute('placeholder', '8192')
                tool.select_option('')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                expect(chat()).to_have_value('')
                expect(chat(new=True)).to_have_value('qwen-image')
                checks.append('First enabled image model is automatic for new chats; explicit Disabled survives reload')

                model_settings('other-image')
                page.get_by_label('Use as default image tool', exact=True).check()
                save_settings()
                expect(page.get_by_label('Use as default image tool', exact=True)).to_be_checked()
                expect(chat()).to_have_value('qwen-image')
                expect(chat(new=True)).to_have_value('other-image')
                checks.append('Model checkbox saves the preference; new chats use it while existing chats keep their selection')

                model_settings('qwen-image')
                page.get_by_label('Use as default image tool', exact=True).check()
                save_settings()
                model_settings('other-image')
                expect(page.get_by_label('Use as default image tool', exact=True)).not_to_be_checked()
                expect(chat(new=True)).to_have_value('qwen-image')
                checks.append('Choosing a different default clears the previous checkbox')

                model_settings('qwen-image')
                page.get_by_label('Expose this model through the API', exact=True).uncheck()
                save_settings()
                expect(chat(new=True)).to_have_value('other-image')
                model_settings('other-image')
                page.get_by_label('Expose this model through the API', exact=True).uncheck()
                save_settings()
                expect(chat(new=True)).to_have_value('')
                checks.append('A disabled preferred model falls back to another enabled image model; no enabled models leaves the tool disabled')

                model_settings('other-image')
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                expect(page.get_by_label('Use as default image tool', exact=True)).to_be_visible()
                out = Path('test-results'); out.mkdir(exist_ok=True)
                page.screenshot(path=str(out/'chat-defaults-mobile.png'))
                page.set_viewport_size({'width': 1440, 'height': 1000})
                page.screenshot(path=str(out/'chat-defaults-desktop.png'))
                assert not errors, errors
                checks.append('Default checkbox wraps on mobile without horizontal overflow; no JavaScript errors')
                (out/'chat-defaults.json').write_text(json.dumps({'checks': checks, 'browser': browser.version}, indent=2))
                browser.close()
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()
        print(json.dumps({'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
