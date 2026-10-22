"""Exercise numeric reservation defaults through the real forms and HTTP API."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service
from test_server_contract import fixture_export
from test_image_edits import export_fixture
from test_image_settings import scheduler_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('test-results/memory-defaults'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture_export(root / 'export')
        export = export_fixture(root / 'image')
        scheduler_file(export)
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=str(root / 'data'), models_dir=str(root))
        config['devices'].append({'id': 'TEST.GPU.1', 'name': 'Second GPU with custom budget',
                                  'budget_gib': 19.8, 'policy': 'budget'})
        config['models'][1].update(path=str(export), runtime_profile='qwen-image-21',
                                   max_width=2752, max_height=2752,
                                   generation={'width': 2048, 'height': 2048, 'steps': 40})
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                for width, height in ((1440, 1000), (390, 844)):
                    context = browser.new_context(viewport={'width': width, 'height': height})
                    page = context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    def shot(name):
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                        page.evaluate('hideToast()')
                        page.screenshot(path=str(args.output_dir / f'{name}-{width}.png'))
                    page.goto(f'http://127.0.0.1:{http.server_port}')
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('#add-model').click()
                    form = page.locator('#register-form')
                    memory = form.locator('[name="memory_gib"]')
                    expect(memory).to_have_value('22')
                    shot('add-model-empty')
                    form.locator('[name="device"]').select_option('TEST.GPU.1')
                    expect(memory).to_have_value('19.8')
                    form.locator('[name="device"]').select_option('TEST.CPU')
                    expect(memory).to_have_value('48')
                    memory.fill('3.5')
                    page.locator('#register-path').fill(str(root / 'export'))
                    page.locator('#inspect-folder').click()
                    expect(page.locator('#inspection-result')).to_contain_text('Export structure checked')
                    expect(memory).to_have_value('3.5')
                    page.locator('#inspect-folder').scroll_into_view_if_needed()
                    shot('inspection-result')
                    alias = f'memory-{width}'
                    form.locator('[name="id"]').fill(alias)
                    form.locator('button[type="submit"]').click()
                    expect(page.locator('#register-dialog')).not_to_be_visible()
                    assert service.model(alias)['memory_gib'] == 3.5

                    page.locator(f'.model-name[data-edit="{alias}"]').click()
                    editor = page.locator('#model-form')
                    expect(editor.locator('[name="memory_gib"]')).to_have_value('3.5')
                    editor.locator('[name="device"]').select_option('TEST.GPU.1')
                    expect(editor.locator('[name="memory_gib"]')).to_have_value('19.8')
                    editor.locator('[name="memory_gib"]').fill('7')
                    editor.locator('[data-memory-default]').click()
                    expect(editor.locator('[name="memory_gib"]')).to_have_value('19.8')
                    shot('model-memory')
                    editor.locator('button[type="submit"]').click()
                    expect(page.locator('#toast')).to_contain_text('Settings saved')
                    assert service.model(alias)['memory_gib'] == 19.8
                    page.reload()
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator(f'.model-name[data-edit="{alias}"]').click()
                    expect(page.locator('#model-form [name="memory_gib"]')).to_have_value('19.8')
                    page.locator(f'[data-model-device="{alias}"]').select_option('TEST.CPU')
                    expect(page.locator('#toast')).to_contain_text('Device saved')
                    assert service.model(alias)['memory_gib'] == 48
                    expect(page.locator('#model-form [name="memory_gib"]')).to_have_value('48')

                    page.locator('nav [data-view="downloads"]').click()
                    imports = page.locator('#download-form')
                    expect(imports.locator('[name="memory_gib"]')).to_have_value('22')
                    imports.locator('[name="device"]').select_option('TEST.GPU.1')
                    expect(imports.locator('[name="memory_gib"]')).to_have_value('19.8')
                    imports.locator('[name="memory_gib"]').fill('4.5')
                    page.wait_for_function('!pendingRefresh')
                    page.evaluate('refresh()')
                    expect(imports.locator('[name="memory_gib"]')).to_have_value('4.5')
                    imports.locator('[data-memory-default]').click()
                    expect(imports.locator('[name="memory_gib"]')).to_have_value('19.8')
                    imports.locator('summary').click()
                    imports.locator('summary').scroll_into_view_if_needed()
                    shot('import-options')
                    submitted = []
                    def capture_import(route):
                        submitted.append(route.request.post_data_json)
                        route.fulfill(json={'id': 'fixture-import'})
                    page.route('**/api/downloads', capture_import)
                    imports.locator('[name="url"]').fill('https://huggingface.co/test/model')
                    imports.locator('[name="destination"]').fill(str(root / f'import-{width}'))
                    imports.locator('button[type="submit"]').click()
                    expect(page.locator('#toast')).to_contain_text('Model import started')
                    assert submitted[-1]['memory_gib'] == 19.8
                    assert submitted[-1]['device'] == 'TEST.GPU.1'

                    # Render other forms touched by the shared spacing rules.
                    page.locator('nav [data-view="models"]').click()
                    page.locator('[data-edit="qwen-image"]').last.click()
                    page.locator('#image-size-preset').scroll_into_view_if_needed()
                    shot('image-generation-options')
                    page.locator('#model-form summary').last.click()
                    page.locator('#model-form summary').last.scroll_into_view_if_needed()
                    shot('model-advanced-options')
                    page.locator('nav [data-view="server"]').click()
                    page.locator('#server-form summary').first.click()
                    page.locator('#server-form summary').first.scroll_into_view_if_needed()
                    shot('server-options')
                    page.locator('nav [data-view="devices"]').click()
                    shot('devices')
                    assert errors == [], errors
                    results.append({'viewport': [width, height], 'status': 'PASS', 'checks': [
                        'new registration uses device budget', 'GPU change uses custom device budget',
                        'CPU change uses its device budget', 'inspection preserves manual override',
                        'registration persists manual number', 'editor preserves saved reservation',
                        'editor device change refills value', 'Default restores device budget',
                        'saved default survives reload', 'table device switch updates reservation',
                        'imports use device budget', 'polling preserves manual import value',
                        'import request contains selected numeric budget'], 'javascript_errors': errors})
                    context.close()
                browser.close()
        finally:
            http.shutdown()
            http.server_close()
            thread.join(3)
            service.close()
    report = {'scope': 'Real HTTP/dashboard/Chromium; fixture exports, no native inference or real import',
              'scenarios': results}
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
