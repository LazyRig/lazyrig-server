"""Exercise multi-device controls with real dashboard/HTTP persistence and fake hardware."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from lazyrig.device_info import support_metadata
from support import DEFAULT, Service
from test_core import wait_job
from test_server_contract import fixture_export
from test_image_edits import export_fixture
from test_image_settings import scheduler_file


def fixture_config(root):
    fixture_export(root / 'export')
    image = export_fixture(root / 'image')
    scheduler_file(image)
    config = copy.deepcopy(DEFAULT)
    config.update(data_dir=str(root / 'data'), models_dir=str(root), terminal_metrics=False)
    config['devices'] = [
        {'id': 'GPU.0', 'name': 'Intel integrated graphics', 'device_type': 'integrated', 'budget_gib': 8},
        {'id': 'GPU.1', 'name': 'Intel Arc Pro B60 discrete graphics', 'device_type': 'discrete', 'budget_gib': 22},
        {'id': 'GPU.2', 'name': 'NVIDIA RTX 5060 Ti', 'device_type': 'discrete', 'budget_gib': 14},
        {'id': 'CPU', 'name': 'Intel Core CPU and system memory', 'device_type': 'cpu', 'budget_gib': 48},
        {'id': 'NPU', 'name': 'Neural processing unit', 'device_type': 'npu', 'budget_gib': 8},
    ]
    for device in config['devices']:
        device.update(detected=True, policy='budget', **support_metadata(device['id'], device['name']))
    for model in config['models']:
        model['device'] = 'GPU.1'
    config['models'][0].update(path=str(root / 'export'), plugin_config={'PERFORMANCE_HINT': 'LATENCY'})
    config['models'][1].update(path=str(image), runtime_profile='qwen-image-21')
    custom = copy.deepcopy(config['models'][0])
    custom.update(id='custom-chat', name='Custom placement properties',
                  plugin_config={'PERFORMANCE_HINT': 'LATENCY', 'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .5}}})
    config['models'].append(custom)
    return config


def scenario(browser, width, height, output_dir):
    with tempfile.TemporaryDirectory() as directory:
        service = Service(fixture_config(Path(directory)))
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        context = browser.new_context(viewport={'width': width, 'height': height})
        # This suite only exercises the local fixture server.
        context.route('**/*', lambda route: route.continue_() if urlsplit(route.request.url).hostname in {'127.0.0.1', 'localhost', '::1'} else route.abort())
        page = context.new_page()
        errors, posts, checks = [], [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: posts.append(request.post_data_json) if urlsplit(request.url).path == '/api/settings' and request.method == 'POST' else None)

        def shot(name):
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Page overflows horizontally'
            page.evaluate('hideToast()')
            page.screenshot(path=str(output_dir / f'{name}-{width}.png'))

        def table_picker(mid='qwen-chat'):
            return page.locator(f'[data-model-device="{mid}"] [data-device-picker]')

        def open_picker(picker):
            picker.locator('summary').scroll_into_view_if_needed()
            page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
            picker.locator('summary').click()
            expect(picker).to_have_attribute('open', '')
            expect(picker.locator('[data-placement-apply]')).to_be_visible()
            # <details> queues its toggle event; let draft initialization finish
            # before select_option dispatches the next synthetic change event.
            page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')

        def choice(picker, device):
            return picker.locator(f'[data-placement-device][value="{device}"]')

        def editor(mid='qwen-chat'):
            page.locator(f'.model-name[data-edit="{mid}"]').click()
            expect(page.locator('#model-form')).to_be_visible()
            return page.locator('#model-form')

        def save_form(form):
            with page.expect_response(lambda response: urlsplit(response.url).path == '/api/settings' and response.request.method == 'POST') as result:
                form.locator('button[type="submit"]').click()
            assert result.value.ok, result.value.text()
            expect(page.locator('#toast')).to_contain_text('Settings saved')
            page.wait_for_function('()=>!pendingRefresh')

        try:
            page.goto(f'http://127.0.0.1:{http.server_port}')
            expect(page.locator('#connection')).to_have_text('Server online')
            picker = table_picker()
            expect(picker.locator('[data-device-caption]')).to_have_text('Auto · GPU.1')
            open_picker(picker)
            expect(choice(picker, 'GPU.1')).to_be_checked()
            expect(choice(picker, 'CPU')).not_to_be_checked()
            expect(choice(picker, 'GPU.2')).not_to_be_checked()
            expect(picker.locator('[data-placement-primary] option[value="GPU.2"]')).to_contain_text('Experimental')
            expect(picker.locator('[data-placement-primary] option[value="GPU.2"]')).not_to_be_disabled()
            choice(picker, 'GPU.2').check()
            choice(picker, 'CPU').check()
            picker.locator('[data-placement-apply]').click()
            page.wait_for_function('()=>!pendingRefresh')
            expected = {'mode': 'auto', 'devices': ['GPU.1', 'GPU.2', 'CPU'], 'shares': {}, 'allow_fallback': False, 'shared_memory': 'budget'}
            assert service.model('qwen-chat')['placement'] == expected
            checks.append('One device by default; experimental GPU and CPU remain selectable for automatic allocation')

            form = editor()
            page.locator('[data-settings-tab="devices"]').click()
            expect(form.locator('[name="placement_mode"], [data-share-range]')).to_have_count(0)
            expect(form.locator('[data-placement-auto]')).to_contain_text('Try the primary device first')
            page.locator('[data-settings-tab="runtime"]').click()
            form.locator('[name="runtime_EXECUTION_MODE_HINT"]').select_option('ACCURACY')
            form.locator('[name="runtime_COMPILATION_NUM_THREADS"]').fill('2')
            form.locator('summary', has_text='CPU scheduling').click()
            form.locator('[name="runtime_SCHEDULING_CORE_TYPE"]').select_option('ANY_CORE')
            form.locator('[name="runtime_ENABLE_HYPER_THREADING"]').select_option('false')
            form.locator('[name="runtime_ENABLE_CPU_PINNING"]').select_option('false')
            shot('automatic-runtime-settings')
            save_form(form)
            plugin = service.model('qwen-chat')['plugin_config']
            assert plugin['EXECUTION_MODE_HINT'] == 'ACCURACY'
            assert plugin['COMPILATION_NUM_THREADS'] == 2
            assert plugin['ENABLE_HYPER_THREADING'] is False
            assert plugin['ENABLE_CPU_PINNING'] is False
            assert plugin['SCHEDULING_CORE_TYPE'] == 'ANY_CORE'
            assert service.model('qwen-chat')['placement'] == expected
            checks.append('Runtime controls persist typed values without restoring manual sliders or percentages')

            form = editor()
            page.locator('[data-settings-tab="devices"]').click()
            picker = form.locator('[data-device-picker]')
            open_picker(picker)
            picker.locator('[data-placement-primary]').select_option('GPU.2')
            expect(picker.locator('[data-placement-note]')).to_contain_text('Experimental')
            picker.locator('[data-placement-apply]').click()
            expect(form.locator('[name="memory_gib"]')).to_have_value('14')
            form.locator('[name="memory_gib"]').fill('9')
            open_picker(picker)
            choice(picker, 'CPU').uncheck()
            picker.locator('[data-placement-apply]').click()
            expect(form.locator('[name="memory_gib"]')).to_have_value('9')
            save_form(form)
            expected = {'mode': 'auto', 'devices': ['GPU.2', 'GPU.1'], 'shares': {}, 'allow_fallback': False, 'shared_memory': 'budget'}
            assert service.model('qwen-chat')['placement'] == expected
            page.reload()
            expect(page.locator('#connection')).to_have_text('Server online')
            form = editor()
            expect(form.locator('[name="device"]')).to_have_value('GPU.2')
            expect(form.locator('[name="runtime_EXECUTION_MODE_HINT"]')).to_have_value('ACCURACY')
            assert service.model('qwen-chat')['placement'] == expected
            page.locator('#close-editor').click()
            checks.append('Experimental primary, explicit selection, reservation and runtime settings survive reload')

            page.locator('#add-model').click()
            expect(page.locator('#register-device')).to_have_value('GPU.1')
            expect(page.locator('#register-form [name="memory_gib"]')).to_have_value('22')
            page.locator('#register-device').select_option('GPU.2')
            expect(page.locator('#register-form [name="memory_gib"]')).to_have_value('14')
            page.locator('#register-dialog .close-dialog').click()
            page.locator('nav [data-view="downloads"]').click()
            expect(page.locator('#download-device')).to_have_value('GPU.1')
            page.locator('#download-device').select_option('GPU.2')
            page.evaluate('renderDownloads()')
            expect(page.locator('#download-device')).to_have_value('GPU.2')
            checks.append('Registration and download prefer Intel GPU; explicit experimental choices persist')

            page.locator('nav [data-view="devices"]').click()
            page.locator('[data-edit-device="GPU.1"]').click()
            expect(page.locator('#device-settings-dialog .device-support')).to_contain_text('Supported target')
            page.keyboard.press('Escape')
            page.locator('[data-edit-device="GPU.2"]').click()
            expect(page.locator('#device-settings-dialog .device-support')).to_contain_text('Experimental')
            page.keyboard.press('Escape')
            shot('device-support')
            checks.append('Support labels render on device cards without blocking controls')
            assert errors == [], errors
            assert all(body.get('placement', {}).get('mode', 'auto') == 'auto' for body in posts)
            return {'viewport': [width, height], 'status': 'PASS', 'checks': checks, 'javascript_errors': errors}
        except Exception:
            page.screenshot(path=str(output_dir / f'failure-{width}.png'), full_page=True)
            raise
        finally:
            context.close()
            http.shutdown()
            http.server_close()
            thread.join(3)
            service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('test-results/device-placement'))
    argv = sys.argv[1:]
    # The local offline runner leaves the target script in argv.
    if argv and Path(argv[0]).name == Path(__file__).name:
        argv = argv[1:]
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            results = [scenario(browser, width, height, args.output_dir) for width, height in ((1440, 1000), (390, 844))]
        finally:
            browser.close()
    report = {'scope': 'Real HTTP/dashboard/Chromium; fixture hardware and worker, no native GPU inference', 'scenarios': results}
    (args.output_dir / 'device-placement-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
