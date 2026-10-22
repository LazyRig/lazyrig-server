"""Exercise modal alerts and the shared folder picker in the real dashboard."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service
from test_server_contract import fixture_export


DENIED = 'Access denied to folder: G:\\. Choose a folder readable by the account running LazyRig.'


def visible_alert(page, dialog, message=DENIED):
    notice = page.locator(f'#{dialog} #toast')
    expect(notice).to_be_visible()
    expect(notice).to_have_text(message)
    assert notice.evaluate('''el => {
        const r=el.getBoundingClientRect();
        const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return r.top>=0 && r.bottom<=innerHeight && (hit===el || el.contains(hit));
    }'''), 'Alert is clipped or covered by another layer'
    assert page.locator('#toast').count() == 1


def aligned_actions(page):
    parent = page.locator('#folder-up').bounding_box()
    select = page.locator('#folder-select').bounding_box()
    assert parent and select
    assert abs(parent['y']-select['y']) < 1, (parent, select)
    assert abs(parent['height']-select['height']) < 1, (parent, select)
    assert page.locator('#folder-dialog').evaluate('(el) => el.scrollWidth<=el.clientWidth+1')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('test-results/dialogs'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture_export(root / 'export')
        (root / 'blocked').mkdir()
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=str(root / 'data'), models_dir=str(root))
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = pw.chromium.launch(**launch)
                for width, height in ((1440, 1000), (390, 844)):
                    context = browser.new_context(viewport={'width': width, 'height': height})
                    page = context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    failures = {'browse': True, 'inspect': True}
                    inspect_requests = []

                    def folders(route):
                        selected = parse_qs(urlparse(route.request.url).query).get('path', [''])[0]
                        if failures['browse'] or selected == str(root / 'blocked'):
                            route.fulfill(status=400, json={'error': DENIED})
                            return
                        response = route.fetch()
                        data = response.json()
                        if response.status == 200:
                            data['roots'].append('C:/Programs/' + 'Long-model-folder-name-' * 8)
                        route.fulfill(response=response, json=data)

                    def inspection(route):
                        inspect_requests.append(route.request.post_data_json)
                        if failures['inspect']:
                            route.fulfill(status=400, json={'error': {'message': DENIED}})
                        else:
                            route.continue_()

                    page.route('**/api/folders**', folders)
                    page.route('**/api/models/inspect', inspection)
                    page.goto(f'http://127.0.0.1:{http.server_port}')
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('#add-model').click()
                    for blank in ('', '   '):
                        page.locator('#register-path').fill(blank)
                        page.locator('#inspect-folder').click()
                        visible_alert(page, 'register-dialog', 'Choose a model folder first.')
                        expect(page.locator('#register-path')).to_be_focused()
                        assert inspect_requests == [], 'Blank paths must not start a server scan'
                    page.screenshot(path=str(args.output_dir / f'empty-path-alert-{width}.png'))
                    page.locator('#register-path').fill(str(root))
                    page.locator('#register-id').fill(f'dialog-test-{width}')
                    page.locator('[data-browse="register-path"]').click()
                    visible_alert(page, 'register-dialog')
                    expect(page.locator('#folder-dialog')).not_to_be_visible()
                    expect(page.locator('#register-id')).to_have_value(f'dialog-test-{width}')
                    page.screenshot(path=str(args.output_dir / f'add-model-alert-{width}.png'))

                    failures['browse'] = False
                    page.locator('[data-browse="register-path"]').click()
                    expect(page.locator('#folder-dialog')).to_be_visible()
                    expect(page.locator('#folder-title')).to_have_text('Choose a model folder')
                    page.locator('#folder-up').scroll_into_view_if_needed()
                    aligned_actions(page)
                    page.screenshot(path=str(args.output_dir / f'folder-controls-{width}.png'))
                    page.locator('#folder-path').fill(str(root / 'blocked'))
                    page.locator('#folder-go').click()
                    visible_alert(page, 'folder-dialog')
                    expect(page.locator('#folder-go')).to_be_focused()
                    page.screenshot(path=str(args.output_dir / f'folder-alert-{width}.png'))

                    # A nested modal's warning must not leak out after Escape.
                    page.keyboard.press('Escape')
                    expect(page.locator('#folder-dialog')).not_to_be_visible()
                    expect(page.locator('#register-dialog')).to_be_visible()
                    expect(page.locator('#toast')).not_to_be_visible()
                    page.locator('#inspect-folder').click()
                    visible_alert(page, 'register-dialog')

                    # Success closes the modal and shows a normal page notice.
                    failures['inspect'] = False
                    page.locator('#register-path').fill(str(root / 'blocked'))
                    page.locator('#inspect-folder').click()
                    visible_alert(page, 'register-dialog',
                                  'No OpenVINO .xml/.bin model pairs found. Choose an exported OpenVINO model.')
                    page.screenshot(path=str(args.output_dir / f'invalid-model-alert-{width}.png'))
                    page.locator('#register-path').fill(str(root / 'export'))
                    page.locator('#inspect-folder').click()
                    expect(page.locator('#inspection-result')).to_contain_text('Export structure checked')
                    page.locator('#register-form button[type="submit"]').click()
                    expect(page.locator('#register-dialog')).not_to_be_visible()
                    expect(page.locator('#toast')).to_contain_text('Model registered')
                    assert page.locator('#toast').evaluate('(el) => el.parentElement===document.body')
                    expect(page.locator(f'.model-name[data-edit="dialog-test-{width}"]')).to_be_visible()

                    # The shared picker uses the purpose of each caller.
                    page.locator(f'.model-name[data-edit="dialog-test-{width}"]').click()
                    page.locator('[data-browse="edit-path"]').click()
                    expect(page.locator('#folder-title')).to_have_text('Choose a model folder')
                    page.locator('#folder-dialog .close-dialog').click()
                    page.locator('#close-editor').click()
                    page.locator('nav [data-view="downloads"]').click()
                    page.locator('#download-destination').fill(str(root))
                    page.locator('[data-browse="download-destination"]').click()
                    expect(page.locator('#folder-title')).to_have_text('Choose an output folder')
                    page.locator('#folder-select').click()
                    expect(page.locator('#download-destination')).to_have_value(str(root))
                    page.locator('nav [data-view="server"]').click()
                    page.locator('#server-settings-open').click()
                    page.locator('[data-browse="models-dir"]').click()
                    expect(page.locator('#folder-title')).to_have_text('Choose the default models folder')
                    page.locator('#folder-dialog .close-dialog').click()

                    page.locator('#server-settings-dialog .close-dialog').first.click()
                    if width == 1440:
                        # Exercise a short viewport and an expired modal notice.
                        page.set_viewport_size({'width': 900, 'height': 460})
                        page.locator('nav [data-view="models"]').click()
                        page.locator('#add-model').click()
                        failures['browse'] = True
                        page.locator('[data-browse="register-path"]').click()
                        visible_alert(page, 'register-dialog')
                        page.screenshot(path=str(args.output_dir / 'short-dialog-alert.png'))
                        expect(page.locator('#toast')).not_to_be_visible(timeout=9000)
                        assert page.locator('#toast').evaluate('(el) => el.parentElement===document.body')

                    assert errors == [], errors
                    results.append({'viewport': [width, height], 'status': 'PASS',
                                    'checks': ['empty-path validation without an HTTP request',
                                               'add-model error visibility', 'nested picker error visibility',
                                               'real invalid-export response stays in the foreground',
                                               'keyboard dismissal and cleanup', 'field values preserved',
                                               'aligned footer and wrapped long roots', 'model inspect/register',
                                               'page notification after modal close', 'context-specific titles'],
                                    'javascript_errors': errors})
                    context.close()
                browser.close()
        finally:
            http.shutdown()
            http.server_close()
            thread.join(3)
            service.close()
    report = {'scope': 'Real HTTP/dashboard/Chromium; fixture exports and injected permission errors; no native inference',
              'scenarios': results, 'short_viewport_and_timer_cleanup': 'PASS'}
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
