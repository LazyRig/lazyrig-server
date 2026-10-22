"""Exercise the real update dialog with simulated package operations."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from support import DEFAULT, Service
from lazyrig.server import handler
from lazyrig import runtime_update as update
from test_runtime_update import ready, CURRENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser')
    parser.add_argument('--output-dir', type=Path, default=Path('test-results/runtime-update'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = str(root / 'data')
        service = Service(config)
        service.runtime_updater = update.RuntimeUpdater(service, root)
        service.runtime_updater.shutdown = lambda: None
        service.runtime_updater.stop_server = lambda: None
        service.runtime_updater.restart = [sys.executable, '-m', 'lazyrig']
        service.runtime_updater.cwd = str(update.ROOT)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        outcomes = []
        try:
            with patch.object(update, 'check_update', return_value=ready()) as check, \
                 patch.object(update, 'managed_environment', return_value=True), \
                 patch.object(update, 'installed_versions', return_value=CURRENT), \
                 patch.object(update, 'launch_helper') as launch, sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=args.browser, args=['--no-sandbox'])
                page = browser.new_page(viewport={'width': 1280, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{http.server_port}')
                expect(page.locator('#connection')).to_have_text('Server online')
                check.assert_not_called()
                page.locator('#runtime-update-open').click()
                expect(page.locator('#runtime-update-install')).to_be_enabled()
                expect(page.locator('#runtime-update-versions')).to_contain_text('2026.4.1.0')
                launch.assert_not_called()
                page.screenshot(path=str(args.output_dir / 'desktop.png'))
                outcomes.append('explicit check, version display, no installation on check')

                service.runtime_requests = 1
                page.locator('#runtime-update-install').click()
                expect(page.locator('#runtime-update-message')).to_contain_text('Wait for preparing/active/queued')
                launch.assert_not_called()
                service.runtime_requests = 0
                outcomes.append('busy-server installation rejected')

                page.locator('#runtime-update-check').click()
                expect(page.locator('#runtime-update-install')).to_be_enabled()
                page.locator('#runtime-update-install').click()
                expect(page.locator('#runtime-update-message')).to_contain_text('reconnect')
                launch.assert_called_once()
                expect(page.locator('#runtime-update-install')).to_be_disabled()
                plan = json.loads(launch.call_args.args[0].read_text())
                update.write_json(update.update_paths(root)[0], {
                    'id': plan['id'], 'status': 'succeeded', 'message': 'Stable runtime updated and verified.'})
                service.runtime_maintenance = False
                service.runtime_updater.result = {'status': 'idle', 'installed': CURRENT, 'can_update': False}
                expect(page.locator('#runtime-update-last')).to_contain_text('updated and verified', timeout=10000)
                expect(page.locator('#runtime-update-check')).to_be_enabled()
                outcomes.append('install receipt and simulated restart completion')

                check.return_value = {**ready(), 'can_update': False, 'message': 'Experimental/nightly runtime detected.'}
                page.locator('#runtime-update-check').click()
                expect(page.locator('#runtime-update-message')).to_contain_text('Experimental/nightly')
                expect(page.locator('#runtime-update-install')).to_be_disabled()
                outcomes.append('experimental profile preserved')

                check.side_effect = OSError('Connection unavailable')
                page.locator('#runtime-update-check').click()
                expect(page.locator('#runtime-update-message')).to_contain_text('Could not check')
                expect(page.locator('#runtime-update-check')).to_be_enabled()
                outcomes.append('failed check can be retried')

                check.side_effect = None
                check.return_value = ready()
                page.locator('#runtime-update-check').click()
                expect(page.locator('#runtime-update-install')).to_be_enabled()
                page.route('**/api/runtime/update', lambda route: route.abort() if route.request.method == 'POST' else route.continue_())
                page.locator('#runtime-update-install').click()
                expect(page.locator('#runtime-update-message')).to_contain_text('not accepted', timeout=10000)
                expect(page.locator('#runtime-update-check')).to_be_enabled()
                launch.assert_called_once()
                outcomes.append('lost request before acceptance does not retry installation')

                page.set_viewport_size({'width': 390, 'height': 844})
                expect(page.locator('#runtime-update-check')).to_be_in_viewport()
                box = page.locator('#runtime-update-dialog').bounding_box()
                assert box['x'] >= 0 and box['x'] + box['width'] <= 390
                page.screenshot(path=str(args.output_dir / 'mobile.png'))
                outcomes.append('mobile dialog fits viewport')
                assert not errors, errors
                browser.close()
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()
    print(json.dumps({'passed': outcomes, 'javascript_errors': errors}, indent=2))


if __name__ == '__main__':
    main()
