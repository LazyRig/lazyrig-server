"""Browser Stop races over real HTTP; deterministic, cancellable image workers."""
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
from lazyrig import __version__
from support import DEFAULT, Service
from test_cancellation import GatedImages
from test_core import wait_job


def main():
    evidence = {'scope': 'Chromium + real HTTP; cancellable test workers', 'checks': []}
    with tempfile.TemporaryDirectory() as root:
        config = {**copy.deepcopy(DEFAULT), 'data_dir': root}
        service = Service(config)
        receipt_entered, receipt_release = threading.Event(), threading.Event()
        hold_receipt = threading.Event()
        Base = handler(service)
        class Handler(Base):
            def reply(self, value, code=200):
                if code == 202 and self.path == '/v1/images/generations' and hold_receipt.is_set():
                    receipt_entered.set()
                    if not receipt_release.wait(10):
                        raise RuntimeError('Browser did not release delayed receipt')
                super().reply(value, code)
        http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                base = 'http://127.0.0.1:' + str(http.server_port)
                page.goto(base)
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-task').select_option('image')

                def start(label):
                    workers = GatedImages()
                    service.scheduler.factory = workers
                    page.locator('#chat-prompt').fill(label)
                    page.locator('#chat-send').click()
                    expect(page.locator('#chat-stop')).to_be_enabled()
                    assert workers.started.wait(3)
                    return workers, service.store.image_jobs()[0]['id']

                workers, jid = start('Stop running generation')
                page.locator('#chat-stop').click()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=10000)
                assert wait_job(service, jid)['status'] == 'cancelled'
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                expect(page.locator('.job-error')).to_contain_text('Stopped')
                assert len(service.store.image_jobs()) == 1
                assert workers.started_actions == ['image']
                evidence['checks'].append('Stop cancels running work; reload never resubmits the image')

                page.locator('#chat-new').click()
                hold_receipt.set()
                workers, jid = start('Stop before acknowledgement')
                assert receipt_entered.wait(2)
                page.locator('#chat-stop').click()
                expect(page.locator('#chat-stop')).to_be_disabled()
                expect(page.locator('.job-progress')).to_contain_text('Cancelling')
                receipt_release.set()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=10000)
                assert wait_job(service, jid)['status'] == 'cancelled'
                assert workers.started_actions == ['image']
                hold_receipt.clear()
                evidence['checks'].append('Stop before receipt is remembered and sent after admission is confirmed')

                page.locator('#chat-new').click()
                attempts = []
                def lose_first_cancel(route):
                    attempts.append(route.request.url)
                    if len(attempts) == 1:
                        route.abort('failed')
                    else:
                        route.continue_()
                page.route('**/api/jobs/*/cancel', lose_first_cancel)
                workers, jid = start('Recover an unconfirmed stop')
                page.locator('#chat-stop').click()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                assert wait_job(service, jid)['status'] == 'cancelled'
                assert len(attempts) >= 2
                assert len(service.store.image_jobs()) == 3
                assert workers.started_actions == ['image']
                evidence['checks'].append('Lost cancel request retries cancellation only, never generation')
                assert not errors, errors
                evidence['browser'] = browser.version
                browser.close()
        finally:
            receipt_release.set()
            http.shutdown()
            http.server_close()
            service.close()
    output = Path('test-results')/('browser-cancellation-'+__version__+'.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
