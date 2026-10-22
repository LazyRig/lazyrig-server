"""Deleting conversations updates History immediately, without a page reload.

Real Chromium, native confirmation dialogs and IndexedDB; no model inference.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service


def main():
    evidence = {'scope': 'Real Chromium, HTTP and IndexedDB; no native inference', 'checks': []}
    output = Path(os.environ.get('HISTORY_DELETE_QA_OUTPUT', 'test-results/history-delete'))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = root
        service = Service(config)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = p.chromium.launch(**launch)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                origin = 'http://127.0.0.1:' + str(server.server_port)
                page.goto(origin + '/favicon.svg')
                page.evaluate('''() => new Promise((resolve, reject) => {
                    const open=indexedDB.open('lazyrig-chat',1);
                    open.onupgradeneeded=()=>open.result.createObjectStore('threads',{keyPath:'id'});
                    open.onerror=()=>reject(open.error);
                    open.onsuccess=()=>{
                        const db=open.result,tx=db.transaction('threads','readwrite');
                        for(const [id,updated] of [['alpha',2000],['beta',1000]])tx.objectStore('threads').put({
                            id,title:id,created:1000,updated,task:'chat',model:'qwen-chat',
                            settings:{temperature:'0.7'},draft:'Keep '+id+' draft',messages:[]
                        });
                        tx.oncomplete=()=>{db.close();localStorage.setItem('lazyrig.chat','alpha');resolve();};
                        tx.onerror=()=>reject(tx.error);
                    };
                })''')
                page.goto(origin)
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()

                def active_id():
                    return page.locator('[data-thread][aria-current="true"]').get_attribute('data-thread')

                def row(thread):
                    return page.locator(f'[data-thread="{thread}"]')

                def stored_ids():
                    return page.evaluate('''() => new Promise((resolve, reject) => {
                        const open=indexedDB.open('lazyrig-chat');
                        open.onerror=()=>reject(open.error);
                        open.onsuccess=()=>{
                            const db=open.result,tx=db.transaction('threads','readonly');
                            const get=tx.objectStore('threads').getAllKeys();
                            get.onsuccess=()=>resolve(get.result);
                            tx.oncomplete=()=>db.close();
                            tx.onerror=()=>reject(tx.error);
                        };
                    })''')

                def delete(thread, accept=True):
                    page.locator(f'[data-thread-menu="{thread}"]').click()
                    page.once('dialog', lambda dialog: dialog.accept() if accept else dialog.dismiss())
                    page.locator('#chat-delete').click()
                    expect(page.locator('#chat-thread-menu')).not_to_be_visible()

                # Exact report: delete a newly created selected conversation
                # while other conversations remain. No reload is allowed here.
                page.locator('#chat-new').click()
                blank = active_id()
                delete(blank)
                expect(row('alpha')).to_have_attribute('aria-current', 'true')
                assert blank not in stored_ids()
                evidence['deleted_blank_still_visible'] = row(blank).count() != 0
                page.screenshot(path=str(output/'after-selected-delete.png'))
                (output/'selected-delete.json').write_text(json.dumps(evidence, indent=2))
                expect(row(blank)).to_have_count(0)
                expect(page.locator('#chat-prompt')).to_have_value('Keep alpha draft')
                evidence['checks'].append('Selected new conversation disappears immediately; saved fallback draft survives')

                delete('beta')
                expect(row('beta')).to_have_count(0)
                assert active_id() == 'alpha' and stored_ids() == ['alpha']
                expect(page.locator('#chat-prompt')).to_have_value('Keep alpha draft')
                evidence['checks'].append('Deleting another conversation preserves selection and draft')

                delete('alpha', accept=False)
                expect(row('alpha')).to_be_visible()
                assert stored_ids() == ['alpha']
                evidence['checks'].append('Cancelling deletion preserves the conversation in History and storage')

                page.locator('#chat-folder-new').click()
                page.locator('#chat-folder-form [name="folder_name"]').fill('Validation')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                folder = page.locator('[data-folder-toggle]').get_attribute('data-folder-toggle')
                page.locator('[data-thread-menu="alpha"]').click()
                page.locator('#chat-move').click()
                page.locator('#chat-folder-form [name="folder_id"]').select_option(folder)
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                expect(page.locator('[data-folder-toggle] small')).to_have_text('1')

                page.locator('#chat-new').click()
                survivor = active_id()
                page.locator(f'[data-thread-menu="{survivor}"]').click()
                page.locator('#chat-move').click()
                page.locator('#chat-folder-form [name="folder_id"]').select_option('')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                page.locator('#chat-prompt').fill('Keep replacement draft')
                row('alpha').click()
                page.locator('#chat-prompt').fill('Discard this deleted draft')
                delete('alpha')
                expect(row('alpha')).to_have_count(0)
                expect(row(survivor)).to_have_attribute('aria-current', 'true')
                expect(page.locator('#chat-prompt')).to_have_value('Keep replacement draft')
                expect(page.locator('[data-folder-toggle] small')).to_have_count(0)
                expect(page.locator('.history-folder [data-thread]')).to_have_count(0)
                assert page.evaluate("localStorage.getItem('lazyrig.draft.alpha')") is None
                evidence['checks'].append('Deleting a selected folder conversation updates its count and preserves the replacement draft')

                page.locator('#chat-prompt').fill('Do not copy this deleted draft')
                delete(survivor)
                expect(row(survivor)).to_have_count(0)
                expect(page.locator('[data-thread]')).to_have_count(1)
                expect(page.locator('#chat-prompt')).to_have_value('')
                replacement = active_id()
                assert replacement not in {'alpha', 'beta', blank, survivor}
                page.wait_for_timeout(600)  # Let the existing 500 ms draft debounce expire.
                assert stored_ids() == [replacement]
                evidence['checks'].append('Deleting the last conversation leaves one fresh empty chat; pending draft save does not restore the deleted one')

                # Only now reload to check durability, after immediate UI checks.
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator('[data-thread]')).to_have_count(1)
                assert active_id() == replacement and stored_ids() == [replacement]
                expect(page.locator('#chat-prompt')).to_have_value('')
                expect(page.locator('[data-folder-toggle]')).to_contain_text('Validation')
                evidence['checks'].append('Deletion remains durable after refresh; the folder survives')
                assert not errors, errors
                evidence['checks'].append('No uncaught JavaScript errors')
                evidence['browser'] = browser.version
                page.screenshot(path=str(output/'after-reload.png'))
                browser.close()
        finally:
            server.shutdown()
            service.close()
    (output/'result.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
