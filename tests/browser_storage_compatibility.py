"""Keep a real old-version connection open while the new chat stores folders.

Uses Chromium/IndexedDB and the HTTP app with deterministic model workers.
No browser storage is deleted or reset during a scenario.
"""
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
    evidence = {'scope': 'Real browser storage and HTTP, deterministic worker', 'checks': []}
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = root
        service = Service(config)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with sync_playwright() as p:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = p.chromium.launch(**launch)
                evidence['browser'] = browser.version
                for version in (1, 2):
                    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                    old = context.new_page()
                    old.goto(origin + '/favicon.svg')
                    old.evaluate('''version => new Promise((resolve, reject) => {
                        const request = indexedDB.open('lazyrig-chat', version);
                        request.onupgradeneeded = () => {
                            request.result.createObjectStore('threads', {keyPath:'id'});
                            if (version === 2) request.result.createObjectStore('folders', {keyPath:'id'});
                        };
                        request.onerror = () => reject(request.error);
                        request.onsuccess = () => {
                            // Deliberately emulate the old app: keep its connection
                            // open without a versionchange handler.
                            window.oldDatabase = request.result;
                            const tx = oldDatabase.transaction(version === 2 ? ['threads','folders'] : ['threads'], 'readwrite');
                            tx.objectStore('threads').put({id:'legacy-thread', title:'Preserved legacy chat',
                                created:1000, updated:2000, task:'chat', model:'qwen-chat',
                                settings:{temperature:'0.7'}, draft:'Stored draft',
                                folderId: version === 2 ? 'legacy-folder' : null,
                                messages:[{id:'legacy-user',role:'user',task:'chat',content:'Existing message',attachments:[]}]});
                            if (version === 2) tx.objectStore('folders').put({id:'legacy-folder',name:'Existing folder',collapsed:false});
                            tx.oncomplete = () => {
                                localStorage.setItem('lazyrig.chat','legacy-thread');
                                localStorage.setItem('lazyrig.draft.legacy-thread',JSON.stringify({draft:'Preserved unsent draft',settings:{temperature:'0.7'}}));
                                resolve();
                            };
                        };
                    })''', version)
                    page = context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(origin)
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    if '--expect-blocked' in sys.argv:
                        expect(page.locator('#chat-notice')).to_contain_text('Close other LazyRig tabs')
                        expect(page.locator('#chat-send')).to_be_disabled()
                        page.locator('#chat-folder-new').click()
                        expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                        print(json.dumps({'reproduced': True, 'notice': page.locator('#chat-notice').inner_text(),
                                          'send_disabled': True, 'add_folder_inert': True}, indent=2))
                        browser.close()
                        return
                    expect(page.locator('#chat-send')).to_be_enabled()
                    expect(page.locator('#chat-prompt')).to_have_value('Preserved unsent draft')
                    expect(page.locator('.message.user .message-body')).to_have_text('Existing message')
                    evidence['checks'].append(f'v{version}: old tab stays open; Send enabled and history/draft preserved')
                    if version == 2:
                        expect(page.locator('[data-folder="legacy-folder"] [data-thread="legacy-thread"]')).to_be_visible()
                        evidence['checks'].append('v2: existing folder and membership imported')
                    page.locator('#chat-folder-new').click()
                    expect(page.locator('#chat-folder-dialog')).to_be_visible()
                    page.locator('#chat-folder-form [name="folder_name"]').fill('Outfits')
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                    folder = page.locator('.history-folder-toggle').filter(has_text='Outfits').get_attribute('data-folder-toggle')
                    page.locator('[data-thread-menu="legacy-thread"]').click()
                    page.locator('#chat-move').click()
                    page.locator('#chat-folder-form [name="folder_id"]').select_option(folder)
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator(f'[data-folder="{folder}"] [data-thread="legacy-thread"]')).to_be_visible()
                    evidence['checks'].append(f'v{version}: Add folder and move work while legacy connection remains open')
                    # An old tab can save a record without the new folder field.
                    old.evaluate('''() => new Promise((resolve, reject) => {
                        const tx=oldDatabase.transaction('threads','readwrite'),store=tx.objectStore('threads');
                        const read=store.get('legacy-thread');
                        read.onsuccess=()=>{delete read.result.folderId;store.put(read.result);};
                        tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);
                    })''')
                    page.reload()
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    expect(page.locator(f'[data-folder="{folder}"] [data-thread="legacy-thread"]')).to_be_visible()
                    evidence['checks'].append(f'v{version}: folder membership survives an old tab saving and reload')
                    page.locator('#chat-prompt').fill('New request while the old tab is still open')
                    page.locator('#chat-send').click()
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                    expect(page.locator('.message.assistant .message-body').last).to_contain_text('Test worker response.')
                    evidence['checks'].append(f'v{version}: real chat send completes')
                    page.locator(f'[data-folder-menu="{folder}"]').click()
                    page.once('dialog', lambda dialog: dialog.accept())
                    page.locator('#chat-folder-remove').click()
                    expect(page.locator(f'[data-folder="{folder}"]')).to_have_count(0)
                    expect(page.locator('[data-thread="legacy-thread"]')).to_be_visible()
                    if version == 2:
                        page.locator('[data-folder-menu="legacy-folder"]').click()
                        page.once('dialog', lambda dialog: dialog.accept())
                        page.locator('#chat-folder-remove').click()
                        expect(page.locator('[data-folder="legacy-folder"]')).to_have_count(0)
                    page.reload()
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    expect(page.locator(f'[data-folder="{folder}"]')).to_have_count(0)
                    expect(page.locator('[data-thread="legacy-thread"]')).to_be_visible()
                    if version == 2:
                        expect(page.locator('[data-folder="legacy-folder"]')).to_have_count(0)
                        # The legacy source remains intact: the migration marker
                        # prevents it from being reimported after explicit removal.
                        assert old.evaluate('''() => new Promise(resolve => {
                            const read=oldDatabase.transaction('folders').objectStore('folders').get('legacy-folder');
                            read.onsuccess=()=>resolve(read.result?.name);
                        })''') == 'Existing folder'
                        evidence['checks'].append('v2: removed imported folder does not resurrect; legacy source preserved')
                    evidence['checks'].append(f'v{version}: removing folder retains conversation; old membership does not resurrect')
                    assert old.evaluate('oldDatabase.version') == version
                    assert errors == [], errors
                    context.close()
                # State may arrive after IndexedDB. Text typed during that wait
                # must survive the first selected-conversation restore.
                context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                page = context.new_page()
                held = []
                page.route('**/api/state?summary=1', lambda route: held.append(route))
                page.goto(origin)
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-prompt').fill('Draft typed during initialization')
                expect(page.locator('#chat-send')).to_be_disabled()
                assert held, 'Initial state request was not intercepted'
                held[0].continue_()
                page.unroute('**/api/state?summary=1')
                expect(page.locator('#connection')).to_have_text('Server online')
                expect(page.locator('#chat-send')).to_be_enabled()
                expect(page.locator('#chat-prompt')).to_have_value('Draft typed during initialization')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                expect(page.locator('#chat-prompt')).to_have_value('Draft typed during initialization')
                evidence['checks'].append('Draft typed while server state loads is restored and persists after reload')
                context.close()
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
