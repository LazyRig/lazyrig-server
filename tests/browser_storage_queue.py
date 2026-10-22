"""Preserve existing history when an earlier dashboard queued a schema upgrade.

A versionless IndexedDB open can wait behind a pending upgrade without receiving
its own blocked event. Keep all three real pages alive until the original holder
is closed, then check automatic recovery and continued chat/folder operation.
"""
import copy
import json
import os
import re
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
    evidence = {'scope': 'Real Chromium, IndexedDB upgrade queue and HTTP; deterministic worker', 'checks': []}
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
                for close_queued in (False, True):
                    scenario = 'queued upgrader remains' if not close_queued else 'queued upgrader closed first'
                    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                    original = context.new_page()
                    original.goto(origin + '/favicon.svg')
                    original.evaluate('''() => new Promise((resolve, reject) => {
                        const request = indexedDB.open('lazyrig-chat', 1);
                        request.onupgradeneeded = () => request.result.createObjectStore('threads', {keyPath:'id'});
                        request.onerror = () => reject(request.error);
                        request.onsuccess = () => {
                            // Original app kept this connection open with no
                            // versionchange handler, so it blocks the upgrade.
                            window.originalDatabase = request.result;
                            const tx = originalDatabase.transaction('threads', 'readwrite');
                            tx.objectStore('threads').put({id:'preserved-thread',title:'Existing conversation',
                                created:1000,updated:2000,task:'chat',model:'qwen-chat',settings:{temperature:'0.7'},
                                draft:'Stored draft',messages:[{id:'preserved-message',role:'user',task:'chat',
                                content:'Message saved before the folder update',attachments:[]}]});
                            tx.oncomplete = () => {
                                localStorage.setItem('lazyrig.chat','preserved-thread');
                                localStorage.setItem('lazyrig.draft.preserved-thread',JSON.stringify({
                                    draft:'Unsent draft before the folder update',settings:{temperature:'0.7'}}));
                                resolve();
                            };
                            tx.onerror = () => reject(tx.error);
                        };
                    })''')
                    queued = context.new_page()
                    queued.goto(origin + '/favicon.svg')
                    queued.evaluate('''() => {
                        window.upgradeRequest = indexedDB.open('lazyrig-chat', 2);
                        upgradeRequest.onblocked = () => window.upgradeBlocked = true;
                        upgradeRequest.onupgradeneeded = () => {
                            upgradeRequest.result.createObjectStore('folders',{keyPath:'id'});
                        };
                        upgradeRequest.onsuccess = () => {
                            window.upgradeDatabase = upgradeRequest.result;
                            window.upgradeFinished = true;
                        };
                    }''')
                    queued.wait_for_function('() => window.upgradeBlocked === true')
                    page = context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(origin)
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    expect(page.locator('#chat-model')).to_have_value('qwen-chat')
                    expect(page.locator('#chat-send')).to_be_disabled()
                    expect(page.locator('#chat-folder-new')).to_be_disabled()
                    expect(page.locator('#chat-storage-warning-text')).to_contain_text(re.compile(r'other LazyRig tab', re.I), timeout=10000)
                    expect(page.locator('#chat-storage-warning')).to_be_visible()
                    expect(page.locator('#chat-storage-warning-title')).to_have_text('Chat blocked — close other LazyRig tabs')
                    assert not re.search(r'loading|still opening', page.locator('#chat-storage-warning').inner_text(), re.I)
                    blocked_notice = page.locator('#chat-storage-warning-text').inner_text()
                    # Opening the chat again calls renderMessages. Polling the
                    # valid API also must not overwrite the storage explanation.
                    page.locator('nav [data-view="models"]').click()
                    page.locator('nav [data-view="chat"]').click()
                    page.evaluate('() => refresh()')
                    expect(page.locator('#chat-storage-warning-text')).to_have_text(blocked_notice)
                    expect(page.locator('#connection')).to_have_text('Server online')
                    evidence['checks'].append(scenario + ': queued open is explained persistently despite successful API and chat navigation')
                    if os.environ.get('WARNING_SCREENSHOTS') and not close_queued:
                        output = Path(os.environ['WARNING_SCREENSHOTS'])
                        output.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(output/'warning-desktop.png'))
                        page.set_viewport_size({'width':390,'height':844})
                        warning = page.locator('#chat-storage-warning').bounding_box()
                        assert warning['x'] >= 0 and warning['x'] + warning['width'] <= 390
                        page.screenshot(path=str(output/'warning-mobile.png'))
                        page.set_viewport_size({'width':1440,'height':1000})
                    if close_queued:
                        queued.close()
                        # Closing only the queued requester does not release the
                        # connection which blocked its upgrade in Chromium.
                        expect(page.locator('#chat-send')).to_be_disabled()
                    original.close()
                    if not close_queued:
                        queued.wait_for_function('() => window.upgradeFinished === true')
                    expect(page.locator('#chat-send')).to_be_enabled(timeout=10000)
                    expect(page.locator('#chat-folder-new')).to_be_enabled()
                    expect(page.locator('#chat-storage-warning')).not_to_be_visible()
                    expect(page.locator('#chat-prompt')).to_have_value('Unsent draft before the folder update')
                    expect(page.locator('.message.user .message-body')).to_have_text('Message saved before the folder update')
                    expect(page.locator('#chat-options [name="temperature"]')).to_have_value('0.7')
                    evidence['checks'].append(scenario + ': closing original old tab resumes without reload; message, draft and settings retained')
                    if not page.locator('#chat-history').is_visible():
                        page.locator('#chat-history-toggle').click()
                    page.locator('#chat-folder-new').click()
                    page.locator('#chat-folder-form [name="folder_name"]').fill('Outfits')
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                    folder = page.locator('.history-folder-toggle').filter(has_text='Outfits').get_attribute('data-folder-toggle')
                    page.locator('[data-thread-menu="preserved-thread"]').click()
                    page.locator('#chat-move').click()
                    page.locator('#chat-folder-form [name="folder_id"]').select_option(folder)
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator(f'[data-folder="{folder}"] [data-thread="preserved-thread"]')).to_be_visible()
                    page.locator('#chat-prompt').fill('Send after old tab closes')
                    page.locator('#chat-send').click()
                    expect(page.locator('.message.assistant .message-body').last).to_contain_text('Test worker response.', timeout=15000)
                    expect(page.locator('#chat-stop')).not_to_be_visible()
                    evidence['checks'].append(scenario + ': Add folder, move and real HTTP chat send complete after recovery')
                    page.reload()
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    expect(page.locator(f'[data-folder="{folder}"] [data-thread="preserved-thread"]')).to_be_visible()
                    expect(page.locator('.message.user .message-body').first).to_have_text('Message saved before the folder update')
                    expect(page.locator('.message.user .message-body').last).to_have_text('Send after old tab closes')
                    expect(page.locator('.message.assistant .message-body').last).to_contain_text('Test worker response.')
                    expect(page.locator('#chat-send')).to_be_enabled()
                    assert errors == [], errors
                    evidence['checks'].append(scenario + ': reload retains original message, new response and folder membership; no JS errors')
                    context.close()
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
