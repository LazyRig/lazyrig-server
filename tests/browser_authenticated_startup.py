"""Real authenticated dashboard startup with existing local history."""
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
    evidence = {'scope': 'Authenticated real HTTP/Chromium; deterministic model workers', 'scenarios': []}
    with tempfile.TemporaryDirectory() as temp:
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=temp, api_key='test-authenticated-startup-key')
        service = Service(config)
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=http.serve_forever, daemon=True).start()
        origin = 'http://127.0.0.1:' + str(http.server_port)
        try:
            with sync_playwright() as p:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = p.chromium.launch(**launch)
                for version, status in ((0, 'fresh'), (1, 'completed'), (2, 'completed'), (1, 'running'), (2, 'running')):
                    context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                    page = context.new_page()
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    if version:
                        page.goto(origin + '/favicon.svg')
                        page.evaluate('''({version,status}) => new Promise((resolve,reject) => {
                            const request=indexedDB.open('lazyrig-chat', version);
                            request.onupgradeneeded=()=>{
                                request.result.createObjectStore('threads',{keyPath:'id'});
                                if(version===2)request.result.createObjectStore('folders',{keyPath:'id'});
                            };
                            request.onerror=()=>reject(request.error);
                            request.onsuccess=()=>{
                                const db=request.result,tx=db.transaction(version===2?['threads','folders']:['threads'],'readwrite');
                                tx.objectStore('threads').put({id:'existing',title:'Existing conversation',created:1000,updated:2000,
                                    task:'chat',model:'qwen-chat',settings:{temperature:'0.7'},draft:'Saved draft',folderId:version===2?'old-folder':null,
                                    messages:[{id:'old-user',role:'user',task:'chat',content:'Existing user message',attachments:[]},
                                      {id:'old-assistant',role:'assistant',task:'chat',model:'qwen-chat',content:'Existing response',status,
                                       jobId:'old-job',artifacts:[],reasoning:''}]});
                                if(version===2)tx.objectStore('folders').put({id:'old-folder',name:'Preserved folder',collapsed:false});
                                tx.oncomplete=()=>{db.close();localStorage.setItem('lazyrig.chat','existing');resolve();};
                                tx.onerror=()=>reject(tx.error);
                            };
                        })''', {'version':version,'status':status})
                    # Hold the first state request so IndexedDB completes before authentication.
                    held = []
                    page.route('**/api/state?summary=1', lambda route: held.append(route))
                    page.goto(origin)
                    page.wait_for_timeout(250)
                    for route in held:
                        route.continue_()
                    page.unroute('**/api/state?summary=1')
                    expect(page.locator('#key-dialog')).to_be_visible()
                    if version == 0:
                        page.locator('#client-key').fill('wrong-key')
                        page.locator('#key-form button').click()
                        expect(page.locator('#key-dialog')).to_be_visible()
                        expect(page.locator('#toast')).to_be_visible()
                        expect(page.locator('#toast')).to_contain_text(re.compile('API key', re.I))
                    page.locator('#client-key').fill(config['api_key'])
                    page.locator('#key-form button').click()
                    expect(page.locator('#connection')).to_have_text('Server online')
                    expect(page.locator('#toast')).not_to_be_visible()
                    # A successful background refresh must leave unrelated feedback visible.
                    if version == 0:
                        page.wait_for_function('() => !pendingRefresh')
                        page.evaluate("() => toast('Unrelated action succeeded')")
                        page.evaluate('() => refresh()')
                        expect(page.locator('#toast')).to_have_text('Unrelated action succeeded')
                        expect(page.locator('#toast')).to_be_visible()
                    page.locator('nav [data-view="chat"]').click()
                    page.locator('#chat-history-toggle').click()
                    page.wait_for_timeout(500)
                    snapshot = page.evaluate('''() => ({
                       notice:document.querySelector('#chat-notice').textContent,
                       sendDisabled:document.querySelector('#chat-send').disabled,
                       sendHidden:document.querySelector('#chat-send').hidden,
                       folderDisabled:document.querySelector('#chat-folder-new').disabled,
                       prompt:document.querySelector('#chat-prompt').value,
                       toast:document.querySelector('#toast').textContent,
                       toastVisible:!document.querySelector('#toast').hidden,
                       keyDialog:document.querySelector('#key-dialog').open,
                       models:document.querySelector('#chat-model').value,
                       messages:[...document.querySelectorAll('.message-body')].map(e=>e.textContent)
                    })''')
                    row={'version':version,'historyStatus':status,'afterLogin':snapshot,'pageErrors':errors}
                    evidence['scenarios'].append(row)
                    print(json.dumps(row), flush=True)
                    expect(page.locator('#chat-folder-new')).to_be_enabled()
                    page.locator('#chat-folder-new').click()
                    expect(page.locator('#chat-folder-dialog')).to_be_visible()
                    page.locator('#chat-folder-form [name="folder_name"]').fill('New folder')
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('.history-folder-toggle').filter(has_text='New folder')).to_be_visible()
                    # Start a fresh conversation, regardless of the pending job fixture.
                    page.locator('#chat-new').click()
                    page.locator('#chat-prompt').fill('Authenticated new request')
                    expect(page.locator('#chat-send')).to_be_enabled()
                    page.locator('#chat-send').click()
                    expect(page.locator('.message.assistant .message-body').last).to_contain_text('Test worker response.', timeout=15000)
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                    row['newFolderAndSend']='passed'
                    row['authenticatedNoticeCleared']='passed'
                    if version == 0:
                        row['wrongKeyStillRejected']='passed'
                        row['unrelatedToastPreservedOnPoll']='passed'
                    assert errors == [], errors
                    context.close()
                browser.close()
        finally:
            http.shutdown()
            http.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
