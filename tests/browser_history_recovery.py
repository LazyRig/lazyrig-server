"""Recent activity opens saved conversations without cloning their messages.

Uses Chromium, real IndexedDB/HTTP and deterministic model workers. Covers
original image attachments, completed workflow steps, repeated recovery and
pending jobs; this does not validate native model inference.
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
from support import DEFAULT, Service, fixture_png


def main():
    evidence = {'scope': 'Real Chromium, IndexedDB and HTTP; deterministic workers', 'checks': []}
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = root
        config['models'][0]['pipeline'] = 'vlm'
        config['models'][1]['generation'] = {'steps': 2}
        config['models'].append({**copy.deepcopy(DEFAULT['models'][0]), 'id': 'qwen-text'})
        service = Service(config)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = 'http://127.0.0.1:' + str(server.server_port)
        photo = Path(root) / 'source.png'
        fixture_png(photo)
        try:
            with sync_playwright() as p:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = p.chromium.launch(**launch)
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors, submissions = [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: submissions.append(request.url)
                        if request.method == 'POST' and '/v1/' in request.url else None)

                def records():
                    return page.evaluate('''() => new Promise((resolve,reject) => {
                        const open=indexedDB.open('lazyrig-chat');
                        open.onerror=()=>reject(open.error);
                        open.onsuccess=()=>{const db=open.result,tx=db.transaction('threads');
                            const read=tx.objectStore('threads').getAll();
                            read.onsuccess=()=>resolve(read.result);
                            tx.oncomplete=()=>db.close();tx.onerror=()=>reject(tx.error);};
                    })''')

                def active_id():
                    return page.locator('[data-thread][aria-current="true"]').get_attribute('data-thread')

                def enter_chat():
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    expect(page.locator('#chat-send')).to_be_enabled()
                    if not page.locator('#chat-history').is_visible():
                        page.locator('#chat-history-toggle').click()

                def view_job(jid):
                    page.locator('nav [data-view="models"]').click()
                    page.evaluate('() => refresh()')
                    page.locator(f'[data-view-job="{jid}"]').click()

                page.goto(origin)
                enter_chat()
                original = active_id()
                page.locator('#chat-folder-new').click()
                page.locator('[name="folder_name"]').fill('Validation')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                folder = page.locator('[data-folder-toggle]').get_attribute('data-folder-toggle')
                page.locator(f'[data-thread-menu="{original}"]').click()
                page.locator('#chat-move').click()
                page.locator('[name="folder_id"]').select_option(folder)
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                page.locator('#chat-files').set_input_files(str(photo))
                expect(page.locator('#chat-send')).to_be_enabled()
                page.locator('#chat-prompt').fill('Read this source screenshot')
                page.locator('#chat-send').click()
                expect(page.locator('.message.assistant .message-body')).to_contain_text('Test worker response.')
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                jid = page.locator('[data-result]').get_attribute('data-result')
                page.locator('#chat-prompt').fill('Keep this unsent draft')
                page.locator('[data-folder-toggle]').click()
                expect(page.locator('[data-folder-toggle]')).to_have_attribute('aria-expanded', 'false')
                before = records()[0]
                view_job(jid)
                if '--expect-duplicate' in sys.argv:
                    expect(page.locator('[data-thread]')).to_have_count(2)
                    expect(page.locator('.message.user .message-images img')).to_have_count(0)
                    evidence['reproduced'] = 'View result creates a second thread and drops the source attachment'
                    print(json.dumps(evidence, indent=2))
                    browser.close()
                    return
                expect(page.locator('[data-thread]')).to_have_count(1)
                assert active_id() == original
                expect(page.locator('[data-folder-toggle]')).to_have_attribute('aria-expanded', 'true')
                expect(page.locator('.message.user .message-images img')).to_have_count(1)
                expect(page.locator('#chat-prompt')).to_have_value('Keep this unsent draft')
                after = records()[0]
                assert after['messages'] == before['messages']
                assert after['updated'] == before['updated']
                evidence['checks'].append('View result selects the original, reveals its folder, and preserves image, messages, draft and recency')
                for _ in range(3):
                    view_job(jid)
                    expect(page.locator('[data-thread]')).to_have_count(1)
                    assert active_id() == original
                page.reload()
                enter_chat()
                view_job(jid)
                expect(page.locator('[data-thread]')).to_have_count(1)
                expect(page.locator('.message.user .message-images img')).to_have_count(1)
                assert len(submissions) == 1, submissions
                evidence['checks'].append('Repeated views and reload keep one conversation and submit no new inference')

                # Direct API jobs have no browser conversation yet. Concurrent
                # opens must create one fully populated recovery, not blanks.
                external = page.evaluate('''async () => {
                    const response=await fetch('/v1/images/generations',{method:'POST',
                        headers:{'Content-Type':'application/json',Prefer:'respond-async'},
                        body:JSON.stringify({model:'qwen-image',prompt:'External orange robot',size:'256x256',steps:2,seed:42})});
                    if(!response.ok)throw new Error(await response.text());return response.json();
                }''')['id']
                page.evaluate('jid => Promise.all(Array.from({length:5},()=>Chat.recover(jid)))', external)
                expect(page.locator('[data-thread]')).to_have_count(2)
                recovered = active_id()
                assert recovered != original
                expect(page.locator(f'[data-thread="{recovered}"]')).to_have_text('External orange robot')
                expect(page.locator('#chat-task')).to_have_value('image')
                expect(page.locator('#chat-model')).to_have_value('qwen-image')
                expect(page.locator('.message.assistant .message-images img')).to_have_count(1, timeout=15000)
                expect(page.locator('#chat-stop')).not_to_be_visible()
                assert len(records()) == 2
                evidence['checks'].append('Five concurrent opens of an external image job create one correctly titled image conversation')
                page.reload()
                enter_chat()
                page.evaluate('jid => Chat.recover(jid)', external)
                expect(page.locator('[data-thread]')).to_have_count(2)
                assert active_id() == recovered
                expect(page.locator('.message.assistant .message-images img')).to_have_count(1)
                evidence['checks'].append('Recovered image conversation and saved artifact are reused after reload')

                # A completed tool stage belongs to the full parent chat even
                # when the assistant's latest receipt is a later continuation.
                page.locator('#chat-new').click()
                page.locator('#chat-task').select_option('chat')
                page.locator('#chat-model').select_option('qwen-text')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-options [name="image_model"]').select_option('qwen-image')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-prompt').fill('Draw a small robot')
                page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                expect(page.locator('.message.assistant .message-body')).to_contain_text('The image job finished')
                parent = active_id()
                workflow = next(t for t in records() if t['id'] == parent)['messages'][-1]['workflow']
                assert len(workflow['completed']) == 3, workflow
                posted = len(submissions)
                for step in workflow['completed']:
                    page.evaluate('jid => Chat.recover(jid)', step['id'])
                    expect(page.locator('[data-thread]')).to_have_count(3)
                    assert active_id() == parent
                    expect(page.locator('.message.assistant .message-body')).to_contain_text('The image job finished')
                assert len(submissions) == posted
                evidence['checks'].append('Every chat/image/continuation receipt selects the complete tool conversation without generating again')

                # Recover a still-running standalone request; opening progress
                # repeatedly must monitor the same job and preserve Stop.
                running = page.evaluate('''async () => (await (await fetch('/v1/images/generations',{
                    method:'POST',headers:{'Content-Type':'application/json',Prefer:'respond-async'},
                    body:JSON.stringify({model:'qwen-image',prompt:'Pending external image',size:'256x256',steps:100})})).json()).id''')
                page.evaluate('jid => Promise.all([Chat.recover(jid),Chat.recover(jid)])', running)
                expect(page.locator('[data-thread]')).to_have_count(4)
                expect(page.locator('#chat-stop')).to_be_visible()
                pending = active_id()
                view_job(running)
                assert active_id() == pending
                expect(page.locator('[data-thread]')).to_have_count(4)
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                expect(page.locator('.message.assistant .message-images img')).to_have_count(1)
                assert len(submissions) == posted + 1
                evidence['checks'].append('View progress reuses the pending conversation and completes the same job')

                # Existing copies are user data: prefer the original receipt
                # owner without deleting either thread or guessing by title.
                page.evaluate('''id => new Promise((resolve,reject) => {
                    const open=indexedDB.open('lazyrig-chat');
                    open.onsuccess=()=>{const db=open.result,tx=db.transaction('threads','readwrite'),store=tx.objectStore('threads');
                        const read=store.get(id);read.onsuccess=()=>{
                            const t=read.result;t.id='previous-recovery-copy';t.created+=1;t.updated+=1;
                            for(const m of t.messages){m.attachments=[];delete m.workflow;}store.put(t);
                        };tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>reject(tx.error);};
                })''', original)
                page.reload()
                enter_chat()
                page.locator('[data-thread="previous-recovery-copy"]').click()
                expect(page.locator('.message.user .message-images img')).to_have_count(0)
                page.evaluate('jid => Chat.recover(jid)', jid)
                assert active_id() == original
                expect(page.locator('.message.user .message-images img')).to_have_count(1)
                expect(page.locator('[data-thread]')).to_have_count(5)
                assert len(records()) == 5
                evidence['checks'].append('An existing recovery copy is preserved while View result selects the older original with its source image')
                assert errors == [], errors
                evidence['browser'] = browser.version
                if os.environ.get('RECOVERY_EVIDENCE'):
                    output = Path(os.environ['RECOVERY_EVIDENCE'])
                    output.mkdir(parents=True, exist_ok=True)
                    page.locator(f'[data-thread="{original}"]').click()
                    page.screenshot(path=str(output / 'original-conversation.png'))
                    (output / 'results.json').write_text(json.dumps(evidence, indent=2) + '\n')
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
