"""Conversation selection must stay responsive without rewriting saved messages.

Uses real Chromium/IndexedDB, eight saved PNGs and native pointer gestures.
Model workers are deterministic; no native inference is tested.
"""
import base64
import copy
import io
import json
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service


def main():
    evidence = {'scope': 'Real Chromium and IndexedDB; deterministic server', 'checks': [], 'profile': []}
    picture = io.BytesIO()
    Image.frombytes('RGB', (1024, 1024), os.urandom(1024 * 1024 * 3)).save(picture, format='PNG')
    photo = 'data:image/png;base64,' + base64.b64encode(picture.getvalue()).decode()
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
                context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                page = context.new_page()
                errors, generation_requests = [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: generation_requests.append(request.url)
                        if request.method == 'POST' and '/v1/' in request.url else None)
                origin = 'http://127.0.0.1:' + str(server.server_port)
                page.goto(origin + '/favicon.svg')
                page.evaluate('''photo => new Promise((resolve, reject) => {
                    const open=indexedDB.open('lazyrig-chat',1);
                    open.onupgradeneeded=()=>open.result.createObjectStore('threads',{keyPath:'id'});
                    open.onerror=()=>reject(open.error);
                    open.onsuccess=()=>{
                        const db=open.result,tx=db.transaction('threads','readwrite');
                        for(const id of ['alpha','photos','beta'])tx.objectStore('threads').put({
                            id,title:id,created:1000,updated:2000,task:'chat',model:'qwen-chat',
                            settings:{temperature:'0.7'},draft:'Draft '+id,
                            messages:id==='photos'?Array.from({length:8},(_,i)=>({
                                id:id+i,role:'user',task:'chat',content:'Saved source photo '+i,
                                attachments:[{name:'source.png',url:photo,bytes:3145728}]
                            })):[{id:id+'-text',role:'user',task:'chat',content:'Message '+id,attachments:[]}]
                        });
                        tx.oncomplete=()=>{db.close();localStorage.setItem('lazyrig.chat','alpha');resolve();};
                        tx.onerror=()=>reject(tx.error);
                    };
                })''', photo)
                page.goto(origin)
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator('[data-thread="alpha"]')).to_have_attribute('aria-current', 'true')
                page.evaluate('''() => {
                    window.navigationWrites=[];
                    const put=IDBObjectStore.prototype.put;
                    IDBObjectStore.prototype.put=function(...args){
                        if(this.name==='threads')navigationWrites.push(args[0].id);
                        return put.apply(this,args);
                    };
                }''')
                for target in ['photos', 'beta', 'photos', 'alpha']:
                    result = page.evaluate('''id => {
                        navigationWrites=[];
                        const button=document.querySelector('[data-thread="'+id+'"]'),start=performance.now();
                        button.click();
                        return {id,handler_ms:performance.now()-start,writes:navigationWrites.length,
                            selected:button.getAttribute('aria-current')==='true',sameButton:button.isConnected};
                    }''', target)
                    evidence['profile'].append(result)
                    assert result['selected'] and result['sameButton'], result
                    assert result['writes'] == 0, result
                    expect(page.locator('#chat-prompt')).to_have_value('Draft ' + target)
                    if target == 'photos':
                        expect(page.locator('.message-images img')).to_have_count(8)
                        page.wait_for_function('() => [...document.querySelectorAll(".message-images img")].every(img=>img.complete&&img.naturalWidth===1024)')
                evidence['checks'].append('Switching image/text history selects immediately, preserves row nodes and writes no full conversation records')
                evidence['checks'].append('Eight stored PNGs render correctly after repeated switches')

                def active_id():
                    return page.locator('[data-thread][aria-current="true"]').get_attribute('data-thread')

                for index in range(20):
                    target = 'beta' if index % 2 == 0 else 'alpha'
                    page.locator(f'[data-thread="{target}"]').click()
                    assert active_id() == target
                    expect(page.locator('#chat-prompt')).to_have_value('Draft ' + target)
                evidence['checks'].append('Twenty rapid alternating selections open the requested conversation and draft')

                page.locator('#chat-prompt').fill('Keep this unsent draft')
                page.locator('[data-thread="beta"]').click()
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                page.locator('[data-thread="alpha"]').click()
                expect(page.locator('#chat-prompt')).to_have_value('Keep this unsent draft')
                expect(page.locator('#chat-options [name="temperature"]')).to_have_value('0.7')
                evidence['checks'].append('Switching before the draft debounce preserves the unsent draft and settings across reload')

                page.locator('#chat-folder-new').click()
                page.locator('[name="folder_name"]').fill('Outfits')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                for delta in [0, 2, 4, 6, 10]:
                    page.locator('[data-thread="alpha"]').click()
                    rect = page.locator('[data-thread="beta"]').bounding_box()
                    x, y = rect['x'] + rect['width']/2, rect['y'] + rect['height']/2
                    page.mouse.move(x, y)
                    page.mouse.down()
                    page.mouse.move(x + delta, y, steps=2)
                    page.mouse.up()
                    expect(page.locator('[data-thread="beta"]')).to_have_attribute('aria-current', 'true')
                    expect(page.locator('#chat-threads .is-dragging,#chat-threads .is-drop-target')).to_have_count(0)
                evidence['checks'].append('Clicks with 0/2/4/6/10px pointer movement still select the conversation, with no stuck drag state')

                page.locator('[data-thread="photos"]').drag_to(page.locator('[data-folder-toggle]'))
                expect(page.locator('.history-folder [data-thread="photos"]')).to_be_visible()
                assert active_id() == 'beta'
                source = page.locator('[data-thread="beta"]').bounding_box()
                destination = page.locator('[data-folder-toggle]').bounding_box()
                page.mouse.move(source['x'] + source['width']/2, source['y'] + source['height']/2)
                page.mouse.down()
                page.mouse.move(destination['x'] + destination['width']/2, destination['y'] + destination['height']/2, steps=12)
                page.mouse.move(destination['x'] + destination['width']/2, destination['y'] + destination['height']/2 + 1)
                page.keyboard.press('Escape')
                page.mouse.up()
                expect(page.locator('#chat-history')).to_be_visible()
                expect(page.locator('#chat-threads .is-dragging,#chat-threads .is-drop-target')).to_have_count(0)
                page.locator('[data-thread="alpha"]').click()
                page.locator('[data-thread="beta"]').drag_to(page.locator('#chat-composer'))
                assert active_id() == 'alpha'
                page.locator('[data-thread="beta"]').click()
                expect(page.locator('[data-thread="beta"]')).to_have_attribute('aria-current', 'true')
                evidence['checks'].append('Real folder moves preserve selection; Escape and cancelled outside drops do not block the next selection')

                # A failed last-selected pointer write must not abort navigation.
                page.evaluate('''() => {const set=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){
                    if(k==='lazyrig.chat')throw new DOMException('Fixture quota limit','QuotaExceededError');
                    return set.call(this,k,v);
                };}''')
                page.locator('[data-thread="alpha"]').click()
                expect(page.locator('[data-thread="alpha"]')).to_have_attribute('aria-current', 'true')
                expect(page.locator('#chat-prompt')).to_have_value('Keep this unsent draft')
                expect(page.locator('#chat-send')).to_be_enabled()
                assert errors == [], errors
                assert generation_requests == [], generation_requests
                evidence['checks'].append('Selection survives a local-storage pointer failure; no page errors or model-generation requests')
                other = page.context.new_page()
                other.goto(origin)
                expect(other.locator('#connection')).to_have_text('Server online')
                other.locator('nav [data-view="chat"]').click()
                expect(other.locator('#chat-send')).to_be_enabled()
                expect(other.locator('#chat-storage-warning')).not_to_be_visible()
                expect(page.locator('#chat-send')).to_be_enabled()
                expect(page.locator('#chat-storage-warning')).not_to_be_visible()
                other.close()
                evidence['checks'].append('Two current dashboard tabs can open chat together without a storage warning or disabled Send')
                evidence['browser'] = browser.version
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
