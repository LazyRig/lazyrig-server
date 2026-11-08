"""Folder removal choices, persistence and failure safety in real Chromium/IndexedDB.

Run with Python Playwright installed. CHROMIUM_EXECUTABLE may select a local browser.
Only synthetic browser history and deterministic server workers are used.
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
    output = Path(os.environ.get('FOLDER_REMOVAL_QA_OUTPUT', 'test-results/folder-removal'))
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'scope': 'Real Chromium, HTTP and IndexedDB; no native inference', 'checks': []}
    with tempfile.TemporaryDirectory() as root:
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=root, terminal_metrics=False)
        service = Service(config)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with sync_playwright() as playwright:
                launch = {'headless': True}
                if os.environ.get('CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
                browser = playwright.chromium.launch(**launch)
                errors = []

                def open_page(only_folder=False):
                    context = browser.new_context(viewport={'width': 1600, 'height': 1000})
                    page = context.new_page()
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(origin + '/favicon.svg')
                    page.evaluate('''async onlyFolder => {
                        const put = (name, stores, data) => new Promise((resolve, reject) => {
                            const open=indexedDB.open(name);
                            open.onupgradeneeded=()=>{for(const s of stores)open.result.createObjectStore(s,{keyPath:'id'});};
                            open.onerror=()=>reject(open.error);
                            open.onsuccess=()=>{
                                const db=open.result,tx=db.transaction(stores,'readwrite');
                                for(const [name,rows] of Object.entries(data))for(const row of rows)tx.objectStore(name).put(row);
                                tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>reject(tx.error);
                            };
                        });
                        const rows=[['alpha','folder'],['beta','folder'],...(!onlyFolder?[['outside',null],['other','other-folder']]:[])].map(([id,folderId],i)=>({
                            id,folderId,title:id,created:1000+i,updated:1000+i,task:'chat',model:'qwen-chat',
                            settings:{},draft:'Draft '+id,messages:[{id:id+'-u',role:'user',content:'Keep '+id,task:'chat'},
                                {id:id+'-a',role:'assistant',content:'Answer '+id,task:'chat',model:'qwen-chat',status:'completed'}]
                        }));
                        await put('lazyrig-chat',['threads'],{threads:rows});
                        await put('lazyrig-chat-folders',['folders','memberships','metadata'],{
                            folders:[{id:'folder',name:'Project',collapsed:false},...(!onlyFolder?[{id:'other-folder',name:'Other',collapsed:false}]:[])],
                            memberships:rows.map(t=>({id:t.id,folderId:t.folderId})),metadata:[{id:'legacy-import',complete:true}]
                        });
                        localStorage.setItem('lazyrig.chat','alpha');
                    }''', only_folder)
                    page.goto(origin)
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.locator('nav [data-view="chat"]').click()
                    if not page.locator('#chat-history').is_visible():
                        page.locator('#chat-history-toggle').click()
                    return context, page

                def records(page, store='threads'):
                    return page.evaluate('''store => new Promise((resolve,reject)=>{
                        const open=indexedDB.open(store==='threads'?'lazyrig-chat':'lazyrig-chat-folders');
                        open.onsuccess=()=>{const db=open.result,tx=db.transaction(store),read=tx.objectStore(store).getAll();
                            read.onsuccess=()=>resolve(Object.fromEntries(read.result.map(r=>[r.id,r])));
                            tx.oncomplete=()=>db.close();tx.onerror=()=>reject(tx.error);};open.onerror=()=>reject(open.error);
                    })''', store)

                def manage(page, removal=True):
                    page.locator('[data-folder-menu="folder"]').click()
                    if removal:
                        page.locator('#chat-folder-remove-start').click()
                    expect(page.locator('#chat-folder-delete-conversations')).to_be_checked()

                def remove(page, accept=True):
                    if not page.locator('#chat-folder-remove').is_visible():
                        page.locator('#chat-folder-remove-start').click()
                    expect(page.locator('#chat-folder-summary')).to_have_text('Project')
                    page.locator('#chat-folder-remove' if accept else '#chat-folder-back').click()

                context, page = open_page()
                baseline = records(page)
                manage(page, removal=False)
                expect(page.locator('#chat-folder-summary')).to_have_text('2 conversations')
                expect(page.locator('#chat-folder-delete-option')).not_to_be_visible()
                typography = page.locator('#chat-folder-dialog .actions > button').evaluate_all('''buttons => buttons.map(button=>{
                    const s=getComputedStyle(button);return [s.fontFamily,s.fontSize,s.fontWeight,s.lineHeight,button.getBoundingClientRect().height];
                })''')
                assert typography[0] == typography[1], typography
                page.wait_for_timeout(250)  # Allow the modal animation to settle for visual QA.
                page.screenshot(path=str(output / 'manage-folder.png'))
                page.locator('#chat-folder-form [name="folder_name"]').fill('Unfinished rename')
                page.locator('#chat-folder-remove-start').click()
                expect(page.locator('#chat-folder-back')).to_be_focused()
                expect(page.locator('#chat-folder-removal-note')).to_contain_text('2 conversations will be deleted')
                expect(page.locator('.folder-removal-switch strong')).to_have_text('Delete conversations')
                page.screenshot(path=str(output / 'delete-enabled.png'))
                for width in [320, 390, 1600]:
                    page.set_viewport_size({'width': width, 'height': 1000})
                    group = page.locator('#chat-folder-delete-option').bounding_box()
                    toggle = page.locator('#chat-folder-delete-conversations').bounding_box()
                    assert group['x'] <= toggle['x'] and toggle['x'] + toggle['width'] <= group['x'] + group['width']
                    assert group['y'] <= toggle['y'] and toggle['y'] + toggle['height'] <= group['y'] + group['height']
                    assert page.locator('#chat-folder-dialog').evaluate('(el) => el.scrollWidth <= el.clientWidth')
                    if width == 390:
                        page.screenshot(path=str(output / 'mobile-remove-control.png'))
                remove(page, accept=False)
                expect(page.locator('#chat-folder-form [name="folder_name"]')).to_have_value('Unfinished rename')
                expect(page.locator('#chat-folder-form [name="folder_name"]')).to_be_focused()
                page.set_viewport_size({'width': 390, 'height': 844})
                page.screenshot(path=str(output / 'mobile-manage-folder.png'))
                page.set_viewport_size({'width': 1600, 'height': 1000})
                assert records(page) == baseline
                assert 'folder' in records(page, 'folders')
                evidence['checks'].append('Editing and removal are separate steps; matching action typography, default-on switch, and Back preserve the unfinished rename and messages')
                page.locator('#chat-folder-dialog .close-dialog').first.click()
                page.locator('#chat-prompt').fill('Pending draft must not resurrect alpha')
                manage(page)
                remove(page)
                expect(page.locator('[data-folder="folder"]')).to_have_count(0)
                expect(page.locator('[data-thread="alpha"],[data-thread="beta"]')).to_have_count(0)
                page.wait_for_timeout(650)  # Let the pending draft debounce fire.
                remaining = records(page)
                assert set(remaining) == {'outside', 'other'}
                assert set(records(page, 'memberships')) == {'outside', 'other'}
                for key in remaining:
                    assert remaining[key]['messages'] == baseline[key]['messages']
                    assert remaining[key]['draft'] == baseline[key]['draft']
                assert page.evaluate('localStorage.getItem("lazyrig.draft.alpha")') is None
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                assert set(records(page)) == {'outside', 'other'}
                evidence['checks'].append('Bulk deletion removes only folder members, memberships and drafts; late saves and reload do not resurrect them')
                context.close()

                context, page = open_page()
                baseline = records(page)
                manage(page)
                switch = page.get_by_role('switch', name='Delete conversations in this folder')
                unexpected_dialogs = []
                def unexpected_confirmation(dialog):
                    unexpected_dialogs.append(dialog.message)
                    dialog.dismiss()
                page.on('dialog', unexpected_confirmation)
                page.locator('.folder-removal-switch strong').click()
                expect(switch).not_to_be_checked()
                switch.focus()
                page.keyboard.press('Space')
                expect(switch).to_be_checked()
                page.keyboard.press('Space')
                expect(switch).not_to_be_checked()
                page.keyboard.press('Enter')
                expect(page.locator('#chat-folder-dialog')).to_be_visible()
                assert not unexpected_dialogs
                assert records(page) == baseline
                page.remove_listener('dialog', unexpected_confirmation)
                expect(page.locator('#chat-folder-removal-note')).to_contain_text('2 conversations will be kept')
                page.wait_for_timeout(250)
                page.screenshot(path=str(output / 'delete-disabled.png'))
                remove(page)
                expect(page.locator('[data-folder="folder"]')).to_have_count(0)
                assert set(records(page)) == set(baseline)
                for key in ['alpha', 'beta']:
                    assert records(page)[key]['folderId'] is None
                    assert records(page)[key]['messages'] == baseline[key]['messages']
                    assert records(page, 'memberships')[key]['folderId'] is None
                assert page.evaluate('localStorage.getItem("lazyrig.chat")') == 'alpha'
                expect(page.locator('#chat-prompt')).to_have_value('Draft alpha')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                assert records(page)['alpha']['folderId'] is None
                evidence['checks'].append('Pointer and keyboard toggles never remove the folder; switch-off keeps messages, active draft and selection through reload')
                context.close()

                context, page = open_page()
                baseline = records(page)
                manage(page)
                page.evaluate('''() => {
                    window.originalDelete=IDBObjectStore.prototype.delete;
                    IDBObjectStore.prototype.delete=function(id){
                        const result=originalDelete.call(this,id);
                        if(this.name==='threads'&&id==='beta')this.transaction.abort();
                        return result;
                    };
                }''')
                remove(page)
                expect(page.locator('#toast')).to_contain_text('Could not delete folder conversations')
                assert records(page) == baseline
                assert 'folder' in records(page, 'folders')
                page.evaluate('() => {IDBObjectStore.prototype.delete=window.originalDelete;}')
                remove(page)
                expect(page.locator('[data-folder="folder"]')).to_have_count(0)
                evidence['checks'].append('Aborted storage transaction keeps every conversation and the folder; retry succeeds')
                context.close()

                context, page = open_page(only_folder=True)
                manage(page)
                remove(page)
                expect(page.locator('[data-folder="folder"]')).to_have_count(0)
                expect(page.locator('[data-thread]')).to_have_count(1)
                rows = records(page)
                assert len(rows) == 1 and not {'alpha', 'beta'} & rows.keys()
                blank = next(iter(rows.values()))
                assert blank['folderId'] is None and blank['messages'] == []
                evidence['checks'].append('Deleting the last folder opens one usable blank conversation at root')
                context.close()

                context, page = open_page()
                blocked = []
                page.route('**/v1/chat/completions', lambda route: blocked.append(route))
                page.locator('#chat-prompt').fill('Pending request')
                with page.expect_request('**/v1/chat/completions'):
                    page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible()
                manage(page)
                page.locator('#chat-folder-remove').click()
                expect(page.locator('#toast')).to_contain_text('Stop or finish active conversations')
                assert set(records(page)) == {'alpha', 'beta', 'outside', 'other'}
                page.get_by_role('switch', name='Delete conversations in this folder').uncheck()
                remove(page)
                expect(page.locator('[data-folder="folder"]')).to_have_count(0)
                assert records(page)['alpha']['folderId'] is None
                evidence['checks'].append('Active conversations block deletion; keeping them still allows folder removal')
                context.close()

                context, page = open_page()
                for width in [1600, 1100, 390]:
                    page.set_viewport_size({'width': width, 'height': 1000})
                    for history_open in [True, False]:
                        if page.locator('#chat-history').is_visible() != history_open:
                            page.locator('#chat-history-toggle').click()
                        expect(page.locator('#chat-history #chat-new')).to_have_count(0)
                        expect(page.locator('.chat-toolbar #chat-new')).to_be_visible()
                        expect(page.locator('.chat-toolbar #chat-new')).to_be_enabled()
                        count = len(records(page))
                        if history_open:
                            page.screenshot(path=str(output / f'toolbar-history-open-{width}.png'))
                        page.locator('#chat-new').click()
                        expect(page.locator('[data-thread]')).to_have_count(count + 1)
                        new_id = page.evaluate('localStorage.getItem("lazyrig.chat")')
                        assert records(page)[new_id]['folderId'] == 'folder'
                    if width == 390:
                        page.locator('#chat-history-toggle').click()
                        page.locator('#chat-new').focus()
                        page.keyboard.press('Enter')
                        expect(page.locator('#chat-history')).not_to_be_visible()
                        expect(page.locator('#chat-prompt')).to_be_focused()
                page.set_viewport_size({'width': 1600, 'height': 1000})
                expect(page.locator('#chat-history')).to_be_visible()
                page.locator('[data-thread="alpha"]').hover()
                page.locator('[data-thread-menu="alpha"]').click()
                expect(page.locator('#chat-thread-menu')).to_be_visible()
                page.locator('#chat-new').click()
                expect(page.locator('#chat-thread-menu')).not_to_be_visible()
                evidence['checks'].append('One permanent toolbar New chat button works with history open/closed, mobile keyboard and conversation menus; selected folder destination is preserved')
                context.close()
                assert not errors, errors
                evidence['checks'].append('No uncaught browser errors')
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    (output / 'result.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
