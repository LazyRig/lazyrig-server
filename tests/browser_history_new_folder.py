"""New chats follow folder selection independently of expansion; real HTTP and IndexedDB."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.environ.get('HISTORY_FOLDER_QA_SOURCE', str(Path(__file__).resolve().parents[1])))
from playwright.sync_api import expect, sync_playwright
from lazyrig.server import handler
from support import DEFAULT, Service


def main():
    evidence = {'scope': 'Real Chromium, HTTP and IndexedDB; no inference', 'checks': []}
    output = Path(os.environ.get('HISTORY_FOLDER_QA_OUTPUT', 'test-results/history-new-folder'))
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
                page.goto('http://127.0.0.1:' + str(server.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()

                def active_id():
                    # A selected folder can differ from the conversation still displayed.
                    return page.evaluate('localStorage.getItem("lazyrig.chat")')

                def folder_button(folder):
                    return page.locator(f'[data-folder-toggle="{folder}"]')

                def expect_destination(name=None):
                    indicator = page.locator('#chat-new-destination')
                    if name:
                        expect(indicator).to_be_visible()
                        expect(page.locator('#chat-new-folder')).to_have_text(name)
                        expect(page.locator('#chat-new')).to_have_attribute('aria-label', f'New chat in folder “{name}”')
                    else:
                        expect(indicator).not_to_be_visible()
                        expect(page.locator('#chat-new')).to_have_attribute('aria-label', 'New chat outside folders')

                def add_folder(name):
                    page.locator('#chat-folder-new').click()
                    page.locator('#chat-folder-form [name="folder_name"]').fill(name)
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                    return page.locator('[data-folder-toggle]').filter(has_text=name).get_attribute('data-folder-toggle')

                def new_chat(folder):
                    previous = active_id()
                    page.locator('#chat-new').click()
                    expect(page.locator(f'[data-thread="{previous}"]')).to_have_attribute('aria-current', 'false')
                    thread = active_id()
                    destination = f'[data-folder="{folder}"]' if folder else '.history-unfiled'
                    item = page.locator(f'{destination} [data-thread="{thread}"]')
                    expect(item).to_have_attribute('aria-current', 'true')
                    if page.locator('#chat-history').is_visible() and (not folder or folder_button(folder).get_attribute('aria-expanded') == 'true'):
                        expect(item).to_be_visible()
                    else:
                        expect(item).not_to_be_visible()
                    return thread

                def records(store):
                    return page.evaluate('''name => new Promise((resolve, reject) => {
                        const open=indexedDB.open(name==='threads'?'lazyrig-chat':'lazyrig-chat-folders');
                        open.onerror=()=>reject(open.error);
                        open.onsuccess=()=>{
                            const db=open.result,tx=db.transaction(name,'readonly');
                            const get=tx.objectStore(name).getAll();
                            get.onsuccess=()=>resolve(Object.fromEntries(get.result.map(r=>[r.id,r])));
                            tx.oncomplete=()=>db.close();
                            tx.onerror=()=>reject(tx.error);
                        };
                    })''', store)

                outside = active_id()
                page.locator('#chat-prompt').fill('Keep the outside draft')
                validation = add_folder('Validation')
                # Expansion affects visibility, not the selected creation destination.
                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                closed_created = new_chat(validation)
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                expect(folder_button(validation)).to_have_attribute('aria-current', 'location')
                expect_destination('Validation')
                expect(folder_button(validation).locator('svg path')).to_have_count(1)
                assert records('memberships')[closed_created]['folderId'] == validation
                evidence['checks'].append('Closed selected folder receives New chat, stays highlighted and closed, and shows its destination')

                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                first = new_chat(validation)
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                expect(folder_button(validation)).to_have_attribute('aria-current', 'location')
                expect(folder_button(validation).locator('small')).to_have_text('2')
                assert records('memberships')[first]['folderId'] == validation
                expect(folder_button(validation).locator('svg path')).to_have_count(2)
                expect(page.locator('#chat-prompt')).to_have_value('')
                evidence['checks'].append('Open selected folder receives New chat, updates count and saves membership')

                page.locator('#chat-prompt').fill('Keep the folder draft')
                second = new_chat(validation)
                expect(folder_button(validation).locator('small')).to_have_text('3')
                expect(page.locator('#chat-prompt')).to_have_value('')
                page.locator(f'[data-thread="{first}"]').click()
                expect(page.locator('#chat-prompt')).to_have_value('Keep the folder draft')
                evidence['checks'].append('Repeated New chat stays in the folder and preserves existing drafts')

                other = add_folder('Other')
                expect(page.locator(f'[data-thread="{first}"]')).to_have_attribute('aria-current', 'false')
                third = new_chat(other)
                expect(folder_button(other)).to_have_attribute('aria-current', 'location')
                folder_button(validation).click()
                expect(page.locator(f'[data-thread="{third}"]')).to_have_attribute('aria-current', 'false')
                expect(page.locator(f'[data-thread="{third}"]')).not_to_have_class('thread-item active')
                assert active_id() == third
                expect_destination('Validation')
                page.locator(f'[data-thread="{third}"]').click()
                expect(folder_button(validation)).to_have_attribute('aria-current', 'false')
                expect(folder_button(other)).to_have_attribute('aria-current', 'location')
                fourth = new_chat(other)
                evidence['checks'].append('Selecting another folder removes the previous conversation highlight without changing its display; selecting that conversation restores its parent destination')

                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                expect(folder_button(other)).to_have_attribute('aria-expanded', 'true')
                page.locator(f'[data-thread="{third}"]').click()
                expect_destination('Other')
                page.locator(f'[data-thread="{first}"]').click()
                expect_destination('Validation')
                page.locator('#chat-history-toggle').click()
                expect(page.locator('#chat-history')).not_to_be_visible()
                hidden_child = new_chat(validation)
                expect_destination('Validation')
                page.locator('#chat-history-toggle').click()
                evidence['checks'].append('With two folders open, selecting an inside conversation sets its folder as the destination even while History is hidden')

                page.locator(f'[data-thread="{outside}"]').click()
                expect(page.locator('#chat-prompt')).to_have_value('Keep the outside draft')
                expect(page.locator('[data-folder-toggle][aria-current="location"]')).to_have_count(0)
                expect_destination()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                expect(folder_button(other)).to_have_attribute('aria-expanded', 'true')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator('#chat-prompt')).to_have_value('Keep the outside draft')
                expect(page.locator('[data-folder-toggle][aria-current="location"]')).to_have_count(0)
                expect_destination()
                page.locator('#chat-history-toggle').click()
                outside_selected_child = new_chat(None)
                expect_destination()
                page.locator('#chat-history-toggle').click()
                evidence['checks'].append('Selecting an outside conversation clears the destination and indicator across refresh and hidden History, despite other folders remaining open')

                expect(page.locator(f'[data-thread="{first}"]')).to_be_visible()
                page.locator(f'[data-thread="{first}"]').click()
                expect_destination('Validation')
                previous = active_id()
                page.evaluate('outside=>{document.querySelector(`[data-thread="${outside}"]`).click();document.querySelector("#chat-new").click();}', outside)
                expect(page.locator(f'[data-thread="{previous}"]')).to_have_attribute('aria-current', 'false')
                immediate_outside = active_id()
                expect(page.locator(f'.history-unfiled [data-thread="{immediate_outside}"]')).to_be_visible()
                expect_destination()
                assert records('memberships')[immediate_outside]['folderId'] is None
                evidence['checks'].append('Immediate outside-conversation click followed by New chat creates outside every folder in the same event turn')
                page.locator(f'[data-thread="{first}"]').click()
                fifth = new_chat(validation)
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator(f'[data-folder="{validation}"] [data-thread="{fifth}"]')).to_have_attribute('aria-current', 'true')
                sixth = new_chat(validation)
                expected = {first: validation, second: validation, third: other, fourth: other, fifth: validation, sixth: validation, hidden_child: validation, outside_selected_child: None, immediate_outside: None, closed_created: validation}
                membership = records('memberships')
                assert {key: membership[key]['folderId'] for key in expected} == expected
                assert {key: records('threads')[key]['folderId'] for key in expected} == expected
                assert folder_button(validation).evaluate('el=>getComputedStyle(el).color') != folder_button(other).evaluate('el=>getComputedStyle(el).color')
                evidence['selected_folder_colors'] = folder_button(validation).evaluate('el=>Object.fromEntries([el,...el.querySelectorAll("span,small,svg")].map(e=>[e.tagName,getComputedStyle(e).color]))')
                page.screenshot(path=str(output/'selected-folder-new-chat.png'))
                evidence['checks'].append('Conversation selection and refresh retain the folder destination and stored memberships')

                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                expect_destination('Validation')
                new_chat(validation)
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                evidence['checks'].append('Reloading a conversation in a closed folder retains its parent destination without expanding the folder')
                page.locator(f'[data-thread="{outside}"]').click()
                current_outside = active_id()
                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-current', 'location')
                expect(page.locator(f'[data-thread="{current_outside}"]')).to_have_attribute('aria-current', 'false')
                expect_destination('Validation')
                page.locator(f'[data-thread="{current_outside}"]').click()
                expect(page.locator('[data-folder-toggle][aria-current="location"]')).to_have_count(0)
                expect_destination()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                page.screenshot(path=str(output/'outside-selected-open.png'))
                folder_button(validation).click()
                expect(folder_button(validation)).to_have_attribute('aria-current', 'location')
                expect(page.locator(f'[data-thread="{current_outside}"]')).to_have_attribute('aria-current', 'false')
                expect(page.locator(f'[data-thread="{current_outside}"]').locator('..')).not_to_have_class('thread-row active')
                expect_destination('Validation')
                expect(page.locator('#chat-prompt')).to_have_value('Keep the outside draft')
                assert active_id() == current_outside
                page.screenshot(path=str(output/'closed-folder-selected-outside-displayed.png'))
                page.locator(f'[data-thread="{current_outside}"]').click()
                expect(page.locator('[data-folder-toggle][aria-current="location"]')).to_have_count(0)
                expect_destination()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                page.screenshot(path=str(output/'outside-selected-closed.png'))
                new_chat(None)
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                evidence['checks'].append('Open or closed folder selection removes the outside conversation highlight; clicking the displayed outside conversation restores its highlight and clears the folder destination')
                folder_button(validation).click()
                page.locator(f'[data-thread="{sixth}"]').click()

                page.locator(f'[data-thread-menu="{sixth}"]').click()
                page.locator('#chat-move').click()
                page.locator('#chat-folder-form [name="folder_id"]').select_option('')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                expect(page.locator('[data-folder-toggle][aria-current="location"]')).to_have_count(0)
                expect_destination()
                new_chat(None)
                evidence['checks'].append('Moving the active conversation outside clears its highlight and destination; subsequent New chat stays outside')

                folder_button(other).click()
                page.locator(f'[data-folder-menu="{other}"]').click()
                page.once('dialog', lambda dialog: dialog.accept())
                page.locator('#chat-folder-remove').click()
                expect(folder_button(other)).to_have_count(0)
                new_chat(None)
                assert other not in records('folders')
                assert records('memberships')[third]['folderId'] is None
                evidence['checks'].append('Removing the selected folder preserves conversations and clears the New chat destination')

                previous = active_id()
                page.evaluate('folder=>{document.querySelector(`[data-folder-toggle="${folder}"]`).click();document.querySelector("#chat-new").click();}', validation)
                expect(page.locator(f'[data-thread="{previous}"]')).to_have_attribute('aria-current', 'false')
                expect(page.locator(f'[data-folder="{validation}"] [data-thread="{active_id()}"]')).to_have_attribute('aria-current', 'true')
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'false')
                expect_destination('Validation')
                previous = active_id()
                page.evaluate('folder=>{document.querySelector(`[data-folder-toggle="${folder}"]`).click();document.querySelector("#chat-new").click();}', validation)
                expect(page.locator(f'[data-thread="{previous}"]')).to_have_attribute('aria-current', 'false')
                expect(page.locator(f'[data-folder="{validation}"] [data-thread="{active_id()}"]')).to_be_visible()
                expect(folder_button(validation)).to_have_attribute('aria-expanded', 'true')
                evidence['checks'].append('Immediate close/New-chat and open/New-chat both create in the selected folder before persistence finishes')
                assert not errors, errors
                evidence['checks'].append('No uncaught JavaScript errors')
                evidence['browser'] = browser.version
                browser.close()
        finally:
            server.shutdown()
            service.close()
    (output/'result.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
