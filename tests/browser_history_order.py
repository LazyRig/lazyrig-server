"""Conversation history recency is changed by sends, not browser navigation.

Runs the real frontend, IndexedDB and HTTP service with deterministic workers.
Set CHROMIUM_EXECUTABLE when using an existing Chromium installation.
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
    evidence = {
        'scope': 'Real Chromium, IndexedDB and HTTP; deterministic worker responses',
        'checks': [],
    }
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

                def check(name, actual, expected):
                    evidence['checks'].append({
                        'name': name, 'passed': actual == expected,
                        'actual': actual, 'expected': expected,
                    })

                def ids():
                    return page.locator('#chat-threads [data-thread]').evaluate_all(
                        'buttons => buttons.map(button => button.dataset.thread)')

                def active_id():
                    return page.locator('#chat-threads [aria-current="true"]').get_attribute('data-thread')

                def records(store='threads', target=None):
                    return (target or page).evaluate('''name => new Promise((resolve, reject) => {
                        const request = indexedDB.open(name === 'threads' ? 'lazyrig-chat' : 'lazyrig-chat-folders');
                        request.onerror = () => reject(request.error);
                        request.onsuccess = () => {
                            const database = request.result;
                            const transaction = database.transaction(name, 'readonly');
                            const read = transaction.objectStore(name).getAll();
                            read.onsuccess = () => resolve(Object.fromEntries(
                                read.result.map(thread => [thread.id, thread])));
                            read.onerror = () => reject(read.error);
                            transaction.oncomplete = () => database.close();
                        };
                    })''', store)

                def timestamps():
                    return {key: thread['updated'] for key, thread in records().items()}

                def select(thread):
                    # Separate UI actions in time, so accidental timestamp changes
                    # cannot be hidden by several events landing in one millisecond.
                    page.wait_for_timeout(60)
                    page.locator(f'#chat-threads [data-thread="{thread}"]').click()
                    expect(page.locator(f'#chat-threads [data-thread="{thread}"]')).to_have_attribute(
                        'aria-current', 'true')

                def send(text):
                    page.locator('#chat-prompt').fill(text)
                    page.locator('#chat-send').click()
                    expect(page.locator('.message.user .message-body').last).to_have_text(text)
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                    expect(page.locator('.message.assistant .message-body').last).to_contain_text(
                        'Test worker response.')

                page.goto('http://127.0.0.1:' + str(server.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                send('Alpha conversation')
                alpha = active_id()
                page.locator('#chat-new').click()
                send('Beta conversation')
                beta = active_id()
                page.locator('#chat-new').click()
                send('Gamma conversation')
                gamma = active_id()
                page.wait_for_timeout(60)
                page.locator('#chat-new').click()
                blank = active_id()
                expected = [blank, gamma, beta, alpha]
                check('New conversations start first', ids(), expected)
                baseline = timestamps()

                # Include both an empty chat and existing conversations. Merely
                # leaving a thread must not touch it either.
                for name, thread in [('Alpha', alpha), ('Gamma', gamma), ('empty chat', blank), ('Beta', beta)]:
                    select(thread)
                    check('Selecting ' + name + ' preserves order', ids(), expected)
                    check('Selecting ' + name + ' preserves recency timestamps', timestamps(), baseline)

                page.locator('#chat-prompt').fill('An unsent draft')
                page.wait_for_timeout(650)  # The actual draft debounce is 500 ms.
                check('Saving a draft preserves recency timestamps', timestamps(), baseline)
                select(blank)
                check('Leaving a draft preserves order', ids(), expected)
                select(beta)
                expect(page.locator('#chat-prompt')).to_have_value('An unsent draft')

                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-options [name="temperature"]').fill('0.4')
                page.locator('#chat-options [name="thinking"]').select_option('false')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                check('Saving parameters preserves recency timestamps', timestamps(), baseline)
                select(alpha)
                check('Leaving parameter changes preserves order', ids(), expected)
                select(beta)
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                expect(page.locator('#chat-options [name="temperature"]')).to_have_value('0.4')
                expect(page.locator('#chat-options [name="thinking"]')).to_have_value('false')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()

                send('Beta follow-up')
                expected = [beta, blank, gamma, alpha]
                check('Sending a new message moves that conversation first', ids(), expected)
                after_send = timestamps()
                check('A new message advances its conversation timestamp', after_send[beta] > baseline[beta], True)
                check('A new message leaves other recency timestamps unchanged',
                      {key: value for key, value in after_send.items() if key != beta},
                      {key: value for key, value in baseline.items() if key != beta})

                # Reload a non-leading selected thread to cover restoring the
                # last-opened conversation and updateModels/updateTask side effects.
                select(alpha)
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                if not page.locator('#chat-history').is_visible():
                    page.locator('#chat-history-toggle').click()
                check('Reload restores the selected conversation', active_id(), alpha)
                check('Reload preserves history order', ids(), expected)
                check('Reload preserves recency timestamps', timestamps(), after_send)
                def folder_id(name):
                    return page.locator('.history-folder-toggle').filter(has_text=name).get_attribute('data-folder-toggle')

                def manage(folder):
                    page.locator(f'[data-folder-menu="{folder}"]').click()
                    expect(page.locator('#chat-folder-dialog')).to_be_visible()

                def create_folder(name):
                    page.locator('#chat-folder-new').click()
                    page.locator('#chat-folder-form [name="folder_name"]').fill(name)
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                    return folder_id(name)

                def move_thread(thread, folder=None):
                    select(thread)
                    page.locator(f'[data-thread-menu="{thread}"]').click()
                    page.locator('#chat-move').click()
                    page.locator('#chat-folder-form [name="folder_id"]').select_option(folder or '')
                    page.locator('#chat-folder-submit').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()

                def remove_folder(folder):
                    manage(folder)
                    page.once('dialog', lambda dialog: dialog.accept())
                    page.locator('#chat-folder-remove').click()
                    expect(page.locator('#chat-folder-dialog')).not_to_be_visible()

                expect(page.locator('#chat-move')).not_to_be_visible()
                projects = create_folder('Projects')
                wardrobe = create_folder('Wardrobe')
                check('Expanded folders keep a borderless highlight',
                      page.locator(f'[data-folder-toggle="{wardrobe}"]').evaluate('button => [getComputedStyle(button).borderTopWidth, getComputedStyle(button).boxShadow]'),
                      ['0px', 'none'])
                check('Folders are optional and flat',
                      [sorted(folder) for folder in records('folders').values()],
                      [['collapsed', 'id', 'name'], ['collapsed', 'id', 'name']])
                move_thread(beta, wardrobe)
                move_thread(gamma, projects)
                check('Moving conversations persists folder membership',
                      [records()[beta]['folderId'], records()[gamma]['folderId']], [wardrobe, projects])
                move_thread(alpha, projects)
                move_thread(alpha)
                check('Conversations can move back to the normal list', records()[alpha]['folderId'], None)
                check('Organizing folders preserves conversation recency', timestamps(), after_send)
                select(alpha)
                page.locator('#chat-prompt').fill('Keep this draft while organizing')
                source = page.locator(f'[data-thread="{beta}"]')
                source.drag_to(page.locator(f'[data-folder-toggle="{projects}"]'))
                expect(page.locator(f'[data-folder="{projects}"] [data-thread="{beta}"]')).to_be_visible()
                check('Dragging moves a conversation directly into a folder',
                      records('memberships')[beta]['folderId'], projects)
                check('Dragging another conversation preserves selection and draft',
                      [active_id(), page.locator('#chat-prompt').input_value()],
                      [alpha, 'Keep this draft while organizing'])
                source.drag_to(page.locator(f'[data-folder-toggle="{projects}"]'))
                check('Dropping into the same folder does not duplicate the conversation',
                      page.locator(f'[data-thread="{beta}"]').count(), 1)
                source.drag_to(page.locator('.history-unfiled'))
                expect(page.locator(f'.history-unfiled [data-thread="{beta}"]')).to_be_visible()
                check('Dragging outside folders removes folder membership',
                      records('memberships')[beta]['folderId'], None)
                page.locator(f'[data-folder-toggle="{wardrobe}"]').click()
                expect(page.locator(f'[data-folder-toggle="{wardrobe}"]')).to_have_attribute('aria-expanded', 'false')
                source.drag_to(page.locator(f'[data-folder-toggle="{wardrobe}"]'))
                expect(page.locator(f'[data-folder="{wardrobe}"] [data-thread="{beta}"]')).to_be_visible()
                check('Dropping onto a collapsed folder reveals the moved conversation',
                      records('folders')[wardrobe]['collapsed'], False)
                source.drag_to(page.locator('#chat-composer'))
                check('Dropping outside history leaves the folder and composer unchanged',
                      [records('memberships')[beta]['folderId'], page.locator('#chat-attachments img').count(),
                       page.locator('#chat-composer').evaluate("el => el.classList.contains('drop-target')")],
                      [wardrobe, 0, False])
                check('Drag moves preserve all conversation recency timestamps', timestamps(), after_send)
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator(f'[data-folder="{wardrobe}"] [data-thread="{beta}"]')).to_be_visible()
                check('Drag destination, selected conversation and draft survive reload',
                      [records('memberships')[beta]['folderId'], active_id(), page.locator('#chat-prompt').input_value()],
                      [wardrobe, alpha, 'Keep this draft while organizing'])
                manage(wardrobe)
                check('Folder dialog uses compact action text',
                      page.locator('#chat-folder-dialog .actions .close-dialog').evaluate('button => getComputedStyle(button).fontSize'), '14px')
                page.locator('#chat-folder-form [name="folder_name"]').fill('Outfits')
                page.locator('#chat-folder-submit').click()
                expect(page.locator('#chat-folder-dialog')).not_to_be_visible()
                check('Renaming preserves folder identity and contents',
                      [records('folders')[wardrobe]['name'], records()[beta]['folderId']], ['Outfits', wardrobe])

                page.locator(f'[data-folder-toggle="{wardrobe}"]').click()
                expect(page.locator(f'[data-folder-toggle="{wardrobe}"]')).to_have_attribute('aria-expanded', 'false')
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                expect(page.locator(f'[data-folder-toggle="{wardrobe}"]')).to_have_attribute('aria-expanded', 'false')
                expect(page.locator(f'[data-thread="{beta}"]')).not_to_be_visible()
                check('Reload preserves folders and collapse state',
                      [records('folders')[wardrobe]['name'], records('folders')[wardrobe]['collapsed']], ['Outfits', True])
                page.locator(f'[data-folder-toggle="{wardrobe}"]').click()
                expect(page.locator(f'[data-thread="{beta}"]')).to_be_visible()
                select(beta)
                page.locator('#chat-new').click()
                new_filed = active_id()
                check('New chats inherit the selected conversation folder', records()[new_filed]['folderId'], wardrobe)
                expect(page.locator(f'[data-folder="{wardrobe}"] [data-thread="{new_filed}"]')).to_be_visible()
                folder_timestamps = timestamps()
                messages_before_remove = {key: thread['messages'] for key, thread in records().items()}
                remove_folder(wardrobe)
                check('Removing a folder returns all its conversations to the normal list',
                      [records()[beta]['folderId'], records()[new_filed]['folderId']], [None, None])
                remove_folder(projects)
                check('Removing another folder preserves its conversation', records()[gamma]['folderId'], None)
                check('All folders can be removed without losing a conversation',
                      {key: thread['messages'] for key, thread in records().items()}, messages_before_remove)
                check('Folder rename/removal do not change recency', timestamps(), folder_timestamps)
                expect(page.locator('#chat-move')).not_to_be_visible()
                page.reload()
                expect(page.locator('#connection')).to_have_text('Server online')
                check('Removed folders stay removed after reload', records('folders'), {})
                check('Final conversation membership persists after reload',
                      {key: thread.get('folderId') for key, thread in records().items()},
                      {key: None for key in folder_timestamps})

                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-history-toggle').click()
                select(alpha)
                actions = page.locator(f'[data-thread-menu="{beta}"]')
                actions.click()
                expect(page.locator('#chat-thread-menu')).to_be_visible()
                check('Conversation actions leave the current selection unchanged', active_id(), alpha)
                check('Add folder remains accessible from the compact History header',
                      page.locator('#chat-history .panel-heading #chat-folder-new').get_attribute('aria-label'), 'Add folder')
                page.keyboard.press('Escape')
                expect(page.locator('#chat-thread-menu')).not_to_be_visible()
                expect(actions).to_be_focused()
                expect(page.locator('#chat-history')).to_be_visible()
                check('Escape closes the action menu without closing history', True, True)

                actions.click()
                with page.expect_download() as download:
                    page.locator('#chat-export').click()
                exported = json.loads(Path(download.value.path()).read_text())
                check('Export uses the chosen conversation, not the active one', exported['id'], beta)
                check('Export does not reorder conversations', timestamps(), folder_timestamps)
                actions.click()
                page.once('dialog', lambda dialog: dialog.accept())
                page.locator('#chat-delete').click()
                expect(page.locator(f'[data-thread="{beta}"]')).to_have_count(0)
                check('Deleting another conversation preserves the active conversation', active_id(), alpha)
                check('Delete removes only the chosen conversation', sorted(records()), sorted(set(folder_timestamps)-{beta}))

                # The same compact menu stays inside a phone-sized viewport.
                page.set_viewport_size({'width': 390, 'height': 844})
                page.locator(f'[data-thread-menu="{alpha}"]').click()
                rect = page.locator('#chat-thread-menu').bounding_box()
                check('Conversation actions fit the mobile viewport',
                      rect['x'] >= 0 and rect['y'] >= 0 and rect['x'] + rect['width'] <= 390 and rect['y'] + rect['height'] <= 844, True)
                page.keyboard.press('Escape')
                page.set_viewport_size({'width': 1440, 'height': 1000})

                # Seed the previous released database schema in a separate
                # browser context before loading the upgraded frontend.
                migration = browser.new_page(viewport={'width': 1440, 'height': 1000})
                migration.goto('http://127.0.0.1:' + str(server.server_port) + '/favicon.svg')
                legacy = {
                    'id': 'legacy-thread', 'title': 'Preserved legacy chat',
                    'created': 1000, 'updated': 2000, 'task': 'chat', 'model': 'qwen-chat',
                    'settings': {'temperature': '0.7'}, 'draft': 'Preserved unsent draft',
                    'messages': [{'id': 'legacy-user', 'role': 'user', 'task': 'chat',
                                  'content': 'Existing message', 'attachments': []}],
                }
                migration.evaluate('''thread => new Promise((resolve, reject) => {
                    const request = indexedDB.open('lazyrig-chat', 1);
                    request.onupgradeneeded = () => request.result.createObjectStore('threads', {keyPath: 'id'});
                    request.onerror = () => reject(request.error);
                    request.onsuccess = () => {
                        const database = request.result, transaction = database.transaction('threads', 'readwrite');
                        transaction.objectStore('threads').put(thread);
                        transaction.oncomplete = () => {database.close(); localStorage.setItem('lazyrig.chat', thread.id); resolve();};
                    };
                })''', legacy)
                migration.goto('http://127.0.0.1:' + str(server.server_port))
                expect(migration.locator('#connection')).to_have_text('Server online')
                migration.locator('nav [data-view="chat"]').click()
                expect(migration.locator('#chat-prompt')).to_have_value('Preserved unsent draft')
                expect(migration.locator('.message.user .message-body')).to_have_text('Existing message')
                migrated = records(target=migration)['legacy-thread']
                check('Version 1 migration preserves legacy content, draft, settings and timestamps',
                      [migrated['messages'], migrated['draft'], migrated['settings']['temperature'], migrated['created'], migrated['updated']],
                      [legacy['messages'], legacy['draft'], '0.7', 1000, 2000])
                check('Version 1 chats use a separate empty folders store', records('folders', migration), {})
                migration.close()
                check('No uncaught browser errors', errors, [])
                evidence['browser'] = browser.version
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            service.close()
    print(json.dumps(evidence, indent=2))
    failed = [item['name'] for item in evidence['checks'] if not item['passed']]
    assert not failed, 'History order regression: ' + '; '.join(failed)


if __name__ == '__main__':
    main()
