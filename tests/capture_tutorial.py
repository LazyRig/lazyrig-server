"""Capture the real dashboard for docs using explicitly illustrative test data.

No model weights are downloaded or loaded. This is a documentation/test utility,
not a production mode. Install Playwright and Chromium; optionally pass --browser
with the path to an already installed Chromium executable.
"""
import argparse
import copy
import json
import os
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright, expect
from support import DEFAULT, Service, FixtureWorker
from test_image_edits import export_fixture
from lazyrig.server import handler


class TutorialService(Service):
    def discover(self):
        import openvino
        return {'devices': self.config['devices'], 'available': True,
                'note': 'Illustrative documentation configuration; not a hardware benchmark',
                'openvino_version': openvino.__version__}


class TutorialWorker(FixtureWorker):
    calls = []

    def run(self, request, timeout, cancelled):
        self.calls.append({'action': request['action'],
                           'images': len(request.get('prepared', {}).get('images', []))})
        if request['action'] != 'text':
            yield from super().run(request, timeout, cancelled)
            return
        prepared = request['prepared']
        last = prepared['messages'][-1]
        functions = [tool['function'] for tool in json.loads(prepared['text']).get('tools') or []]
        if last['role'] == 'tool' or str(last['content']).startswith('Generated outputs from completed image tools.'):
            reply = 'The image tool finished and returned the result to this chat.'
        elif functions:
            edit = next((tool for tool in functions if tool['name'] == 'edit_image'), None)
            arguments = {'prompt': last['content']}
            if edit and 'blue' in last['content'].lower():
                arguments['image_id'] = edit['parameters']['properties']['image_id']['enum'][-1]
            else:
                edit = None
            reply = '<tool_call>' + json.dumps({'name': 'edit_image' if edit else 'generate_image',
                                                'arguments': arguments}) + '</tool_call>'
        else:
            reply = ('You can use this VLM for text chat and image understanding. '
                     'Enable an Image tool in Parameters to let it call image generation '
                     'and editing from the same conversation.')
        yield {'kind': 'text', 'text': reply, 'generated_tokens': 47, 'accuracy': 'estimated-test'}
        yield {'kind': 'result', 'text': reply, 'generated_tokens': 47, 'accuracy': 'estimated-test'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/images/tutorial')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    captures = []
    with tempfile.TemporaryDirectory(prefix='lazyrig-guide-') as temp:
        temp = Path(temp); os.chdir(temp)
        try:
            Path('models').mkdir()
            export_fixture('models/my-image', qwen=False)
            config = copy.deepcopy(DEFAULT)
            config.update(data_dir=str(temp / 'data'), models_dir='models', port=args.port, terminal_metrics=False)
            config['devices'] = [
                {'id': 'GPU.0', 'name': 'Example GPU', 'budget_gib': 20, 'policy': 'budget'},
                {'id': 'CPU', 'name': 'Example CPU', 'budget_gib': 64, 'policy': 'budget'},
            ]
            chat, image = config['models']
            chat.update(id='my-chat', name='Chat & vision · example', pipeline='vlm',
                        device='GPU.0', path='models/my-chat', memory_gib=8)
            image.update(id='my-image', name='Image model · example', device='GPU.0', path='models/my-image', memory_gib=12)
            image['generation'] = {'width': 512, 'height': 512, 'steps': 20, 'seed': -1}
            TutorialWorker.calls = []
            service = TutorialService(config); service.scheduler.factory = TutorialWorker
            http = ThreadingHTTPServer(('127.0.0.1', args.port), handler(service))
            thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(executable_path=args.browser, args=['--no-sandbox'])
                    page = browser.new_page(viewport={'width': 1280, 'height': 960}, device_scale_factor=1)
                    errors = []; page.on('pageerror', lambda err: errors.append(str(err)))
                    page.goto(f'http://127.0.0.1:{http.server_port}')
                    expect(page.locator('#connection')).to_have_text('Server online')
                    page.wait_for_function('() => window.Chat && state && !pendingRefresh')

                    def nav(name):
                        page.locator(f'nav [data-view="{name}"]').click()
                        page.evaluate('window.scrollTo(0,0)')
                        page.wait_for_timeout(180)

                    def annotate(targets):
                        page.evaluate('''targets => {
                          document.querySelectorAll('[data-guide-overlay]').forEach(e=>e.remove());
                          targets.forEach((target,index)=>{
                            const el=document.querySelector(target);
                            if(!el)throw new Error('Missing screenshot target '+target);
                            const r=el.getBoundingClientRect();
                            if(!r.width || !r.height)throw new Error('Hidden screenshot target '+target);
                            const box=document.createElement('div');box.dataset.guideOverlay='';
                            box.style.cssText=`position:absolute;pointer-events:none;z-index:2147483646;border:2px solid #f5ab62;border-radius:6px;left:${r.left+scrollX-4}px;top:${r.top+scrollY-4}px;width:${r.width+8}px;height:${r.height+8}px;box-shadow:0 0 0 2px #101316b3;`;
                            const badge=document.createElement('span');badge.textContent=String(index+1);
                            badge.style.cssText='position:absolute;right:-11px;top:-13px;width:25px;height:25px;border-radius:50%;background:#f5ab62;border:2px solid #101316;color:#101316;font:700 13px/21px system-ui;text-align:center;box-shadow:0 2px 6px #0007';
                            box.append(badge);
                            // A dialog's top layer needs annotations in that same layer.
                            const dialog=el.closest('dialog[open]');
                            if(dialog){const dr=dialog.getBoundingClientRect();box.style.left=(r.left-dr.left-4)+'px';box.style.top=(r.top-dr.top-4+dialog.scrollTop)+'px';dialog.append(box);}
                            else document.body.append(box);
                          });
                        }''', targets)

                    def capture(name, targets, clip=None):
                        page.mouse.move(1260, 945)
                        page.locator('#toast').evaluate('(el)=>{el.hidden=true}')
                        page.wait_for_timeout(160)
                        bounds = clip or {'x': 0, 'y': 0, **page.viewport_size}
                        for target in targets:
                            rect = page.locator(target).bounding_box()
                            assert rect and all((
                                rect['x'] >= bounds['x'],
                                rect['y'] >= bounds['y'],
                                rect['x'] + rect['width'] <= bounds['x'] + bounds['width'],
                                rect['y'] + rect['height'] <= bounds['y'] + bounds['height'],
                            )), f'{name}: highlight outside screenshot: {target}'
                        annotate(targets)
                        path=output / name
                        page.screenshot(path=str(path), clip=clip, animations='disabled')
                        page.evaluate("document.querySelectorAll('[data-guide-overlay]').forEach(e=>e.remove())")
                        captures.append(name); print('Captured', name, flush=True)

                    capture('01-dashboard.png', ['#connection', '#add-model', 'nav [data-view="devices"]', 'nav [data-view="chat"]'])
                    nav('downloads')
                    page.locator('#download-destination').fill('models/new-model')
                    capture('02-import-model.png', ['#download-url', '#download-destination', '#download-form .grid.three', '#download-form button[type="submit"]'])
                    nav('models'); page.locator('#add-model').click()
                    page.locator('#register-path').fill('models/my-chat')
                    page.locator('#register-id').fill('new-chat')
                    page.locator('#register-name').fill('My local chat and vision model')
                    page.locator('#register-device').select_option('GPU.0')
                    page.locator('#register-form [name="memory_gib"]').fill('8')
                    dialog = page.locator('#register-dialog').bounding_box()
                    capture('03-local-export.png', ['#register-path', '#inspect-folder', '#register-id', '#register-form button[type="submit"]'],
                            {'x': dialog['x']-20, 'y': dialog['y']-20, 'width': dialog['width']+40, 'height': dialog['height']+40})
                    page.keyboard.press('Escape')
                    nav('devices');page.set_viewport_size({'width':1800,'height':960});page.wait_for_timeout(180)
                    device_box=page.locator('#devices-grid form').first.bounding_box()
                    capture('04-device-memory.png', ['#devices-grid form:first-child [name="budget_gib"]', '#devices-grid form:first-child [name="policy"]', '#devices-grid form:first-child button[type="submit"]' if page.locator('#devices-grid form:first-child button[type="submit"]').count() else '#devices-grid form:first-child .actions button'],
                            {'x':device_box['x']-15,'y':device_box['y']-15,'width':device_box['width']+30,'height':device_box['height']+30})
                    page.set_viewport_size({'width':1280,'height':960})
                    nav('models'); page.locator('[data-edit="my-image"]').last.click()
                    expect(page.locator('[name="keep_image_model_loaded"]')).to_have_value('auto')
                    page.evaluate("window.scrollTo(0,document.querySelector('#model-form').getBoundingClientRect().top+scrollY-95)")
                    box=page.locator('#model-form').bounding_box()
                    generation=page.get_by_role('heading',name='Default generation settings').bounding_box()
                    capture('05-image-retention.png', ['#model-form [name="device"]','#model-form [name="memory_gib"]','#model-form [name="keep_image_model_loaded"]','#model-form [name="restore_text_after_image"]'],
                            {'x':box['x']-10,'y':box['y']-10,'width':box['width']+20,'height':generation['y']-box['y']-6})
                    # Save the actual control through the production settings endpoint.
                    page.locator('#model-form button[type="submit"]').click()
                    expect(page.locator('#toast')).to_contain_text('Settings saved')
                    assert service.model('my-image')['keep_image_model_loaded']=='auto'
                    nav('chat')
                    page.locator('#chat-task').select_option('chat'); page.locator('#chat-model').select_option('my-chat')
                    page.locator('#chat-prompt').fill('What can I do in this workspace?')
                    page.locator('#chat-send').click()
                    expect(page.locator('.message.assistant .message-body')).to_contain_text('You can use this VLM',timeout=15000)
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                    page.wait_for_function("() => state.scheduler.states['my-chat']==='loaded'")
                    capture('06-chat.png', ['#chat-model', '#chat-settings-toggle', '#chat-prompt', '#chat-history-toggle'])
                    page.locator('#chat-prompt').fill('Describe the main objects in this image.')
                    expect(page.locator('#chat-model')).to_have_value('my-chat')
                    capture('06b-vision.png', ['#chat-model', '#chat-attach', '#chat-prompt'])
                    page.locator('#chat-new').click()
                    page.locator('#chat-settings-toggle').click()
                    options = page.locator('#chat-options')
                    options.locator('[name="image_model"]').select_option('my-image')
                    expect(options.locator('[name="enhance_prompt"]')).not_to_be_checked()
                    page.set_viewport_size({'width':1280,'height':1400})
                    page.wait_for_timeout(180)
                    toolbar=page.locator('.chat-toolbar').bounding_box()
                    controls=options.bounding_box()
                    capture('06c-vlm-image-tools.png', ['#chat-model', '#chat-options [name="image_model"]',
                            '#chat-options [name="size"]', '#chat-options [name="enhance_prompt"]'],
                            {'x':toolbar['x']-10,'y':toolbar['y']-15,'width':toolbar['width']+20,
                             'height':controls['y']+controls['height']-toolbar['y']+30})
                    page.set_viewport_size({'width':1280,'height':960})
                    page.locator('#chat-settings-toggle').click()
                    # Exercise the documented VLM -> image tool -> VLM loop.
                    def send_tool_prompt(prompt, expected_actions, expected_images):
                        before = len(TutorialWorker.calls)
                        page.locator('#chat-prompt').fill(prompt)
                        page.locator('#chat-send').click()
                        expect(page.locator('#chat-stop')).to_be_visible()
                        expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                        expect(page.locator('.job-error')).to_have_count(0)
                        expect(page.locator('.message.assistant .message-images img')).to_have_count(expected_images)
                        calls = TutorialWorker.calls[before:]
                        assert [call['action'] for call in calls] == expected_actions, calls
                        assert calls[-1]['images'], 'VLM continuation must receive output pixels'
                        expect(page.locator('#chat-model')).to_have_value('my-chat')
                    send_tool_prompt('Generate an image of a small red cabin beside a lake, soft morning light.',
                                     ['text', 'image', 'text'], 1)
                    page.locator('.artifact-use').click()
                    expect(page.locator('#chat-attachments img')).to_be_visible()
                    send_tool_prompt('Edit this image: change the cabin door to blue. Keep the composition and lighting.',
                                     ['text', 'image_edit', 'text'], 2)
                    page.locator('#chat-new').click();page.locator('#chat-task').select_option('image')
                    page.locator('#chat-model').select_option('my-image')
                    if not page.locator('#chat-options').is_visible():page.locator('#chat-settings-toggle').click()
                    page.locator('#chat-prompt').fill('A small red cabin beside a lake, soft morning light, simple composition.')
                    capture('07-image-generation.png', ['#chat-task', '#chat-model', '#chat-options [name="size"]', '#chat-send'])
                    page.locator('#chat-task').select_option('edit')
                    page.locator('#chat-prompt').fill('Change the cabin door to blue. Keep the composition and lighting.')
                    if page.locator('#chat-options').is_visible():page.locator('#chat-settings-toggle').click()
                    capture('08-image-editing.png', ['#chat-task', '#chat-model', '#chat-attach', '#chat-prompt'])
                    nav('server')
                    page.set_viewport_size({'width':1280,'height':1500})
                    page.wait_for_timeout(180)
                    panel=page.locator('#server-view > .panel').first.bounding_box()
                    page.set_viewport_size({'width':1280,'height':round(panel['y']+panel['height']+16)})
                    capture('09-api-client.png', ['#base-url', '#api-model', '#example-type', '#copy-example'])
                    page.set_viewport_size({'width':1280,'height':960})
                    page.locator('#server-form').scroll_into_view_if_needed()
                    page.evaluate("window.scrollTo(0,document.querySelector('#server-form').getBoundingClientRect().top+scrollY-95)")
                    box=page.locator('#server-form').bounding_box()
                    capture('10-server-access.png', ['#server-form [name="host"]', '#server-api-key'],
                            {'x':box['x']-10,'y':box['y']-10,'width':box['width']+20,'height':min(box['height']+20,850)})
                    nav('models');page.locator('[data-edit="my-chat"]').last.click()
                    heading=page.get_by_role('heading', name='Default generation settings')
                    heading.scroll_into_view_if_needed()
                    page.evaluate("window.scrollTo(0,Array.from(document.querySelectorAll('#model-form h2')).find(e=>e.textContent==='Default generation settings').getBoundingClientRect().top+scrollY-105)")
                    x=page.locator('#model-form').bounding_box()['x'];w=page.locator('#model-form').bounding_box()['width']
                    capture('11-text-settings.png', ['#model-form [name="gen_temperature"]','#model-form [name="mtp_enabled"]','#model-form [name="runtime_KV_CACHE_PRECISION"]'],
                            {'x':x-8,'y':92,'width':w+16,'height':790})
                    assert not errors,errors
                    browser.close()
            finally:
                http.shutdown();http.server_close();thread.join(3);service.close()
        finally:os.chdir(original_cwd)
    print(json.dumps({'screenshots':captures,'count':len(captures),'page_errors':errors,
                      'vlm_tool_calls': TutorialWorker.calls},indent=2))


if __name__=='__main__':main()
