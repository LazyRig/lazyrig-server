"""Live chat edit/enhancement controls; real HTTP/Chromium and deterministic workers."""
import base64
import copy
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from lazyrig import __version__
from lazyrig.server import handler
from lazyrig.worker import Cancelled
from support import DEFAULT, Service, FixtureWorker
from test_chat_image_tools import source_image, rewritten, edit_call
from test_image_edits import export_fixture


def main():
    out = Path('test-results'); out.mkdir(exist_ok=True)
    evidence = {'scope': 'Real Chromium and HTTP; deterministic workers, not model-quality testing', 'checks': []}
    generation_rewrite = json.loads(rewritten())
    generation_rewrite['rewritten_prompt'] = 'An orange bicycle.\nA lake in the background.'
    with tempfile.TemporaryDirectory() as directory:
        config = copy.deepcopy(DEFAULT)
        config['data_dir'] = directory
        config['models'][0]['pipeline'] = 'vlm'
        config['models'].append({**copy.deepcopy(config['models'][0]), 'id': 'text-chat', 'pipeline': 'llm'})
        config['models'][1].update(path=str(export_fixture(Path(directory)/'qwen')), runtime_profile='qwen-image-21',
            max_width=2752, max_height=2752, generation={'width': 512, 'height': 512, 'steps': 3})
        service = Service(config)
        calls = []

        def factory(*args):
            worker = FixtureWorker(*args); run = worker.run
            def tracked(request, timeout, cancelled):
                calls.append(copy.deepcopy(request))
                if request['action'] == 'text':
                    last = request['prepared']['messages'][-1]
                    if request.get('response_format', {}).get('type') == 'json_schema':
                        if 'Cancel enhancement' in last['content']:
                            while not cancelled.wait(.01):
                                pass
                            raise Cancelled()
                        text = rewritten(True) if request['prepared']['images'] else json.dumps(generation_rewrite)
                    elif json.loads(request['prepared']['text']).get('tools') and last['role'] != 'tool' and not str(last['content']).startswith('Generated outputs from completed image tools.'):
                        tools = json.loads(request['prepared']['text'])['tools']
                        edit = next((t['function'] for t in tools if t['function']['name'] == 'edit_image'), None)
                        if 'Repair tool name' in last['content'] and not any('Use only the declared functions' in str(m['content']) for m in request['prepared']['messages'] if m['role'] == 'system'):
                            text = '{"function":"image_gen","args":{"prompt":"A train"}}'
                        elif 'Two images' in last['content']:
                            text = ''.join('<tool_call>'+json.dumps({'name': 'generate_image', 'arguments': {'prompt': subject}})+'</tool_call>' for subject in ('A train', 'A bridge'))
                        elif edit and any(word in last['content'].lower() for word in ('blue', 'warmer', 'edit')):
                            text = edit_call(edit['parameters']['properties']['image_id']['enum'][-1], last['content'])
                        else:
                            text = '<tool_call>'+json.dumps({'name': 'generate_image', 'arguments': {'prompt': last['content']}})+'</tool_call>'
                    elif last['role']=='tool' or str(last['content']).startswith('Generated outputs from completed image tools.'):
                        text = 'The image file is ready.'
                    else:
                        yield from run(request, timeout, cancelled)
                        return
                    yield {'kind': 'result', 'text': text, 'generated_tokens': 100, 'accuracy': 'estimated-test'}
                    return
                yield from run(request, timeout, cancelled)
            worker.run = tracked
            return worker
        service.scheduler.factory = factory
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('response', lambda response: print('HTTP failure:', response.status, response.text(), flush=True) if response.status >= 400 else None)
                inference_requests = []
                page.on('request', lambda request: inference_requests.append((request.url, request.post_data_json)) if request.method == 'POST' and '/v1/' in request.url else None)
                page.goto('http://127.0.0.1:'+str(http.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                invalid_edit = {**generation_rewrite, 'wh_ratio': '', 'ratio_follow': '<image1>'}
                assert page.evaluate('''value => {
                    try { InferenceClient.parseEnhancement({finish_reason:'stop', message:{content:JSON.stringify(value)}}, true); }
                    catch { return true; }
                    return false;
                }''', invalid_edit), 'Editing must continue to reject multiline enhancements'
                page.locator('nav [data-view="chat"]').click()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                options = page.locator('#chat-options')
                expect(options.locator('[name="image_model"]')).to_have_value('qwen-image')
                expect(options.locator('[name="max_tokens"]')).to_have_attribute('placeholder', '8192')
                expect(options.locator('[name="enhance_prompt"]')).not_to_be_checked()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()

                def send(prompt):
                    page.locator('#chat-prompt').fill(prompt)
                    page.locator('#chat-send').click()
                    expect(page.locator('#chat-stop')).to_be_visible()
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                    assert not page.locator('.job-error').count(), page.locator('.job-error').all_text_contents()
                    expect(page.locator('.job-error')).to_have_count(0)

                send('Draw an orange bicycle.')
                expect(page.locator('.message.assistant .message-images img')).to_have_count(1)
                assert calls[0]['max_tokens'] == 8192
                expect(page.locator('.artifact-edit-chat')).to_have_count(0)
                page.locator('#chat-prompt').fill('Keep my draft')
                original_thread = page.evaluate("localStorage.getItem('lazyrig.chat')")
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                options.locator('[name="image_model"]').select_option('')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('.artifact-use').click()
                expect(page.locator('#chat-task')).to_have_value('chat')
                expect(page.locator('#chat-model')).to_have_value('qwen-chat')
                expect(options.locator('[name="image_model"]')).to_have_value('qwen-image')
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                original_source = page.locator('#chat-attachments img').get_attribute('src')
                page.locator('.artifact-use').click(click_count=3)
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                page.locator('.artifact-edit').click()
                expect(page.locator('#chat-task')).to_have_value('edit')
                expect(page.locator('#chat-model')).to_have_value('qwen-image')
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                page.locator('.artifact-use').click()
                expect(page.locator('#chat-task')).to_have_value('chat')
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                expect(page.locator('#chat-prompt')).to_have_value('Keep my draft')
                assert page.evaluate("localStorage.getItem('lazyrig.chat')") == original_thread
                assert page.locator('#chat-attachments img').get_attribute('src') == original_source
                page.locator('[data-remove-file]').click()
                page.evaluate('''() => {
                    window.savedFileRead = FileReader.prototype.readAsDataURL;
                    window.delayedFileReads = [];
                    FileReader.prototype.readAsDataURL = function(blob) {
                        window.delayedFileReads.push(() => savedFileRead.call(this, blob));
                    };
                    for (let n=0;n<4;n++) document.querySelector('.artifact-use').click();
                }''')
                page.locator('#chat-files').set_input_files({'name':'different-source.png', 'mimeType':'image/png',
                    'buffer':base64.b64decode(source_image().split(',')[1])})
                expect(page.locator('#chat-send')).to_be_disabled()
                page.evaluate('''() => {
                    FileReader.prototype.readAsDataURL = savedFileRead;
                    delayedFileReads.forEach(read => read());
                }''')
                expect(page.locator('#chat-send')).to_be_enabled()
                expect(page.locator('#chat-attachments img')).to_have_count(2)
                page.locator('#chat-attachments .attachment').filter(has=page.locator('img[alt="different-source.png"]')).locator('button').click()
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                assert page.locator('#chat-attachments img').get_attribute('src') == original_source
                page.locator('.artifact-edit').click()
                page.evaluate('''() => {
                    window.delayedFileReads = [];
                    FileReader.prototype.readAsDataURL = function(blob) {
                        delayedFileReads.push(() => savedFileRead.call(this, blob));
                    };
                }''')
                page.locator('#chat-files').set_input_files({'name':'earlier-selection.png', 'mimeType':'image/png',
                    'buffer':base64.b64decode(source_image().split(',')[1])})
                page.locator('.artifact-edit').click()
                page.evaluate('''() => {
                    FileReader.prototype.readAsDataURL = savedFileRead;
                    delayedFileReads.forEach(read => read());
                }''')
                expect(page.locator('#chat-send')).to_be_enabled()
                assert page.locator('#chat-attachments img').get_attribute('src') == original_source
                page.locator('.artifact-use').click()
                page.locator('[data-remove-file]').click()
                page.evaluate('''() => {
                    window.delayedFileReads = [];
                    FileReader.prototype.readAsDataURL = function(blob) {
                        delayedFileReads.push(() => savedFileRead.call(this, blob));
                    };
                    document.querySelector('.artifact-use').click();
                }''')
                page.locator('#chat-new').click()
                page.evaluate('''() => {
                    FileReader.prototype.readAsDataURL = savedFileRead;
                    delayedFileReads.forEach(read => read());
                }''')
                expect(page.locator('#chat-send')).to_be_enabled()
                expect(page.locator('#chat-attachments img')).to_have_count(0)
                if page.locator('#chat-history').is_hidden():
                    page.locator('#chat-history-toggle').click()
                page.locator('[data-thread="'+original_thread+'"]').click()
                page.locator('.artifact-use').click()
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                expect(page.locator('#chat-prompt')).to_have_value('Keep my draft')
                evidence['checks'].append('Late reads cannot replace the newest direct-edit source or attach to a different conversation; a removed source can be attached again')
                evidence['checks'].append('Use in chat enables editing and preserves draft/thread; repeated and overlapping clicks attach once, task switching reuses pixels, and concurrent distinct uploads survive')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                options.locator('[name="enhance_prompt"]').check()
                options.locator('[name="enhance_prompt"]').locator('..').get_by_role('button',name='About Enhance image prompt').click()
                expect(page.locator('#setting-help')).to_contain_text('enhanced edits require a vision model')
                page.keyboard.press('Escape')
                page.screenshot(path=str(out/('enhancement-controls-'+__version__+'.png')))
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                before = len(calls)
                send('Make this image blue.')
                expect(page.locator('.message.assistant .message-images img')).to_have_count(2)
                assert [c['action'] for c in calls[before:]] == ['text', 'text', 'image_edit', 'text']
                assert calls[before+1]['prepared']['images'][0]['sha256'] == calls[before+2]['input_images'][0]['sha256']
                assert calls[before+3]['prepared']['images'], 'Vision continuation must receive output pixels'
                assert calls[before+2]['seed'] != next(c['seed'] for c in calls if c['action']=='image')
                assert calls[before+1]['prepared']['messages'][0]['content'] == (Path('lazyrig/prompts/image_editing.txt').read_text())
                expect(page.locator('.image-prompt')).to_have_count(1)
                page.locator('.image-prompt summary').click()
                expect(page.locator('.image-prompt')).to_contain_text('Change the subject to deep blue')
                page.screenshot(path=str(out/('chat-edit-enhanced-'+__version__+'.png')))
                assert not page.locator('.message-body').filter(has_text='rewritten_prompt').count()
                evidence['checks'].append('Use in chat retains conversation, uses exact selected pixels and exposes the enhanced prompt without raw JSON in chat')

                image_count = len([c for c in calls if c['action'] in {'image', 'image_edit'}])
                page.reload(); expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                expect(options.locator('[name="enhance_prompt"]')).to_be_checked()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                assert len([c for c in calls if c['action'] in {'image', 'image_edit'}]) == image_count
                send('Make the last image warmer.')
                assert [c for c in calls if c['action'] in {'image', 'image_edit'}][-1]['action'] == 'image_edit'
                evidence['checks'].append('Enhancement toggle/history survive reload with no resubmission; a follow-up can edit a saved result without reattaching it')

                page.locator('#chat-new').click(); page.locator('#chat-task').select_option('image')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                options.locator('[name="enhance_prompt"]').check()
                options.locator('[name="enhancement_model"]').select_option('text-chat')
                options.locator('[name="size"]').select_option('custom')
                options.locator('[name="width"]').fill('512'); options.locator('[name="height"]').fill('768')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                send('A small orange bicycle.')
                image = [c for c in calls if c['action']=='image'][-1]
                assert (image['width'], image['height']) == (512,768)
                assert image['prompt'] == generation_rewrite['rewritten_prompt']
                assert [c for c in calls if c.get('response_format', {}).get('type') == 'json_schema'][-1]['prepared']['messages'][0]['content'] == Path('lazyrig/prompts/image_generation.txt').read_text()
                evidence['checks'].append('Direct generation can use a selected text enhancer; explicit dimensions override suggested aspect ratio')
                evidence['checks'].append('Multiline generation enhancements reach the image worker intact; editing still requires one paragraph')

                page.locator('#chat-new').click(); page.locator('#chat-task').select_option('edit')
                page.locator('#chat-files').set_input_files({'name':'source.png', 'mimeType':'image/png',
                    'buffer':base64.b64decode(source_image().split(',')[1])})
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click(); options.locator('[name="enhance_prompt"]').check()
                expect(options.locator('[name="enhancement_model"] option[value="text-chat"]')).to_have_count(0)
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                send('Make the source blue.')
                image = [c for c in calls if c['action']=='image_edit'][-1]
                assert (image['width'],image['height']) == (256,512)
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click(); options.locator('[name="enhance_prompt"]').uncheck()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-files').set_input_files({'name':'source.png', 'mimeType':'image/png',
                    'buffer':base64.b64decode(source_image().split(',')[1])})
                before = len(calls); send('Leave the edit prompt unchanged.')
                assert [c['action'] for c in calls[before:]] == ['image_edit']
                assert calls[-1]['prompt'] == 'Leave the edit prompt unchanged.'
                evidence['checks'].append('Direct enhanced edits select a VLM and follow source dimensions; disabling enhancement adds no LLM pass')

                for prompt, expected in [('Repair tool name: draw a train.', ['text','text','image','text']), ('Two images: a train and a bridge.', ['text','image','image','text'])]:
                    page.locator('#chat-new').click(); page.locator('#chat-task').select_option('chat')
                    page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    options.locator('[name="image_model"]').select_option('qwen-image')
                    options.locator('[name="enhance_prompt"]').uncheck()
                    page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    before = len(calls); send(prompt)
                    assert [c['action'] for c in calls[before:]] == expected
                    expect(page.locator('.message-images img')).to_have_count(expected.count('image'))
                evidence['checks'].append('An undeclared raw JSON function gets one client repair, never an alias; multiple declared calls execute once each before one continuation')

                page.locator('#chat-new').click(); page.locator('#chat-task').select_option('image')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click(); options.locator('[name="enhance_prompt"]').check()
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                image_count = len([c for c in calls if c['action'] in {'image','image_edit'}])
                page.locator('#chat-prompt').fill('Cancel enhancement'); page.locator('#chat-send').click()
                expect(page.locator('.job-progress')).to_contain_text('Enhancing', timeout=10000)
                page.locator('#chat-stop').click(); expect(page.locator('#chat-stop')).not_to_be_visible(timeout=10000)
                assert len([c for c in calls if c['action'] in {'image','image_edit'}]) == image_count
                page.set_viewport_size({'width':390,'height':844})
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.screenshot(path=str(out/('image-tools-mobile-'+__version__+'.png')))
                assert not errors, errors
                assert inference_requests and all('auto_image' not in body and 'enhance_prompt' not in body for _, body in inference_requests)
                continuations = [body for url, body in inference_requests if '/chat/completions' in url and (body['messages'][-1]['role'] == 'tool' or (len(body['messages']) > 1 and body['messages'][-2]['role'] == 'tool'))]
                assert continuations
                assert all(body['seed']==-1 for url,body in inference_requests if '/v1/images/' in url and '/chat/completions' not in url and 'seed' in body)
                for body in continuations:
                    calls_by_id = {call['id']: call for message in body['messages'] for call in message.get('tool_calls', [])}
                    tool = next(m for m in reversed(body['messages']) if m['role'] == 'tool')
                    assert tool['tool_call_id'] in calls_by_id
                    report = json.loads(tool['content'])
                    assert report['status'] == 'completed' and report['visual_verification'] == 'not_performed'
                    assert 'requested_prompt' in report and 'prompt' not in report
                    if body['model'] == 'qwen-chat':
                        preview = body['messages'][-1]
                        assert preview['role'] == 'user' and any(p['type'] == 'image_url' for p in preview['content'])
                    else:
                        assert body['messages'][-1]['role'] == 'tool'
                        assert not any(isinstance(m.get('content'),list) and any(p['type']=='image_url' for p in m['content']) for m in body['messages'])
                evidence['checks'].append('All stages use public /v1 endpoints, no auto_image/enhance_prompt routing flags; exact instruction files and matching tool-call IDs are preserved')
                evidence['checks'].append('Stop during enhancement prevents image inference; mobile controls fit; no JavaScript errors')
                evidence['browser'] = browser.version
                browser.close()
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()
    (out/('browser-image-tools-'+__version__+'.json')).write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
