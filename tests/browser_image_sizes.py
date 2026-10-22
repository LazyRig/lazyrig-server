"""Exact image dimensions across the dashboard, public API and saved PNGs.

Uses real Chromium/HTTP with deterministic workers; does not test LLM obedience
or native image quality. The fake LLM supplies known tool and enhancement output.
"""
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
from support import DEFAULT, Service, FixtureWorker
from test_chat_image_tools import source_image, rewritten
from test_image_edits import export_fixture


def main():
    out = Path('test-results'); out.mkdir(exist_ok=True)
    evidence = {'scope': 'Real Chromium, public HTTP and saved PNG dimensions; deterministic workers', 'checks': []}
    with tempfile.TemporaryDirectory() as directory:
        config = copy.deepcopy(DEFAULT); config['data_dir'] = directory
        config['models'][0]['pipeline'] = 'vlm'
        config['models'][1].update(path=str(export_fixture(Path(directory) / 'qwen')),
            runtime_profile='qwen-image-21', max_width=2752, max_height=2752,
            generation={'width': 2048, 'height': 2048, 'steps': 1})
        config['models'].append({**copy.deepcopy(DEFAULT['models'][1]), 'id': 'generic-image',
            'generation': {'width': 512, 'height': 512, 'steps': 1}})
        service = Service(config); calls = []; scenario = {}

        def factory(*args):
            worker = FixtureWorker(*args); run = worker.run
            def tracked(request, timeout, cancelled):
                calls.append(copy.deepcopy(request))
                if request['action'] == 'text':
                    last = request['prepared']['messages'][-1]
                    tools = json.loads(request['prepared']['text']).get('tools')
                    if request.get('response_format', {}).get('type') == 'json_schema':
                        text = rewritten(bool(request['prepared']['images']))
                    elif tools and last['role'] != 'tool' and not str(last['content']).startswith('Generated outputs from completed image tools.'):
                        name = 'edit_image' if scenario.get('edit') else 'generate_image'
                        declaration = next(t['function'] for t in tools if t['function']['name'] == name)
                        assert 'size' in declaration['parameters']['properties']
                        arguments = {'prompt': last['content'], **scenario.get('arguments', {})}
                        if scenario.get('edit'):
                            arguments['image_id'] = declaration['parameters']['properties']['image_id']['enum'][-1]
                        text = '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'
                    elif last['role']=='tool' or str(last['content']).startswith('Generated outputs from completed image tools.'):
                        text = 'The image file is ready.'
                    else:
                        yield from run(request, timeout, cancelled); return
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
                requests = []
                page.on('request', lambda r: requests.append((r.url, r.post_data_json))
                    if r.method == 'POST' and '/v1/' in r.url else None)
                page.goto('http://127.0.0.1:' + str(http.server_port))
                expect(page.locator('#connection')).to_have_text('Server online')
                page.locator('nav [data-view="chat"]').click()

                # Capability-based validation also applies to non-Qwen adapters.
                page.evaluate('''() => {
                    const c=InferenceClient, q=state.models.find(m=>m.id==='qwen-image'), g=state.models.find(m=>m.id==='generic-image');
                    const check=(condition,message)=>{if(!condition)throw new Error(message);};
                    check(c.dimensions(q,'1376:928')==='2464x1664','Reproduce the old ratio-only calculation');
                    check(c.exactSize(q,'1376x928')==='1376x928','Preserve exact pixels');
                    check(c.dimensions(q,'16:9',null,'1376x928')==='1376x928','Exact size overrides enhancement');
                    check(c.dimensions(q,'', {width:256,height:512},'1376x928')==='1376x928','Exact size overrides source-following');
                    check(c.exactSize(g,'640x960')==='640x960','Generic dimensions');
                    for (const [model,size] of [[q,'1377x928'],[q,'2784x928'],[q,'128x928'],[q,'1376:928'],[q,'1376.5x928'],[q,null],[q,1376],[q,''],[g,'1376x928'],[g,'1056x512']]) {
                        let failed=false;try{c.exactSize(model,size);}catch{failed=true;}check(failed,'Must reject unsupported exact size: '+size);
                    }
                    for(const p of q.image_size_constraints.presets)check(c.dimensions(q,p.aspect_ratio)===p.width+'x'+p.height,'Preserve official ratio presets');
                }''')
                evidence['checks'].append('Reproduced 2464x1664 from ratio-only input; explicit pixels, all seven ratio presets and model-specific limits verified')

                cases = []
                for edit in (False, True):
                    for enhance in (False, True):
                        cases.append({'name': f'Chat {"edit" if edit else "generate"}, enhancement {enhance}',
                            'task': 'chat', 'edit': edit, 'enhance': enhance,
                            'arguments': {'size': '1376x928', 'aspect_ratio': '16:9'}, 'expected': (1376, 928)})
                cases += [
                    {'name': 'Chat custom setting overrides conflicting tool dimensions', 'task': 'chat', 'enhance': True,
                     'custom': (1376, 928), 'arguments': {'size': '2048x2048', 'aspect_ratio': '1:1'}, 'expected': (1376, 928)},
                    {'name': 'Direct enhanced generation', 'task': 'image', 'enhance': True, 'custom': (1376, 928), 'expected': (1376, 928)},
                    {'name': 'Direct enhanced edit overrides source dimensions', 'task': 'edit', 'edit': True,
                     'enhance': True, 'custom': (1376, 928), 'expected': (1376, 928)},
                    {'name': 'Ratio-only request keeps published preset', 'task': 'chat', 'arguments': {'aspect_ratio': '3:2'}, 'expected': (2528, 1696)},
                    {'name': 'Unspecified dimensions keep model default', 'task': 'chat', 'expected': (2048, 2048)},
                    {'name': 'Generic model uses its own custom-size constraints', 'task': 'chat', 'model': 'generic-image',
                     'custom': (640, 960), 'expected': (640, 960)},
                    {'name': 'Invalid alignment stops before enhancement or generation', 'task': 'chat', 'enhance': True,
                     'arguments': {'size': '1377x928', 'aspect_ratio': '3:2'}, 'error': '1377x928'},
                    {'name': 'Oversized request is rejected without resizing', 'task': 'chat', 'enhance': True,
                     'arguments': {'size': '2784x928'}, 'error': '2784x928'},
                ]
                for case in cases:
                    scenario.clear(); scenario.update(case)
                    page.locator('#chat-new').click(); page.locator('#chat-task').select_option(case['task'])
                    page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    options = page.locator('#chat-options')
                    if case['task'] == 'chat':
                        expect(options.locator('[name="size"]')).not_to_be_visible()
                        options.locator('[name="image_model"]').select_option(case.get('model', 'qwen-image'))
                    expect(options.locator('[name="size"]')).to_be_visible()
                    options.locator('[name="enhance_prompt"]').set_checked(case.get('enhance', False))
                    if 'custom' in case:
                        options.locator('[name="size"]').select_option('custom')
                        for axis, value in zip(('width', 'height'), case['custom']):
                            field = options.locator('[name="' + axis + '"]')
                            expect(field).to_be_visible()
                            expect(field).to_have_attribute('step', '64' if case.get('model') == 'generic-image' else '32')
                            field.fill(str(value))
                    page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    if case.get('edit'):
                        page.locator('#chat-files').set_input_files({'name': 'source.png', 'mimeType': 'image/png',
                            'buffer': base64.b64decode(source_image().split(',')[1])})
                    before = len(calls); sent = len(requests)
                    page.locator('#chat-prompt').fill(case['name'] + ': a street photograph, 1376x928 pixels.')
                    page.locator('#chat-send').click()
                    expect(page.locator('.message.assistant')).to_have_count(1)
                    expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                    images = [c for c in calls[before:] if c['action'] in {'image', 'image_edit'}]
                    if case.get('error'):
                        expect(page.locator('.job-error')).to_contain_text(case['error'])
                        expect(page.locator('.job-error')).to_contain_text('size was not changed')
                        assert not images and len(calls[before:]) == 1
                    else:
                        expect(page.locator('.job-error')).to_have_count(0)
                        assert len(images) == 1, case['name']
                        assert (images[0]['width'], images[0]['height']) == case['expected'], case['name']
                        image = page.locator('.message.assistant .message-images img')
                        expect(image).to_be_visible()
                        assert tuple(image.evaluate('(img)=>[img.naturalWidth,img.naturalHeight]')) == case['expected']
                        image_requests = [body for url, body in requests[sent:] if '/v1/images/' in url]
                        expected_size = 'x'.join(map(str, case['expected'])) if ('custom' in case or 'arguments' in case) else 'auto'
                        assert len(image_requests) == 1 and image_requests[0]['size'] == expected_size
                        if case['task'] == 'chat':
                            continuation = [body for url, body in requests[sent:] if '/chat/completions' in url and (body['messages'][-1]['role'] == 'tool' or (len(body['messages']) > 1 and body['messages'][-2]['role'] == 'tool'))][-1]
                            result = json.loads(next(m['content'] for m in reversed(continuation['messages']) if m['role']=='tool'))
                            assert (result['width'], result['height']) == case['expected']
                        if case.get('enhance'):
                            expect(page.locator('.image-prompt')).to_contain_text(f'{case["expected"][0]} × {case["expected"][1]}')
                        if 'custom' in case and case['task'] == 'chat':
                            page.reload(); expect(page.locator('#connection')).to_have_text('Server online')
                            page.locator('nav [data-view="chat"]').click(); page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                            expect(options.locator('[name="size"]')).to_have_value('custom')
                            expect(options.locator('[name="width"]')).to_have_value(str(case['custom'][0]))
                            expect(options.locator('[name="height"]')).to_have_value(str(case['custom'][1]))
                            page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                    evidence['checks'].append(case['name'])
                assert not errors, errors
                evidence['browser'] = browser.version
                browser.close()
        finally:
            http.shutdown(); http.server_close(); thread.join(3); service.close()
    (out / ('browser-image-sizes-' + __version__ + '.json')).write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
