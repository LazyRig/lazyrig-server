"""Real browser + HTTP checks using test-owned deterministic workers."""
import copy
import io
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright, expect
from PIL import Image
from support import DEFAULT, Service, FixtureWorker
from lazyrig.server import handler
from lazyrig import __version__
from test_image_edits import export_fixture
from test_image_settings import scheduler_file


def main():
    out=Path('test-results');out.mkdir(exist_ok=True)
    evidence={'scope':'Real Chromium and HTTP; deterministic test workers, not model-quality testing','checks':[]}
    def screenshot(page, name):
        page.screenshot(path=str(out/(name+'-'+__version__+'.png')))
    with tempfile.TemporaryDirectory() as temp:
        config=copy.deepcopy(DEFAULT);config['data_dir']=temp
        export=export_fixture(Path(temp)/'qwen');scheduler_file(export)
        config['models'][1].update(path=str(export),runtime_profile='qwen-image-21',max_width=2752,max_height=2752,generation={'width':2048,'height':2048,'steps':40,'seed':42})
        config['models'].append({**copy.deepcopy(config['models'][0]),'id':'vision-chat','name':'Vision model','pipeline':'vlm'})
        classic=export_fixture(Path(temp)/'classic',False)
        config['models'].append({**copy.deepcopy(DEFAULT['models'][1]),'id':'diffusion-model','name':'Classic image model','path':str(classic),'generation':{'width':512,'height':512,'steps':2}})
        service=Service(config)
        calls=[];lifecycle=[]
        def factory(*args):
            w=FixtureWorker(*args);run=w.run
            model_id=args[1]['id'];lifecycle.append(('load',model_id));close=w.close
            def tracked_close():
                lifecycle.append(('close',model_id));return close()
            w.close=tracked_close
            def tracked(request,*args):
                calls.append(copy.deepcopy(request))
                yield from run(request,*args)
            w.run=tracked;return w
        service.scheduler.factory=factory
        http=ThreadingHTTPServer(('127.0.0.1',0),handler(service));threading.Thread(target=http.serve_forever,daemon=True).start()
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                base='http://127.0.0.1:'+str(http.server_port)
                page.goto(base);expect(page.locator('#connection')).to_have_text('Server online')
                expect(page.locator('#models-view')).to_be_visible()
                expect(page.locator('#version')).to_have_text('v'+__version__.replace('beta', ' beta'))
                expect(page.locator('#registered-count')).to_have_text('4')
                expect(page.locator('.brand-logo')).to_have_count(1)
                expect(page.locator('.brand-logo .logo-cutout')).to_have_attribute('mask','url(#lr-cutout)')
                assert page.locator('.brand-logo').bounding_box()['width']<=36
                screenshot(page,'models')
                page.locator('#add-model').click()
                register=page.locator('#register-dialog')
                register.get_by_role('button',name='About Memory reservation (GiB)').click()
                expect(page.locator('#setting-help')).to_contain_text('does not cap actual memory use')
                page.keyboard.press('Escape');expect(register).to_be_visible()
                page.keyboard.press('Escape');expect(register).not_to_be_visible()
                page.locator('nav [data-view="models"]').click();page.locator('[data-edit="qwen-image"]').last.click()
                form=page.locator('#model-form');form.locator('#qwen-reference-settings').click()
                expect(form.locator('[name="keep_image_model_loaded"]')).to_have_value('auto')
                expect(form.locator('[name="keep_image_model_loaded"] option')).to_have_text(['Automatic (default)', 'Unload after each request', 'Keep loaded between requests'])
                expect(form.locator('[name="gen_width"]')).to_have_value('2048');expect(form.locator('[name="gen_guidance_scale"]')).to_have_value('')
                form.locator('button[type="submit"]').click();expect(page.locator('#toast')).to_contain_text('Settings saved')
                assert service.model('qwen-image')['diffusion_scheduler']=={'type':'AUTO','config':{}}
                assert 'guidance_scale' not in service.model('qwen-image')['generation']
                assert service.model('qwen-image')['keep_image_model_loaded']=='auto'
                page.wait_for_function('() => !pendingRefresh');page.locator('[data-edit="qwen-chat"]').last.click()
                field=page.locator('[name="gen_temperature"]');field.scroll_into_view_if_needed();page.mouse.move(0,0);field.hover()
                page.wait_for_timeout(1150)
                expect(page.locator('#setting-help')).not_to_be_visible()
                question=field.locator('..').get_by_role('button',name='About Temperature')
                question.focus();expect(page.locator('#setting-help')).not_to_be_visible()
                question.press('Enter');expect(page.locator('#setting-help')).to_contain_text('Higher values')
                screenshot(page,'setting-help')
                page.keyboard.press('Escape');expect(page.locator('#setting-help')).not_to_be_visible()
                assert question.evaluate('e => e === document.activeElement')
                # Controls in the same grid row share a baseline and a height.
                temperature=field.bounding_box();top_p=page.locator('[name="gen_top_p"]').bounding_box()
                assert abs(temperature['y']-top_p['y'])<1
                assert temperature['height']==top_p['height']==42
                evidence['checks'].append('Official Qwen defaults, inherited guidance/export scheduler; click-only help, keyboard and modal behavior, aligned fields')
                page.locator('nav [data-view="server"]').click()
                expect(page.locator('[name="default_auto_image"], [name="default_image_model"]')).to_have_count(0)
                page.locator('#server-form button[type="submit"]').click()
                expect(page.locator('#toast')).to_contain_text('Server settings saved')
                assert 'default_auto_image' not in service.config
                page.locator('nav [data-view="devices"]').click()
                screenshot(page,'devices')
                page.locator('nav [data-view="downloads"]').click()
                destination=page.locator('#download-destination').locator('..').locator('..')
                expect(destination.locator('.help-button')).to_have_count(0)
                screenshot(page,'imports')
                page.locator('nav [data-view="chat"]').click()
                page.locator('#chat-prompt').fill('Hello, retain this conversation.');page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                expect(page.locator('.message.assistant .message-body')).to_contain_text('Test worker response')
                expect(page.locator('.message-metrics')).to_contain_text('Prompt ≈')
                assert not [c for c in calls if c['action']=='text'][-1]['tools_enabled']
                page.locator('#chat-prompt').fill('What did I say?');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                assert len([c for c in calls if c['action']=='text'][-1]['prepared']['messages'])==3, [c['prepared']['messages'] for c in calls if c['action']=='text']
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click();page.locator('#chat-options [name="temperature"]').fill('0.7');page.locator('#chat-options [name="system_prompt"]').fill('Be concise');page.locator('#chat-options [name="system_prompt"]').press('Tab')
                page.reload();expect(page.locator('#connection')).to_have_text('Server online');page.locator('nav [data-view="chat"]').click()
                expect(page.locator('.message')).to_have_count(4)
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click();expect(page.locator('#chat-options [name="system_prompt"]')).to_have_value('Be concise');page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                screenshot(page,'chat')
                evidence['checks'].append('Multi-turn messages, metrics, persisted history and conversation settings across reload')
                resident=service.scheduler.resident['qwen-chat'][1];before_switch=list(lifecycle)
                page.locator('#chat-new').click();expect(page.locator('#chat-model')).to_have_value('qwen-chat')
                page.locator('#chat-prompt').fill('An independent conversation.');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                page.locator('#chat-history-toggle').click()
                page.locator('.thread-item').filter(has_text='Hello, retain this conversation.').click()
                page.locator('#chat-history-toggle').click()
                page.locator('#chat-prompt').fill('Continue the original conversation.');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                assert service.scheduler.resident['qwen-chat'][1] is resident
                assert lifecycle==before_switch, lifecycle
                assert 'An independent conversation.' not in str(calls[-1]['prepared']['messages'])
                evidence['checks'].append('Two independent conversations and returning to the first reuse the same LLM worker despite different template settings')
                for image_format, extension, mime in [('PNG', 'png', 'png'), ('JPEG', 'jpg', 'jpeg')]:
                    page.locator('#chat-new').click();page.locator('#chat-model').select_option('vision-chat')
                    upload=io.BytesIO();Image.new('RGB',(320,200),'#407084').save(upload,format=image_format)
                    page.locator('#chat-files').set_input_files({'name':'source.'+extension,'mimeType':'image/'+mime,'buffer':upload.getvalue()})
                    expect(page.locator('#chat-attachments img')).to_be_visible();page.locator('#chat-prompt').fill('Describe this image.');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                    assert [c for c in calls if c['action']=='text'][-1]['prepared']['images']
                    expect(page.locator('.job-error')).to_have_count(0)
                    for width,height in [(1440,1000),(980,640),(390,844)]:
                        page.set_viewport_size({'width':width,'height':height})
                        # Capture both rectangles together: responsive scrolling
                        # can move the conversation between separate browser calls.
                        body,image=page.locator('.message.user').evaluate('''row =>
                            ['.message-body','.message-images img'].map(selector => {
                                const r=row.querySelector(selector).getBoundingClientRect();
                                return {x:r.x,y:r.y,width:r.width,height:r.height};
                            })''')
                        assert image['y']-(body['y']+body['height'])>=16, (width, height, body, image)
                        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                        scroll=page.locator('#chat-messages')
                        assert scroll.evaluate("e => parseFloat(getComputedStyle(e).paddingRight)")>=8
                        assert scroll.evaluate("e => getComputedStyle(e,'::-webkit-scrollbar-track').backgroundColor")=='rgba(0, 0, 0, 0)'
                        assert scroll.evaluate("e => getComputedStyle(e,'::-webkit-scrollbar-button').display")=='none'
                        if image_format=='PNG':
                            scroll.evaluate('e => e.scrollTop = 0')
                            screenshot(page,'vision-upload-'+str(width))
                    page.set_viewport_size({'width':1440,'height':1000})
                evidence['checks'].append('PNG/JPEG uploads reach vision input; 16px text/image gap, clear subtle scrollbar and no overflow at desktop, short and phone sizes')
                page.locator('#chat-new').click()
                page.locator('#chat-prompt').fill('Keep this earlier context.');page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                good=io.BytesIO();Image.new('RGB',(32,20),'red').save(good,format='PNG')
                bad=io.BytesIO();Image.new('RGB',(32,20),'blue').save(bad,format='BMP')
                page.locator('#chat-files').set_input_files([
                    {'name':'rejected-turn-valid.png','mimeType':'image/png','buffer':good.getvalue()},
                    {'name':'rejected-turn-bad.png','mimeType':'image/png','buffer':bad.getvalue()}])
                expect(page.locator('#chat-attachments img')).to_have_count(2)
                page.locator('#chat-prompt').fill('Rejected two attachments.');page.locator('#chat-send').click()
                expect(page.locator('.job-error')).to_contain_text('Image 2 in message 3: Unsupported image format: BMP')
                expect(page.locator('#chat-stop')).not_to_be_visible()
                # Existing persisted failures also recover; no new per-turn flag
                # or deletion of visible history is needed.
                page.reload();expect(page.locator('#connection')).to_have_text('Server online');page.locator('nav [data-view="chat"]').click()
                expect(page.locator('.message')).to_have_count(4)
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click();page.locator('#chat-options [name="image_model"]').select_option('qwen-image');page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-files').set_input_files({'name':'replacement.jpg','mimeType':'image/jpeg','buffer':upload.getvalue()})
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                page.locator('#chat-prompt').fill('Describe the replacement.');page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                expect(page.locator('.message.assistant .message-body').last).to_contain_text('Test worker response')
                expect(page.locator('.job-error')).to_have_count(1)
                expect(page.locator('.message')).to_have_count(6)
                prepared=calls[-1]['prepared']
                assert len(prepared['images'])==1
                assert 'Keep this earlier context.' in str(prepared['messages'])
                assert 'Rejected two attachments.' not in str(prepared['messages'])
                tools=json.loads(prepared['text'])['tools']
                edit=next(t for t in tools if t['function']['name']=='edit_image')['function']
                assert edit['parameters']['properties']['image_id']['enum']==['image_1']
                assert 'replacement.jpg' in edit['description'] and 'rejected-turn-' not in edit['description']
                evidence['checks'].append('Rejected two-image turn stays visible after reload but is excluded from later context and image-tool sources; a replacement JPEG succeeds in the same conversation and earlier successful context remains')
                page.locator('#chat-new').click();page.locator('#chat-task').select_option('image')
                page.locator('#chat-prompt').fill('An orange geometric landscape.');page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible();page.reload();page.locator('nav [data-view="chat"]').click()
                expect(page.locator('.message.assistant .message-images img')).to_be_visible(timeout=20000)
                expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                assert len([c for c in calls if c['action']=='image'])==1
                screenshot(page,'image-chat')
                page.locator('.artifact-edit').click();expect(page.locator('#chat-task')).to_have_value('edit');expect(page.locator('#chat-attachments img')).to_be_visible()
                page.evaluate('''bytes => {
                    const data = new DataTransfer();
                    for (const name of ['one.png','two.png']) data.items.add(new File([new Uint8Array(bytes)],name,{type:'image/png'}));
                    document.querySelector('#chat-composer').dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));
                }''',list(good.getvalue()))
                expect(page.locator('#toast')).to_contain_text('Image editing accepts one source image per request')
                expect(page.locator('#chat-attachments img')).to_have_count(1)
                page.locator('#chat-prompt').fill('Make the landscape blue.');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                assert len([c for c in calls if c['action']=='image_edit'])==1
                expect(page.locator('.message.assistant .message-images img')).to_have_count(2)
                evidence['checks'].append('Image job reconnect after reload without duplicate generation; generated image reused for editing')
                page.locator('#chat-new').click();page.locator('#chat-task').select_option('chat')
                page.locator('#chat-model').select_option('qwen-chat');page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-options [name="image_model"]').select_option('qwen-image')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                before=len(calls)
                page.locator('#chat-prompt').fill('Generate an image of an orange bicycle.');page.locator('#chat-send').click()
                expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=20000)
                expect(page.locator('.message.assistant .message-images img')).to_be_visible()
                assert [c['action'] for c in calls[before:]]==['text','image','text']
                assert not calls[before]['tools_enabled']  # No server-owned image orchestration.
                assert json.loads(calls[before]['prepared']['text'])['tools'][0]['function']['name'] == 'generate_image'
                evidence['checks'].append('Ordinary chat stays text-only; explicitly selecting Image tool performs one text/image/text handoff')
                page.locator('nav [data-view="models"]').click();page.locator('[data-edit="diffusion-model"]').last.click()
                expect(page.locator('#qwen-reference-settings')).to_have_count(0);expect(page.locator('#image-size-preset')).to_have_count(0)
                page.locator('[data-edit="vision-chat"]').last.click();expect(page.locator('#sampling-preset')).to_have_count(0)
                page.locator('nav [data-view="chat"]').click();page.locator('#chat-new').click();page.locator('#chat-task').select_option('image');page.locator('#chat-model').select_option('diffusion-model')
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click();page.locator('#chat-options [name="size"]').select_option('custom');page.locator('#chat-options [name="width"]').fill('512');page.locator('#chat-options [name="height"]').fill('768');page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-prompt').fill('A classic diffusion landscape.');page.locator('#chat-send').click();expect(page.locator('#chat-stop')).to_be_visible();expect(page.locator('#chat-stop')).not_to_be_visible(timeout=15000)
                last=[c for c in calls if c['action']=='image'][-1];assert (last['width'],last['height'],last['steps'])==(512,768,2)
                evidence['checks'].append('Non-Qwen LLM/image settings omit Qwen presets; classic image requests keep their own defaults and custom dimensions')
                page.set_viewport_size({'width':390,'height':844});page.locator('nav [data-view="chat"]').click();screenshot(page,'mobile')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                send=page.locator('#chat-send').bounding_box()
                assert send['y']+send['height']<=844
                page.locator('#done-chat-settings' if page.locator('#chat-settings-dialog').is_visible() else '#chat-settings-toggle').click()
                page.locator('#chat-options [name="steps"]').locator('..').get_by_role('button',name='About Steps').click()
                help_box=page.locator('#setting-help').bounding_box()
                assert help_box['x']>=0 and help_box['x']+help_box['width']<=390
                page.keyboard.press('Escape')
                screenshot(page,'mobile-parameters')
                assert not errors,errors
                evidence['checks'].append('Single inverted SVG logo, no global routing controls, server settings save, mobile layout/help and zero JavaScript page errors')
                evidence['browser']=browser.version;browser.close()
        finally:http.shutdown();http.server_close();service.close()
    (out/('browser-workspace-'+__version__+'.json')).write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
