"""Vision transport and bundled-MTP contracts; fixture tests do not claim model quality."""
import base64
import copy
import io
import json
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers, qwen_call
from support import DEFAULT
from lazyrig.config import validate
from lazyrig.context import TokenizerService
from lazyrig.mtp import configure_generation, ensure_export, model_capabilities
from lazyrig.runtime_settings import pipeline_properties
from lazyrig.vision import ImageStore, check_content, fetch_image
from lazyrig.worker import _child


def image_url(color='red', size=(32, 20)):
    output=io.BytesIO();Image.new('RGB',size,color).save(output,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()


def parts(*urls):
    return [{'type':'text','text':'Describe these images.'}]+[{'type':'image_url','image_url':{'url':url}} for url in urls]


class VisionHTTPTests(unittest.TestCase):
    setUp=test_http.HTTPTests.setUp
    tearDown=test_http.HTTPTests.tearDown
    request=test_http.HTTPTests.request
    json=test_http.HTTPTests.json

    def vlm(self,responses=('A red image.',)):
        self.service.config['models'][0]['pipeline']='vlm'
        self.workers=FixtureWorkers([(text,8) for text in responses])
        self.service.scheduler.factory=self.workers

    def body(self,**extra):
        return {'model':'qwen-chat','messages':[{'role':'user','content':parts(image_url())}], 'max_tokens':128, **extra}

    def test_all_advertised_still_formats_reach_vision_input(self):
        formats = [('PNG', 'png'), ('JPEG', 'jpeg'), ('WEBP', 'webp'), ('GIF', 'gif')]
        self.vlm(('Image received.',) * len(formats))
        for image_format, mime in formats:
            with self.subTest(format=image_format):
                output = io.BytesIO()
                Image.new('RGB', (32, 20), 'red').save(output, format=image_format)
                url = f'data:image/{mime};base64,' + base64.b64encode(output.getvalue()).decode()
                result = self.json('/v1/chat/completions', self.body(
                    messages=[{'role': 'user', 'content': parts(url)}]))
                self.assertEqual(result['choices'][0]['message']['content'], 'Image received.')
                inputs = self.workers.calls[-1]['prepared']['images']
                self.assertEqual(len(inputs), 1)
                with Image.open(inputs[0]['path']) as normalized:
                    self.assertEqual((normalized.format, normalized.mode, normalized.size),
                                     ('PNG', 'RGB', (32, 20)))

    def test_bad_earlier_attachment_is_identified_and_fresh_jpeg_works(self):
        self.vlm(('Image received.',))
        output = io.BytesIO()
        Image.new('RGB', (32, 20), 'red').save(output, format='BMP')
        mislabeled = 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()
        output = io.BytesIO()
        Image.new('RGB', (32, 20), 'blue').save(output, format='JPEG')
        jpeg = 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()
        latest = {'role': 'user', 'content': parts(jpeg)}
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/v1/chat/completions', self.body(messages=[
                {'role': 'user', 'content': parts(mislabeled)}, latest]))
        self.assertEqual(error.exception.code, 400)
        self.assertIn('Image 1 in message 1: Unsupported image format: BMP',
                      error.exception.read().decode())
        self.assertEqual(self.workers.calls, [])
        result = self.json('/v1/chat/completions', self.body(messages=[latest]))
        self.assertEqual(result['choices'][0]['message']['content'], 'Image received.')

    def test_two_images_in_one_vision_message_are_supported(self):
        self.vlm(('Two images received.',))
        result = self.json('/v1/chat/completions', self.body(messages=[
            {'role': 'user', 'content': parts(image_url('red'), image_url('blue'))}]))
        self.assertEqual(result['choices'][0]['message']['content'], 'Two images received.')
        inputs = self.workers.calls[-1]['prepared']['images']
        self.assertEqual(len(inputs), 2)
        self.assertNotEqual(inputs[0]['sha256'], inputs[1]['sha256'])

    def test_image_parts_reach_worker_in_order_and_survive_saved_result_and_history(self):
        self.vlm(('Two images.','Text reply.','Streamed vision reply.'))
        messages=[{'role':'user','content':parts(image_url('red'))},
                  {'role':'assistant','content':'Earlier image.'},
                  {'role':'user','content':parts(image_url('blue'))}]
        receipt=self.json('/api/chat/jobs',self.body(messages=messages))
        job=wait_job(self.service,receipt['id']);self.assertEqual(job['status'],'completed',job.get('error'))
        result=self.json(receipt['result_url'])
        self.assertEqual(result['choices'][0]['message']['content'],'Two images.')
        payload=self.workers.calls[-1]['prepared'];images=payload['images']
        self.assertEqual(len(images),2)
        for info,color in zip(images,[(255,0,0),(0,0,255)]):
            with Image.open(info['path']) as im:self.assertEqual(im.getpixel((0,0)),color)
        self.assertIn('<ov_genai_image_0>',payload['messages'][0]['content'])
        self.assertIn('<ov_genai_image_1>',payload['messages'][-1]['content'])
        history=self.service.store.messages(job['session_id'])
        self.assertIn('lazyrig-input:',history[0]['content'][1]['image_url']['url'])
        self.assertNotIn('base64',json.dumps(job))
        self.assertEqual(self.json(receipt['result_url']),result)
        plain=self.json('/v1/chat/completions',self.body(messages=[{'role':'user','content':'Hello'}]))
        self.assertEqual(self.workers.calls[-1]['prepared']['images'],[])
        with self.request('/v1/chat/completions',self.body(stream=True)) as response:
            raw=response.read().decode()
        values=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        self.assertFalse(any('error' in v for v in values),values)
        self.assertEqual(''.join(v['choices'][0]['delta'].get('content','') for v in values if v.get('choices')),'Streamed vision reply.')
        self.assertEqual(len(self.workers.calls[-1]['prepared']['images']),1)

    def test_image_inputs_replayed_through_image_generation_handoff(self):
        self.vlm((qwen_call(),'Generated with the reference image.'))
        result=self.json('/v1/chat/completions',self.body(auto_image=True,image_model='qwen-image',steps=1))
        text_calls=[c for c in self.workers.calls if c['action']=='text']
        self.assertEqual(len(text_calls),2)
        self.assertEqual(text_calls[0]['prepared']['images'],text_calls[1]['prepared']['images'])
        self.assertEqual(len(result['x_artifacts']),1)

    def test_preflight_marks_vision_pending_and_digest_changes_with_pixels(self):
        self.vlm()
        first=self.json('/api/context',self.body())
        second=self.json('/api/context',self.body(messages=[{'role':'user','content':parts(image_url('blue'))}]))
        self.assertEqual(first['accuracy'],'estimated-multimodal')
        self.assertIsNone(first['fits']);self.assertFalse(first['visual_tokens_included'])
        self.assertEqual(first['image_count'],1)
        self.assertNotEqual(first['prompt_sha256'],second['prompt_sha256'])
        self.assertFalse(self.service.scheduler.resident)
        models=self.json('/v1/models')['data']
        self.assertEqual(models[0]['input_modalities'],['text','image'])
        self.assertTrue(models[0]['capabilities']['vision'])

    def test_llm_allows_text_arrays_but_rejects_images_and_malformed_vlm_parts(self):
        self.service.scheduler.factory=FixtureWorkers([('Text array works.',5)])
        body=self.body(messages=[{'role':'user','content':[{'type':'text','text':'test'}]}])
        self.assertEqual(self.json('/v1/chat/completions',body)['choices'][0]['message']['content'],'Text array works.')
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/v1/chat/completions',self.body())
        self.assertEqual(error.exception.code,400)
        self.assertIn('VLM',error.exception.read().decode())
        self.vlm()
        invalid=[parts('file:///etc/passwd'),parts('http://127.0.0.1/a.png'),parts('data:image/png;base64,aGVsbG8='),
                 parts('lazyrig-input:'+'0'*64),[{'type':'image_url'}],[{'type':'input_audio','data':'abc'}]]
        for content in invalid:
            with self.subTest(content=str(content)[:50]),self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/chat/jobs',self.body(messages=[{'role':'user','content':content}]))
            self.assertEqual(error.exception.code,400)


class ImageIngestionTests(unittest.TestCase):
    def test_normalized_images_persist_and_low_detail_resizes(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ImageStore(directory);model={'pipeline':'vlm'}
            messages=[{'role':'user','content':parts(image_url(size=(1024,512)))}]
            messages[0]['content'][1]['image_url']['detail']='low'
            captured=store.capture(messages,model);ref=captured[0]['content'][1]['image_url']['url']
            info=ImageStore(directory).describe(ref)
            self.assertEqual((info['width'],info['height']),(512,256))
            self.assertEqual(store.capture(messages,model),captured)
            self.assertEqual(len(list(Path(directory,'inputs').glob('*.png'))),1)
            self.assertTrue(messages[0]['content'][1]['image_url']['url'].startswith('data:'))

    def test_url_redirects_cannot_reach_private_addresses(self):
        class Response:
            status=302
            def getheader(self,key):return 'http://127.0.0.1/private' if key=='Location' else None
        class Connection:
            def __init__(self,*a,**k):pass
            def request(self,*a,**k):pass
            def getresponse(self):return Response()
            def close(self):pass
        def lookup(host,*args,**kwargs):return [(2,1,6,'',('127.0.0.1' if host=='127.0.0.1' else '8.8.8.8',80))]
        with patch('lazyrig.vision.socket.getaddrinfo',side_effect=lookup),patch('lazyrig.vision.socket.create_connection') as connect,patch('lazyrig.vision.http.client.HTTPConnection',Connection):
            with self.assertRaisesRegex(ValueError,'Private/local'):fetch_image('http://public.example/image')
            self.assertEqual(connect.call_count,1)
            self.assertEqual(connect.call_args.args[0],('8.8.8.8',80))


class MTPHTTPTests(unittest.TestCase):
    setUp=test_http.HTTPTests.setUp
    tearDown=test_http.HTTPTests.tearDown
    request=test_http.HTTPTests.request
    json=test_http.HTTPTests.json

    def test_settings_persist_and_draft_count_applies_to_every_handoff_pass(self):
        export = Path(self.temp.name) / 'mtp'; export.mkdir()
        for suffix in ('xml', 'bin'):
            (export / ('openvino_mtp_model.' + suffix)).write_bytes(b'fixture')
        self.service.config['models'][0]['path'] = str(export)
        factory=FixtureWorkers([(qwen_call(),15),('Image ready.',8)])
        self.service.scheduler.factory=factory
        self.json('/api/settings',{'model':'qwen-chat','mtp':{'enabled':True,'num_assistant_tokens':2},'generation':{'temperature':0},'scheduler_config':{'enable_prefix_caching':False}})
        persisted=json.loads(self.service.settings_path.read_text())['models'][0]
        self.assertTrue(persisted['mtp']['enabled'])
        result=self.json('/v1/chat/completions',{'model':'qwen-chat','messages':[{'role':'user','content':'Draw an image'}],
                         'num_assistant_tokens':3,'auto_image':True,'image_model':'qwen-image','max_tokens':128,'steps':1})
        self.assertEqual(result['x_mtp'],{'enabled':True,'num_assistant_tokens':3})
        for c in factory.calls:
            if c['action']=='text':self.assertEqual((c['temperature'],c['generation']['num_assistant_tokens']),(0,3))
        self.assertEqual(self.service.model('qwen-chat')['mtp']['num_assistant_tokens'],2)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/chat/jobs',{'model':'qwen-chat','messages':[{'role':'user','content':'Hi'}],'temperature':.7})
        self.assertEqual(error.exception.code,400)

    def test_mtp_conflicts_and_missing_head_fail_explicitly(self):
        for extra in [{'generation':{'temperature':.7}},{'scheduler_config':{'enable_prefix_caching':True}},
                      {'mtp':{'enabled':True,'num_assistant_tokens':0}},{'mtp':{'enabled':'true'}}]:
            with self.subTest(extra=extra),self.assertRaises(urllib.error.HTTPError):
                self.request('/api/settings',{'model':'qwen-chat','mtp':{'enabled':True},**extra})
        with self.assertRaisesRegex(ValueError,'openvino_mtp_model'):
            ensure_export({'path':'nonexistent','mtp':{'enabled':True}})
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/v1/chat/completions',{'model':'qwen-chat','messages':[{'role':'user','content':'Hi'}],'num_assistant_tokens':2})

    def test_native_mtp_constructor_and_generation_contract(self):
        try:import openvino_genai as genai
        except ImportError:self.skipTest('Optional native OpenVINO runtime')
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('xml','bin'):Path(directory,'openvino_mtp_model.'+suffix).write_text('fixture only')
            model={'id':'fixture','path':directory,'device':'CPU','pipeline':'vlm','task':'text-generation','mtp':{'enabled':True,'num_assistant_tokens':2}}
            calls=[];sentinel=object()
            fake=types.SimpleNamespace(SchedulerConfig=genai.SchedulerConfig,draft_model=lambda *a,**k:(calls.append((a,k)) or sentinel))
            props=pipeline_properties(fake,model)
            self.assertIs(props['draft_model'],sentinel)
            self.assertEqual(calls,[((directory,'CPU'),{})])
            self.assertFalse(props['scheduler_config'].enable_prefix_caching)
            config=genai.GenerationConfig();config.max_new_tokens=100
            configure_generation(config,model,{'temperature':0,'num_assistant_tokens':4});config.validate()
            self.assertEqual(config.num_assistant_tokens,4);self.assertFalse(config.do_sample)
            self.assertTrue(model_capabilities(model)['capabilities']['mtp_available'])
            self.assertEqual(pipeline_properties(fake,{**model,'mtp':{'enabled':False}}),{})

    def test_legacy_vlm_pipeline_migrates_without_losing_model_settings(self):
        config=copy.deepcopy(DEFAULT);config['models'][0]['pipeline']='vlm-text'
        self.assertEqual(validate(config)['models'][0]['pipeline'],'vlm')


if __name__=='__main__':unittest.main()
