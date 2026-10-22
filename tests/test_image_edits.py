"""Image-edit HTTP, upload, lifecycle and Qwen native-call contract regressions."""
from support import fixture_png
import base64
import copy
import io
import json
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import test_http
from test_core import wait_job
from test_handoff import FixtureWorkers
from test_vision_mtp import image_url
from lazyrig.image_edit import capabilities, parse_multipart
from lazyrig.worker import _child, _native_image


def export_fixture(root, qwen=True):
    root = Path(root); root.mkdir(exist_ok=True)
    (root/'model_index.json').write_text(json.dumps({'_class_name':'QwenImage21Pipeline' if qwen else 'StableDiffusionPipeline'}))
    for name in ('vae_encoder','vision_encoder','text_encoder_i2i'):
        (root/name).mkdir(exist_ok=True)
        for suffix in ('xml','bin'):
            (root/name/('openvino_model.'+suffix)).write_bytes(b'Contract fixture, not model weights')
    return root


def multipart(fields, files):
    boundary='lazyrig-edit-test-boundary'
    chunks=[]
    for name,value in fields:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name,data in files:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="../../source.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+data+b'\r\n')
    chunks.append(f'--{boundary}--\r\n'.encode())
    return 'multipart/form-data; boundary='+boundary,b''.join(chunks)


class EditWorkers(FixtureWorkers):
    def __call__(self,mode,model,timeout,cancelled):
        worker=super().__call__(mode,model,timeout,cancelled)
        run=worker.run
        def edit_run(payload,timeout,cancelled):
            if payload['action']!='image_edit':
                yield from run(payload,timeout,cancelled)
                return
            self.calls.append(copy.deepcopy(payload))
            with Image.open(payload['input_images'][0]['path']) as image:
                assert image.getpixel((0,0))==(255,0,0)
            fixture_png(payload['output_path'], 'EDIT CONTRACT FIXTURE', payload['width'], payload['height'])
            yield {'kind':'progress','step':1,'steps':payload['steps']}
            yield {'kind':'result','path':payload['output_path']}
        worker.run=edit_run
        return worker


class ImageEditHTTPTests(unittest.TestCase):
    setUp=test_http.HTTPTests.setUp
    tearDown=test_http.HTTPTests.tearDown
    request=test_http.HTTPTests.request
    json=test_http.HTTPTests.json

    def fixture(self):
        self.service.config['models'][1]['path']=str(export_fixture(Path(self.temp.name)/'edit-model'))
        self.service.config['models'][1]['runtime_profile']='qwen-image-21'
        self.workers=EditWorkers([])
        self.service.scheduler.factory=self.workers

    def body(self,**extra):
        return {'model':'qwen-image','prompt':'Add a scarf while preserving the subject.','image':image_url(),'steps':2,**extra}

    def upload(self,fields=None,files=None,path='/v1/images/edits',headers=None):
        raw=base64.b64decode(image_url().split(',',1)[1])
        content_type,data=multipart(fields or [('model','qwen-image'),('prompt','Add a scarf.'),('n','1'),('size','512x512')],files or [('image',raw)])
        return urllib.request.urlopen(urllib.request.Request(self.base+path,data,{'Content-Type':content_type,**(headers or {})}),timeout=15)

    def test_hermes_multipart_edits_default_to_base64_and_restore_chat(self):
        self.fixture()
        self.service.config['models'][1]['restore_text_after_image'] = True
        wait_job(self.service,self.service.control('qwen-chat','load')['id'])
        with self.upload() as response:
            result=json.load(response)
        self.assertEqual(result['x_operation'],'edit')
        self.assertIn(b'EDIT CONTRACT FIXTURE',base64.b64decode(result['data'][0]['b64_json']))
        self.assertEqual(self.workers.calls[-1]['action'],'image_edit')
        self.assertNotIn('strength',self.workers.calls[-1])
        wait_job(self.service, result['x_job_id'])
        self.assertEqual(set(self.service.scheduler.resident),{'qwen-chat'})
        self.assertEqual(self.workers.lifecycle,[('load','qwen-chat'),('close','qwen-chat'),('load','qwen-image'),('close','qwen-image'),('load','qwen-chat')])
        self.assertNotIn('input_images',result['x_artifacts'][0])
        self.assertNotIn('base64',json.dumps(self.service.store.job(result['x_job_id'])))

    def test_generation_edit_generation_switch_closes_old_pipeline_and_reuses_edit(self):
        self.fixture()
        self.service.config['models'][1]['keep_image_model_loaded'] = True
        plain={'model':'qwen-image','prompt':'A lake','steps':1}
        self.json('/v1/images/generations',plain)
        self.json('/v1/images/edits',self.body())
        self.json('/v1/images/edits',self.body())
        self.json('/v1/images/generations',plain)
        self.assertEqual([c['action'] for c in self.workers.calls],['image','image_edit','image_edit','image'])
        self.assertEqual(self.workers.lifecycle,[('load','qwen-image'),('close','qwen-image'),('load','qwen-image'),('close','qwen-image'),('load','qwen-image')])

    def test_async_json_saved_result_and_authenticated_cors(self):
        self.fixture()
        self.service.config.update(api_key='test-edit-key-1234567890',browser_access='api-key')
        headers={'Origin':'https://frontend.example','Authorization':'Bearer test-edit-key-1234567890'}
        request=urllib.request.Request(self.base+'/v1/images/edits',method='OPTIONS',headers={
            'Origin':headers['Origin'],'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
        with urllib.request.urlopen(request) as response:self.assertEqual(response.status,204)
        with self.upload(path='/api/image/edits',headers=headers) as response:
            self.assertEqual(response.status,202)
            self.assertEqual(response.headers['Access-Control-Allow-Origin'],headers['Origin'])
            receipt=json.load(response)
        self.assertEqual(wait_job(self.service,receipt['id'])['status'],'completed')
        with self.request(receipt['result_url'],headers=headers) as response:result=json.load(response)
        self.assertIn('b64_json',result['data'][0])
        with self.request('/v1/images/edits',self.body(quality='high',input_fidelity='high'),{**headers,'Prefer':'respond-async'}) as response:
            self.assertEqual(response.status,202); second=json.load(response)
        job=wait_job(self.service,second['id'])
        self.assertEqual(job['status'],'completed')
        self.assertEqual(len(job['parameter_notes']),2)
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/v1/images/edits',self.body(),{'Origin':headers['Origin']})
        self.assertEqual(error.exception.code,401)

    def test_json_public_source_and_local_artifact_reuse(self):
        self.fixture()
        raw=base64.b64decode(image_url().split(',',1)[1])
        with patch('lazyrig.vision.fetch_image',return_value=raw) as fetch:
            result=self.json('/v1/images/edits',self.body(image='https://images.example/source.png',response_format='url'))
            fetch.assert_called_once_with('https://images.example/source.png')
        self.assertTrue(result['data'][0]['url'].startswith(self.base+'/artifacts/'))
        name='a'*32+'.png';(self.service.store.directory/'artifacts'/name).write_bytes(raw)
        for url in ('/artifacts/'+name,self.base+'/artifacts/'+name):
            with patch('lazyrig.vision.fetch_image',side_effect=AssertionError('Own artifact must not use HTTP')):
                self.assertEqual(self.json('/v1/images/edits',self.body(image=url))['x_operation'],'edit')

    def test_random_edit_seeds_are_recorded_once_and_explicit_seeds_remain_exact(self):
        self.fixture()
        # Existing fixed model preferences survive an upgrade. Clients can
        # explicitly request fresh noise without rewriting those preferences.
        self.service.config['models'][1].setdefault('generation',{})['seed']=42
        body=self.body(seed=-1,response_format='url')
        headers={'Idempotency-Key':'random-edit-once'}
        with patch('lazyrig.service.secrets.randbelow',side_effect=[12345,777]) as random:
            with self.request('/v1/images/edits',body,headers) as response:first=json.load(response)
            wait_job(self.service,first['x_job_id'])
            with self.request('/v1/images/edits',body,headers) as response:retry=json.load(response)
            second=self.json('/v1/images/edits',body)
            wait_job(self.service,second['x_job_id'])
            fixed=self.json('/v1/images/edits',self.body(seed=0))
        self.assertEqual(first['x_job_id'],retry['x_job_id'])
        self.assertEqual([first['x_seed'],retry['x_seed'],second['x_seed'],fixed['x_seed']],[12345,12345,777,0])
        self.assertEqual(random.call_count,2)
        self.assertEqual([c['seed'] for c in self.workers.calls],[12345,777,0])
        self.assertEqual(first['x_artifacts'][0]['seed'],12345)

    def test_invalid_edit_options_fail_before_worker_load(self):
        self.fixture()
        invalid=[{'image':[]},{'image':[image_url(),image_url()]},{'image':'http://127.0.0.1/private'},
                 {'image':'file:///client/source.png'},{'image':'data:image/png;base64,aGVsbG8='},
                 {'strength':.8},{'mask':image_url()},{'n':2},{'output_format':'webp'},
                 {'background':'transparent'},{'image_url':image_url()},{'size':'1536x1024'}]
        for extra in invalid:
            with self.subTest(fields=list(extra)),self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/v1/images/edits',self.body(**extra))
            self.assertEqual(error.exception.code,400)
        self.assertFalse(self.workers.lifecycle)
        self.assertFalse(self.service.store.jobs())

    def test_edit_capabilities_follow_complete_export_and_classic_strength(self):
        self.fixture()
        model=self.service.model('qwen-image');self.assertTrue(capabilities(model)['available'])
        self.assertFalse(capabilities(model)['supports_strength'])
        rows=self.json('/v1/models')['data'];entry=next(m for m in rows if m['id']=='qwen-image')
        self.assertTrue(entry['capabilities']['image_editing']);self.assertEqual(entry['input_modalities'],['text','image'])
        missing=Path(model['path'])/'text_encoder_i2i/openvino_model.bin';missing.unlink()
        self.assertFalse(capabilities(model)['available'])
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/v1/images/edits',self.body())
        self.assertIn('text_encoder_i2i',error.exception.read().decode())
        classic=export_fixture(Path(self.temp.name)/'classic',False)
        self.service.config['models'][1].update(path=str(classic),runtime_profile='default')
        self.json('/api/settings',{'model':'qwen-image','generation':{'strength':.6}})
        self.json('/v1/images/edits',self.body())
        self.assertEqual(self.workers.calls[-1]['strength'],.6)


class MultipartTests(unittest.TestCase):
    def test_file_array_alias_and_malformed_or_duplicate_fields(self):
        content_type,data=multipart([('model','qwen-image'),('steps','4'),('guidance_scale','1.0')],[('image[]',b'png bytes')])
        parsed=parse_multipart(content_type,data)
        self.assertEqual(parsed['image[]'],[b'png bytes']);self.assertEqual(parsed['steps'],4)
        self.assertEqual(parsed['guidance_scale'],1.0)
        for fields,files in [([('model','a'),('model','b')],[]),([('steps','nan')],[]),([],[('mask',b'mask')])]:
            with self.assertRaises(ValueError):parse_multipart(*multipart(fields,files))
        with self.assertRaises(ValueError):parse_multipart('multipart/form-data',b'missing boundary')


class NativeEditContractTests(unittest.TestCase):
    def test_qwen_worker_uses_image_pipeline_rgb_conditioning_and_torch_seed_without_strength(self):
        import numpy as np
        import openvino as ov
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.png';Image.new('RGB',(32,20),'red').save(source)
            output=Path(directory)/'output.png';calls=[];events=[]
            class Pipeline:
                def __init__(self,*args,**kwargs):calls.append(('construct',args,kwargs))
                def generate(self,prompt,image,**kwargs):
                    calls.append(('generate',prompt,kwargs))
                    self.assertion=image.data.copy()
                    assert image.shape==[1,20,32,3]
                    assert tuple(image.data[0,0,0])==(255,0,0)
                    assert 'strength' not in kwargs and kwargs['generator']==('torch-seed',42)
                    assert kwargs['max_sequence_length']==512 and kwargs['negative_prompt']=='blur'
                    kwargs['callback'](0,2,None)
                    kwargs['callback'](1,2,None)
                    return ov.Tensor(np.zeros((1,256,256,3),dtype=np.uint8))
            class Connection:
                sent=False
                def send(self,event):events.append(event)
                def recv(self):
                    if self.sent:raise EOFError()
                    self.sent=True
                    return {'action':'image_edit','input_images':[{'path':str(source)}],'prompt':'Add a scarf',
                            'width':256,'height':256,'steps':2,'seed':42,'guidance_scale':4.0,
                            'max_sequence_length':512,'negative_prompt':'blur','output_path':str(output)}
                def close(self):pass
            fake=types.SimpleNamespace(LLMPipeline=object,VLMPipeline=object,Text2ImagePipeline=object,
                                       Image2ImagePipeline=Pipeline,TorchGenerator=lambda seed:('torch-seed',seed))
            model={'pipeline':'image','_image_operation':'edit','path':directory,'device':'CPU','runtime_profile':'qwen-image-21'}
            with patch.dict(sys.modules,{'openvino_genai':fake}):_child(Connection(),'openvino',model)
            self.assertEqual([e['kind'] for e in events],
                             ['worker_diagnostics','ready','progress','progress','result'],events)
            self.assertEqual(events[-1]['inference']['denoising'], {'step':2,'steps':2,'callbacks':2})
            self.assertEqual(events[-1]['inference']['seed'],42)
            self.assertTrue(events[-1]['inference']['classifier_free_guidance'])
            self.assertEqual(calls[0][1],(directory,'CPU'))
            with Image.open(output) as im:self.assertEqual(im.size,(256,256))

    def test_partial_denoising_return_is_not_published_as_a_completed_image(self):
        import numpy as np
        import openvino as ov
        class Pipeline:
            def generate(self, prompt, **kwargs):
                kwargs['callback'](0,40,None)
                return ov.Tensor(np.zeros((1,256,256,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output.png'
            request={'action':'image','prompt':'test','width':256,'height':256,'steps':40,'seed':12345,'output_path':str(output)}
            with self.assertRaisesRegex(RuntimeError,'1/40 denoising steps'):
                _native_image(types.SimpleNamespace(send=lambda event:None),Pipeline(),{'runtime_profile':'default'},request)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.png.part').exists())


if __name__=='__main__':unittest.main(verbosity=2)
