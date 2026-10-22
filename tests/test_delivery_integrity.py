"""Delivery regressions for the serving workflow."""
import json
import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
import test_http
from test_core import wait_job
from lazyrig.worker import _native_image
from lazyrig.config import validate, SERVER_DEFAULT
import copy


class ContinueTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    json = test_http.HTTPTests.json
    request = test_http.HTTPTests.request

    def test_expect_continue_is_followed_by_final_receipt_and_saved_result(self):
        body=json.dumps({'model':'qwen-image','prompt':'A geometric shape','steps':1}).encode()
        with socket.create_connection(('127.0.0.1',self.server.server_port),timeout=5) as conn:
            conn.sendall((f'POST /api/image/jobs HTTP/1.1\r\nHost: 127.0.0.1:{self.server.server_port}\r\nExpect: 100-continue\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode())
            interim=conn.recv(4096)
            self.assertEqual(interim,b'HTTP/1.1 100 Continue\r\n\r\n')
            conn.sendall(body);chunks=[]
            while data:=conn.recv(4096):chunks.append(data)
        header,raw=b''.join(chunks).split(b'\r\n\r\n',1)
        self.assertIn(b'202 Accepted',header)
        receipt=json.loads(raw);self.assertIn('events_url',receipt)
        self.assertFalse(receipt['x_handoff_complete'])
        wait_job(self.service,receipt['id'])
        result=self.json(receipt['result_url'])
        self.assertTrue(result['x_result_ready']);self.assertTrue(result['x_handoff_complete'])
        with self.request(result['data'][0]['url'].removeprefix(self.base)) as response:self.assertEqual(response.read()[:8],b'\x89PNG\r\n\x1a\n')

    def test_removed_demo_mode_is_rejected(self):
        config=copy.deepcopy(SERVER_DEFAULT);config['mode']='demo'
        with self.assertRaisesRegex(ValueError,'Only the OpenVINO'):validate(config)


class AtomicImageTests(unittest.TestCase):
    def test_png_is_not_public_until_encoding_finishes_and_failed_encoding_cleans_up(self):
        pipeline=types.SimpleNamespace(generate=lambda *a,**kw:types.SimpleNamespace(data=np.zeros((1,16,16,3),dtype=np.uint8)))
        conn=types.SimpleNamespace(send=lambda e:None)
        original=Image.Image.save
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'output.png'
            request={'action':'image','prompt':'test','width':256,'height':256,'steps':1,'seed':42,'output_path':str(target)}
            waiting,release=threading.Event(),threading.Event();errors=[]
            def gated(image,stream,*args,**kwargs):
                stream.write(b'partial');stream.flush();waiting.set();release.wait(3);stream.seek(0);stream.truncate();return original(image,stream,*args,**kwargs)
            def run():
                try:_native_image(conn,pipeline,{},request)
                except BaseException as e:errors.append(e)
            with patch.object(Image.Image,'save',gated):
                thread=threading.Thread(target=run);thread.start()
                try:self.assertTrue(waiting.wait(2));self.assertFalse(target.exists())
                finally:release.set();thread.join(3)
            self.assertFalse(errors);self.assertFalse(target.with_suffix('.png.part').exists())
            with Image.open(target) as image:image.verify()
            target.unlink()
            def fail(image,stream,*args,**kwargs):stream.write(b'partial');raise OSError('Disk write failed')
            with patch.object(Image.Image,'save',fail),self.assertRaisesRegex(OSError,'Disk write'):_native_image(conn,pipeline,{},request)
            self.assertFalse(target.exists());self.assertFalse(target.with_suffix('.png.part').exists())



class IncrementalToolStreamTests(unittest.TestCase):
    setUp = test_http.HTTPTests.setUp
    tearDown = test_http.HTTPTests.tearDown
    request = test_http.HTTPTests.request
    json = test_http.HTTPTests.json

    def test_tool_enabled_chat_streams_before_native_generation_finishes(self):
        from support import FixtureWorker
        from lazyrig.context import IMAGE_TOOL
        release, blocked = threading.Event(), threading.Event()
        def factory(*args):
            worker=FixtureWorker(*args)
            def run(payload,timeout,cancelled):
                yield {'kind':'text','text':'<think>Check.</think>Hello','generated_tokens':6,'accuracy':'estimated-test'}
                blocked.set()
                if not release.wait(5):raise RuntimeError('Client did not receive the incremental text')
                yield {'kind':'result','text':'<think>Check.</think>Hello there.','generated_tokens':10,'accuracy':'estimated-test'}
            worker.run=run;return worker
        self.service.scheduler.factory=factory
        body={'model':'qwen-chat','messages':[{'role':'user','content':'Hello'}],'stream':True,'tools':[IMAGE_TOOL]}
        try:
            with self.request('/v1/chat/completions',body) as response:
                self.assertEqual(response.headers['X-Accel-Buffering'],'no')
                self.assertIn('no-transform',response.headers['Cache-Control'])
                deltas=[]
                while True:
                    line=response.readline().decode()
                    if line.startswith('data: {'):
                        value=json.loads(line[6:]);self.assertNotIn('error',value)
                        delta=value['choices'][0]['delta'];deltas.append(delta)
                        if delta.get('content')=='Hello':break
                self.assertTrue(blocked.wait(1));self.assertFalse(release.is_set());release.set()
                for line in response.read().decode().splitlines():
                    if line.startswith('data: {'):
                        value=json.loads(line[6:]);self.assertNotIn('error',value);deltas.append(value['choices'][0]['delta'])
                self.assertEqual(''.join(d.get('content','') for d in deltas),'Hello there.')
                self.assertEqual(''.join(d.get('reasoning_content','') for d in deltas),'Check.')
        finally:release.set()

if __name__=='__main__':unittest.main()
