"""Native VLM HTTP smoke test with a supplied export; no weight download."""
import argparse
import base64
import copy
import io
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-vision-') as directory:
        service=Service({**copy.deepcopy(SERVER_DEFAULT),'data_dir':directory})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def request(path,body=None):
            req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
            try:return urllib.request.urlopen(req,timeout=120)
            except urllib.error.HTTPError as exc:raise RuntimeError(exc.read().decode()) from exc
        def api(path,body=None):
            with request(path,body) as response:return json.load(response)
        def user(color):
            buffer=io.BytesIO();Image.new('RGB',(48,32),color).save(buffer,format='PNG')
            return {'role':'user','content':[{'type':'text','text':'Describe this image.'},{'type':'image_url','image_url':{
                'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}}]}
        try:
            model=api('/api/models',{'id':'native-vlm','path':str(Path(args.vlm_dir).resolve()),'device':'CPU',
                     'memory_gib':1,'context_limit':2048,'plugin_config':{'INFERENCE_NUM_THREADS':2}})
            assert model['pipeline']=='vlm'
            metadata=api('/v1/models')['data'][0];assert metadata['capabilities']['vision']
            body={'model':'native-vlm','messages':[user('red')],'max_tokens':4,'temperature':0}
            preflight=api('/api/context',body)
            assert preflight['fits'] is None and not preflight['visual_tokens_included']
            result=api('/v1/chat/completions',body)
            assert result['x_context']['visual_tokens_included'] and result['x_context']['accuracy']=='exact',result
            assert result['usage']['prompt_tokens']>preflight['input_tokens'],(preflight,result)
            assert result['usage']['completion_tokens']>0
            with request('/v1/chat/completions',{**body,'stream':True,'stream_options':{'include_usage':True}}) as response:
                raw=response.read().decode()
            records=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            assert not any('error' in r for r in records),records
            stream=''.join(r['choices'][0]['delta'].get('content','') for r in records if r.get('choices'))
            assert stream==result['choices'][0]['message']['content']
            assert records[-1]['usage']==result['usage']
            unload=api('/api/models/native-vlm/unload',{})
            import time
            for _ in range(600):
                job=api('/api/jobs/'+unload['id'])
                if job['status'] in ('completed','failed'):break
                time.sleep(.05)
            assert job['status']=='completed',job
            history=[user('red'),{'role':'assistant','content':'An earlier image.'},user('blue')]
            resumed=api('/v1/chat/completions',{**body,'messages':history})
            assert resumed['x_context']['image_count']==2 and resumed['x_context']['visual_tokens_included']
            assert resumed['usage']['prompt_tokens']>result['usage']['prompt_tokens']
            plain=api('/v1/chat/completions',{**body,'messages':[{'role':'user','content':'Hello.'}]})
            assert plain['usage']['prompt_tokens']<result['usage']['prompt_tokens']
            tool_history=[{'role':'user','content':'Inspect both images.'},
                {'role':'assistant','content':'','tool_calls':[
                    {'id':'vision-'+color,'type':'function','function':{
                        'name':'vision_analyze','arguments':json.dumps({'image_url':'/client/'+color+'.png','question':'What is depicted?'})}}
                    for color in ('red','blue')]},
                *[{'role':'tool','name':'vision_analyze','tool_call_id':'vision-'+color,'content':user(color)['content']}
                    for color in ('red','blue')]]
            tool_body={**body,'messages':tool_history,'auto_image':False}
            tool_preflight=api('/api/context',tool_body)
            tool_result=api('/v1/chat/completions',tool_body)
            assert tool_result['x_context']['image_count']==2 and tool_result['x_context']['visual_tokens_included']
            assert tool_result['usage']['prompt_tokens']>tool_preflight['input_tokens']
            assert tool_result['usage']['completion_tokens']>0
            unload=api('/api/models/native-vlm/unload',{})
            for _ in range(600):
                job=api('/api/jobs/'+unload['id'])
                if job['status'] in ('completed','failed'):break
                time.sleep(.05)
            assert job['status']=='completed',job
            replay=api('/v1/chat/completions',{**tool_body,'messages':tool_history+[
                {'role':'assistant','content':'Both images were inspected.'},
                {'role':'user','content':'Compare the two earlier images again.'}]})
            assert replay['x_context']['image_count']==2 and replay['x_context']['visual_tokens_included']
            assert replay['usage']['completion_tokens']>0
            evidence={'environment':service.detected,'model_note':args.model_note,'model_directory':str(Path(args.vlm_dir).resolve()),
                      'metadata':metadata,'preflight':preflight,'image_result':result,'sse_content_and_usage_match':True,
                      'two_images_after_unload_reload':resumed,'following_text_only_result':plain,
                      'tool_image_preflight':tool_preflight,'tool_image_result':tool_result,
                      'tool_images_replayed_after_reload':replay}
            Path(args.evidence).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
            print('PASS: real VLM user/tool image input via HTTP, native visual token counts, SSE, user/tool history after reload, and subsequent text-only input.')
        finally:
            server.shutdown();server.server_close();thread.join(3);service.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlm-dir',required=True)
    parser.add_argument('--model-note',default='User-supplied VLM. This test checks execution, not visual answer quality.')
    parser.add_argument('--evidence',default='native-vision-evidence.json')
    run(parser.parse_args())
