"""Opt-in native CPU check with an existing OpenVINO LLM; downloads nothing."""
import argparse
import copy
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lazyrig.config import SERVER_DEFAULT
from lazyrig.server import handler
from lazyrig.service import Service


def run(args):
    with tempfile.TemporaryDirectory(prefix='lazyrig-format-') as directory:
        config={**copy.deepcopy(SERVER_DEFAULT),'data_dir':directory}
        service=Service(config)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def request(path,body):
            req=urllib.request.Request(base+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
            try:return urllib.request.urlopen(req,timeout=120)
            except urllib.error.HTTPError as exc:raise RuntimeError(exc.read().decode()) from exc
        def api(path,body):
            with request(path,body) as response:return json.load(response)
        try:
            plugin={'INFERENCE_NUM_THREADS':2}
            if args.attention_backend:plugin['ATTENTION_BACKEND']=args.attention_backend
            api('/api/models',{'id':'native-format','path':str(Path(args.llm_dir).resolve()),'device':'CPU',
                              'memory_gib':1,'context_limit':2048,'plugin_config':plugin})
            body={'model':'native-format','messages':[{'role':'user','content':'Return a short title.'}],
                  'max_tokens':128,'temperature':0,'response_format':{'type':'json_schema','json_schema':{
                    'name':'bounded_title','strict':True,'schema':{'type':'object','properties':{
                       'title':{'type':'string','enum':['ok']}},'required':['title'],'additionalProperties':False}}}}
            result=api('/v1/chat/completions',body)
            assert json.loads(result['choices'][0]['message']['content'])=={'title':'ok'},result
            assert result['x_json_valid'] is True and result['x_context']['accuracy']=='exact'
            preflight=api('/api/context',{k:v for k,v in body.items() if k!='temperature'})
            assert preflight['input_tokens']==result['usage']['prompt_tokens'],(preflight,result['usage'])
            job=service.store.job(result['x_job_id'])
            assert job['content']==result['choices'][0]['message']['content']
            native_raw=job['text']
            worker=service.scheduler.resident['native-format']
            with request('/v1/chat/completions',{**body,'stream':True,'stream_options':{'include_usage':True}}) as response:
                data=response.read().decode()
            records=[json.loads(line[6:]) for line in data.splitlines() if line.startswith('data: {')]
            assert not any('error' in r for r in records),records
            text=''.join(r['choices'][0]['delta'].get('content','') for r in records if r.get('choices'))
            assert json.loads(text)=={'title':'ok'} and data.rstrip().endswith('[DONE]'),data
            assert records[-1]['usage']==result['usage']
            plain=api('/v1/chat/completions',{**body,'response_format':{'type':'text'},'max_tokens':4})
            assert plain['x_response_format']=={'type':'text'}
            assert service.scheduler.resident['native-format'] is worker
            assert plain['usage']['completion_tokens']>0
            evidence={'environment':service.detected,'model_directory':str(Path(args.llm_dir).resolve()),
                      'model_note':args.model_note,'plugin_config':plugin,'json_result':result,
                      'native_raw_output':native_raw,'sse_matches':True,'exact_context_matches':True,
                      'plain_request_reused_worker_and_cleared_constraint':True,'plain_result':plain}
            Path(args.evidence).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
            print('PASS: native schema generation, clean content, raw terminal marker retained, exact context, SSE, and constraint reset on reused CPU worker.')
        finally:
            server.shutdown();server.server_close();thread.join(3);service.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-dir',required=True)
    parser.add_argument('--attention-backend',choices=['SDPA','PA'])
    parser.add_argument('--model-note',default='User-supplied OpenVINO export; generation quality is not assessed.')
    parser.add_argument('--evidence',default='native-response-evidence.json')
    run(parser.parse_args())
