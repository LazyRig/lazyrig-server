"""Deterministic workers for contract tests; never installed as a server mode."""
import json
import re
import time
from pathlib import Path
from lazyrig.context import TokenizerService as NativeTokenizer
from lazyrig.service import Service as NativeService
from lazyrig.worker import Cancelled
from lazyrig.metrics import GenerationTimer
from lazyrig.response_format import is_json

DEFAULT = {
    "mode": "openvino", "data_dir": "data", "port": 8765,
    "load_timeout_seconds": 600, "request_timeout_seconds": 1800,
    "devices": [
        {"id": "TEST.GPU", "name": "Test 24 GiB GPU", "budget_gib": 22, "policy": "exclusive"},
        {"id": "TEST.CPU", "name": "Test CPU", "budget_gib": 48, "policy": "budget"}
    ],
    "models": [
        {"id": "qwen-chat", "name": "Qwen3.8 · 27B", "task": "text-generation", "pipeline": "llm",
         "device": "TEST.GPU", "path": "", "context_limit": 32768, "memory_gib": 20,
         "plugin_config": {}, "template_context": {"enable_thinking": False}},
        {"id": "qwen-image", "name": "Qwen-Image · 2.1", "task": "text-to-image", "pipeline": "image",
         "device": "TEST.GPU", "path": "", "memory_gib": 18, "max_width": 1024, "max_height": 1024,
         "plugin_config": {}}
    ]
}


class TokenizerService(NativeTokenizer):
    def encode(self, model, messages, tools):
        details = {"template_context": model.get("template_context", {}), "template_notes": []}
        text = json.dumps({"messages": messages, "tools": tools, "template_context": details["template_context"]}, ensure_ascii=False)
        return text, [], max(1, (len(text.encode()) + 3) // 4), "estimated-test", details


def fixture_png(path, label="TEST OUTPUT", width=256, height=256):
    from PIL import Image, PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("fixture", label)
    with Image.new('RGB', (width, height), '#e6a06c') as image:
        image.save(path, format='PNG', pnginfo=info)


class FixtureWorker:
    def __init__(self, mode, model, timeout, cancelled):
        self.process = self
        self.alive = True
        self.worker_pid = None

    def is_alive(self): return self.alive
    def close(self): self.alive = False

    def run(self, request, timeout, cancelled):
        deadline = time.monotonic() + timeout
        def tick(delay):
            time.sleep(delay)
            if cancelled.is_set(): raise Cancelled("Job cancelled")
            if time.monotonic() > deadline: raise TimeoutError("Model operation exceeded its configured deadline")
        if request['action'] in {'image', 'image_edit'}:
            for step in range(1, request['steps'] + 1):
                tick(.035)
                yield {'kind': 'progress', 'step': step, 'steps': request['steps']}
            fixture_png(request['output_path'], width=request['width'], height=request['height'])
            yield {'kind':'result', 'path':request['output_path']}
            return
        if is_json(request.get('response_format')):
            raise ValueError('Test worker does not enforce JSON schemas')
        last=request['prepared']['messages'][-1]
        if last['role'] == 'tool':
            output='The image job finished and the chat model is available again. Your conversation survived the model switch.'
        elif json.loads(request['prepared']['text']).get('tools') and (request.get('force_image') or re.search(r'\b(draw|image|picture|illustration|render|wallpaper|photo)\b', last['content'], re.I)):
            output='<tool_call>'+json.dumps({'name':'generate_image','arguments':{'prompt':last['content'][:8000]}})+'</tool_call>'
        else:
            output='Test worker response. The request was received and conversation context has been retained. This deterministic reply validates delivery, history and token limits.'
        output=output[:request['max_tokens']*4]
        partial=''
        timer=GenerationTimer(request['prepared']['context']['input_tokens'], 'estimated-test') if request.get('timings_per_token') else None
        for chunk in re.findall(r'.{1,12}',output, flags=re.S):
            tick(.014)
            partial+=chunk
            generated=(len(partial.encode())+3)//4
            if timer:timer.observe(generated)
            yield {'kind':'text','text':partial,'generated_tokens':generated,'accuracy':'estimated-test', **({'timings':timer.snapshot(generated)} if timer else {})}
        if timer:timer.finish()
        yield {'kind':'result','text':partial,'generated_tokens':generated,'accuracy':'estimated-test', **({'timings':timer.snapshot(generated)} if timer else {})}


class Service(NativeService):
    def __init__(self, config, restore=True):
        super().__init__(config, restore, worker_factory=FixtureWorker, tokenizer_factory=TokenizerService)

    def discover(self):
        return {'devices':self.config['devices'], 'available':True, 'note':'Contract test fixtures', 'openvino_version':'test'}
