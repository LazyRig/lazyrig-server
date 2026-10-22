"""Image-edit capabilities, multipart uploads and source-image preparation."""
import json
import math
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

SOURCE_FIELDS = {'image', 'image[]', 'images', 'image_url'}
EDIT_FIELDS = SOURCE_FIELDS | {'mask', 'strength', 'quality', 'input_fidelity', 'output_format', 'background'}


def capabilities(model):
    kind, reason = None, 'This export has no supported image-edit adapter.'
    if model.get('pipeline') == 'image':
        root = Path(model.get('path') or '__no_model__')
        architecture = model.get('inspection', {}).get('architecture', '')
        try:
            architecture = json.loads((root / 'model_index.json').read_text()).get('_class_name', architecture)
        except (OSError, ValueError, TypeError):
            pass
        if model.get('runtime_profile') == 'qwen-image-21' or 'qwenimage21' in str(architecture).lower().replace('_', ''):
            kind = 'qwen-image-21'
        elif architecture in {'StableDiffusionPipeline', 'StableDiffusionXLPipeline', 'LatentConsistencyModelPipeline'}:
            kind = 'img2img'
        required = ['vae_encoder', 'vision_encoder', 'text_encoder_i2i'] if kind == 'qwen-image-21' else ['vae_encoder']
        missing = [name for name in required if not all((root / name / ('openvino_model.' + suffix)).is_file() for suffix in ('xml', 'bin'))]
        if kind and missing:
            reason = 'Image editing needs a complete export. Missing XML/BIN components: ' + ', '.join(missing)
            kind = None
        elif kind:
            reason = 'Uses the same export through OpenVINO Image2ImagePipeline; runtime/device support is checked when loading.'
    return {'available': kind is not None, 'kind': kind, 'max_input_images': 1 if kind else 0,
            'supports_strength': kind == 'img2img', 'supports_mask': False, 'note': reason}


def parse_multipart(content_type, data):
    """Python 3.13-compatible parser. Uploaded filenames never become paths."""
    if '\r' in content_type or '\n' in content_type:
        raise ValueError('Invalid multipart Content-Type')
    try:
        message = BytesParser(policy=policy.default).parsebytes(
            b'Content-Type: ' + content_type.encode('ascii') + b'\r\nMIME-Version: 1.0\r\n\r\n' + data)
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Invalid multipart request') from exc
    if message.get_content_type() != 'multipart/form-data' or not message.is_multipart() or message.defects:
        raise ValueError('Malformed multipart body or missing boundary')
    body = {}
    for index, part in enumerate(message.iter_parts()):
        if index >= 32 or part.is_multipart() or part.defects or part.get_content_disposition() != 'form-data':
            raise ValueError('Invalid multipart field')
        name = part.get_param('name', header='content-disposition')
        if not name:
            raise ValueError('Multipart fields need a name')
        value = part.get_payload(decode=True)
        if name == 'mask':
            raise ValueError('Mask-based inpainting is not supported by this image-edit endpoint; describe the edit in prompt')
        if name in SOURCE_FIELDS:
            if part.get_filename() is None:
                raise ValueError('Multipart image must be a file upload; send URLs using application/json')
            body.setdefault(name, []).append(value)
        else:
            if name in body or part.get_filename() is not None or len(value) > 65536:
                raise ValueError('Duplicate, oversized or unexpected multipart field: ' + name)
            try:
                text = value.decode('utf-8')
                if name in {'n', 'steps', 'num_inference_steps', 'seed', 'rng_seed', 'max_sequence_length'}:
                    value = int(text)
                elif name in {'guidance_scale', 'strength'}:
                    value = float(text)
                    if not math.isfinite(value):
                        raise ValueError('Non-finite value')
                elif name == 'enhance_prompt':
                    if text not in {'true', 'false'}:
                        raise ValueError('Expected true or false')
                    value = text == 'true'
                else:
                    value = text
            except (UnicodeError, ValueError) as exc:
                raise ValueError('Invalid multipart value for ' + name) from exc
            body[name] = value
    return body


def prepare(store, model, body, origin=''):
    info = capabilities(model)
    if not info['available']:
        raise ValueError(info['note'])
    if body.get('mask') is not None:
        raise ValueError('Mask-based inpainting is not supported; describe the edit in prompt')
    keys = SOURCE_FIELDS & body.keys()
    if len(keys) != 1:
        raise ValueError('Provide one source image in image (or image_url/images), not multiple source fields')
    source = body[next(iter(keys))]
    if isinstance(source, list):
        if len(source) != 1:
            raise ValueError('This OpenVINO edit adapter accepts exactly one source image per request')
        source = source[0]
    if isinstance(source, dict):
        if set(source) == {'image_url'}:
            source = source['image_url']
        if isinstance(source, dict) and set(source) == {'url'}:
            source = source['url']
    if not isinstance(source, (str, bytes)) or not source:
        raise ValueError('Provide an image file, image data URL, public HTTP(S) URL, or this server\'s artifact URL')

    options, notes = {}, []
    strength = body.get('strength', model.get('generation', {}).get('strength'))
    if strength is not None:
        if info['kind'] == 'qwen-image-21':
            raise ValueError('Qwen-Image-2.1 uses image conditioning, not denoising strength; omit strength')
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError('strength must be greater than 0 and at most 1')
    if info['supports_strength']:
        options['strength'] = .8 if strength is None else strength
    for key, values in (('quality', {'auto', 'low', 'medium', 'high', 'xhigh', 'max', 'standard', 'hd'}),
                        ('input_fidelity', {'low', 'high'})):
        if key in body:
            if not isinstance(body[key], str) or body[key] not in values:
                raise ValueError('Invalid ' + key)
            notes.append(key + ' is a client hint without a direct OpenVINO equivalent; saved model settings and explicit inference steps are used.')
    if body.get('output_format', 'png') != 'png':
        raise ValueError('This endpoint supports output_format=png')
    if body.get('background', 'auto') not in {'auto', 'opaque'}:
        raise ValueError('This endpoint produces opaque images; use background=auto or opaque')

    data = read_source(store, source, origin)
    ref = store.save(data, 'auto')
    return {**options, 'input_images': [store.describe(ref)]}, notes


def read_source(store, source, origin=''):
    """Resolve a public upload/URL or a bounded local artifact, never a file path."""
    if isinstance(source, str):
        url = urlsplit(source)
        base = urlsplit(origin)
        own = not url.scheme and not url.netloc or (origin and (url.scheme, url.netloc) == (base.scheme, base.netloc))
        if own and not url.query and not url.fragment and re.fullmatch(r'/artifacts/[a-f0-9]{32}\.png', url.path):
            path = store.directory.parent / 'artifacts' / Path(url.path).name
            if not path.is_file():
                raise ValueError('Source artifact does not exist on this server')
            source = path.read_bytes()
    return store.read_source(source)
