"""Best-effort native fault evidence without recording prompts or tensor data."""
import faulthandler
import os
import re
import sys
import tempfile


FAULT_LOG_BYTES = 8192


def create_fault_log():
    try:
        fd, path = tempfile.mkstemp(prefix='lazyrig-worker-', suffix='.fault')
        os.close(fd)
        return path
    except OSError:
        return None


def enable_fault_log(path):
    if not path:
        return None
    stream = None
    try:
        stream = open(path, 'ab', buffering=0)
        faulthandler.enable(file=stream, all_threads=True)
        return stream
    except (OSError, RuntimeError):
        if stream is not None:
            stream.close()
        return None


def read_fault_log(path):
    if not path:
        return {'status': 'unavailable'}
    try:
        with open(path, 'rb') as stream:
            data = stream.read(FAULT_LOG_BYTES + 1)
    except OSError:
        return {'status': 'unavailable'}
    if not data:
        return {'status': 'empty'}
    text = data[:FAULT_LOG_BYTES].decode('utf-8', errors='replace')
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    return {'status': 'captured', 'text': text,
            'truncated': len(data) > FAULT_LOG_BYTES,
            'scope': 'Python fault-handler trace; not a native crash dump or proof of root cause'}


def runtime_versions(ov, genai):
    return {'python': sys.version.split()[0],
            'openvino': str(getattr(ov, '__version__', 'unknown'))[:160],
            'openvino_genai': str(getattr(genai, '__version__', 'unknown'))[:160]}
