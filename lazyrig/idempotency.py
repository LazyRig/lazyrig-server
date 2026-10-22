"""Explicit retry identities for inference; identical prompts remain independent."""
import hashlib
import json


class IdempotencyConflict(ValueError):
    def __init__(self, job_id):
        super().__init__('Idempotency-Key already belongs to a different request. Recover its result or use a new key for a new operation.')
        self.job_id = job_id


def request_identity(key, body, operation):
    if not isinstance(key, str) or not 1 <= len(key) <= 128 or any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError('Idempotency-Key must be 1–128 visible ASCII characters without spaces')
    def canonical(value):
        if isinstance(value, bytes):
            return ['bytes', len(value), hashlib.sha256(value).hexdigest()]
        if isinstance(value, dict):
            return ['object', [[k, canonical(v)] for k, v in sorted(value.items())]]
        if isinstance(value, list):
            return ['array', [canonical(v) for v in value]]
        return ['json', value]
    payload = json.dumps([operation, canonical(body)], ensure_ascii=False,
                         allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(key.encode()).hexdigest(), hashlib.sha256(payload).hexdigest()


def image_identity(key, body, editing):
    return request_identity(key, body, 'edit' if editing else 'generate')
