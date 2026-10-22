"""Submit once, check or recover a LazyRig image job without a long terminal call.

Uses only Python's standard library. Credentials come from LAZYRIG_API_KEY.
Examples (set LAZYRIG_URL or pass --url before the command):
  python image_job.py submit --request request.json --state subway-job.json
  python image_job.py status --state subway-job.json --output subway.png
  python image_job.py list --model my-image-model
  python image_job.py status --job JOB_ID --output recovered.png
Run the same submit command after a lost connection: the saved retry key is reused.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def base_url(value):
    value = value.rstrip('/')
    if value.endswith('/v1'):
        value = value[:-3]
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTP(S) server URL; provide its API key through LAZYRIG_API_KEY')
    return value


def request(base, path, body=None, retry_key=None, raw=False):
    headers = {}
    if os.environ.get('LAZYRIG_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['LAZYRIG_API_KEY']
    if retry_key:
        headers['Idempotency-Key'] = retry_key
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
            data = response.read()
            return data if raw else json.loads(data)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode('utf-8', 'replace')
        raise RuntimeError(f'HTTP {exc.code}: {text}') from exc


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as target:
            os.chmod(temporary, 0o600)
            target.write(data); target.flush(); os.fsync(target.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_state(path, value):
    write_atomic(path, json.dumps(value, indent=2).encode())


def summary(base, result):
    jid = result.get('x_job_id') or result.get('id')
    artifacts = result.get('x_artifacts', [])
    ready = bool(result.get('x_result_ready') or result.get('data'))
    status = result.get('x_status', result.get('status'))
    failed = status in {'failed', 'cancelled', 'interrupted'}
    next_action = ('Use the saved image; do not generate it again.' if ready else
                   'Inspect the job error before deciding whether to start a new generation.' if failed else
                   'Check this job again; do not submit a new generation.')
    return {'job_id': jid, 'image_status': 'completed' if ready else status if failed else 'pending',
            'job_status': status, 'phase': result.get('x_phase', result.get('phase')),
            'image_urls': [base + a['url'] for a in artifacts],
            'result_url': base + '/api/jobs/' + jid + '/result', 'warning': result.get('x_warning'),
            'next_action': next_action}


def run(args):
    base = base_url(args.url or os.environ.get('LAZYRIG_URL', 'http://127.0.0.1:8765'))
    if args.command == 'list':
        query = '?model=' + urllib.parse.quote(args.model, safe='') if args.model else ''
        result = request(base, '/api/image/jobs' + query)
        return {'jobs': [{**summary(base, row), 'prompt': row.get('prompt'), 'created': row.get('created')} for row in result['data']]}
    if args.command == 'submit':
        state_path = Path(args.state)
        body = json.loads(Path(args.request).read_text(encoding='utf-8')) if args.request else None
        if body is not None:
            if not isinstance(body, dict):
                raise ValueError('Request file must contain a JSON object')
            body.setdefault('response_format', 'b64_json')
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding='utf-8'))
            if args.url and base != saved['server']:
                raise ValueError('This state file belongs to another server')
            if body is not None and body != saved['request']:
                raise ValueError('This state file belongs to a different request; use a new state file for a new image')
            if args.request_id and args.request_id != saved['request_id']:
                raise ValueError('The saved retry key cannot be changed')
            if args.edit and saved['endpoint'] != '/api/image/edits':
                raise ValueError('This state file belongs to a generation request')
            base = saved['server']
        else:
            if body is None:
                raise ValueError('Initial submission needs --request FILE')
            saved = {'server': base, 'endpoint': '/api/image/edits' if args.edit else '/api/image/jobs',
                     'request_id': args.request_id or uuid.uuid4().hex, 'request': body}
            state_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents two clients inventing different keys
            # for the same state file. The identity reaches disk before POST.
            with state_path.open('x', encoding='utf-8') as target:
                os.chmod(state_path, 0o600)
                json.dump(saved, target, indent=2); target.flush(); os.fsync(target.fileno())
        if not request(base, '/api/health').get('features', {}).get('image_job_idempotency'):
            raise ValueError('Server does not advertise safe image request retries; submission stopped')
        try:
            result = request(base, saved['endpoint'], saved['request'], saved['request_id'])
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f'{exc}\nState retained in {state_path}. Run the SAME submit command to recover this job; do not create a new state file.') from exc
        saved['job_id'] = result['x_job_id']
        save_state(state_path, saved)
        return {**summary(base, result), 'state_file': str(state_path), 'request_id': saved['request_id']}
    if args.state:
        saved = json.loads(Path(args.state).read_text(encoding='utf-8'))
        if args.url and base != saved['server']:
            raise ValueError('This state file belongs to another server')
        base, jid = saved['server'], saved.get('job_id')
        if not jid:
            raise ValueError('The submission receipt was not saved. Run submit with this SAME --state file to recover the job.')
    else:
        jid = args.job
    if not re.fullmatch(r'[a-f0-9]{32}', jid):
        raise ValueError('Job ID must be 32 lowercase hexadecimal characters')
    result = request(base, '/api/jobs/' + jid + '/result')
    status = summary(base, result)
    if args.output and result.get('data'):
        first = result['data'][0]
        if first.get('b64_json'):
            data = base64.b64decode(first['b64_json'], validate=True)
        else:
            artifact = result['x_artifacts'][0]['url']
            if not re.fullmatch(r'/artifacts/[a-f0-9]{32}\.(png|svg)', artifact):
                raise ValueError('Unexpected artifact URL')
            data = request(base, artifact, raw=True)
        output = Path(args.output)
        if output.exists() and output.read_bytes() != data:
            raise ValueError('Output file already exists with different contents; choose another output path')
        if not output.exists():
            write_atomic(output, data)
        status.update(saved_file=str(output.resolve()), sha256=hashlib.sha256(data).hexdigest())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Server URL; defaults to LAZYRIG_URL, then localhost:8765')
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit', help='Submit once or recover a lost submission receipt using its saved key')
    submit.add_argument('--request', help='Image request JSON file; optional when resuming a saved state')
    submit.add_argument('--state', required=True, help='Durable local receipt, one per intended image')
    submit.add_argument('--request-id', help='Optional explicit retry key; otherwise saved automatically')
    submit.add_argument('--edit', action='store_true')
    status = commands.add_parser('status', help='Read the current result once; never queues generation')
    choice = status.add_mutually_exclusive_group(required=True)
    choice.add_argument('--state')
    choice.add_argument('--job', help='Recover an existing server job without a local receipt')
    status.add_argument('--output', help='Save the image when it is ready')
    listing = commands.add_parser('list', help='Find recent image jobs and artifacts after a client timeout')
    listing.add_argument('--model')
    try:
        print(json.dumps(run(parser.parse_args()), indent=2))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({'error': str(exc), 'next_action': 'Recover or check the existing job before starting another generation.'}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
