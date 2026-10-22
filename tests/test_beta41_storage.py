"""Production boundary tests: real SQLite/OS locks; no native GPU required."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from lazyrig.context import TokenizerService
from lazyrig.deadlines import DeadlineEvent, RequestDeadline, deadline_context
from lazyrig.storage import Store
from lazyrig.storage_lock import DataDirectoryInUse


class StorageHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def store(self, path=None):
        store = Store(path or self.path)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def job(jid='job', status='completed'):
        return {'id': jid, 'status': status, 'prompt': 'Test prompt', 'model': 'fixture',
                'created': 1, 'updated': 2, 'request': {}}

    def test_second_owner_cannot_run_recovery_in_same_process(self):
        first = self.store()
        first.save_job(self.job(status='running'))
        with self.assertRaisesRegex(DataDirectoryInUse, 'Another server may already own it'):
            Store(self.path / '.' / 'artifacts' / '..')
        self.assertEqual(first.job('job')['status'], 'running')
        first.close()
        reopened = self.store()
        self.assertEqual(reopened.job('job')['status'], 'interrupted')

    def test_owner_lock_is_cross_process_and_releases_after_close(self):
        first = self.store()
        first.save_job(self.job(status='running'))
        script = '''import sys
from lazyrig.storage import Store
from lazyrig.storage_lock import DataDirectoryInUse
try:
    store=Store(sys.argv[1])
except DataDirectoryInUse:
    print('owned', flush=True)
    sys.exit(7)
print(store.job('job')['status'], flush=True)
store.close()
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)], capture_output=True, text=True, timeout=15)
        self.assertEqual((result.returncode, result.stdout.strip()), (7, 'owned'), result.stderr)
        self.assertEqual(first.job('job')['status'], 'running')
        first.close()
        result = subprocess.run([sys.executable, '-c', script, str(self.path)], capture_output=True, text=True, timeout=15)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, 'interrupted'), result.stderr)

    def test_killed_owner_does_not_leave_stale_lock(self):
        script = '''import sys
from lazyrig.storage import Store
store=Store(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
'''
        child = subprocess.Popen([sys.executable, '-c', script, str(self.path)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            with self.assertRaises(DataDirectoryInUse):
                Store(self.path)
            child.kill()
            child.wait(timeout=15)
            self.assertTrue(self.store().health(probe=True)['ok'])
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=15)

    def test_canonical_symlink_alias_cannot_bypass_owner(self):
        first = self.store(self.path / 'real')
        alias = self.path / 'alias'
        try:
            alias.symlink_to(first.directory, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('Directory symlink privilege unavailable')
        with self.assertRaises(DataDirectoryInUse):
            Store(alias)

    def test_failed_initialization_releases_ownership(self):
        with patch('lazyrig.storage.sqlite3.connect', side_effect=sqlite3.OperationalError('injected open failure')):
            with self.assertRaisesRegex(sqlite3.OperationalError, 'injected open failure'):
                Store(self.path)
        self.assertTrue(self.store().health()['ok'])

    def test_storage_error_latches_until_real_write_probe_succeeds(self):
        store = self.store()
        store.db.execute('PRAGMA query_only=ON')
        with self.assertRaises(sqlite3.OperationalError):
            store.session('Must fail')
        self.assertFalse(store.health()['ok'])
        self.assertEqual(store.sessions(), [])
        self.assertFalse(store.health()['ok'], 'A successful read cannot prove writes work')
        self.assertFalse(store.health(probe=True)['writable'])
        store.db.execute('PRAGMA query_only=OFF')
        self.assertTrue(store.health(probe=True)['writable'])
        self.assertIsNone(store.health()['error'])
        store.close()
        store.close()
        self.assertFalse(store.health(probe=True)['ok'])

    def test_stream_replay_preserves_resets_missing_channels_and_cursors(self):
        store = self.store()
        samples = [
            {'text': '<think>reason', 'content': '', 'reasoning_content': 'reason', 'stage': 'initial'},
            {'text': '<think>reason</think>answer', 'content': 'answer', 'reasoning_content': 'reason', 'stage': 'initial'},
            {'text': 'new', 'content': 'new', 'reasoning_content': '', 'stage': 'final'},
            {'text': 'new response 🧠', 'content': 'new response 🧠', 'reasoning_content': '', 'stage': 'final'},
            {'text': 'new corrected', 'content': 'new corrected', 'reasoning_content': 'scratch', 'stage': 'final'},
            {'text': 'no optional fields'},
            {'text': '', 'content': '', 'reasoning_content': ''},
        ]
        with patch('lazyrig.storage.time.time', return_value=10.5):
            for sample in samples:
                store.event('job', 'text', **sample)
                store.event('other', 'progress', step=1)
        expected = [{'seq': i * 2 + 1, 'kind': 'text', 'time': 10.5, **sample} for i, sample in enumerate(samples)]
        self.assertEqual(store.events('job'), expected)
        for event in expected:
            self.assertEqual(store.events('job', event['seq']), [item for item in expected if item['seq'] > event['seq']])
        store.close()
        reopened = self.store()
        self.assertEqual(reopened.events('job'), expected)
        reopened.event('job', 'text', text='restart suffix', content='restart suffix', reasoning_content='')
        self.assertEqual(reopened.events('job')[-1]['text'], 'restart suffix')

    def test_stream_storage_growth_is_linear_and_replay_is_paginated(self):
        store = self.store()
        sizes = []
        for count in (1000, 2000):
            jid = str(count)
            for i in range(1, count + 1):
                text = 'a' * (64 * i)
                store.event(jid, 'text', text=text, content=text, reasoning_content='', stage='initial')
            sizes.append(store.db.execute('SELECT SUM(length(body)) FROM events WHERE job=?', (jid,)).fetchone()[0])
        self.assertLess(sizes[1], sizes[0] * 2.7, sizes)
        self.assertLess(sizes[1], 2_000_000, sizes)
        store.close()
        store = self.store()
        cursor, seen = 0, 0
        while batch := store.events('2000', cursor):
            self.assertLessEqual(len(batch), 300)
            for event in batch:
                seen += 1
                self.assertEqual(event['text'], 'a' * (64 * seen))
                self.assertEqual(event['content'], event['text'])
            cursor = batch[-1]['seq']
        self.assertEqual(seen, 2000)
        # An arbitrary cursor must also reconstruct from durable state rather
        # than depend on an earlier subscriber having visited that position.
        middle = store.events('2000', 2300)
        self.assertEqual(len(middle[0]['text']), 64 * 1301)

    def test_legacy_database_migrates_without_changing_old_events_or_retry_key(self):
        connection = sqlite3.connect(self.path / 'state.sqlite3')
        connection.executescript('''CREATE TABLE jobs(id TEXT PRIMARY KEY,body TEXT);
            CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,body TEXT);
            CREATE TABLE image_request_keys(key_hash TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,job TEXT NOT NULL);''')
        job = self.job()
        connection.execute('INSERT INTO jobs VALUES(?,?)', ('job', json.dumps(job)))
        connection.execute('INSERT INTO image_request_keys VALUES(?,?,?)', ('key', 'fingerprint', 'job'))
        old = [{'kind': 'context', 'time': 1, 'input_tokens': 2},
               {'kind': 'text', 'time': 2, 'text': 'legacy'},
               {'kind': 'text', 'time': 3, 'text': 'legacy answer', 'content': 'legacy answer', 'reasoning_content': ''}]
        for event in old:
            connection.execute('INSERT INTO events(job,body) VALUES(?,?)', ('job', json.dumps(event)))
        connection.commit()
        connection.close()
        store = self.store()
        self.assertEqual(store.events('job'), [{'seq': i + 1, **event} for i, event in enumerate(old)])
        store.event('job', 'text', text='legacy answer continued', content='legacy answer continued', reasoning_content='')
        self.assertEqual(store.events('job', 2)[-1]['text'], 'legacy answer continued')
        self.assertEqual(store.inference_request(('key', 'fingerprint')), job)
        self.assertEqual(store.jobs(summary=True)[0]['prompt'], job['prompt'])
        store.close()
        self.assertEqual(self.store().events('job', 2)[-1]['text'], 'legacy answer continued')

    def test_job_summary_preserves_contract_without_loading_full_job_bodies(self):
        store = self.store()
        job = self.job()
        job.update(prompt='p' * 1000, text='x' * 1_000_000, history=[{'content': 'y' * 1000}],
                   request={'max_tokens': 123, 'width': 64, 'temperature': .7})
        store.save_job(job)
        statements = []
        store.db.set_trace_callback(statements.append)
        summary = store.jobs(summary=True)[0]
        store.db.set_trace_callback(None)
        self.assertEqual(summary['prompt'], 'p' * 160)
        self.assertEqual(summary['request'], {'max_tokens': 123, 'width': 64})
        self.assertNotIn('text', summary)
        self.assertNotIn('history', summary)
        self.assertTrue(any('SELECT summary FROM jobs' in query for query in statements))
        self.assertEqual(store.job('job')['text'], job['text'])

    def test_opt_in_compaction_retains_final_text_lifecycle_and_idempotency(self):
        store = self.store()
        store.save_job(self.job(), retry_identity=('key', 'fingerprint'))
        store.save_job(self.job('active', 'running'))
        artifact = self.path / 'artifacts' / ('a' * 32 + '.png')
        artifact.write_bytes(b'fixture artifact must remain')
        for jid in ('job', 'active'):
            store.event(jid, 'queued')
            for count in range(1, 6):
                store.event(jid, 'text', text='a' * count, content='a' * count, reasoning_content='')
            store.event(jid, 'finished', status='completed')
        original = store.events('job')
        result = store.compact_events(before=3)
        self.assertEqual(result, {'jobs': 1, 'text_events_removed': 4})
        final = store.events('job')
        self.assertEqual(final, [original[0], original[-2], original[-1]])
        self.assertEqual(store.events('job', original[3]['seq']), final[1:])
        self.assertEqual(len(store.events('active')), 7)
        self.assertEqual(store.inference_request(('key', 'fingerprint'))['id'], 'job')
        self.assertEqual(artifact.read_bytes(), b'fixture artifact must remain')
        self.assertEqual(store.compact_events(before=3)['jobs'], 0)
        store.close()
        self.assertEqual(self.store().events('job'), final)


class VisionContextBoundTests(unittest.TestCase):
    def prepare(self, count):
        class Images:
            def flatten(self, messages, model):
                return messages, [{'sha256': 'fixture-image'}]

        class Tokens(TokenizerService):
            def encode(self, model, messages, tools):
                return 'prompt', [], count, 'estimated-template', {'template_context': {}, 'template_notes': []}

        return Tokens('test', Images()).prepare({'context_limit': 128, 'pipeline': 'vlm'},
                                                [{'role': 'user', 'content': 'Hello'}], 32,
                                                tools_enabled=False)['context']

    def test_known_text_overflow_is_not_hidden_by_unknown_vision_token_cost(self):
        self.assertIs(self.prepare(100)['fits'], False)
        self.assertIs(self.prepare(96)['fits'], None)
        self.assertIs(self.prepare(10)['fits'], None)
        self.assertFalse(self.prepare(100)['visual_tokens_included'])

    def test_waiting_for_tokenizer_lock_honors_deadline_and_cancellation(self):
        from lazyrig.worker import Cancelled

        class Tokens(TokenizerService):
            calls = 0

            def encode(self, model, messages, tools):
                self.calls += 1
                return 'prompt', [], 10, 'exact', {'template_context': {}, 'template_notes': []}

        model = {'context_limit': 128, 'pipeline': 'llm'}
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                tokens = Tokens('test')
                signal = DeadlineEvent(timeout=None if cancel else .08)
                entered, errors = threading.Event(), []

                def wait_for_tokenizer():
                    try:
                        with deadline_context(signal):
                            entered.set()
                            tokens.prepare(model, [], 16, tools_enabled=False)
                    except Exception as exc:
                        errors.append(exc)

                thread = threading.Thread(target=wait_for_tokenizer, daemon=True)
                try:
                    with tokens.lock:
                        thread.start()
                        self.assertTrue(entered.wait(1))
                        if cancel:
                            signal.set()
                        thread.join(1)
                        # The other thread still owns the lock. The waiter
                        # must finish without waiting for that owner or encode.
                        self.assertFalse(thread.is_alive())
                        self.assertEqual(tokens.calls, 0)
                        self.assertEqual(len(errors), 1)
                        self.assertIsInstance(errors[0], Cancelled if cancel else RequestDeadline)
                finally:
                    thread.join(1)
                self.assertEqual(tokens.prepare(model, [], 16, tools_enabled=False)['context']['input_tokens'], 10)
                self.assertEqual(tokens.calls, 1)


if __name__ == '__main__':
    unittest.main()
