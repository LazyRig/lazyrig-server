"""Generation writes stay small without sacrificing live reads or crash recovery."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from lazyrig.compat import start_chat
from lazyrig.storage import Store
from lazyrig.worker import Cancelled
from support import DEFAULT, FixtureWorker, Service


def running_job(jid='job'):
    return {'id': jid, 'status': 'running', 'phase': 'Generating response',
            'created': 1, 'updated': 2, 'prompt': 'p' * 100_000, 'model': 'fixture',
            'request': {'max_tokens': 4000}, 'text': '', 'content': '',
            'reasoning_content': '', 'thinking_open': False, 'reasoning_incomplete': False,
            'context': {'generated_tokens': 0}}


def text_update(store, job, text, count=1):
    job.update(text=text, content=text, reasoning_content='', updated=count + 2,
               context={'generated_tokens': count})
    store.event(job['id'], 'text', job_update=job, text=text, content=text,
                reasoning_content='', context=job['context'], final=False)


class StreamStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def store(self):
        store = Store(self.path)
        self.addCleanup(store.close)
        return store

    def test_streaming_never_writes_full_jobs_and_live_reads_stay_current(self):
        store = self.store()
        job = running_job()
        store.save_job(job)

        def forbid_snapshots(action, table, *args):
            return sqlite3.SQLITE_DENY if table == 'jobs' and action in {
                sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE} else sqlite3.SQLITE_OK

        store.db.set_authorizer(forbid_snapshots)
        for n in range(1, 97):
            text_update(store, job, 'chunk ' * n, n)
        current = store.job('job')
        self.assertEqual(current, job)
        self.assertEqual(store.jobs()[0], job)
        summary = store.jobs(summary=True)[0]
        self.assertEqual(summary['context']['generated_tokens'], 96)
        self.assertEqual(summary['updated'], job['updated'])
        self.assertNotIn('text', summary)
        self.assertEqual(json.loads(store.db.execute('SELECT body FROM jobs').fetchone()[0])['text'], '')
        small = store.db.execute('SELECT body FROM job_streams').fetchone()[0]
        self.assertLess(len(small), 1000)
        self.assertNotIn('prompt', json.loads(small))
        self.assertNotIn('text', json.loads(small))
        self.assertEqual(store.events('job')[-1]['text'], current['text'])
        store.db.set_authorizer(None)
        current.update(status='completed', phase='Complete')
        store.save_job(current)
        self.assertEqual(store.db.execute('SELECT COUNT(*) FROM job_streams').fetchone()[0], 0)
        self.assertEqual(store.job('job'), current)

    def test_failed_progress_write_rolls_back_event_and_cached_prefix(self):
        store = self.store()
        job = running_job()
        store.save_job(job)
        text_update(store, job, 'before')
        previous = store.job('job')
        replay = store.events('job')

        def deny_progress(action, table, *args):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == 'job_streams' else sqlite3.SQLITE_OK

        store.db.set_authorizer(deny_progress)
        with self.assertRaises(sqlite3.DatabaseError):
            text_update(store, job, 'before failed write', 2)
        store.db.set_authorizer(None)
        self.assertEqual(store.job('job'), previous)
        self.assertEqual(store.events('job'), replay)
        self.assertFalse(store.health()['ok'])
        self.assertTrue(store.health(probe=True)['ok'])
        text_update(store, job, 'before successful retry', 3)
        self.assertEqual(store.job('job')['text'], 'before successful retry')
        self.assertEqual(store.events('job')[-1]['text'], 'before successful retry')

    def test_failed_full_snapshot_keeps_the_committed_projection(self):
        store = self.store()
        job = running_job()
        store.save_job(job)
        text_update(store, job, 'committed output')
        prior = store.job('job')

        def deny_clear(action, table, *args):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DELETE and table == 'job_streams' else sqlite3.SQLITE_OK

        store.db.set_authorizer(deny_clear)
        with self.assertRaises(sqlite3.DatabaseError):
            store.save_job({**job, 'status': 'completed'})
        store.db.set_authorizer(None)
        self.assertEqual(store.job('job'), prior)
        self.assertEqual(json.loads(store.db.execute('SELECT body FROM jobs').fetchone()[0])['status'], 'running')

    def test_phase_reset_and_image_progress_do_not_restore_an_older_text_pass(self):
        store = self.store()
        job = running_job()
        store.save_job(job)
        text_update(store, job, 'first tool pass')
        job.update(text='', content='', reasoning_content='', phase='Generating image')
        store.save_job(job)
        job.update(progress={'step': 1, 'steps': 40}, updated=10)
        store.event('job', 'progress', job_update=job, **job['progress'])
        self.assertEqual(store.job('job')['text'], '')
        self.assertEqual(store.job('job')['progress'], job['progress'])
        job['phase'] = 'Generating final response'
        text_update(store, job, 'continuation', 11)
        job['updated'] = 12
        store.event('job', 'enhancement_progress', job_update=job, generated_tokens=2)
        self.assertEqual(store.job('job')['text'], 'continuation')
        self.assertEqual(store.job('job')['updated'], 12)
        self.assertEqual(store.events('job')[-2]['text'], 'continuation')

    def test_interleaved_jobs_reconstruct_after_writer_cache_eviction(self):
        store = self.store()
        jobs = [running_job(str(n)) for n in range(36)]
        for job in jobs:
            job['prompt'] = 'small fixture'
            store.save_job(job)
        for n in range(1, 4):
            for job in jobs:
                text_update(store, job, job['id'] + ': ' + 'text ' * n, n)
        self.assertNotIn('0', store._write_heads)
        for job in jobs:
            self.assertEqual(store.job(job['id']), job)
        self.assertEqual(len(store.jobs(summary=True)), len(jobs))

    def test_streaming_jobs_remain_visible_after_200_newer_jobs(self):
        store = self.store()
        job = running_job()
        job.update(kind='image', prompt='live job')
        store.save_job(job)
        for n in range(205):
            store.save_job({**job, 'id': str(n), 'status': 'completed', 'updated': n + 3})
        job.update(progress={'step': 10, 'steps': 40}, updated=1000)
        store.event('job', 'progress', job_update=job, **job['progress'])
        for listing in (store.jobs(), store.jobs(summary=True), store.image_jobs()):
            self.assertEqual(len(listing), 200)
            self.assertEqual(listing[0]['id'], 'job')
            self.assertEqual(listing[0]['progress'], job['progress'])
            self.assertEqual(len({item['id'] for item in listing}), 200)

    def test_abrupt_process_exit_recovers_latest_committed_text_and_cancellation(self):
        script = '''import os, sys
from lazyrig.storage import Store
store = Store(sys.argv[1])
job = {'id':'job','status':'running','phase':'Generating response','created':1,'updated':2,
       'prompt':'fixture','request':{},'text':'','content':'','reasoning_content':''}
store.save_job(job)
for n in range(1, 65):
    text = 'saved fragment ' * n
    job.update(text=text, content=text, updated=n+2, context={'generated_tokens':n})
    store.event('job','text',job_update=job,text=text,content=text,reasoning_content='')
    if n == 32 and sys.argv[2] == 'cancel':
        job['cancel_requested'] = True
        store.save_job(job)
os._exit(0)
'''
        for requested in (False, True):
            with self.subTest(cancelled=requested):
                directory = self.path / str(requested)
                result = subprocess.run([sys.executable, '-c', script, str(directory),
                                         'cancel' if requested else 'running'],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((directory / 'state.sqlite3-wal').exists())
                store = Store(directory)
                self.addCleanup(store.close)
                restored = store.job('job')
                self.assertEqual(restored['status'], 'cancelled' if requested else 'interrupted')
                self.assertEqual(restored['text'], 'saved fragment ' * 64)
                self.assertEqual(restored['content'], restored['text'])
                self.assertEqual(restored['context']['generated_tokens'], 64)
                self.assertEqual(len(store.events('job')), 64)
                self.assertEqual(store.db.execute('SELECT COUNT(*) FROM job_streams').fetchone()[0], 0)


class StreamServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = copy.deepcopy(DEFAULT)
        config.update(data_dir=self.temp.name, terminal_metrics=False)
        self.service = Service(config, restore=False)
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        self.service.close()
        self.temp.cleanup()

    def wait_terminal(self, jid):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.service.job(jid)
            if job['status'] in {'completed', 'failed', 'cancelled'}:
                return job
            time.sleep(.01)
        self.fail('Generation did not become terminal')

    def generation(self, outcome):
        entered = threading.Event()
        counts = {'snapshots': 0, 'commits': 0}
        output = 'response fragment ' * 512

        def trace(statement):
            if statement.startswith('INSERT OR REPLACE INTO jobs'):
                counts['snapshots'] += 1
            if statement == 'COMMIT':
                counts['commits'] += 1

        def factory(*args):
            worker = FixtureWorker(*args)

            def run(request, timeout, cancelled):
                self.service.store.db.set_trace_callback(trace)
                for n in range(1, 513):
                    yield {'kind': 'text', 'text': 'response fragment ' * n,
                           'generated_tokens': n, 'accuracy': 'estimated-test'}
                entered.set()
                while not self.release.wait(.01):
                    if cancelled.is_set():
                        raise Cancelled()
                if outcome == 'failed':
                    raise RuntimeError('Injected generation failure')
                yield {'kind': 'result', 'text': output,
                       'generated_tokens': 512, 'accuracy': 'estimated-test'}

            worker.run = run
            return worker

        self.service.scheduler.factory = factory
        receipt = start_chat(self.service, {'model': 'qwen-chat', 'max_tokens': 1024,
                            'messages': [{'role': 'user', 'content': 'Generate text.'}]})
        self.assertTrue(entered.wait(5))
        live = self.service.job(receipt['id'])
        self.assertEqual(live['status'], 'running')
        self.assertEqual(live['text'], output)
        self.assertEqual(live['context']['generated_tokens'], 512)
        self.assertEqual(self.service.state(summary=True)['jobs'][0]['context']['generated_tokens'], 512)
        self.assertEqual(counts, {'snapshots': 0, 'commits': 512})
        self.service.store.db.set_trace_callback(None)
        if outcome == 'cancelled':
            self.service.cancel(receipt['id'])
        else:
            self.release.set()
        terminal = self.wait_terminal(receipt['id'])
        self.assertEqual(terminal['status'], outcome)
        self.assertEqual(terminal['text'], output)
        self.assertEqual(self.service.store.db.execute('SELECT COUNT(*) FROM job_streams').fetchone()[0], 0)
        self.assertFalse(self.service.running)
        self.assertEqual(self.service.store.job(receipt['id']), terminal)
        if outcome == 'completed':
            self.assertEqual(terminal['usage']['completion_tokens'], 512)

    def test_completion_keeps_exact_output_with_one_small_commit_per_update(self):
        self.generation('completed')

    def test_cancellation_keeps_latest_text_without_per_chunk_snapshots(self):
        self.generation('cancelled')

    def test_worker_failure_keeps_latest_text_without_per_chunk_snapshots(self):
        self.generation('failed')

    def test_image_progress_remains_live_without_rewriting_full_jobs(self):
        job = self.service._new_job(None, 'qwen-image', 'image fixture', {'width': 512, 'height': 512})
        job['kind'] = 'image'
        entered = threading.Event()

        def operation(cancel, emit):
            job['phase'] = 'Generating image'
            emit('image_started', model='qwen-image')
            for n in range(1, 41):
                emit('progress', step=n, steps=40)
            entered.set()
            self.release.wait(5)

        receipt = self.service._launch(job, operation)
        self.assertTrue(entered.wait(5))
        live = self.service.job(receipt['id'])
        self.assertEqual(live['progress'], {'step': 40, 'steps': 40})
        self.assertEqual(self.service.store.image_jobs()[0]['progress'], live['progress'])
        self.assertEqual(self.service.state(summary=True)['jobs'][0]['progress'], live['progress'])
        raw = json.loads(self.service.store.db.execute('SELECT body FROM jobs WHERE id=?', (job['id'],)).fetchone()[0])
        self.assertNotIn('progress', raw)
        self.release.set()
        self.assertEqual(self.wait_terminal(job['id'])['progress'], live['progress'])


if __name__ == '__main__':
    unittest.main()
