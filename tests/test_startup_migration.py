"""Large legacy history must not be read/re-written as a startup side effect."""
from contextlib import redirect_stdout
import builtins
import io
import json
from pathlib import Path
import runpy
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from lazyrig.storage import Store


class StartupMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'state.sqlite3'

    def legacy(self, records, partial=False):
        with sqlite3.connect(self.path) as db:
            db.executescript('''
                CREATE TABLE jobs(id TEXT PRIMARY KEY,body TEXT);
                CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,body TEXT);
                CREATE INDEX events_job_seq ON events(job,seq);
                CREATE TABLE image_request_keys(key_hash TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,job TEXT NOT NULL);
            ''')
            job = {'id': 'job', 'status': 'completed', 'prompt': 'private prompt',
                   'created': 1, 'updated': 2, 'request': {}}
            db.execute('INSERT INTO jobs VALUES(?,?)', ('job', json.dumps(job)))
            db.execute('INSERT INTO image_request_keys VALUES(?,?,?)', ('key', 'fingerprint', 'job'))
            for record in records:
                db.execute('INSERT INTO events(job,body) VALUES(?,?)', ('job', json.dumps(record)))
            if partial:
                # The original beta committed ALTER TABLE before its expensive
                # backfill. A stopped upgrade can have columns but no metadata.
                db.execute('ALTER TABLE events ADD COLUMN encoding INTEGER NOT NULL DEFAULT 1')
                db.execute('ALTER TABLE events ADD COLUMN checkpoint INTEGER NOT NULL DEFAULT 0')
                db.execute("ALTER TABLE events ADD COLUMN kind TEXT NOT NULL DEFAULT ''")
                db.execute('ALTER TABLE jobs ADD COLUMN summary TEXT')

    @staticmethod
    def event(text, **extra):
        return {'kind': 'text', 'time': 1, 'text': text, 'content': text, 'reasoning_content': '', **extra}

    def open(self):
        store = Store(self.directory)
        self.addCleanup(store.close)
        return store

    def test_large_legacy_startup_cannot_read_update_or_index_event_payloads(self):
        # ~32 MiB of repeated legacy payload, with the actual v0.4 (job,seq) index.
        record = self.event('x' * (2 * 1024 * 1024))
        self.legacy([record] * 8)
        self.assertGreater(self.path.stat().st_size, 32 * 1024 * 1024)
        connect = sqlite3.connect
        queries = []

        def authorize(action, first, second, database, source):
            if action == sqlite3.SQLITE_READ and first == 'events' and second == 'body':
                return sqlite3.SQLITE_DENY
            if action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE) and first == 'events':
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_CREATE_INDEX and second == 'events':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def guarded(*args, **kwargs):
            db = connect(*args, **kwargs)
            db.set_authorizer(authorize)
            db.set_trace_callback(queries.append)
            return db

        with patch('lazyrig.storage.sqlite3.connect', side_effect=guarded), \
             self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            store = self.open()
        # Job summaries and schema marker only: no per-event writes.
        self.assertLessEqual(store.db.total_changes, 3)
        self.assertFalse(any('json_extract' in query for query in queries))
        self.assertTrue(any('database ready' in message for message in logs.output))
        self.assertFalse(any('private prompt' in message for message in logs.output))
        store.db.set_authorizer(None)
        store.db.set_trace_callback(None)
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM events WHERE kind='' AND checkpoint=0").fetchone()[0], 8)
        store.event('job', 'text', **{k: v for k, v in self.event('new output').items() if k not in ('kind', 'time')})
        self.assertEqual(store.events('job', 8)[0]['text'], 'new output')
        self.assertEqual(store.inference_request(('key', 'fingerprint'))['id'], 'job')

    def test_interrupted_column_migration_and_old_beta_delta_replay(self):
        self.legacy([self.event('before')], partial=True)
        # An original-beta delta may depend on a legacy row that has no kind tag.
        with sqlite3.connect(self.path) as db:
            delta = {'item': {'kind': 'text', 'time': 2},
                     'fields': {'text': [6, ' after'], 'content': [6, ' after'], 'reasoning_content': [0, '']}}
            db.execute("INSERT INTO events(job,body,encoding,checkpoint,kind) VALUES(?,?,2,0,'text')",
                       ('job', json.dumps(delta)))
        store = self.open()
        self.assertEqual(store.events('job', 1)[0]['text'], 'before after')
        store.close()
        store = self.open()
        self.assertEqual(store.events('job', 1)[0]['text'], 'before after')
        self.assertEqual(store.jobs(summary=True)[0]['status'], 'completed')

    def test_partial_legacy_channels_mix_with_existing_compact_rows(self):
        self.legacy([self.event('a', reasoning_content='reason'),
                     {'kind': 'text', 'time': 2, 'text': 'ab'}], partial=True)
        with sqlite3.connect(self.path) as db:
            delta = {'item': {'kind': 'text', 'time': 3}, 'fields': {
                'text': [2, 'c'], 'content': [1, 'bc'], 'reasoning_content': [6, 'ing']}}
            db.execute("INSERT INTO events(job,body,encoding,checkpoint,kind) VALUES(?,?,2,0,'text')",
                       ('job', json.dumps(delta)))
        store = self.open()
        event = store.events('job', 2)[0]
        self.assertEqual((event['text'], event['content'], event['reasoning_content']), ('abc', 'abc', 'reasoning'))

    def test_legacy_cursor_reads_only_the_requested_page(self):
        self.legacy([self.event(str(i)) for i in range(700)])
        store = self.open()
        with patch.object(store, '_state_at', side_effect=AssertionError('Legacy replay must not rescan earlier history')):
            events = store.events('job', 500)
        self.assertEqual(len(events), 200)
        self.assertEqual(events[0]['text'], '500')
        self.assertEqual(events[-1]['text'], '699')

    def test_explicit_compaction_still_handles_unmodified_legacy_rows(self):
        original = [{'kind': 'queued', 'time': 0}, self.event('a'), self.event('ab'),
                    {'kind': 'artifact', 'time': 3, 'artifact': {'url': '/artifacts/saved.png'}},
                    self.event('abc'), {'kind': 'finished', 'time': 4, 'status': 'completed'}]
        self.legacy(original, partial=True)
        store = self.open()
        self.assertEqual(store.compact_events(3), {'jobs': 1, 'text_events_removed': 2})
        expected = [{'seq': i + 1, **event} for i, event in enumerate(original) if i not in (1, 2)]
        self.assertEqual(store.events('job'), expected)
        self.assertEqual(store.compact_events(3)['jobs'], 0)
        store.close()
        self.assertEqual(self.open().events('job'), expected)


class StartupVisibilityTests(unittest.TestCase):
    def test_banner_is_flushed_before_server_import(self):
        class Output(io.StringIO):
            flushed = ''
            def flush(self):
                self.flushed = self.getvalue()

        output = Output()
        original_import = builtins.__import__

        class ImportReached(Exception):
            pass

        def importing(name, *args, **kwargs):
            if name == 'server':
                self.assertIn('starting; loading application modules', output.flushed)
                raise ImportReached()
            return original_import(name, *args, **kwargs)

        with redirect_stdout(output), patch('builtins.__import__', side_effect=importing):
            with self.assertRaises(ImportReached):
                runpy.run_module('lazyrig', run_name='__main__')


if __name__ == '__main__':
    unittest.main()
