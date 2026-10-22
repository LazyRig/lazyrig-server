import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from lazyrig.placement_report import read_report


class PlacementReportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name, 'state.sqlite3')
        self.jid = '71c8ad05' + '0' * 24
        self.preflight = {'runtime_version': '2026.4.1', 'calibration': {'attempts': [
            {'ratios': {'GPU.1': .5}, 'measured_weight_shares': {'GPU.1': 88.2, 'CPU': 11.8}}]}}
        self.job = {'id': self.jid, 'model': 'test', 'status': 'failed', 'error_code': 'manual_split_rejected',
                    'prompt': 'private prompt', 'request': {'secret': 'private secret'},
                    'placement_events': [{'kind': 'placement_rejected', 'target': 'HETERO:GPU.1,CPU',
                                          'preflight': self.preflight, 'placement': {'path': '/private/path'}}]}
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,body TEXT)')
            db.execute('INSERT INTO jobs VALUES(?,?)', (self.jid, json.dumps(self.job)))

    def test_existing_report_is_read_without_changes_or_private_request_data(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = read_report(self.path, '71c8ad05')
        self.assertEqual(report['placement_reports'][0]['preflight'], self.preflight)
        self.assertEqual(report['job_id'], self.jid)
        self.assertNotIn('private', json.dumps(report))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        self.assertEqual(read_report(self.path, self.jid), report)

    def test_missing_and_ambiguous_jobs_fail_without_selecting_another_job(self):
        with self.assertRaisesRegex(ValueError, 'No saved job'):
            read_report(self.path, 'ffffffff')
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT INTO jobs VALUES(?,?)', ('71c8ad05' + 'f' * 24, '{}'))
        with self.assertRaisesRegex(ValueError, 'Several jobs'):
            read_report(self.path, '71c8ad05')

    def test_bounded_reader_and_invalid_prefixes(self):
        with patch('lazyrig.placement_report.MAX_JOB_BYTES', 20), self.assertRaisesRegex(ValueError, 'too large'):
            read_report(self.path, '71c8ad05')
        for prefix in ('', '71c8', '71c8ad05\n', "71c8ad05' OR 1=1"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                read_report(self.path, prefix)

    def test_missing_database_is_not_created(self):
        missing = self.path.with_name('missing.sqlite3')
        with self.assertRaises(FileNotFoundError):
            read_report(missing, '71c8ad05')
        self.assertFalse(missing.exists())

    def test_job_without_preflight_is_reported_explicitly(self):
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE jobs SET body=?', (json.dumps({'id': self.jid}),))
        with self.assertRaisesRegex(ValueError, 'no saved manual-split'):
            read_report(self.path, '71c8ad05')
