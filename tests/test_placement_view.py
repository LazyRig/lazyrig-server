"""Manual placement views share weight files and survive until worker shutdown."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lazyrig.placement_view import prepare_manual_view
from lazyrig.worker import Cancelled, NativeWorkerError, ProcessWorker, WorkerCleanupError


class PlacementViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.source = Path(self.directory) / 'model'
        self.source.mkdir()
        self.xml = b'<net name="original"/>'
        self.normalized = b'<net name="normalized"/>'
        self.report = {'method': 'assign_dependency_depth', 'assign_count': 2, 'reordered': True}
        (self.source / 'openvino_language_model.xml').write_bytes(self.xml)
        (self.source / 'openvino_language_model.bin').write_bytes(bytes(range(64)))
        (self.source / 'generation_config.json').write_text('{"eos_token_id": 42}')
        (self.source / 'openvino_text_embeddings_model.xml').write_bytes(b'<net name="embeddings"/>')
        (self.source / 'cache').mkdir()
        self.model = {'pipeline': 'vlm', 'path': str(self.source), 'device': 'GPU.1',
                      'placement': {'mode': 'manual'},
                      '_placement': {'devices': ['GPU.1', 'CPU']}}
        self.normalize = Mock(return_value=(self.normalized, self.report))
        self.tag_xml = self.enterContext(patch('lazyrig.manual_affinity.prepare_xml', side_effect=lambda xml: xml))
        self.enterContext(patch.dict('sys.modules', {
            'lazyrig.state_order': SimpleNamespace(MAX_XML_BYTES=1024,
                                                 normalize_state_sink_order=self.normalize)}))

    def views(self):
        return list(Path(self.directory).glob('.lazyrig-placement-*'))

    def test_view_changes_only_language_xml_and_shares_original_weights(self):
        original = copy.deepcopy(self.model)
        runtime, view = prepare_manual_view(self.model)
        try:
            self.assertEqual(self.model, original)
            self.assertEqual(runtime['_state_order'], self.report)
            self.assertEqual(Path(runtime['path']), view.path)
            self.assertEqual(view.path.parent, self.source.parent)
            self.assertEqual((view.path / 'openvino_language_model.xml').read_bytes(), self.normalized)
            self.assertEqual((self.source / 'openvino_language_model.xml').read_bytes(), self.xml)
            self.assertFalse(os.path.samefile(self.source / 'openvino_language_model.xml',
                                             view.path / 'openvino_language_model.xml'))
            for name in ('openvino_language_model.bin', 'generation_config.json',
                         'openvino_text_embeddings_model.xml'):
                self.assertTrue(os.path.samefile(self.source / name, view.path / name))
            self.assertFalse((view.path / 'cache').exists())
        finally:
            view.close()
        self.assertFalse(self.views())
        self.assertEqual((self.source / 'openvino_language_model.bin').read_bytes(), bytes(range(64)))

    def test_ineligible_models_never_read_or_normalize_an_export(self):
        for change in ({'placement': {'mode': 'auto'}}, {'pipeline': 'image'},
                       {'_placement': {'devices': ['GPU.1']}}):
            with self.subTest(change=change):
                model = {**self.model, **change, 'path': '/does-not-exist'}
                result, view = prepare_manual_view(model)
                self.assertIs(result, model)
                self.assertIsNone(view)
        self.normalize.assert_not_called()

    def test_unchanged_order_still_creates_owned_affinity_view(self):
        self.normalize.return_value = self.xml, {}
        result, view = prepare_manual_view(self.model)
        self.assertTrue(result['_manual_affinity'])
        self.assertEqual((view.path / 'openvino_language_model.xml').read_bytes(), self.xml)
        view.close()
        self.assertFalse(self.views())

    def test_oversized_language_xml_is_rejected_before_read_or_normalization(self):
        (self.source / 'openvino_language_model.xml').write_bytes(b'x' * 1025)
        with patch.object(Path, 'open', side_effect=AssertionError('oversized XML must not be read')), \
             self.assertRaisesRegex(ValueError, 'size limit'):
            prepare_manual_view(self.model)
        self.normalize.assert_not_called()
        self.assertFalse(self.views())

    def test_llm_uses_its_own_language_filename(self):
        (self.source / 'openvino_model.xml').write_bytes(b'<net name="llm"/>')
        (self.source / 'openvino_model.bin').write_bytes(b'llm weights')
        result, view = prepare_manual_view({**self.model, 'pipeline': 'llm'})
        try:
            self.normalize.assert_called_once_with(b'<net name="llm"/>')
            self.assertEqual((view.path / 'openvino_model.xml').read_bytes(), self.normalized)
            self.assertTrue(os.path.samefile(self.source / 'openvino_model.bin', view.path / 'openvino_model.bin'))
        finally:
            view.close()

    def test_snapshot_file_symlink_becomes_a_hardlink_to_its_target(self):
        target = Path(self.directory) / 'weights-blob'
        target.write_bytes(b'unchanged blob')
        link = self.source / 'extra.bin'
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest('File symlinks unavailable on this host')
        _, view = prepare_manual_view(self.model)
        try:
            self.assertFalse((view.path / 'extra.bin').is_symlink())
            self.assertTrue(os.path.samefile(target, view.path / 'extra.bin'))
        finally:
            view.close()

    def test_failed_hardlink_removes_partial_view_without_copying(self):
        with patch('lazyrig.placement_view.os.link', side_effect=OSError('unsupported filesystem')), \
             patch('shutil.copyfile') as copy_file, \
             self.assertRaisesRegex(RuntimeError, 'weights will not be copied'):
            prepare_manual_view(self.model)
        copy_file.assert_not_called()
        self.assertFalse(self.views())
        self.assertEqual((self.source / 'openvino_language_model.xml').read_bytes(), self.xml)

    def test_cancellation_during_preparation_removes_partial_view(self):
        def checkpoint():
            if self.views():
                raise Cancelled('cancelled during preparation')
        with self.assertRaises(Cancelled):
            prepare_manual_view(self.model, checkpoint)
        self.assertFalse(self.views())

    def worker_environment(self, event=None, cannot_stop=False):
        stack = self.enterContext(ExitStack())
        state = {'alive': True, 'cannot_stop': cannot_stop}
        process = Mock(pid=12345)
        process.is_alive.side_effect = lambda: state['alive']
        def stop():
            if not state['cannot_stop']:
                state['alive'] = False
        process.terminate.side_effect = stop
        process.kill.side_effect = stop
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.return_value = event or {'kind': 'ready', 'pid': process.pid}
        context = Mock()
        context.Pipe.return_value = connection, Mock()
        context.Process.return_value = process
        guard = Mock(allocation_limit_bytes=None)
        guard.diagnostics.return_value = {'minimum_available_gib': 32}
        stack.enter_context(patch('lazyrig.worker.LoadMemoryGuard', return_value=guard))
        stack.enter_context(patch('lazyrig.worker.mp.get_context', return_value=context))
        stack.enter_context(patch('lazyrig.worker.os.killpg'))
        # This suite exercises portable parent lifecycle logic. Windows Job
        # membership and OS termination are covered by their dedicated tests.
        stack.enter_context(patch('lazyrig.worker.os.name', 'posix'))
        return process, context, state

    def test_parent_passes_view_to_worker_and_removes_it_after_handle_release(self):
        process, context, _ = self.worker_environment()
        worker = ProcessWorker('openvino', self.model, 5, threading.Event())
        runtime = context.Process.call_args.kwargs['args'][2]
        view = Path(runtime['path'])
        self.assertTrue(view.is_dir())
        self.assertEqual(runtime['_state_order'], self.report)
        self.assertEqual(self.model['path'], str(self.source))
        process.close.side_effect = lambda: self.assertTrue(view.is_dir())
        worker.close()
        self.assertFalse(view.exists())
        worker.close()
        process.close.assert_called_once()

    def test_native_load_failure_cleans_prepared_view(self):
        process, _, _ = self.worker_environment({'kind': 'error', 'error': 'native load rejected'})
        with self.assertRaisesRegex(NativeWorkerError, 'native load rejected'):
            ProcessWorker('openvino', self.model, 5, threading.Event())
        process.close.assert_called_once()
        self.assertFalse(self.views())

    def test_cancellation_before_spawn_creates_no_view_or_worker(self):
        _, context, _ = self.worker_environment()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            ProcessWorker('openvino', self.model, 5, cancelled)
        self.normalize.assert_not_called()
        context.Process.assert_not_called()
        self.assertFalse(self.views())

    def test_preparation_failure_retains_cleanup_ownership_for_parent_retry(self):
        self.worker_environment()
        from lazyrig.placement_view import PlacementView
        cleanup = PlacementView.close
        calls = []
        def fail_once(view):
            calls.append(view.path)
            if len(calls) == 1:
                raise OSError('temporary cleanup lock')
            cleanup(view)
        with patch('lazyrig.placement_view.os.link', side_effect=OSError('hardlink unsupported')), \
             patch.object(PlacementView, 'close', fail_once), \
             self.assertRaisesRegex(RuntimeError, 'weights will not be copied'):
            ProcessWorker('openvino', self.model, 5, threading.Event())
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.views())

    def test_unconfirmed_termination_keeps_view_until_successful_retry(self):
        _, _, state = self.worker_environment({'kind': 'error', 'error': 'native load rejected'}, cannot_stop=True)
        with self.assertRaises(WorkerCleanupError) as caught:
            ProcessWorker('openvino', self.model, 5, threading.Event())
        worker = caught.exception.worker
        view = worker._placement_view.path
        self.assertTrue(view.is_dir())
        state['cannot_stop'] = False
        worker.close()
        self.assertFalse(view.exists())

    def test_view_cleanup_retry_does_not_repeat_native_shutdown(self):
        process, _, _ = self.worker_environment()
        worker = ProcessWorker('openvino', self.model, 5, threading.Event())
        view = worker._placement_view
        with patch.object(view, 'close', side_effect=OSError('file temporarily locked')):
            with self.assertRaisesRegex(OSError, 'temporarily locked'):
                worker.close()
        self.assertTrue(worker._closed)
        self.assertTrue(view.path.exists())
        worker.close()
        self.assertFalse(view.path.exists())
        process.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
