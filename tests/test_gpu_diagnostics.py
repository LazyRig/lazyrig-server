"""Compiler failure reporting and bounded diagnostics; no RTX hardware simulated as validated."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyrig import gpu_diagnostics as diagnostic
from lazyrig.gpu_errors import kernel_build_failure
from lazyrig.placement import retryable_load_error, load_failure_reason
from lazyrig.service import _record_failure
from lazyrig.terminal import Reporter
from lazyrig.worker import _child, NativeWorkerError, ProcessWorker


ERROR = ('Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_kernel_builder.hpp:73: '
         '[GPU] clBuildProgram, error code: -11 CL_BUILD_PROGRAM_FAILURE')


def _gpu_error_child(conn, mode, model, job_name):
    if os.name != 'nt':
        os.setsid()
    def fail(*args, **kwargs):
        raise RuntimeError(ERROR)
    genai = SimpleNamespace(LLMPipeline=fail, VLMPipeline=fail, Text2ImagePipeline=fail)
    with patch.dict(sys.modules, {'openvino': SimpleNamespace(), 'openvino_genai': genai}), \
         patch('lazyrig.worker._pipeline_setup', return_value=({}, 'GPU.0', {})):
        _child(conn, 'openvino', model)


class KernelFailureTests(unittest.TestCase):
    def test_failure_survives_real_worker_ipc_and_reports_without_claiming_out_of_memory(self):
        with patch('lazyrig.worker._spawn_child', _gpu_error_child), \
             self.assertRaises(NativeWorkerError) as caught:
            ProcessWorker('fixture', {'pipeline': 'llm', 'path': '/fixture'}, 10, threading.Event())
        error = caught.exception
        self.assertEqual(error.error_code, 'gpu_kernel_build_failed')
        self.assertIn(ERROR, str(error))
        self.assertIn('does not establish a VRAM shortage', str(error))
        self.assertIn('diagnose-gpu.bat', str(error))
        self.assertTrue(retryable_load_error(error))
        self.assertEqual(load_failure_reason(error), 'gpu_kernel_build_failed')
        job = {'id': 'a' * 32, 'model': 'test'}
        _record_failure(job, error)
        self.assertEqual(job['phase'], 'GPU kernel compilation failed')
        with self.assertLogs('lazyrig.runtime', level='INFO') as logs:
            Reporter({'terminal_metrics': True}, job).finish('failed')
        self.assertIn('gpu_kernel_build_failed', '\n'.join(logs.output))
        self.assertNotIn('ocl_kernel_builder.hpp', '\n'.join(logs.output))

    def test_allocation_cancel_cache_and_unrelated_compilers_are_not_reclassified(self):
        for error in ('[GPU] CL_OUT_OF_RESOURCES', '[GPU] CL_MEM_OBJECT_ALLOCATION_FAILURE',
                      'Requested linear attention cache allocation exceeds the configured cache size.',
                      'CL_BUILD_PROGRAM_FAILURE', '[GPU] CL_BUILD_PROGRAM_FAILURE_OTHER', 'Cancelled'):
            self.assertEqual(kernel_build_failure(RuntimeError(error)), {})


class DriverProbeTests(unittest.TestCase):
    def compiler(self, build_status=-11, program_status=0):
        cl = diagnostic.OpenCL.__new__(diagnostic.OpenCL)
        self.released = []
        self.sources = []
        def context(props, count, devices, callback, userdata, error):
            ct.cast(error, ct.POINTER(ct.c_int32))[0] = 0
            return 123
        def program(context, count, strings, sizes, error):
            self.sources.append(strings[0].decode())
            ct.cast(error, ct.POINTER(ct.c_int32))[0] = program_status
            return 456 if program_status == 0 else None
        def log(program, device, key, capacity, output, size):
            value = b'fixture compiler detail: half is unavailable\0'
            if output is None:
                ct.cast(size, ct.POINTER(ct.c_size_t))[0] = len(value)
            else:
                ct.memmove(output, value, len(value))
            return 0
        cl.clCreateContext, cl.clCreateProgramWithSource = context, program
        cl.clBuildProgram = lambda *args: build_status
        cl.clGetProgramBuildInfo = log
        cl.clReleaseProgram = lambda p: self.released.append(('program', p))
        cl.clReleaseContext = lambda p: self.released.append(('context', p))
        return cl

    def test_direct_compiler_log_and_resources_survive_failed_build(self):
        result = self.compiler().compile_probe(12, 'fp16')
        self.assertEqual(result['error_code'], -11)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('half is unavailable', result['build_log'])
        self.assertIn('cl_khr_fp16', self.sources[0])
        self.assertEqual(self.released, [('program', 456), ('context', 123)])

    def test_success_is_labelled_compiled_not_model_supported(self):
        result = self.compiler(0).compile_probe(12, 'fp32')
        self.assertEqual(result['status'], 'compiled')
        self.assertNotIn('supported', result)
        self.assertNotIn('cl_khr_fp16', self.sources[0])
        self.assertEqual(len(self.released), 2)

    def test_program_creation_failure_still_releases_context(self):
        with self.assertRaisesRegex(RuntimeError, '-6'):
            self.compiler(program_status=-6).compile_probe(12, 'fp16')
        self.assertEqual(self.released, [('context', 123)])

    def test_build_log_is_bounded_and_marked_truncated(self):
        cl = self.compiler()
        result = cl.text(cl.clGetProgramBuildInfo, 456, 12, 0x1183, limit=8)
        self.assertEqual(result, 'fixture \n[truncated]')

    def test_driver_timeout_is_reported_and_child_is_reaped(self):
        process = SimpleNamespace(returncode=-9)
        calls = []
        def wait(timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired('fixture', timeout)
        process.wait, process.kill = wait, lambda: calls.append('kill')
        with patch.object(diagnostic.subprocess, 'Popen', return_value=process):
            self.assertEqual(diagnostic.run_probe('opencl', 1), {'status': 'timed_out', 'timeout_seconds': 1})
        self.assertEqual(calls, [1, 'kill', None])

    def test_report_retains_failed_check_and_explains_scope(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
                diagnostic, 'run_probe', side_effect=[{'status': 'ok', 'data': {'devices': []}},
                                                     {'status': 'failed', 'error': 'OpenCL missing'}]):
            path = Path(folder) / 'report.json'
            self.assertEqual(diagnostic.main(['--output', str(path)]), 0)
            report = json.loads(path.read_text())
            self.assertEqual(report['opencl']['error'], 'OpenCL missing')
            self.assertIn('No model inference', report['scope'])
            self.assertIn('CUDA', ' '.join(report['interpretation']))


if __name__ == '__main__':
    unittest.main()
