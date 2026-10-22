"""Read runtime/driver metadata and compile tiny OpenCL kernels in timed children.

No models, saved settings, chats, weights, network access or extra dependencies.
The independent OpenCL inventory must not be indexed as OpenVINO GPU.N devices.
Passing these compilation probes does not validate inference or model support.
"""
import argparse
import ctypes as ct
import ctypes.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


MARKER = 'LAZYRIG_GPU_DIAGNOSTICS='
MAX_LOG = 16384
MAX_DEVICES = 8
FEATURES = ('cl_khr_fp16', 'cl_khr_fp64', 'cl_khr_subgroups', 'cl_khr_il_program',
            'cl_intel_subgroups', 'cl_intel_subgroups_short', 'cl_intel_subgroups_char',
            'cl_intel_required_subgroup_size', 'cl_intel_unified_shared_memory')


def runtime_report():
    import openvino as ov
    core = ov.Core()
    result = {'openvino_version': getattr(ov, '__version__', 'unknown'), 'devices': []}
    try:
        import openvino_genai as genai
        result['genai_version'] = getattr(genai, '__version__', 'unknown')
    except Exception as exc:
        result['genai_error'] = str(exc)[:4096]
    for device in list(core.available_devices)[:MAX_DEVICES]:
        item = {'id': device, 'properties': {}, 'unavailable_properties': []}
        for key in ('FULL_DEVICE_NAME', 'DEVICE_TYPE', 'OPTIMIZATION_CAPABILITIES',
                    'GPU_DEVICE_TOTAL_MEM_SIZE', 'GPU_UARCH_VERSION'):
            try:
                value = core.get_property(device, key)
                if isinstance(value, (list, tuple)):
                    value = [str(v)[:256] for v in value[:64]]
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    value = str(value)[:1024]
                item['properties'][key] = value
            except Exception:
                item['unavailable_properties'].append(key)
        try:
            item['plugins'] = {name: {'description': version.description, 'build': version.build_number}
                               for name, version in core.get_versions(device).items()}
        except Exception:
            item['plugins'] = None
        result['devices'].append(item)
    return result


class OpenCL:
    """The small OpenCL C API subset needed for discovery and compilation only."""
    def __init__(self):
        if os.name == 'nt':
            # System OpenCL loader; never load a DLL from the current directory.
            self.lib = ct.WinDLL('OpenCL.dll', winmode=0x800)
        else:
            name = ct.util.find_library('OpenCL')
            if not name:
                raise OSError('OpenCL loader is not installed')
            self.lib = ct.CDLL(name)
        ptr, uint, integer, size = ct.c_void_p, ct.c_uint32, ct.c_int32, ct.c_size_t
        pptr, puint, pint, psize = ct.POINTER(ptr), ct.POINTER(uint), ct.POINTER(integer), ct.POINTER(size)
        signatures = {
            'clGetPlatformIDs': (integer, [uint, pptr, puint]),
            'clGetPlatformInfo': (integer, [ptr, uint, size, ptr, psize]),
            'clGetDeviceIDs': (integer, [ptr, ct.c_uint64, uint, pptr, puint]),
            'clGetDeviceInfo': (integer, [ptr, uint, size, ptr, psize]),
            'clCreateContext': (ptr, [ptr, uint, pptr, ptr, ptr, pint]),
            'clReleaseContext': (integer, [ptr]),
            'clCreateProgramWithSource': (ptr, [ptr, uint, ct.POINTER(ct.c_char_p), psize, pint]),
            'clBuildProgram': (integer, [ptr, uint, pptr, ct.c_char_p, ptr, ptr]),
            'clGetProgramBuildInfo': (integer, [ptr, ptr, uint, size, ptr, psize]),
            'clReleaseProgram': (integer, [ptr]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, argtypes
            setattr(self, name, function)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'OpenCL error {code}')

    def text(self, function, *args, limit=MAX_LOG):
        length = ct.c_size_t()
        self.check(function(*args, 0, None, ct.byref(length)))
        if length.value == 0:
            return ''
        # OpenCL requires a buffer large enough for the complete property.
        if length.value > 1024 * 1024:
            return '[Property exceeds diagnostic size limit]'
        value = ct.create_string_buffer(length.value)
        self.check(function(*args, length, value, None))
        raw = value.value
        return raw[:limit].decode('utf-8', 'replace') + ('\n[truncated]' if len(raw) > limit else '')

    def number(self, device, key, value_type):
        value = value_type()
        self.check(self.clGetDeviceInfo(device, key, ct.sizeof(value), ct.byref(value), None))
        return value.value

    def compile_probe(self, device, precision):
        scalar = {'fp32': 'float', 'fp16': 'half'}[precision]
        source = ('#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n' if precision == 'fp16' else '')
        source += f'__kernel void probe(__global {scalar} *x) {{ x[0] = x[0] + ({scalar})1; }}'
        error = ct.c_int32()
        devices = (ct.c_void_p * 1)(device)
        context = self.clCreateContext(None, 1, devices, None, None, ct.byref(error))
        self.check(error.value)
        if not context:
            raise RuntimeError('OpenCL returned a null context')
        program = None
        try:
            strings = (ct.c_char_p * 1)(source.encode('ascii'))
            program = self.clCreateProgramWithSource(context, 1, strings, None, ct.byref(error))
            self.check(error.value)
            if not program:
                raise RuntimeError('OpenCL returned a null program')
            status = self.clBuildProgram(program, 1, devices, b'', None, None)
            result = {'status': 'compiled' if status == 0 else 'failed', 'error_code': int(status)}
            try:
                result['build_log'] = self.text(self.clGetProgramBuildInfo, program, device, 0x1183)
            except Exception as exc:
                result['build_log_error'] = str(exc)[:4096]
            return result
        finally:
            if program:
                self.clReleaseProgram(program)
            self.clReleaseContext(context)

    def report(self):
        count = ct.c_uint32()
        status = self.clGetPlatformIDs(0, None, ct.byref(count))
        if status == -1001 or (status == 0 and count.value == 0):
            return {'platforms': [], 'note': 'No OpenCL platforms found'}
        self.check(status)
        ids = (ct.c_void_p * min(count.value, 16))()
        self.check(self.clGetPlatformIDs(len(ids), ids, None))
        result = {'platforms': [], 'scope': 'OpenCL inventory; its indices are not OpenVINO GPU.N IDs'}
        seen = 0
        for platform_id in ids:
            item = {name: self.text(self.clGetPlatformInfo, platform_id, key, limit=1024)
                    for name, key in (('name', 0x902), ('vendor', 0x903), ('version', 0x901))}
            item['devices'] = []
            result['platforms'].append(item)
            count = ct.c_uint32()
            status = self.clGetDeviceIDs(platform_id, 4, 0, None, ct.byref(count))  # GPU devices only
            if status == -1:
                continue
            if status:
                item['enumeration_error'] = status
                continue
            if seen >= MAX_DEVICES or count.value == 0:
                if count.value:
                    item['truncated'] = True
                continue
            devices = (ct.c_void_p * min(count.value, MAX_DEVICES - seen))()
            self.check(self.clGetDeviceIDs(platform_id, 4, len(devices), devices, None))
            item['truncated'] = len(devices) < count.value
            for device in devices:
                seen += 1
                info = {name: self.text(self.clGetDeviceInfo, device, key, limit=1024)
                        for name, key in (('name', 0x102B), ('vendor', 0x102C), ('version', 0x102F),
                                          ('driver', 0x102D), ('opencl_c_version', 0x103D))}
                info['vendor_id'] = self.number(device, 0x1001, ct.c_uint32)
                info['global_memory_bytes'] = self.number(device, 0x101F, ct.c_uint64)
                info['max_allocation_bytes'] = self.number(device, 0x1010, ct.c_uint64)
                extensions = set(self.text(self.clGetDeviceInfo, device, 0x1030).split())
                info['extensions'] = {name: name in extensions for name in FEATURES}
                info['compile_probes'] = {}
                for precision in ('fp32', 'fp16'):
                    try:
                        info['compile_probes'][precision] = self.compile_probe(device, precision)
                    except Exception as exc:
                        info['compile_probes'][precision] = {'status': 'failed', 'error': str(exc)[:4096]}
                item['devices'].append(info)
        return result


def run_probe(kind, timeout):
    # Discovery and driver compilation can crash or hang. Neither runs inside
    # the server or this coordinator, and each child has a fixed time limit.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen([sys.executable, '-m', 'lazyrig.gpu_diagnostics', '--probe', kind],
                                   env=env, stdout=output, stderr=output, stdin=subprocess.DEVNULL)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        output.seek(0, 2)
        output.seek(max(0, output.tell() - 1024 * 1024))
        raw = output.read().decode('utf-8', 'replace')
    if timed_out:
        return {'status': 'timed_out', 'timeout_seconds': timeout}
    if process.returncode == 0:
        for line in reversed(raw.splitlines()):
            if line.startswith(MARKER):
                try:
                    return json.loads(line[len(MARKER):])
                except (TypeError, ValueError):
                    break
    return {'status': 'failed', 'exit_code': process.returncode, 'output': raw[-MAX_LOG:]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='logs/gpu-diagnostics.json')
    parser.add_argument('--timeout', type=int, default=45, help='Seconds per isolated check (1–120)')
    parser.add_argument('--probe', choices=('runtime', 'opencl'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 120:
        parser.error('--timeout must be between 1 and 120 seconds')
    if args.probe:
        try:
            value = runtime_report() if args.probe == 'runtime' else OpenCL().report()
            result = {'status': 'ok', 'data': value}
        except Exception as exc:
            result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)[:4096]}
        print(MARKER + json.dumps(result, ensure_ascii=True), flush=True)
        return 0
    report = {'schema': 1, 'system': platform.platform(), 'python': platform.python_version(),
              'scope': 'Device metadata and tiny kernel compilation only. No model inference was tested.',
              'interpretation': [
                  'Device enumeration does not guarantee that a model can compile or run.',
                  'OpenCL feature support is reported by the driver, independently of CUDA support.',
                  'Missing FP16/subgroup features can limit models even when their weights fit in VRAM.',
                  'Successful tiny probes do not establish support for Qwen or other model kernels.',
                  'A failed probe needs its own build log; it is not proof of the original model failure cause.']}
    for kind in ('runtime', 'opencl'):
        print('Checking ' + kind + '…', flush=True)
        report[kind] = run_probe(kind, args.timeout)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print('Diagnostic report saved: ' + str(destination.resolve()))
    print('This report does not load or validate a model.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
