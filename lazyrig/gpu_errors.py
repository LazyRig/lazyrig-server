"""Identify kernel compilation failures without diagnosing a vendor or memory fault."""
import re


def kernel_build_failed(error):
    message = str(error)
    return bool(re.search(r'\bCL_BUILD_PROGRAM_FAILURE\b', message, re.IGNORECASE)
                and ('[gpu]' in message.lower() or 'intel_gpu' in message.lower()))


def kernel_build_failure(error):
    if not kernel_build_failed(error):
        return {}
    return {
        'error_code': 'gpu_kernel_build_failed',
        'error': (
            'GPU kernel compilation failed (CL_BUILD_PROGRAM_FAILURE). '
            'This error alone does not establish a VRAM shortage. A detected GPU may lack '
            'OpenCL features required by this model, or its driver/runtime may fail to compile '
            'a kernel. Run diagnose-gpu.bat and share logs/gpu-diagnostics.json. '
            'For CPU fallback, select CPU in Automatic device allocation; Manual keeps your '
            'chosen devices. Original error: ' + type(error).__name__ + ': ' + str(error)),
    }
