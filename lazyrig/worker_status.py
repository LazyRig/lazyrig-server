"""Bounded process-exit diagnostics, independent of native exception text."""
import os
import signal


def exit_status(report):
    """Safe terminal text from numeric OS status; never print arbitrary fields."""
    if not isinstance(report, dict):
        return 'exit code unavailable before cleanup'
    code = report.get('exit_code')
    if type(code) is not int or not -(2**31) <= code < 2**32:
        return 'exit code unavailable before cleanup'
    if report.get('platform') == 'nt':
        return f'Windows exit code {code} (0x{code & 0xFFFFFFFF:08X})'
    if report.get('platform') == 'posix' and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = 'unknown signal'
        return f'signal {-code} ({name})'
    return f'exit code {code}'


class WorkerProcessError(RuntimeError):
    """A worker disappeared without returning a native exception or result."""
    def __init__(self, exit_code, *, pid=None, stage='generation', platform=None, exited=True):
        platform = os.name if platform is None else platform
        stage = stage if stage in {'loading', 'validation', 'generation', 'connection'} else 'operation'
        self.worker_stage = stage
        self.worker_exit = {
            'exit_code': exit_code if type(exit_code) is int and -(2**31) <= exit_code < 2**32 else None,
            'pid': pid if type(pid) is int and pid > 0 else None,
            'stage': stage, 'platform': platform if platform in {'nt', 'posix'} else 'unknown',
            'exit_observed': exited is True,
        }
        if exited:
            self.error_code = 'worker_process_exited'
            message = f'Model worker exited unexpectedly during {stage}'
        else:
            self.error_code = 'worker_connection_closed'
            message = f'Model worker connection closed during {stage}'
        message += '; ' + exit_status(self.worker_exit)
        if self.worker_exit['pid'] is not None:
            message += f' (PID {pid})'
        super().__init__(message + '. No native error was returned.')
