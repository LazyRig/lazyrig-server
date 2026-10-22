"""Isolated, cancellable Optimum export jobs. Never execute model repository code."""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from .worker import Cancelled
from .deadlines import RequestDeadline, check_deadline, time_budget

ROOT = Path(__file__).resolve().parent.parent
PROFILES = {"standard": ROOT / "requirements-converter.txt",
            "qwen-image-21": ROOT / "requirements-qwen-image-21.txt"}
_profile_lock = threading.Lock()
_export_lock = threading.Lock()


def profile_dir(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown conversion runtime profile")
    digest = hashlib.sha256(PROFILES[profile].read_bytes()).hexdigest()[:12]
    return ROOT / ".runtime" / "converters" / (profile + "-" + digest)


def profile_python(profile):
    return _runtime_python(profile_dir(profile))


def _runtime_python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def quiet_environment():
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", HF_HUB_DISABLE_TELEMETRY="1",
               DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", NNCF_TELEMETRY="0")
    # A profile must use its own dependencies, never another environment's imports.
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = str(ROOT)
    return env


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(5)


def run_logged(command, logfile, cancel, report, env=None, timeout=86400):
    """Log native output without blocking cancellation on a readline()."""
    check_deadline(cancel)
    timeout = time_budget(timeout, cancel)
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    environment = env or quiet_environment()
    with logfile.open("ab", buffering=0) as output, logfile.open("rb") as reader:
        reader.seek(0, 2)
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen([str(x) for x in command], cwd=str(ROOT), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, **kwargs)
        start = time.monotonic()
        try:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    text = re.sub(r"hf_[A-Za-z0-9]+", "[HF token]", text)
                    report(text)
                if cancel.is_set():
                    raise Cancelled()
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Runtime/export command exceeded {timeout:g} seconds")
                code = process.poll()
                if code is not None and not chunk:
                    if code:
                        raise RuntimeError(f"Converter process exited with code {code}. See the download job log.")
                    return
                cancel.wait(.2)
        finally:
            stop_process(process)


def _acquire(lock, cancel):
    while not lock.acquire(timeout=.2):
        if cancel.is_set():
            raise Cancelled()
    try:
        if cancel.is_set():
            raise Cancelled()
    except BaseException:
        lock.release()
        raise


def _ready_marker(directory):
    try:
        value = json.loads((directory / 'lazyrig-ready.json').read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _mark_ready(directory, profile, digest):
    # Preserve the virtual environment's original path; Windows launchers may
    # contain absolute paths. Remember validation against the new requirements.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                     prefix='lazyrig-ready-', suffix='.tmp', delete=False) as out:
        temporary = Path(out.name)
        json.dump({'profile': profile, 'python': str(_runtime_python(directory)),
                   'requirements_sha256': digest}, out)
    try:
        temporary.replace(directory / 'lazyrig-ready.json')
    finally:
        temporary.unlink(missing_ok=True)


def _profile_compatible(python, requirements, cancel, report, logfile):
    # Both commands are read-only and make no package/index requests. Verify
    # versions, pinned source URLs, imports and dependency consistency before
    # trusting an environment created for a different requirements-file hash.
    env = quiet_environment()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    try:
        run_logged([python, '-m', 'lazyrig.profile_check', requirements],
                   logfile, cancel, lambda text: None, env=env, timeout=120)
        run_logged([python, '-m', 'pip', '--disable-pip-version-check', '--isolated', 'check'],
                   logfile, cancel, lambda text: None, env=env, timeout=120)
        return True
    except (Cancelled, RequestDeadline):
        raise
    except (RuntimeError, OSError, TimeoutError):
        report('Existing runtime did not pass compatibility checks; details are in the runtime setup log.\n')
        return False


def ensure_profile(profile, cancel, report, logfile):
    _acquire(_profile_lock, cancel)
    try:
        directory, python = profile_dir(profile), profile_python(profile)
        digest = hashlib.sha256(PROFILES[profile].read_bytes()).hexdigest()
        candidates = [directory]
        if directory.parent.is_dir():
            candidates += sorted(p for p in directory.parent.iterdir()
                                 if p != directory and p.is_dir() and not p.is_symlink()
                                 and re.fullmatch(re.escape(profile) + r'-[0-9a-f]{12}', p.name))
        unverified = []
        # Inspect every saved readiness marker before starting a subprocess.
        # A partial current-hash install must not be probed again on every load
        # when another directory already passed checks for these requirements.
        for candidate in candidates:
            candidate_python = _runtime_python(candidate)
            if not candidate_python.is_file():
                continue
            ready = _ready_marker(candidate)
            if ready.get('profile') == profile and (
                    ready.get('requirements_sha256') == digest or
                    (candidate == directory and 'requirements_sha256' not in ready)):
                return candidate_python
            unverified.append((candidate, candidate_python))
        for candidate, candidate_python in unverified:
            report(f'Checking existing {profile} runtime for reuse: {candidate.name}\n')
            if _profile_compatible(candidate_python, PROFILES[profile], cancel, report, logfile):
                _mark_ready(candidate, profile, digest)
                report(f'Reusing existing {profile} runtime; no packages were installed.\n')
                return candidate_python
        directory.parent.mkdir(parents=True, exist_ok=True)
        report(f'No compatible installed {profile} runtime was found.\n'
               'Preparing an isolated runtime for conversion and model loading '
               '(first use can take several minutes).\n')
        # The existing supported Python 3.13 is reused; no Python 3.12 requirement.
        run_logged([sys.executable, "-m", "venv", str(directory)], logfile, cancel, report)
        pip = [python, "-m", "pip", "--disable-pip-version-check", "--isolated", "install"]
        run_logged(pip + ["--upgrade", "pip", "setuptools", "wheel", "--index-url", "https://pypi.org/simple"], logfile, cancel, report)
        # CPU-only PyTorch is sufficient for export and avoids CUDA dependencies.
        run_logged(pip + ["torch", "torchvision", "numpy>=1.24,<2.5", "--index-url", "https://download.pytorch.org/whl/cpu"], logfile, cancel, report)
        run_logged(pip + ["-r", PROFILES[profile]], logfile, cancel, report)
        run_logged([python, "-m", "pip", "check"], logfile, cancel, report)
        run_logged([python, "-c", "import openvino, openvino_genai, optimum.intel, diffusers; print('Conversion runtime ready')"], logfile, cancel, report)
        _mark_ready(directory, profile, digest)
        return python
    finally:
        _profile_lock.release()


def select_profile(model_index, requested="auto"):
    if requested not in {"auto", *PROFILES}:
        raise ValueError("Invalid conversion profile")
    is_qwen21 = "qwenimage21" in str(model_index.get("_class_name", "")).lower().replace("_", "")
    if is_qwen21:
        return "qwen-image-21"
    return "standard" if requested == "auto" else requested


def conversion_task(config, model_index, pipeline_tag, requested="auto"):
    if requested != "auto":
        if requested not in {"text-generation-with-past", "image-text-to-text", "text-to-image"}:
            raise ValueError("Select an LLM, VLM or image generation conversion task")
        return requested
    if model_index:
        return "text-to-image"
    if pipeline_tag in {"image-text-to-text", "image-to-text"} or config.get("vision_config"):
        return "image-text-to-text"
    if pipeline_tag not in {None, "", "text-generation", "conversational"}:
        raise ValueError("Automatic conversion currently supports LLMs, VLMs and image generation models")
    return "text-generation-with-past"


def convert(source, output, profile, task, precision, cancel, report, logfile, set_phase):
    if precision not in {"fp16", "int8", "int4"}:
        raise ValueError("Weights must be fp16, int8 or int4")
    set_phase("waiting-converter")
    _acquire(_export_lock, cancel)
    try:
        set_phase("preparing-runtime")
        python = ensure_profile(profile, cancel, report, logfile)
        command = [python, "-m", "lazyrig.export_model", "--source", source, "--output", output,
                   "--task", task, "--precision", precision, "--profile", profile]
        set_phase("converting")
        env = quiet_environment()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        report(f"Exporting {task} with {precision.upper()} weights using {profile}.\n")
        run_logged(command, logfile, cancel, report, env)
    finally:
        _export_lock.release()
