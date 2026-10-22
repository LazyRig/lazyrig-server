"""HF URL -> inspect, download, optionally convert, publish, register."""
import json
import re
import shutil
import stat
import threading
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse
from .catalog import inspect_export
from .storage import new_id
from .worker import Cancelled
from . import conversion, __version__

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}

def source_document(name):
    """Recognize common upstream notices and model cards; do not infer a license."""
    path = PurePosixPath(name)
    if path.suffix.lower() not in {"", ".txt", ".md", ".markdown", ".rst"}:
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    legal_dirs = {"license", "licenses", "licence", "licences", "legal"}
    return (any(part.lower() in legal_dirs for part in path.parts[:-1]) or
            bool(re.fullmatch(r"(?:licen[cs]e|notice|copying|copyright|authors|readme|"
                              r"model[_-]?card|use[_-]?policy|acceptable[_-]?use[_-]?policy)"
                              r"(?:[._-].*)?", path.name, flags=re.I)))

def hf_documents(manifest, prefix):
    """Keep documents in the selected subtree and its ancestor directories."""
    ancestors = {PurePosixPath(".")}
    root = PurePosixPath(prefix)
    ancestors.update(root.parents)
    documents = []
    for file in manifest["siblings"]:
        name = file["rfilename"]
        path = safe_member(name)
        if not source_document(name):
            continue
        legal_parent = next((parent.parent for parent in path.parents
                             if parent.name.lower() in {"license", "licenses", "licence", "licences", "legal"}), None)
        if name.startswith(prefix) or path.parent in ancestors or legal_parent in ancestors:
            documents.append(file)
    return documents

def hf_files(manifest, subfolder):
    """Return a single export root, or source weights. Avoid duplicate formats."""
    prefix = subfolder.strip("/") + "/" if subfolder else ""
    siblings = [f for f in manifest["siblings"] if f["rfilename"].startswith(prefix)]
    names = {f["rfilename"][len(prefix):] for f in siblings}
    markers = {"openvino_model.xml", "openvino_language_model.xml", "transformer/openvino_model.xml", "unet/openvino_model.xml"}
    exported = bool(names & markers)
    if not exported:
        roots = {n.rsplit("/", 1)[0] for n in names if n.endswith(("/openvino_model.xml", "/openvino_language_model.xml"))}
        roots = {r.rsplit("/", 1)[0] if r.endswith(("/unet", "/transformer")) else r for r in roots if r not in {"unet", "transformer"}}
        if len(roots) == 1:
            prefix += next(iter(roots)) + "/"
            siblings = [f for f in siblings if f["rfilename"].startswith(prefix)]
            exported = True
    extensions = {".xml", ".bin", ".json", ".jinja", ".txt", ".model", ".tiktoken", ".vocab", ".merges"}
    if not exported:
        extensions = (extensions - {".bin", ".xml"}) | {".safetensors"}
    files = []
    for f in siblings:
        relative = f["rfilename"][len(prefix):]
        safe_member(relative)
        p = PurePosixPath(relative)
        if p.name in {"all_results.json", "eval_results.json", "train_results.json", "trainer_state.json"} or "onnx" in p.parts:
            continue
        if p.suffix.lower() not in extensions or any(x.startswith(".") for x in p.parts):
            continue
        if exported and p.name.startswith(("pytorch_model", "diffusion_pytorch_model")):
            continue
        if not exported and (p.name.startswith(("optimizer", "training_args")) or p.name.endswith((".fp16.safetensors", ".bf16.safetensors"))):
            continue
        files.append(f)
    if not exported and not any(f["rfilename"].endswith(".safetensors") for f in files):
        raise ValueError("No supported source weights found. Choose a Safetensors HF model or an OpenVINO export (GGUF and pickle-only checkpoints are not converted).")
    return files, prefix, exported

def safe_member(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe archive path: " + name)
    return p

def source_spec(url, revision="main", subfolder=""):
    u = urlparse(url)
    if u.scheme not in {"https", "http"} or not u.hostname or u.username or u.password:
        raise ValueError("Provide an HTTP(S) URL without embedded credentials")
    if u.hostname.lower() == "huggingface.co":
        parts = [unquote(s) for s in u.path.strip("/").split("/")]
        if len(parts) < 2:
            raise ValueError("Use a Hugging Face model repository URL")
        if len(parts) > 2:
            if parts[2] != "tree" or len(parts) < 4:
                raise ValueError("Use a repository URL or /tree/revision/subfolder URL")
            revision = parts[3]
            subfolder = "/".join(parts[4:])
        if subfolder:
            safe_member(subfolder)
        return {"kind": "huggingface", "repo": "/".join(parts[:2]), "revision": revision, "subfolder": subfolder.strip("/")}
    return {"kind": "zip", "url": url}

def extract_zip(archive, destination, max_bytes, cancel=None):
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 30000 or sum(i.file_size for i in entries) > max_bytes:
            raise ValueError("Archive exceeds configured extraction limits")
        for i in entries:
            safe_member(i.filename)
            if stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
        for i in entries:
            if cancel and cancel.is_set():
                raise Cancelled()
            target = destination / i.filename
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as src, target.open("wb") as out:
                    while chunk := src.read(1024 * 1024):
                        if cancel and cancel.is_set():
                            raise Cancelled()
                        out.write(chunk)

class DownloadManager:
    def __init__(self, service):
        self.service = service
        self.lock = threading.RLock()
        self.jobs, self.running = {}, {}
        self.path = service.store.directory / "downloads.json"
        if self.path.exists():
            self.jobs = json.loads(self.path.read_text())
            for j in self.jobs.values():
                if j["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    j.update(status="interrupted", error="Server restarted; submit again to a new or empty destination.")

    def save(self, job):
        with self.lock:
            self.jobs[job["id"]] = job
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.jobs, indent=2))
            temp.replace(self.path)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)))

    def start(self, body):
        spec = source_spec(body["url"], body.get("revision", "main"), body.get("subfolder", ""))
        if not body.get("destination"):
            raise ValueError("Choose a destination directory on the server")
        dest = Path(body["destination"]).expanduser().resolve()
        precision = body.get("precision", "fp16").lower()
        if precision not in {"fp16", "int8", "int4"}:
            raise ValueError("Precision must be fp16, int8 or int4")
        if body.get("conversion", "auto") not in {"auto", "never"}:
            raise ValueError("Conversion must be auto or never")
        if body.get("profile", "auto") not in {"auto", *conversion.PROFILES}:
            raise ValueError("Unknown conversion profile")
        conversion.conversion_task({}, {}, None, body.get("task", "auto"))
        register = body.get("register", False)
        if not isinstance(register, bool):
            raise ValueError("register must be true or false")
        if register:
            if body.get("device") not in {d["id"] for d in self.service.config["devices"]}:
                raise ValueError("Select a detected device for this model")
            alias = body.get("id") or re.sub(r"[^A-Za-z0-9_.-]", "-", dest.name)[:80].lstrip("-_.")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", alias):
                raise ValueError("Enter an API model alias")
            if any(m["id"] == alias for m in self.service.config["models"]):
                raise ValueError("That model alias is already registered")
            from .memory import device_reservation
            reservation = float(body['memory_gib']) if 'memory_gib' in body else device_reservation(self.service.config['devices'], body['device'])
            if not 0 < reservation < 10000:
                raise ValueError("Invalid memory reservation")
            registration = {"id": alias, "device": body["device"], "memory_gib": reservation}
        else:
            registration = None
        with self.lock:
            if len(self.running) >= 2:
                raise ValueError("Two downloads are already running")
            if any(Path(j["destination"]) == dest for j in self.jobs.values() if j["status"] not in TERMINAL):
                raise ValueError("This destination is already being downloaded")
            if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
                raise ValueError("Choose a new or empty directory; existing files are never overwritten")
            job = {"id": new_id(), "source": body["url"], "destination": str(dest), "created": time.time(),
                   "status": "queued", "bytes": 0, "total_bytes": None, "file": "", "error": None,
                   "precision": precision, "conversion": body.get("conversion", "auto"),
                   "requested_profile": body.get("profile", "auto"), "requested_task": body.get("task", "auto"),
                   "registration": registration, "log": [], "converted": False}
            cancel = threading.Event()
            thread = threading.Thread(target=self._run, args=(job, spec, cancel), daemon=True)
            self.running[job["id"]] = (cancel, thread)
            self.save(job)
            thread.start()
            return dict(job)

    def cancel(self, jid):
        with self.lock:
            if jid not in self.jobs:
                raise KeyError("Download not found")
            if jid in self.running:
                self.running[jid][0].set()
        return {"ok": True}

    def _run(self, job, spec, cancel):
        dest = Path(job["destination"])
        stage = dest.parent / (".lazyrig-download-" + job["id"])
        maximum = int(self.service.config.get("download_max_gib", 100) * 1024**3)
        last_save = [0.0]
        logfile = self.service.store.directory / "conversion-logs" / (job["id"] + ".log")
        def report_log(text):
            with self.lock:
                job["log"] = (job["log"] + text.splitlines())[-120:]
            self.save(job)
        def phase(status):
            job["status"] = status
            if status != "downloading":
                job["file"] = ""
            self.save(job)
        def open_url(request):
            for attempt in range(4):
                if cancel.is_set():
                    raise Cancelled()
                try:
                    return urllib.request.urlopen(request, timeout=30)
                except urllib.error.HTTPError as exc:
                    if exc.code not in {429, 502, 503, 504} or attempt == 3:
                        raise
                    retry_after = exc.headers.get("Retry-After")
                    delay = 5 * (2 ** attempt)
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            from email.utils import parsedate_to_datetime
                            delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    if delay > 120:
                        raise ValueError(f"Remote server asks to retry after {int(delay)} seconds; retry this download later") from exc
                    job["retry"] = f"HTTP {exc.code}; waiting {int(delay)} seconds before retry"
                    self.save(job)
                    if cancel.wait(delay):
                        raise Cancelled()
                except TimeoutError:
                    if attempt == 3:
                        raise
                    if cancel.wait(5 * (2 ** attempt)):
                        raise Cancelled()
        def fetch(url, target, expected=None):
            target.parent.mkdir(parents=True, exist_ok=True)
            job["file"] = target.name
            for attempt in range(3):
                if cancel.is_set():
                    raise Cancelled()
                received = 0
                request = urllib.request.Request(url, headers={"User-Agent": "LazyRig/" + __version__})
                with open_url(request) as response, target.open("wb") as out:
                    advertised = response.headers.get("Content-Length")
                    size = expected if expected is not None else (int(advertised) if advertised else None)
                    while chunk := response.read(1024 * 1024):
                        if cancel.is_set():
                            raise Cancelled()
                        received += len(chunk)
                        job["bytes"] += len(chunk)
                        if job["bytes"] > maximum:
                            raise ValueError("Download exceeds configured size limit")
                        out.write(chunk)
                        if time.monotonic() - last_save[0] > .3:
                            self.save(job); last_save[0] = time.monotonic()
                if size is None or received == size:
                    job.pop("retry", None)
                    return
                job["bytes"] -= received
                if attempt == 2:
                    raise ValueError(f"Incomplete download for {target.name}: expected {size} bytes, received {received}")
                job["retry"] = f"Incomplete transfer for {target.name}; retrying"
                self.save(job)
                if cancel.wait(5 * (attempt + 1)):
                    raise Cancelled()
        try:
            stage.mkdir(parents=True)
            documents_dir = stage / "source-documents"
            job["status"] = "downloading"; self.save(job)
            content = stage / "content"; content.mkdir()
            if spec["kind"] == "zip":
                archive = stage / "download.zip"
                fetch(spec["url"], archive)
                job["status"] = "validating"; self.save(job)
                extract_zip(archive, content, maximum, cancel)
            else:
                repo, revision = spec["repo"], spec["revision"]
                request = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true", headers={"User-Agent": "LazyRig/" + __version__})
                with open_url(request) as response:
                    manifest = json.load(response)
                commit = manifest["sha"]
                prefix = spec["subfolder"] + "/" if spec["subfolder"] else ""
                files, prefix, exported = hf_files(manifest, spec["subfolder"])
                documents = hf_documents(manifest, prefix)
                document_names = {f["rfilename"] for f in documents}
                files = [f for f in files if f["rfilename"] not in document_names]
                transfers = files + documents
                job["source_format"] = "openvino" if exported else "huggingface"
                if not exported and job["conversion"] == "never":
                    raise ValueError("This is a source model. Enable automatic conversion, or choose an OpenVINO export.")
                job["total_bytes"] = sum(f.get("size", f.get("lfs", {}).get("size", 0)) for f in transfers) or None
                if job["total_bytes"] and job["total_bytes"] > maximum:
                    raise ValueError("Model exceeds configured download limit")
                job["revision"] = commit
                required_space = (job["total_bytes"] or 0) * (1 if exported else 3)
                if required_space > shutil.disk_usage(stage).free:
                    raise ValueError("Insufficient disk space for source weights and conversion output")
                # Download architecture metadata first so the correct environment is ready
                # before transferring tens of gigabytes of source weights.
                metadata = {}
                for filename in ("config.json", "model_index.json"):
                    entry = next((f for f in files if f["rfilename"] == prefix + filename), None)
                    if entry:
                        fetch(f"https://huggingface.co/{repo}/resolve/{commit}/{quote(entry['rfilename'], safe='/')}", content / filename, entry.get("size"))
                        metadata[filename] = json.loads((content / filename).read_text(encoding="utf-8"))
                job["profile"] = conversion.select_profile(metadata.get("model_index.json", {}), job["requested_profile"])
                if not exported:
                    job["task"] = conversion.conversion_task(metadata.get("config.json", {}), metadata.get("model_index.json", {}), manifest.get("pipeline_tag"), job["requested_task"])
                if not exported or job["profile"] == "qwen-image-21":
                    phase("preparing-runtime")
                    conversion.ensure_profile(job["profile"], cancel, report_log, logfile)
                phase("downloading")
                for i, file in enumerate(transfers):
                    name = file["rfilename"]
                    job["files_done"], job["files_total"] = i, len(transfers)
                    if name in document_names:
                        fetch(f"https://huggingface.co/{repo}/resolve/{commit}/{quote(name, safe='/')}?download=true", documents_dir / name, file.get("size", file.get("lfs", {}).get("size")))
                        continue
                    relative = name[len(prefix):]
                    safe_member(relative)
                    if relative not in metadata:
                        fetch(f"https://huggingface.co/{repo}/resolve/{commit}/{quote(name, safe='/')}?download=true", content / relative, file.get("size", file.get("lfs", {}).get("size")))
                job["files_done"] = len(transfers)
                if not exported:
                    output = stage / "converted"
                    conversion.convert(content, output, job["profile"], job["task"], job["precision"], cancel, report_log, logfile, phase)
                    content = output
                    job["converted"] = True
            if cancel.is_set():
                raise Cancelled()
            job["status"] = "validating"; self.save(job)
            # Permit a single enclosing directory in an exported-model ZIP.
            try:
                report = inspect_export(content)
                root = content
            except ValueError:
                children = [p for p in content.iterdir() if p.is_dir()]
                if len(children) != 1:
                    raise
                root = children[0]
                report = inspect_export(root)
            # ZIPs may have legal documents outside the enclosing export folder.
            # Preserve original paths so different notices never overwrite each other.
            if spec["kind"] == "zip":
                for path in content.rglob("*"):
                    relative = path.relative_to(content)
                    if path.is_file() and source_document(relative.as_posix()):
                        target = documents_dir / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
            document_paths = []
            if documents_dir.exists():
                target = root / "lazyrig-source-documents"
                suffix = 2
                while target.exists():
                    target = root / f"lazyrig-source-documents-{suffix}"
                    suffix += 1
                shutil.copytree(documents_dir, target)
                document_paths = sorted(p.relative_to(root).as_posix() for p in target.rglob("*") if p.is_file())
            # Keep provenance with the export; it can be moved/re-registered later.
            provenance = {"source": job["source"], "revision": job.get("revision"), "converted": job["converted"],
                          "precision": job["precision"] if job["converted"] else "original",
                          "runtime_profile": "qwen-image-21" if job.get("profile") == "qwen-image-21" else "default"}
            if document_paths:
                provenance["source_documents"] = document_paths
            (root / "lazyrig-export.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            if cancel.is_set():
                raise Cancelled()
            if dest.exists():
                dest.rmdir()  # succeeds only for an empty directory
            root.rename(dest)
            report["path"] = str(dest)
            job.update(inspection=report, file="")
            if job["registration"]:
                try:
                    model = self.service.register({**job["registration"], "path": str(dest)}, from_download=True)
                    job["model_id"] = model["id"]
                except Exception as exc:
                    job["registration_error"] = str(exc) + ". Model saved; use Register to retry."
            job["status"] = "completed"
        except Cancelled:
            job.update(status="cancelled", error="Download/conversion cancelled")
        except Exception as exc:
            job.update(status="failed", error=str(exc))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            self.save(job)
            with self.lock:
                self.running.pop(job["id"], None)

    def close(self):
        with self.lock:
            work = list(self.running.values())
            for cancel, _ in work:
                cancel.set()
        for _, thread in work:
            thread.join(35)
