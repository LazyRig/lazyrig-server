"""Inspect existing exports and browse directories on the server machine."""
import json
import os
from pathlib import Path

def inspect_export(directory):
    if directory is None or (isinstance(directory, str) and not directory.strip()):
        raise ValueError("Choose a model folder first.")
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Model directory does not exist on the server")
    xmls = list(path.rglob("*.xml"))
    irs = [p for p in xmls if p.with_suffix(".bin").is_file()]
    if not irs:
        raise ValueError("No OpenVINO .xml/.bin model pairs found. Choose an exported OpenVINO model.")
    missing = [str(p.relative_to(path)) for p in xmls if not p.with_suffix(".bin").is_file()]
    if missing:
        raise ValueError("Missing .bin weights for: " + ", ".join(missing[:5]))
    if (path / "model_index.json").is_file():
        index = json.loads((path / "model_index.json").read_text())
        pipeline, task = "image", "text-to-image"
        architecture = index.get("_class_name", "Image pipeline")
        for name in ("scheduler", "vae_decoder"):
            if not (path / name).is_dir():
                raise ValueError("Image export is missing component directory: " + name)
        if not any((path / name).is_dir() for name in ("transformer", "unet")):
            raise ValueError("Image export is missing its transformer/unet component")
    else:
        if not (path / "openvino_tokenizer.xml").is_file() or not (path / "openvino_detokenizer.xml").is_file():
            raise ValueError("Text export requires openvino_tokenizer.xml and openvino_detokenizer.xml")
        if (path / "openvino_language_model.xml").is_file():
            pipeline = "vlm"
        elif (path / "openvino_model.xml").is_file():
            pipeline = "llm"
        else:
            raise ValueError("Choose the root of the text model export")
        task, architecture = "text-generation", pipeline
    cfg = {}
    if (path / "config.json").is_file():
        cfg = json.loads((path / "config.json").read_text())
    text_config = cfg.get("text_config", cfg)
    limit = text_config.get("max_position_embeddings", 32768)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    runtime_profile = "qwen-image-21" if "qwenimage21" in str(architecture).lower().replace("_", "") else "default"
    mtp_available = all((path / ("openvino_mtp_model." + suffix)).is_file() for suffix in ("xml", "bin"))
    from .image_edit import capabilities as edit_capabilities
    edit = edit_capabilities({'path': str(path), 'pipeline': pipeline, 'runtime_profile': runtime_profile,
                             'inspection': {'architecture': architecture}})
    return {"path": str(path), "pipeline": pipeline, "task": task, "architecture": architecture,
            "input_modalities": ["text", "image"] if pipeline == "vlm" or edit['available'] else ["text"], "mtp_available": mtp_available,
            "runtime_profile": runtime_profile,
            'image_editing': edit,
            "size_bytes": total, "ir_components": len(irs), "declared_context_limit": limit,
            "validation": "Export structure checked; loading validates runtime compatibility"}

def browse(directory=None):
    try:
        path = Path(directory or Path.home()).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("Directory not found on the server")
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise ValueError(
            f"Access denied to folder: {directory or 'home directory'}. "
            "Choose a folder readable by the account running LazyRig."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot read folder: {directory or 'home directory'}. "
            "Check that the drive is connected and the folder is accessible."
        ) from exc
    folders = []
    for p in entries:
        try:
            if p.is_dir() and not p.name.startswith("."):
                folders.append({"name": p.name, "path": str(p)})
        except OSError:
            pass
    roots = [str(Path.home()), str(Path.cwd())]
    if os.name == "nt":
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:/"
            try:
                if Path(root).exists():
                    roots.append(root)
            except OSError:
                # An unrelated protected or unavailable drive must not prevent
                # browsing folders that this server account can access.
                continue
    else:
        roots.append("/")
    return {"path": str(path), "parent": str(path.parent), "folders": folders[:500], "roots": list(dict.fromkeys(roots))}
