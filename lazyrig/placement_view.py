"""Worker-owned directory views for manual language-graph normalization.

Only the language XML is rewritten. Hardlinks let GenAI keep its normal folder
loading path without duplicating model weights or changing the registered export.
"""
import copy
import os
from pathlib import Path
import shutil
import tempfile


class PlacementView:
    """An explicitly owned directory; cleanup must follow native worker exit.

    There is deliberately no destructor/finalizer: a failed worker termination
    must leave its backing files available until termination can be confirmed.
    """
    def __init__(self, path):
        self.path = Path(path)

    def close(self):
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            if self.path.exists():
                raise


def prepare_manual_view(model, checkpoint=None):
    """Return (worker configuration, owned view), or the unchanged configuration.

    ``checkpoint`` enforces the parent's cancellation, deadline and memory guard.
    No native runtime or weight tensor is loaded while preparing this view.
    """
    if (model.get('pipeline') not in {'llm', 'vlm', 'vlm-text'} or
            model.get('placement', {}).get('mode') != 'manual' or
            len(model.get('_placement', {}).get('devices', [])) < 2):
        return model, None
    from .state_order import MAX_XML_BYTES, normalize_state_sink_order
    from .manual_affinity import prepare_xml
    check = checkpoint or (lambda: None)
    check()
    source = Path(model['path']).resolve(strict=True)
    filename = 'openvino_model.xml' if model['pipeline'] == 'llm' else 'openvino_language_model.xml'
    xml_path = source / filename
    if xml_path.stat().st_size > MAX_XML_BYTES:
        raise ValueError('Cannot prepare manual model split: language XML exceeds the normalization size limit')
    with xml_path.open('rb') as stream:
        # Bound the read too: a concurrent file replacement must not defeat the
        # metadata check and allocate unbounded memory in the server process.
        original = stream.read(MAX_XML_BYTES + 1)
    if len(original) > MAX_XML_BYTES:
        raise ValueError('Cannot prepare manual model split: language XML exceeds the normalization size limit')
    check()
    normalized, report = normalize_state_sink_order(original)
    normalized = prepare_xml(normalized)
    check()
    try:
        view = PlacementView(tempfile.mkdtemp(prefix='.lazyrig-placement-', dir=str(source.parent)))
    except OSError as exc:
        raise RuntimeError('Cannot prepare manual model split: a temporary directory on the '
                           'model filesystem could not be created; the original export was not changed') from exc
    try:
        # Root files preserve the exported-model layout, including arbitrary
        # config/tokenizer metadata. Resolve HF snapshot links to regular files.
        for entry in source.iterdir():
            check()
            if entry.name == filename or not entry.is_file():
                continue
            try:
                os.link(entry.resolve(strict=True), view.path / entry.name)
            except OSError as exc:
                raise RuntimeError('Cannot prepare manual model split: the model filesystem '
                                   'must support same-volume hardlinks for ' + entry.name +
                                   '; model weights will not be copied') from exc
        check()
        (view.path / filename).write_bytes(normalized)
        check()
        runtime_model = copy.deepcopy(model)
        runtime_model['path'] = str(view.path)
        runtime_model['_state_order'] = copy.deepcopy(report)
        runtime_model['_manual_affinity'] = True
        return runtime_model, view
    except BaseException as exc:
        # Nothing has been spawned yet, so partial preparation is safe to remove.
        try:
            view.close()
        except BaseException as cleanup_error:
            # Transfer explicit ownership to ProcessWorker's failure path when
            # a transient file lock prevents this first cleanup attempt.
            exc.placement_view = view
            exc.placement_view_cleanup_error = cleanup_error
        raise
