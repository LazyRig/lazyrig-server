"""Preserve state semantics while making IR sink traversal follow data flow.

OpenVINO's ratio query walks a topological ordering and looks for boundaries
with complete ReadValue/Assign pairs. IR files can list all Assign sinks in an
unrelated order. That can keep a pair open across most of the model even when
each variable belongs to one layer. The IR reader preserves sink list order,
and its topological walk visits the last sink first.

Only top-level Assign element order changes here. IDs, connections, attributes,
subgraphs, variable identifiers and constant offsets remain untouched. No
weight file is opened. The resulting graph must still pass native placement
verification; this is not an explicit affinity map or a guaranteed split.
"""
from collections import Counter, deque
import xml.etree.ElementTree as ET


MAX_XML_BYTES = 128 * 1024 * 1024


def normalize_state_sink_order(xml):
    """Return (XML bytes, change report), or the original bytes with {}.

    Restrict normalization to independent variables with exactly one top-level
    read and write each. More complex state ownership is left to the runtime.
    """
    if len(xml) > MAX_XML_BYTES:
        raise ValueError('Manual split graph preparation requires language XML smaller than 128 MiB')
    if b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
        raise ValueError('Manual split graph preparation does not accept XML document type declarations')
    try:
        root = ET.fromstring(xml, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    except ET.ParseError as exc:
        raise ValueError('Cannot prepare manual split: invalid language model XML') from exc
    layers = root.find('layers')
    edges = root.find('edges')
    if root.tag != 'net' or layers is None or edges is None:
        return xml, {}

    operations = [node for node in layers if node.tag == 'layer']
    assignments = [node for node in operations if node.get('type') == 'Assign']
    if len(assignments) < 2:
        return xml, {}
    top_level = set(operations)
    if any(node not in top_level and node.get('type') in {'ReadValue', 'Assign'}
           for node in root.iter('layer')):
        return xml, {}  # Nested state ownership needs its own validation.
    reads = [node for node in operations if node.get('type') == 'ReadValue']

    def variable(node):
        data = node.find('data')
        return data.get('variable_id') if data is not None else None

    read_ids = Counter(variable(node) for node in reads)
    assign_ids = Counter(variable(node) for node in assignments)
    if (None in read_ids or None in assign_ids or '' in read_ids or '' in assign_ids or
            read_ids != assign_ids or any(count != 1 for count in read_ids.values())):
        return xml, {}

    by_id = {node.get('id'): node for node in operations}
    if None in by_id or '' in by_id or len(by_id) != len(operations):
        raise ValueError('Cannot prepare manual split: missing or duplicate language operation IDs')
    parents = {key: set() for key in by_id}
    children = {key: set() for key in by_id}
    for edge in edges:
        if edge.tag != 'edge':
            continue
        source, destination = edge.get('from-layer'), edge.get('to-layer')
        if source not in by_id or destination not in by_id:
            raise ValueError('Cannot prepare manual split: language edge references a missing operation')
        parents[destination].add(source)
        children[source].add(destination)

    # Data dependency depth only: Assign->ReadValue control dependencies are
    # restored by the IR reader, not added to the serialized data graph here.
    remaining = {key: len(values) for key, values in parents.items()}
    pending = deque(key for key in by_id if not remaining[key])
    depth = {key: 0 for key in by_id}
    visited = 0
    while pending:
        key = pending.popleft()
        visited += 1
        for child in children[key]:
            depth[child] = max(depth[child], depth[key] + 1)
            remaining[child] -= 1
            if not remaining[child]:
                pending.append(child)
    if visited != len(by_id):
        raise ValueError('Cannot prepare manual split: cyclic language data graph')

    positions = {node.get('id'): index for index, node in enumerate(operations)}
    ordered = sorted(assignments, key=lambda node: (depth[node.get('id')], positions[node.get('id')]),
                     reverse=True)
    if ordered == assignments:
        return xml, {}
    # Keep every non-Assign element in its original position, including comments.
    slots = [index for index, node in enumerate(layers) if node.tag == 'layer' and node.get('type') == 'Assign']
    for index, node in zip(slots, ordered):
        layers[index] = node
    normalized = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    return normalized, {'method': 'assign_dependency_depth', 'assign_count': len(assignments), 'reordered': True}
