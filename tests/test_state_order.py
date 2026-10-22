import unittest
import xml.etree.ElementTree as ET

from lazyrig.state_order import normalize_state_sink_order


def graph_xml():
    # IDs deliberately do not describe depth; two independent cache variables
    # sit at different positions in the data flow.
    return b'''<net name="test" version="11"><layers>
    <layer id="input" name="input" type="Parameter"/>
    <layer id="read_a" name="read a" type="ReadValue"><data variable_id="a"/></layer>
    <layer id="z" name="first" type="Add"/>
    <layer id="read_b" name="read b" type="ReadValue"><data variable_id="b"/></layer>
    <layer id="a" name="second" type="Add"/>
    <layer id="weight" name="weight" type="Const"><data offset="9000000000" size="8000000000" element_type="u4"/></layer>
    <!-- keep this comment -->
    <layer id="assign_a" name="write a" type="Assign"><data variable_id="a"/></layer>
    <layer id="assign_b" name="write b" type="Assign"><data variable_id="b"/></layer>
    <layer id="out" name="output" type="Result"/>
    </layers><edges>
    <edge from-layer="input" from-port="0" to-layer="z" to-port="0"/>
    <edge from-layer="read_a" from-port="0" to-layer="z" to-port="1"/>
    <edge from-layer="z" from-port="0" to-layer="a" to-port="0"/>
    <edge from-layer="read_b" from-port="0" to-layer="a" to-port="1"/>
    <edge from-layer="z" from-port="0" to-layer="assign_a" to-port="0"/>
    <edge from-layer="a" from-port="0" to-layer="assign_b" to-port="0"/>
    <edge from-layer="a" from-port="0" to-layer="out" to-port="0"/>
    </edges><rt_info><version value="original"/></rt_info></net>'''


class StateOrderTests(unittest.TestCase):
    def test_changes_only_assign_element_order(self):
        original = graph_xml()
        changed, report = normalize_state_sink_order(original)
        self.assertEqual(report, {'method': 'assign_dependency_depth', 'assign_count': 2, 'reordered': True})
        before, after = ET.fromstring(original), ET.fromstring(changed)
        nodes_before = list(before.find('layers'))
        nodes_after = list(after.find('layers'))
        self.assertEqual([node.get('id') for node in nodes_after if node.get('type') == 'Assign'], ['assign_b', 'assign_a'])
        for left, right in zip(nodes_before, nodes_after):
            if left.get('type') != 'Assign':
                self.assertEqual(ET.tostring(left), ET.tostring(right))
        self.assertEqual({node.get('id'): ET.tostring(node).strip() for node in nodes_before},
                         {node.get('id'): ET.tostring(node).strip() for node in nodes_after})
        self.assertEqual(ET.tostring(before.find('edges')), ET.tostring(after.find('edges')))
        self.assertEqual(ET.tostring(before.find('rt_info')), ET.tostring(after.find('rt_info')))
        self.assertIn(b'keep this comment', changed)

    def test_already_ordered_does_not_change_again(self):
        once, _ = normalize_state_sink_order(graph_xml())
        twice, report = normalize_state_sink_order(once)
        self.assertEqual(twice, once)
        self.assertEqual(report, {})

    def test_ambiguous_or_unpaired_states_are_left_intact(self):
        for xml in [graph_xml().replace(b'variable_id="b"', b'variable_id="a"'),
                    graph_xml().replace(b'<data variable_id="b"/>', b'<data/>', 1),
                    graph_xml().replace(b'name="read b" type="ReadValue"', b'name="read b" type="Parameter"')]:
            self.assertEqual(normalize_state_sink_order(xml), (xml, {}))

    def test_stateless_graph_does_not_change(self):
        xml = graph_xml().replace(b'type="Assign"', b'type="Result"')
        self.assertEqual(normalize_state_sink_order(xml), (xml, {}))

    def test_nested_state_ownership_is_left_intact(self):
        root = ET.fromstring(graph_xml())
        operation = root.find('layers/layer')
        body = ET.SubElement(operation, 'body')
        nested = ET.SubElement(body, 'layers')
        ET.SubElement(nested, 'layer', {'id': 'nested', 'type': 'ReadValue'})
        xml = ET.tostring(root)
        self.assertEqual(normalize_state_sink_order(xml), (xml, {}))

    def test_invalid_edge_and_cycle_stop_preparation(self):
        for xml in [graph_xml().replace(b'from-layer="input"', b'from-layer="missing"'),
                    graph_xml().replace(b'from-layer="input"', b'from-layer="out"'),
                    graph_xml().replace(b'id="read_b"', b'id="read_a"')]:
            with self.assertRaises(ValueError):
                normalize_state_sink_order(xml)

    def test_rejects_invalid_xml_and_entity_expansion(self):
        for xml in [b'<net>', b'<!DOCTYPE net [<!ENTITY repeat "data">]><net/>']:
            with self.assertRaises(ValueError):
                normalize_state_sink_order(xml)


if __name__ == '__main__':
    unittest.main()
