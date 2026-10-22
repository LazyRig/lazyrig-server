"""Manual partition preflight failures must not silently become one-device loads."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from lazyrig.placement_probe import (ManualSplitCalibrationError, verify_manual_split,
                                    _partition_report, MAX_CALIBRATION_QUERIES)

try:
    import openvino as native_ov
except ImportError:
    native_ov = None


class Node:
    def __init__(self, name, size=None):
        self.name, self.size = name, size

    def get_friendly_name(self):
        return self.name

    def get_type_name(self):
        return 'Constant' if self.size is not None else 'Add'

    def get_byte_size(self):
        return self.size

    def get_data(self):
        raise AssertionError('Preflight must not copy weight data')


class Parameter(Node):
    def __init__(self, name, tensor_names=()):
        super().__init__(name)
        self.tensor_names = set(tensor_names)

    def get_type_name(self):
        return 'Parameter'

    def output(self, index):
        assert index == 0
        return types.SimpleNamespace(get_names=lambda: self.tensor_names)


class PlacementProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename in ('openvino_model.xml', 'openvino_language_model.xml'):
            (self.root / filename).touch()
        self.model = {'pipeline': 'llm', 'path': str(self.root),
                      'placement': {'mode': 'manual', 'shares': {'GPU.1': 60, 'CPU': 40}},
                      '_placement': {'devices': ['GPU.1', 'CPU'], 'target': 'HETERO:GPU.1,CPU'}}
        self.properties = {'ATTENTION_BACKEND': 'SDPA', 'DEVICE_PROPERTIES': {
            'GPU.1': {'QUERY_MODEL_RATIO': .6, 'UNRELATED': 'not part of probe'},
            'CPU': {'NUM_STREAMS': 4}}, 'OTHER_GENAI_OPTION': object()}
        self.nodes = [Node('weights_a', 60), Node('weights_b', 40), Node('output')]
        self.graph = Mock(get_ordered_ops=lambda: self.nodes)
        self.core = Mock()
        self.core.read_model.return_value = self.graph
        self.core.query_model.return_value = {'weights_a': 'GPU.1', 'weights_b': 'CPU', 'output': 'CPU'}
        self.runtime = types.SimpleNamespace(__version__='2026.4.1-22982-example')
        patcher = patch.dict(sys.modules, {'openvino': self.runtime})
        patcher.start()
        self.addCleanup(patcher.stop)

    def verify(self):
        return verify_manual_split(self.core, self.model, self.properties)

    def use_generic_gpu(self):
        self.model['placement']['shares'] = {'GPU': 50, 'CPU': 50}
        self.model['_placement'] = {'devices': ['GPU', 'CPU'], 'target': 'HETERO:GPU,CPU'}
        self.properties['DEVICE_PROPERTIES'] = {'GPU': {'QUERY_MODEL_RATIO': .5}}
        self.nodes[:] = [Node('170', 50), Node('weights_b', 50), Node('output')]
        self.core.query_model.return_value = {'170': 'GPU.0', 'weights_b': 'CPU', 'output': 'CPU'}
        self.core.get_property.return_value = '0'

    def test_real_assignment_report_and_only_managed_query_properties(self):
        before = copy.deepcopy(self.model)
        report = self.verify()
        self.core.read_model.assert_called_once_with(str(self.root / 'openvino_model.xml'))
        self.core.query_model.assert_called_once_with(self.graph, 'HETERO:GPU.1,CPU',
            {'DEVICE_PROPERTIES': {'GPU.1': {'QUERY_MODEL_RATIO': .6}}})
        self.assertEqual(report['weight_bytes'], 100)
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 60)
        self.assertEqual(report['devices']['CPU']['weight_bytes'], 40)
        self.assertEqual(report['status'], 'preflight_verified')
        self.assertEqual(report['scope'], 'exported_language_graph')
        self.assertEqual(report['requested_weight_shares'], {'GPU.1': 60, 'CPU': 40})
        self.assertIn('not VRAM', report['note'])
        self.assertEqual(self.model, before)
        self.assertEqual(self.properties['DEVICE_PROPERTIES']['GPU.1']['UNRELATED'], 'not part of probe')
        self.assertEqual(report['resolved_query_devices'], {'GPU.1': 'GPU.1', 'CPU': 'CPU'})
        self.core.get_property.assert_not_called()

    def test_report_keeps_requested_and_queried_shares_distinct(self):
        self.nodes[:] = [Node('weights_a', 99), Node('weights_b', 1), Node('output')]
        with self.assertRaises(ManualSplitCalibrationError) as raised:
            self.verify()
        report = raised.exception.placement_preflight
        self.assertEqual(report['requested_weight_shares'], {'GPU.1': 60, 'CPU': 40})
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 99)
        self.assertEqual(report['devices']['CPU']['weight_share_percent'], 1)
        self.assertEqual(report['scope'], 'exported_language_graph')
        self.assertEqual(report['status'], 'calibration_failed')
        self.assertEqual(report['calibration']['query_count'], 3)
        self.assertIn('Closest tested: GPU.1 99%, CPU 1%', str(raised.exception))
        self.assertIn('No model was compiled', str(raised.exception))

    def weighted_query(self, shares_for_properties):
        """A deterministic quantized graph; no graph mutation or weight reads."""
        self.nodes[:] = [Node(str(index), 1) for index in range(1000)]

        def query(graph, target, properties):
            self.assertIs(graph, self.graph)
            counts = shares_for_properties(properties)
            self.assertEqual(sum(counts.values()), 1000)
            mapping = {}
            for device, count in counts.items():
                for _ in range(count):
                    mapping[str(len(mapping))] = device
            return mapping

        self.core.query_model.side_effect = query

    def test_calibrates_observed_88_2_percent_split_and_keeps_user_configuration(self):
        self.model['placement']['shares'] = {'GPU.1': 50, 'CPU': 50}
        self.properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'] = .5
        before_model = copy.deepcopy(self.model)
        before_device_properties = copy.deepcopy(self.properties['DEVICE_PROPERTIES'])

        def distribution(properties):
            ratio = properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO']
            gpu = 882 if ratio >= .4 else 500 if ratio > .2 else 100
            return {'GPU.1': gpu, 'CPU': 1000 - gpu}

        self.weighted_query(distribution)
        report = self.verify()
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 50)
        self.assertEqual(report['calibration']['query_count'], 2)
        self.assertEqual(report['calibration']['attempts'][0]['measured_weight_shares'],
                         {'GPU.1': 88.2, 'CPU': 11.8})
        selected = report['calibrated_properties']['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO']
        self.assertLess(selected, .4)
        self.assertGreater(selected, .2)
        self.assertEqual(self.model, before_model)
        self.assertEqual(self.properties['DEVICE_PROPERTIES'], before_device_properties)
        self.core.read_model.assert_called_once()

    def test_final_gpu_consumes_remaining_graph_after_first_ratio_calibration(self):
        self.model['placement']['shares'] = {'GPU.1': 50, 'GPU.0': 50}
        self.model['_placement'] = {'devices': ['GPU.1', 'GPU.0'], 'target': 'HETERO:GPU.1,GPU.0'}
        self.properties['DEVICE_PROPERTIES'] = {'GPU.1': {'QUERY_MODEL_RATIO': .5},
                                               'GPU.0': {'QUERY_MODEL_RATIO': 1.0}}

        def distribution(properties):
            ratios = properties['DEVICE_PROPERTIES']
            self.assertEqual(ratios['GPU.0']['QUERY_MODEL_RATIO'], 1.0)
            first = 882 if ratios['GPU.1']['QUERY_MODEL_RATIO'] >= .4 else 500
            return {'GPU.1': first, 'GPU.0': 1000 - first}

        self.weighted_query(distribution)
        report = self.verify()
        self.assertEqual(report['target'], 'HETERO:GPU.1,GPU.0')
        self.assertEqual(report['calibration']['query_count'], 2)
        self.assertEqual(report['calibrated_properties']['DEVICE_PROPERTIES']['GPU.0'],
                         {'QUERY_MODEL_RATIO': 1.0})

    def test_three_device_calibration_uses_conditional_remaining_weights(self):
        selected = ['GPU.1', 'GPU.0', 'CPU']
        self.model['placement']['shares'] = {'GPU.1': 40, 'GPU.0': 30, 'CPU': 30}
        self.model['_placement'] = {'devices': selected, 'target': 'HETERO:' + ','.join(selected)}
        self.properties['DEVICE_PROPERTIES'] = {'GPU.1': {'QUERY_MODEL_RATIO': .4},
                                               'GPU.0': {'QUERY_MODEL_RATIO': .5}}

        def distribution(properties):
            ratios = properties['DEVICE_PROPERTIES']
            first = round(1000 * min(1, ratios['GPU.1']['QUERY_MODEL_RATIO'] + .3))
            second = round((1000 - first) * min(1, ratios['GPU.0']['QUERY_MODEL_RATIO'] + .2))
            return {'GPU.1': first, 'GPU.0': second, 'CPU': 1000 - first - second}

        self.weighted_query(distribution)
        report = self.verify()
        self.assertLessEqual(report['calibration']['query_count'], MAX_CALIBRATION_QUERIES)
        for device, expected in self.model['placement']['shares'].items():
            self.assertLessEqual(abs(report['devices'][device]['weight_share_percent'] - expected), 5)
        ratios = report['calibration']['selected_ratios']
        self.assertLess(ratios['GPU.1'], .4)
        self.assertLess(ratios['GPU.0'], .5)

    def test_four_device_opposite_errors_do_not_cancel_out_of_tolerance_shares(self):
        selected = ['GPU.2', 'GPU.1', 'GPU.0', 'CPU']
        self.model['placement']['shares'] = dict.fromkeys(selected, 25)
        self.model['_placement'] = {'devices': selected, 'target': 'HETERO:' + ','.join(selected)}
        self.properties['DEVICE_PROPERTIES'] = {device: {'QUERY_MODEL_RATIO': ratio}
                                               for device, ratio in zip(selected, (.25, 1 / 3, .5))}

        def distribution(properties):
            ratio = properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO']
            second = 170 if ratio < .4 else 210
            return {'GPU.2': 290, 'GPU.1': second, 'GPU.0': 460 - second, 'CPU': 250}

        self.weighted_query(distribution)
        report = self.verify()
        self.assertEqual(report['calibration']['query_count'], 2)
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 21)

    def test_initial_zero_weight_assignment_can_be_calibrated(self):
        def distribution(properties):
            ratio = properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO']
            first = 1000 if ratio > .4 else 600
            return {'GPU.1': first, 'CPU': 1000 - first}

        self.weighted_query(distribution)
        report = self.verify()
        self.assertEqual(report['calibration']['query_count'], 2)
        self.assertEqual(report['devices']['CPU']['weight_share_percent'], 40)

    def test_zero_gpu_result_probes_neighbor_before_assuming_ratio_must_increase(self):
        # Captured from OpenVINO2026.4.1 get_supported_nodes on a ten-layer
        # stateful graph with sinks ordered by reverse dependency depth.
        # Its ratio response is nonmonotonic: .5 gives no GPU weights, while
        # .475 gives the requested half. Ordinary bisection misses that result.
        observed = {0.125: 200, .25: 300, .45: 500, .475: 500,
                    .49: 0, .5: 0, .51: 0, .55: 600, .625: 0, .75: 800, .875: 0, 1: 1000}
        for final_device in ('CPU', 'GPU.0'):
            with self.subTest(final_device=final_device):
                self.core.query_model.reset_mock()
                self.model['placement']['shares'] = {'GPU.1': 50, final_device: 50}
                self.model['_placement'] = {'devices': ['GPU.1', final_device],
                                           'target': 'HETERO:GPU.1,' + final_device}
                self.properties['DEVICE_PROPERTIES'] = {'GPU.1': {'QUERY_MODEL_RATIO': .5}}
                if final_device == 'GPU.0':
                    self.properties['DEVICE_PROPERTIES'][final_device] = {'QUERY_MODEL_RATIO': 1}

                def distribution(properties):
                    gpu = observed.get(round(properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'], 3), 0)
                    return {'GPU.1': gpu, final_device: 1000 - gpu}

                self.weighted_query(distribution)
                report = self.verify()
                self.assertEqual(report['status'], 'preflight_verified')
                self.assertEqual(report['calibration']['query_count'], 2)
                self.assertEqual(report['calibration']['selected_ratios']['GPU.1'], .475)
                self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 50)
                self.assertEqual(report['devices'][final_device]['weight_share_percent'], 50)
                self.assertEqual(report['calibration']['max_queries'], 9)
                self.assertEqual(report['calibration']['tolerance_percentage_points'], 5)

    def test_zero_gpu_neighbor_is_attempted_once_before_normal_bounded_search(self):
        self.model['placement']['shares'] = {'GPU.1': 50, 'CPU': 50}
        self.properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'] = .5
        self.weighted_query(lambda properties: {'GPU.1': 0, 'CPU': 1000})
        with self.assertRaises(ManualSplitCalibrationError) as raised:
            self.verify()
        calibration = raised.exception.placement_preflight['calibration']
        ratios = [attempt['ratios']['GPU.1'] for attempt in calibration['attempts']]
        self.assertEqual(ratios[:2], [.5, .475])
        self.assertEqual(ratios.count(.475), 1)
        self.assertLessEqual(len(ratios), 9)
        self.assertEqual(calibration['stop_reason'], 'tested graph-boundary plateau')

    def test_state_order_metadata_is_copied_into_success_and_failure_reports(self):
        for gpu_weight, succeeds in ((60, True), (99, False)):
            with self.subTest(succeeds=succeeds):
                self.nodes[:] = [Node('weights_a', gpu_weight), Node('weights_b', 100 - gpu_weight), Node('output')]
                self.model['_state_order'] = {'status': 'normalized', 'sink_count': 10, 'details': {'changed': True}}
                if succeeds:
                    report = self.verify()
                else:
                    with self.assertRaises(ManualSplitCalibrationError) as raised:
                        self.verify()
                    report = raised.exception.placement_preflight
                self.assertEqual(report['state_order'], self.model['_state_order'])
                self.model['_state_order']['details']['changed'] = False
                self.assertTrue(report['state_order']['details']['changed'])

    def test_tolerance_is_explicit_and_applies_to_every_device(self):
        for gpu_weight, accepted in ((65, True), (66, False)):
            with self.subTest(gpu_weight=gpu_weight):
                self.core.query_model.reset_mock()
                self.nodes[:] = [Node('weights_a', gpu_weight), Node('weights_b', 100 - gpu_weight), Node('output')]
                if accepted:
                    report = self.verify()
                    self.assertEqual(report['calibration']['tolerance_percentage_points'], 5)
                    self.assertEqual(report['calibration']['query_count'], 1)
                else:
                    with self.assertRaises(ManualSplitCalibrationError) as raised:
                        self.verify()
                    self.assertEqual(raised.exception.placement_preflight['calibration']['maximum_deviation_percentage_points'], 6)

    def test_query_budget_is_global_and_failure_reports_closest_tested_distribution(self):
        attempts = []

        def distribution(properties):
            attempts.append(properties)
            first = 900 - 10 * len(attempts)
            return {'GPU.1': first, 'CPU': 1000 - first}

        self.weighted_query(distribution)
        with self.assertRaises(ManualSplitCalibrationError) as raised:
            self.verify()
        report = raised.exception.placement_preflight
        self.assertEqual(len(attempts), MAX_CALIBRATION_QUERIES)
        self.assertEqual(report['calibration']['query_count'], MAX_CALIBRATION_QUERIES)
        self.assertEqual(report['calibration']['stop_reason'], 'query budget reached')
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 81)
        self.assertIn('Closest tested: GPU.1 81%', str(raised.exception))

    def test_deadline_stops_between_native_queries_and_preserves_existing_measurement(self):
        clock = [0]

        def distribution(properties):
            clock[0] = 100
            return {'GPU.1': 800, 'CPU': 200}

        self.weighted_query(distribution)
        with patch('lazyrig.placement_probe.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaises(ManualSplitCalibrationError) as raised:
                self.verify()
        report = raised.exception.placement_preflight
        self.assertEqual(report['calibration']['query_count'], 1)
        self.assertEqual(report['calibration']['stop_reason'], 'calibration time budget reached')
        self.assertEqual(report['devices']['GPU.1']['weight_share_percent'], 80)

    def test_external_deadline_can_expire_before_first_query(self):
        with patch('lazyrig.placement_probe.time.monotonic', return_value=10):
            with self.assertRaisesRegex(TimeoutError, 'before querying the model'):
                verify_manual_split(self.core, self.model, self.properties, deadline=9)
        self.core.query_model.assert_not_called()

    def test_original_parameter_names_report_resolved_device_without_guessing(self):
        self.use_generic_gpu()
        self.nodes.extend([Parameter('Parameter_4', ['inputs_embeds', 'embedding_alias']),
                           Parameter('attention_mask', ['attention_mask'])])
        self.core.query_model.return_value.update({'Parameter_4': 'GPU.0', 'attention_mask': 'CPU',
                                                  'Parameter_generated': 'CPU'})
        report = self.verify()
        self.assertEqual(report['language_input_device'], 'GPU.0')
        self.assertEqual(report['input_devices'], {
            'Parameter_4': {'device': 'GPU', 'query_device': 'GPU.0',
                            'tensor_names': ['embedding_alias', 'inputs_embeds']},
            'attention_mask': {'device': 'CPU', 'query_device': 'CPU', 'tensor_names': ['attention_mask']}})

    def test_language_input_device_requires_unambiguous_exact_input_name(self):
        for parameters, expected in (([Parameter('inputs_embeds')], 'CPU'),
                                     ([Parameter('input_ids')], None),
                                     ([Parameter('inputs_embeds'), Parameter('other', ['inputs_embeds'])], None)):
            with self.subTest(parameters=parameters):
                self.nodes[:] = [Node('weights_a', 60), Node('weights_b', 40), Node('output'), *parameters]
                self.core.query_model.return_value = {'weights_a': 'GPU.1', 'weights_b': 'CPU', 'output': 'CPU',
                    **{node.name: 'CPU' for node in parameters}}
                report = self.verify()
                if expected:
                    self.assertEqual(report['language_input_device'], expected)
                else:
                    self.assertNotIn('language_input_device', report)

    def test_single_gpu_alias_from_runtime_device_id_keeps_selected_report_labels(self):
        self.use_generic_gpu()
        self.core.query_model.return_value.update({'Parameter_boundary': 'GPU.0', 'Result_boundary': 'GPU'})
        original = self.core.query_model.return_value.copy()
        report = self.verify()
        self.core.get_property.assert_called_once_with('GPU', 'DEVICE_ID')
        self.core.get_default_context.assert_not_called()
        self.core.query_model.assert_called_once_with(self.graph, 'HETERO:GPU,CPU',
            {'DEVICE_PROPERTIES': {'GPU': {'QUERY_MODEL_RATIO': .5}}})
        self.assertEqual(report['resolved_query_devices'], {'GPU': 'GPU.0', 'CPU': 'CPU'})
        self.assertEqual(report['devices']['GPU']['weight_share_percent'], 50)
        self.assertEqual(report['devices']['CPU']['weight_share_percent'], 50)
        self.assertEqual(report['additional_query_operations'], {'GPU': 2, 'CPU': 0})
        self.assertEqual(self.core.query_model.return_value, original)

    def test_gpu_alias_follows_proven_nonzero_or_tile_default(self):
        for device_id in ('1', '0.1'):
            with self.subTest(device_id=device_id):
                self.use_generic_gpu()
                self.core.get_property.return_value = device_id
                self.core.query_model.return_value['170'] = 'GPU.' + device_id
                report = self.verify()
                self.assertEqual(report['resolved_query_devices']['GPU'], 'GPU.' + device_id)

    def test_generic_gpu_rejects_other_devices_and_internal_hetero_markers(self):
        for device in ('GPU.1', 'GPU.0.0', 'CPU.0', 'HETERO', 'HETERO-TEMP', None):
            with self.subTest(device=device):
                self.use_generic_gpu()
                self.core.query_model.return_value['170'] = device
                with self.assertRaises(ValueError) as raised:
                    self.verify()
                message = str(raised.exception)
                self.assertIn('operation 170', message)
                self.assertIn('outside the selected devices', message)
                self.assertIn('returned ' + repr(device), message)
                self.assertIn("selected ['GPU', 'CPU']", message)
                self.assertIn("resolved query devices {'GPU': 'GPU.0', 'CPU': 'CPU'}", message)

    def test_single_available_device_proves_alias_when_default_property_is_unavailable(self):
        self.use_generic_gpu()
        self.core.get_property.side_effect = [RuntimeError('No DEVICE_ID property'), ['0']]
        report = self.verify()
        self.assertEqual(report['resolved_query_devices']['GPU'], 'GPU.0')
        self.assertEqual([call.args for call in self.core.get_property.call_args_list],
                         [('GPU', 'DEVICE_ID'), ('GPU', 'AVAILABLE_DEVICES')])

    def test_unknown_default_does_not_guess_from_multiple_gpus_or_malformed_properties(self):
        for default, available in ((RuntimeError('missing'), ['0', '1']),
                                   ('', []), ('not-a-device', ['GPU.0']),
                                   (0, '0'), (None, RuntimeError('missing'))):
            with self.subTest(default=default, available=available):
                self.use_generic_gpu()
                self.core.get_property.side_effect = [default, available]
                with self.assertRaisesRegex(ValueError, "returned 'GPU.0'"):
                    self.verify()

    def test_exact_selected_label_is_valid_even_if_generic_alias_cannot_be_resolved(self):
        self.use_generic_gpu()
        self.core.get_property.side_effect = RuntimeError('properties unavailable')
        self.core.query_model.return_value['170'] = 'GPU'
        report = self.verify()
        self.assertEqual(report['resolved_query_devices']['GPU'], 'GPU')

    def test_same_physical_gpu_selected_twice_is_rejected_before_reading_graph(self):
        self.use_generic_gpu()
        self.model['placement']['shares'] = {'GPU': 25, 'GPU.0': 25, 'CPU': 50}
        self.model['_placement'] = {'devices': ['GPU', 'GPU.0', 'CPU'],
                                   'target': 'HETERO:GPU,GPU.0,CPU'}
        self.properties['DEVICE_PROPERTIES']['GPU.0'] = {'QUERY_MODEL_RATIO': 1 / 3}
        with self.assertRaisesRegex(ValueError, 'same physical device'):
            self.verify()
        self.core.read_model.assert_not_called()

    def test_unresolved_generic_gpu_with_explicit_gpu_is_ambiguous(self):
        self.use_generic_gpu()
        self.core.get_property.side_effect = RuntimeError('properties unavailable')
        self.model['placement']['shares'] = {'GPU': 25, 'GPU.1': 25, 'CPU': 50}
        self.model['_placement'] = {'devices': ['GPU', 'GPU.1', 'CPU'],
                                   'target': 'HETERO:GPU,GPU.1,CPU'}
        self.properties['DEVICE_PROPERTIES']['GPU.1'] = {'QUERY_MODEL_RATIO': 1 / 3}
        with self.assertRaisesRegex(ValueError, 'could not be resolved alongside explicit GPU'):
            self.verify()
        self.core.read_model.assert_not_called()

    def test_unused_device_rejected_even_when_it_owns_unweighted_operations(self):
        self.core.query_model.return_value['weights_b'] = 'GPU.1'
        with self.assertRaisesRegex(ValueError, 'no model weights were assigned to CPU'):
            self.verify()

    def test_incomplete_original_operation_map_rejected_even_with_generated_boundaries(self):
        for mapping in ({'weights_a': 'GPU.1', 'weights_b': 'CPU'},
                        {'weights_a': 'GPU.1', 'weights_b': 'CPU', 'Parameter_251208': 'CPU'}):
            with self.subTest(mapping=mapping):
                self.core.query_model.return_value = mapping
                with self.assertRaisesRegex(ValueError, 'unassigned operations: output'):
                    self.verify()

    def test_generated_boundaries_are_validated_but_do_not_change_weight_accounting(self):
        self.core.query_model.return_value.update({
            'Parameter_251208': 'CPU', 'Parameter_251210': 'CPU', 'Result_251212': 'GPU.1'})
        report = self.verify()
        self.assertEqual(report['weight_bytes'], 100)
        self.assertEqual(report['devices']['GPU.1'],
                         {'operations': 1, 'weight_bytes': 60, 'weight_share_percent': 60})
        self.assertEqual(report['devices']['CPU'],
                         {'operations': 2, 'weight_bytes': 40, 'weight_share_percent': 40})
        self.assertEqual(report['additional_query_operations'], {'GPU.1': 1, 'CPU': 2})

    def test_generated_boundary_outside_selected_devices_rejected(self):
        for device in ('GPU.0', 'HETERO', '', None):
            with self.subTest(device=device):
                self.core.query_model.return_value['Parameter_251208'] = device
                with self.assertRaisesRegex(ValueError, 'outside the selected devices'):
                    self.verify()

    def test_invalid_query_operation_names_rejected(self):
        original = self.core.query_model.return_value.copy()
        for name in ('', None, 42):
            with self.subTest(name=name):
                self.core.query_model.return_value = {**original, name: 'CPU'}
                with self.assertRaisesRegex(ValueError, 'invalid query operation name'):
                    self.verify()

    def test_ambiguous_names_and_assignment_outside_target_rejected(self):
        self.nodes.append(Node('weights_a', 20))
        with self.assertRaisesRegex(ValueError, 'duplicate operation names'):
            self.verify()
        self.nodes.pop()
        self.core.query_model.return_value['weights_b'] = 'GPU.0'
        with self.assertRaisesRegex(ValueError, 'outside the selected devices'):
            self.verify()

    def test_graph_without_weights_cannot_claim_verified_split(self):
        self.nodes[:] = [Node('output')]
        self.core.query_model.return_value = {'output': 'CPU'}
        with self.assertRaisesRegex(ValueError, 'no constant weights'):
            self.verify()

    def test_old_or_unknown_runtime_fails_before_reading_weights(self):
        for version in ('2026.3.0', '2025.4.1', 'unknown', ''):
            with self.subTest(version=version):
                self.runtime.__version__ = version
                with self.assertRaisesRegex(ValueError, '2026.4 or newer'):
                    self.verify()
        self.core.read_model.assert_not_called()

    def test_vlm_modes_probe_language_component_only(self):
        for pipeline in ('vlm', 'vlm-text'):
            with self.subTest(pipeline=pipeline):
                self.model['pipeline'] = pipeline
                self.verify()
                self.core.read_model.assert_called_with(str(self.root / 'openvino_language_model.xml'))

    def test_inactive_modes_need_no_runtime_or_native_calls(self):
        self.runtime.__version__ = 'unknown'
        self.model['placement']['mode'] = 'auto'
        self.assertEqual(self.verify(), {})
        self.model['placement']['mode'] = 'manual'
        self.model['_placement']['devices'] = ['GPU.1']
        self.assertEqual(self.verify(), {})
        self.core.read_model.assert_not_called()
        self.core.query_model.assert_not_called()

    def test_automatic_policy_cannot_override_manual_ratio(self):
        self.properties['MODEL_DISTRIBUTION_POLICY'] = 'PIPELINE_PARALLEL'
        with self.assertRaisesRegex(ValueError, 'cannot combine'):
            self.verify()
        self.core.read_model.assert_not_called()

    def test_missing_or_invalid_ratio_rejected(self):
        for ratio in (None, 0, -1, 1.1, float('nan'), True):
            with self.subTest(ratio=ratio):
                self.properties['DEVICE_PROPERTIES']['GPU.1']['QUERY_MODEL_RATIO'] = ratio
                with self.assertRaisesRegex(ValueError, 'no valid weight-share target'):
                    self.verify()
        self.core.read_model.assert_not_called()

    def test_missing_language_graph_fails_before_native_calls(self):
        (self.root / 'openvino_model.xml').unlink()
        with self.assertRaisesRegex(ValueError, 'missing language graph'):
            self.verify()
        self.core.read_model.assert_not_called()

    def test_query_failure_propagates_without_a_verified_report(self):
        self.core.query_model.side_effect = RuntimeError('No legal partition')
        with self.assertRaisesRegex(RuntimeError, 'No legal partition'):
            self.verify()


@unittest.skipIf(native_ov is None, 'OpenVINO is not installed')
class NativeCPUControlTests(unittest.TestCase):
    def test_real_ir_query_metadata_matches_constant_storage(self):
        from openvino import opset13 as ops
        core = native_ov.Core()
        if 'CPU' not in core.available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        parameter = ops.parameter([1, 4], native_ov.Type.f32, name='input')
        parameter.output(0).get_tensor().set_names({'inputs_embeds', 'embedding_alias'})
        weights = ops.constant([[1., 2., 3., 4.]], native_ov.Type.f32, name='weights')
        graph = native_ov.Model([ops.add(parameter, weights, name='sum')], [parameter])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'openvino_model.xml'
            native_ov.serialize(graph, path)
            loaded = core.read_model(str(path))
            assignments = core.query_model(loaded, 'HETERO:CPU')
            report = _partition_report(loaded, assignments, ['CPU'])
        self.assertEqual(report['weight_bytes'], 16)
        self.assertEqual(report['devices']['CPU']['weight_share_percent'], 100)
        self.assertEqual(report['devices']['CPU']['operations'], len(graph.get_ordered_ops()))
        self.assertEqual(report['language_input_device'], 'CPU')
        self.assertEqual(report['input_devices']['input'], {
            'device': 'CPU', 'query_device': 'CPU', 'tensor_names': ['embedding_alias', 'inputs_embeds']})

    def test_real_hetero_query_generated_result_does_not_invalidate_original_graph(self):
        from openvino import opset13 as ops
        core = native_ov.Core()
        if 'CPU' not in core.available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        parameter = ops.parameter([1, 4], native_ov.Type.f32, name='input')
        unused_parameter = ops.parameter([1, 4], native_ov.Type.f32, name='unused_input')
        weights = ops.constant([[1., 2., 3., 4.]], native_ov.Type.f32, name='weights')
        graph = native_ov.Model([ops.add(parameter, weights, name='sum')],
                               [parameter, unused_parameter])
        original_names = {node.get_friendly_name() for node in graph.get_ordered_ops()}
        assignments = core.query_model(graph, 'HETERO:CPU')
        # OpenVINO2026.4 HETERO adds a Result for the unused Parameter in its
        # cloned query graph and includes that new operation in its public map.
        additional_names = set(assignments) - original_names
        self.assertTrue(additional_names)
        self.assertTrue(all(name.startswith('Result_') for name in additional_names))
        report = _partition_report(graph, assignments, ['CPU'])
        self.assertEqual(report['weight_bytes'], 16)
        self.assertEqual(report['devices']['CPU']['operations'], len(original_names))
        self.assertEqual(report['additional_query_operations'], {'CPU': len(additional_names)})


if __name__ == '__main__':
    unittest.main()
