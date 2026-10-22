import copy
from pathlib import Path
import tempfile
import unittest

from lazyrig.manual_affinity import partition_records, graph_records, AffinityReader, prepare_xml, scope_compiled_cache
from lazyrig.state_order import normalize_state_sink_order

try:
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
except ImportError:
    ov = None


class PartitionTests(unittest.TestCase):
    def test_changed_assignments_never_reuse_the_same_compiled_cache(self):
        original = {'CACHE_DIR':'cache', 'DEVICE_PROPERTIES':{'GPU.1':{'CACHE_DIR':'per-device'}},
                    'MODEL_PROPERTIES':{'language_model':{'CACHE_DIR':'per-role'}, 'text_embeddings':{'CACHE_DIR':''}}}
        outputs = []
        for mapping in ({'a':'GPU.1','b':'GPU.0'}, {'b':'GPU.0','a':'GPU.1'}, {'a':'GPU.1','b':'GPU.1'}):
            properties = copy.deepcopy(original)
            scope_compiled_cache(properties, mapping)
            outputs.append(properties)
            scope = Path(properties['CACHE_DIR']).name
            self.assertEqual(Path(properties['DEVICE_PROPERTIES']['GPU.1']['CACHE_DIR']).name, scope)
            self.assertEqual(Path(properties['MODEL_PROPERTIES']['language_model']['CACHE_DIR']).name, scope)
            self.assertEqual(properties['MODEL_PROPERTIES']['text_embeddings']['CACHE_DIR'], '')
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotEqual(outputs[0]['CACHE_DIR'], outputs[2]['CACHE_DIR'])
        self.assertEqual(original['CACHE_DIR'], 'cache')

    def test_complete_states_follow_requested_shares(self):
        records = [{'name':'input', 'kind':'Parameter', 'parents':[], 'weight_bytes':0}]
        previous = 'input'
        for i in range(10):
            records.extend([
                {'name':f'read{i}', 'kind':'ReadValue', 'parents':[], 'weight_bytes':0,'variable':str(i)},
                {'name':f'w{i}', 'kind':'Constant', 'parents':[], 'weight_bytes':100000},
                {'name':f'op{i}', 'kind':'MatMul', 'parents':[previous,f'w{i}',f'read{i}'],'weight_bytes':0},
                {'name':f'write{i}', 'kind':'Assign', 'parents':[f'op{i}'],'weight_bytes':0,'variable':str(i)},
            ])
            previous = f'op{i}'
        for first in (10, 49, 50, 60, 90):
            with self.subTest(first=first):
                mapping, report = partition_records(records, ['GPU.1','CPU'], {'GPU.1':first,'CPU':100-first})
                actual = sum(r['weight_bytes'] for r in records if mapping[r['name']] == 'GPU.1') / 10000
                self.assertLessEqual(abs(actual-first), 5)
                for i in range(10):
                    self.assertEqual(mapping[f'read{i}'],mapping[f'write{i}'])
                    self.assertEqual(mapping[f'w{i}'],mapping[f'op{i}'])
                self.assertEqual(report['method'],'explicit_affinity')

    def test_three_devices_and_zero_weight_segments_are_not_faked(self):
        records = [{'name':str(i),'kind':'Constant','parents':[],'weight_bytes':100000} for i in range(10)]
        mapping, _ = partition_records(records,['GPU.1','GPU.0','CPU'],{'GPU.1':50,'GPU.0':30,'CPU':20})
        self.assertEqual([list(mapping.values()).count(d) for d in ('GPU.1','GPU.0','CPU')],[5,3,2])
        with self.assertRaisesRegex(ValueError,'No explicit whole-state partition'):
            partition_records(records[:1],['GPU.1','CPU'],{'GPU.1':50,'CPU':50})

    def test_weight_decompression_stays_with_its_consumer(self):
        records = [
            {'name':'x','kind':'Parameter','parents':[],'weight_bytes':0},
            {'name':'a','kind':'Constant','parents':[],'weight_bytes':100000},
            {'name':'convert','kind':'Convert','parents':['a'],'weight_bytes':0},
            {'name':'b','kind':'Constant','parents':[],'weight_bytes':100000},
            {'name':'multiply','kind':'Multiply','parents':['b','convert'],'weight_bytes':0},
            {'name':'result','kind':'MatMul','parents':['x','multiply'],'weight_bytes':0},
        ]
        with self.assertRaisesRegex(ValueError,'No explicit whole-state partition'):
            partition_records(records,['GPU.1','CPU'],{'GPU.1':50,'CPU':50})

    def test_malformed_state_is_rejected(self):
        records = [
            {'name':'a','kind':'Constant','parents':[],'weight_bytes':100},
            {'name':'r','kind':'ReadValue','parents':['a'],'weight_bytes':0,'variable':'cache'},
        ]
        with self.assertRaisesRegex(ValueError,'one read and update'):
            partition_records(records,['GPU.1','CPU'],{'GPU.1':50,'CPU':50})


@unittest.skipIf(ov is None, 'Native OpenVINO unavailable')
class NativeAffinityTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.folder)
        self.core = ov.Core()
        p = ops.parameter([1,16],ov.Type.f32,name='input')
        value, sinks = p, []
        for i in range(10):
            info = ov.op.util.VariableInfo()
            info.variable_id = f'cache.{i}'
            info.data_shape, info.data_type = ov.PartialShape([1,16]), ov.Type.f32
            variable = ov.op.util.Variable(info)
            read = ops.read_value(ops.constant(np.zeros([1,16],dtype=np.float32)), variable)
            weight = ops.constant(np.eye(16,dtype=np.float32)*.5)
            value = ops.matmul(ops.add(value,read),weight,False,False)
            sinks.append(ops.assign(value,variable))
        self.original = ov.Model([ops.result(value)],sinks,[p])
        ov.serialize(self.original,self.path/'model.xml')
        normalized, _ = normalize_state_sink_order((self.path/'model.xml').read_bytes())
        (self.path/'view.xml').write_bytes(prepare_xml(normalized))

    def test_reader_preserves_native_types_state_and_all_assignments(self):
        reader = AffinityReader(ov)
        self.core.add_extension(reader.extensions)
        graph = self.core.read_model(self.path/'view.xml',self.path/'model.bin')
        mapping, _ = partition_records(graph_records(graph),['CPUA','CPUB'],{'CPUA':50,'CPUB':50})
        # Two independent CPU plugin instances exercise real HETERO subgraphs
        # and transport; this is not a test of physical GPU execution.
        library = Path(ov.__file__).parent/'libs/libopenvino_intel_cpu_plugin.so'
        if not library.is_file():
            self.skipTest('Two-plugin native probe currently runs on Linux')
        for d in ('CPUA','CPUB'):
            self.core.register_plugin(str(library),d)
        reader.assignments = mapping
        placed = self.core.read_model(self.path/'view.xml',self.path/'model.bin')
        report = {'partition':{}}
        reader.verify_loaded(report)
        self.assertTrue(report['partition']['native_loader_verified'])
        self.assertEqual({n.get_type_name() for n in graph.get_ops()}, {n.get_type_name() for n in placed.get_ops()})
        self.assertEqual({n.get_friendly_name():n.get_rt_info()['affinity'].value for n in placed.get_ops()}, mapping)
        requests = [self.core.compile_model(m,d,{'INFERENCE_NUM_THREADS':2,'INFERENCE_PRECISION_HINT':'f32'}).create_infer_request()
                    for m,d in ((self.original,'CPU'),(placed,'HETERO:CPUA,CPUB'))]
        self.assertEqual(requests[1].get_compiled_model().get_property('EXECUTION_DEVICES'),['CPUA','CPUB'])
        for cycle in range(2):
            for step in range(5):
                output = [next(iter(r.infer([np.full([1,16],step+1,dtype=np.float32)]).values())).copy() for r in requests]
                np.testing.assert_array_equal(*output)
                states = [{s.name:s.state.data.copy() for s in r.query_state()} for r in requests]
                self.assertEqual(states[0].keys(),states[1].keys())
                for name in states[0]: np.testing.assert_array_equal(states[0][name],states[1][name])
            for r in requests: r.reset_state()

    def test_missing_assignment_is_not_silently_defaulted(self):
        reader = AffinityReader(ov)
        self.core.add_extension(reader.extensions)
        graph = self.core.read_model(self.path/'view.xml',self.path/'model.bin')
        reader.assignments = {n.friendly_name:'CPU' for n in graph.get_ops() if n.friendly_name != 'input'}
        with self.assertRaisesRegex(RuntimeError,'Unassigned manual split operation'):
            self.core.read_model(self.path/'view.xml',self.path/'model.bin')

    def test_nested_weights_stay_with_parent_and_nested_state_is_rejected(self):
        then = ov.Model([ops.constant(np.ones(16, dtype=np.float32))], [])
        other = ov.Model([ops.constant(np.zeros(16, dtype=np.float32))], [])
        branch = ops.if_op(ops.constant(True))
        branch.set_then_body(then)
        branch.set_else_body(other)
        branch.set_output(then.get_results()[0], other.get_results()[0])
        graph = ov.Model([branch.output(0)], [])
        records = graph_records(graph)
        self.assertEqual(next(r for r in records if r['kind'] == 'If')['weight_bytes'], 128)
        nested = b'<net><layers><layer type="If"><then_body><layers><layer type="ReadValue"/></layers></then_body></layer></layers></net>'
        with self.assertRaisesRegex(ValueError, 'state variables inside nested subgraphs'):
            prepare_xml(nested)


if __name__ == '__main__': unittest.main()
