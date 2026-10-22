"""Native CPU checks for state-sink ordering; these do not validate GPU placement."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from lazyrig.state_order import normalize_state_sink_order

try:
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
except ImportError:
    np = ov = ops = None


@unittest.skipIf(ov is None, 'OpenVINO is not installed')
class NativeStateOrderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.core = ov.Core()
        parameter = ops.parameter([1, 16], ov.Type.f32, name='input')
        value = parameter
        sinks = []
        for layer in range(10):
            info = ov.op.util.VariableInfo()
            info.data_shape = ov.PartialShape([1, 16])
            info.data_type = ov.Type.f32
            info.variable_id = f'cache.{layer}'
            variable = ov.op.util.Variable(info)
            initial = ops.constant(np.zeros([1, 16], dtype=np.float32),
                                   name=f'initial.{layer}')
            read = ops.read_value(initial, variable, name=f'read.{layer}')
            weight = ops.constant(np.full([16, 16], .01, dtype=np.float32),
                                  name=f'weight.{layer}')
            value = ops.matmul(ops.add(value, read), weight, False, False,
                               name=f'layer.{layer}')
            assign = ops.assign(value, variable)
            assign.friendly_name = f'assign.{layer}'
            sinks.append(assign)
        graph = ov.Model([ops.result(value)], sinks, [parameter])
        self.original_xml = self.path / 'original.xml'
        self.weights = self.path / 'original.bin'
        ov.serialize(graph, self.original_xml, self.weights)
        changed_xml, self.report = normalize_state_sink_order(self.original_xml.read_bytes())
        self.normalized_xml = self.path / 'normalized.xml'
        self.normalized_xml.write_bytes(changed_xml)
        self.original = self.core.read_model(self.original_xml, self.weights)
        self.normalized = self.core.read_model(self.normalized_xml, self.weights)

    @staticmethod
    def state_order(graph):
        return [node.friendly_name for node in graph.get_ordered_ops()
                if node.get_type_name() in {'ReadValue', 'Assign'}]

    def test_native_reader_and_clone_keep_each_cache_pair_near_its_layer(self):
        before = self.state_order(self.original)
        self.assertEqual(before[0], 'read.9')
        self.assertGreater(before.index('assign.0'), before.index('read.9'))
        expected = [name for layer in range(10)
                    for name in (f'read.{layer}', f'assign.{layer}')]
        self.assertEqual(self.state_order(self.normalized), expected)
        self.assertEqual(self.state_order(self.normalized.clone()), expected)
        self.assertEqual(self.normalized.get_parameters()[0].friendly_name, 'input')
        self.assertEqual(len(self.normalized.get_results()), 1)

    def test_repeated_cpu_inference_and_all_cache_states_remain_identical(self):
        if 'CPU' not in self.core.available_devices:
            self.skipTest('OpenVINO CPU plugin is unavailable')
        properties = {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2}
        requests = [self.core.compile_model(graph, 'CPU', properties).create_infer_request()
                    for graph in (self.original, self.normalized)]
        for cycle in range(2):
            for step in range(6):
                value = np.arange(16, dtype=np.float32).reshape(1, 16) * (step + 1) / 10
                results = [next(iter(request.infer({0: value}).values())).copy()
                           for request in requests]
                with self.subTest(cycle=cycle, step=step):
                    np.testing.assert_array_equal(results[0], results[1])
                    states = [{state.name: state.state.data.copy()
                               for state in request.query_state()} for request in requests]
                    self.assertEqual(set(states[0]), {f'cache.{layer}' for layer in range(10)})
                    self.assertEqual(states[0].keys(), states[1].keys())
                    for name in states[0]:
                        np.testing.assert_array_equal(states[0][name], states[1][name])
            for request in requests:
                for state in request.query_state():
                    state.reset()

    def test_native_supported_nodes_can_split_the_reordered_stateful_graph(self):
        # Exercise the same exported native helper that GPU query_model calls, but
        # with an identity transform and an all-supported predicate. This isolates
        # state-pair ordering from GPU compilation and requires no GPU hardware.
        compiler = shutil.which('c++') or shutil.which('g++')
        if sys.platform != 'linux' or compiler is None:
            self.skipTest('This native helper probe requires Linux and a C++ compiler')
        if not ov.__version__.startswith('2026.4.1'):
            self.skipTest('The internal native helper probe is pinned to OpenVINO 2026.4.1')
        package = Path(ov.__file__).resolve().parent
        libraries = list((package / 'libs').glob('libopenvino.so.*'))
        if not libraries or not (package / 'include/openvino/openvino.hpp').is_file():
            self.skipTest('OpenVINO wheel C++ headers and runtime library are unavailable')
        source = self.path / 'query_state_order.cpp'
        source.write_text(r'''
#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <functional>
#include <unordered_set>
#include <iostream>
namespace ov {
std::unordered_set<std::string> get_supported_nodes(
    const std::shared_ptr<const ov::Model>&,
    std::function<void(std::shared_ptr<ov::Model>&)>,
    std::function<bool(const std::shared_ptr<ov::Node>)>, float);
}
int main(int argc, char** argv) {
    ov::Core core;
    auto model = core.read_model(argv[1], argv[2]);
    auto names = ov::get_supported_nodes(model,
        [](std::shared_ptr<ov::Model>&) {},
        [](const std::shared_ptr<ov::Node>) { return true; }, .45f);
    size_t selected = 0, total = 0;
    for (auto node : model->get_ordered_ops()) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            total += constant->get_byte_size();
            if (names.count(node->get_friendly_name())) selected += constant->get_byte_size();
        }
    }
    std::cout << double(selected) / total;
}
''', encoding='utf-8')
        executable = self.path / 'query_state_order'
        subprocess.run([compiler, '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1',
                        f'-I{package / "include"}', str(source), str(libraries[0]),
                        f'-Wl,-rpath,{package / "libs"}', '-o', str(executable)],
                       check=True, capture_output=True, text=True, timeout=30)
        shares = [float(subprocess.run(
            [str(executable), str(xml), str(self.weights)], check=True,
            capture_output=True, text=True, timeout=10).stdout)
            for xml in (self.original_xml, self.normalized_xml)]
        self.assertEqual(shares, [1.0, .5])


if __name__ == '__main__':
    unittest.main()
