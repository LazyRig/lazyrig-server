# Third-party software and trademarks

LazyRig's [Apache License 2.0](LICENSE) applies to its original code, documentation
and bundled enhancement instructions. Dependencies, downloaded models and other
third-party material retain their own licenses.

## OpenVINO

LazyRig uses the [OpenVINO™ toolkit](https://docs.openvino.ai/) and OpenVINO GenAI
for inference. Their upstream projects are licensed under Apache License 2.0:

- [OpenVINO license](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)
- [OpenVINO GenAI license](https://github.com/openvinotoolkit/openvino.genai/blob/master/LICENSE)
- [OpenVINO third-party notices](https://github.com/openvinotoolkit/openvino/blob/master/licensing/third-party-programs.txt)
- [OpenVINO GenAI third-party notices](https://github.com/openvinotoolkit/openvino.genai/blob/master/third-party-programs.txt)

This source archive does not bundle those runtime packages. Setup downloads
packages separately. If you distribute an installation, container, runtime bundle
or modified upstream code, retain the license texts, copyright statements and
applicable third-party notices from the exact versions you distribute. A UI
acknowledgement or link is not a substitute for those files.

OpenVINO is a trademark of Intel Corporation or its subsidiaries.
LazyRig is an independent project and is not affiliated with or endorsed by Intel.
The OpenVINO name identifies the runtime; no Intel logo is included.

Trademark references follow Intel's [usage guidance](https://www.intel.com/content/www/us/en/trademarks/trademarks.html)
and [OpenVINO trademark entry](https://www.intel.com/content/www/us/en/trademarks/trademark-list.html),
reviewed on 2026-10-04.

## Other dependencies

The `requirements-*.txt` files list dependencies for the included package profiles.
Installed packages and their transitive dependencies
retain their upstream terms. Use the license and notice files supplied with the
versions you install or redistribute; this document is not an inventory of every
possible dependency environment.

Windows setup may also download [uv](https://github.com/astral-sh/uv) and a private
Python runtime. They are not bundled in this source archive. See uv's
[Apache-2.0](https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE) and
[MIT](https://github.com/astral-sh/uv/blob/main/LICENSE-MIT) license files and
[Python's license documentation](https://docs.python.org/3/license.html).

For redistributed Apache-2.0 components, include the license, retain applicable
copyright, patent, trademark and attribution notices, preserve relevant upstream
NOTICE content, and mark modifications to upstream files. Other dependency
licenses may have different requirements. The licenses and notices shipped with
the exact redistributed versions are authoritative.

## Models

No model weights are included. Downloaded models, tokenizers, model cards and
associated materials, including user-supplied prompts, configurations and
documentation, retain their upstream terms. Compatibility adapters and conversion
profiles do not endorse a model or grant rights to it.
See [Model licenses](docs/MODEL-LICENSES.md) for user responsibilities and import
notice preservation.
