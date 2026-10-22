# Models, downloads and licenses

LazyRig is software for running models you obtain separately. Its source package
contains no model weights. The Apache License 2.0 for LazyRig does not license
third-party models, datasets, tokenizers, prompts, configurations, documentation
or generated content. User-supplied materials retain their applicable third-party
terms.

## Before downloading or using a model

Review the source repository's license, model card and applicable terms. You are
responsible for obtaining any required access or permissions and complying with
restrictions on use, commercial activity, modification and redistribution.
A publicly downloadable model is not necessarily open source or available for
commercial use. Different models from the same publisher may have different terms.

Importing, quantizing or converting a model does not remove its license conditions.
A supported model architecture or conversion profile means technical support;
it is not a legal review or permission to use the model. If terms are unclear or
missing, contact the publisher before relying on a particular permission.

For gated repositories, complete the publisher's access process. LazyRig's current
URL importer has no authenticated Hugging Face token flow. Obtain a permitted
compatible export separately and register its local folder if needed.

## Qwen Image 2.1 prompt enhancers

The current licenses for the specific
[Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/blob/main/LICENSE)
and
[Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE)
repositories were reviewed on 2026-10-04. Their Qwen Research License Agreement
defines non-commercial use as research or evaluation only and requires separate
permission for commercial use. Redistribution of covered materials or adaptations
also requires compliance with that agreement, including its license-copy,
modification-notice and attribution requirements. These notes do not describe all
Qwen releases; check the exact repository and revision you use.

These optional models and their upstream instruction files are not bundled with
LazyRig. Using LazyRig's built-in enhancement instructions does not change the
terms of any separately downloaded model or associated material.

## Documents kept with imports

For Hugging Face imports, LazyRig preserves common license, notice, copyright,
model-card and use-policy documents from the selected model subtree and its
ancestor directories, including common license directories. Documents are fetched
from the same resolved commit as the model and retain their repository paths under
`lazyrig-source-documents/` in the saved export. They survive source-to-OpenVINO
conversion. ZIP imports also preserve recognized documents outside a single
enclosing export folder. Existing files are not overwritten; a numbered documents
folder is used if necessary.

`lazyrig-export.json` records the source URL, resolved revision when available,
conversion information and paths to preserved documents. It does not declare a
model license or certify that every relevant term was found. File-name recognition
cannot capture every custom license file, external agreement or gated-access term;
review the original source as well. Existing local exports are not rewritten.

## Sharing or deploying models

Before sharing weights, converted exports, containers or a server installation,
check the terms covering everything you distribute. Preserve the required upstream
license and attribution files. Hosted use and rights in outputs can also depend
on the model's terms and applicable law. LazyRig does not grant or guarantee those
rights.

The import notice communicates this division of responsibility; it does not waive
third-party rights or replace obligations that apply to LazyRig's own code and
distributed dependencies. See [Third-party notices](../THIRD_PARTY_NOTICES.md).

## References

Reviewed on 2026-10-04:

- [Hugging Face repository licenses](https://huggingface.co/docs/hub/repositories-licenses)
- [Hugging Face terms of service](https://huggingface.co/terms-of-service)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
