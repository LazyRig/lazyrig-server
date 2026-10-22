# Model settings

Open **Models → Settings** for a registered model. The controls and advanced JSON edit the same stored properties. A blank control removes that override and leaves OpenVINO's default in effect. Saving requires an idle server. Display names, prompt templates, sampling defaults and idle timers reuse healthy resident workers. Changes to the model path, device, pipeline, native plugin/KV settings, MTP enablement or image scheduler unload only the affected worker; **Load** or the next request compiles it. Device budget/policy changes evict only the occupants needed to satisfy the new policy. Failed validation does not alter the saved configuration or unload a working model.

The model name or gear opens a foreground settings dialog. Its tabs group General,
Devices, Generation, Runtime and Advanced controls; **Save model settings** applies
the complete form. Cancel, Escape or clicking the backdrop discards unsaved edits.
Invalid controls reveal their tab before receiving focus. Folder browsing opens
above the settings dialog and preserves the draft beneath it.

Devices and API & server have their own gear buttons for settings. Conversation
parameters use the sliders icon in Chat and save automatically to that conversation;
**Done** closes the dialog. The Image tool chip toggles the current image tool. All
icon-only buttons have accessible labels and hover titles. Memory rings on Devices
show configured reservations, not live RAM or VRAM measurements.

## Device allocation and offload

CPU support follows the installed OpenVINO CPU plugin regardless of vendor.
Intel GPUs are LazyRig's supported graphics target. Other runtime-detected
accelerators remain selectable and are labelled **Experimental**,
including unvalidated NPU paths. Support labels describe project scope, not a
promise that every model works. Detection is not a model-compatibility test;
OpenVINO may enumerate non-Intel GPUs through OpenCL without providing CUDA or
all kernels required by a particular export.

For **GPU kernel compilation failed / CL_BUILD_PROGRAM_FAILURE**, run
`diagnose-gpu.bat` from the application folder and share
`logs/gpu-diagnostics.json`. On other systems, use
`python -m lazyrig.gpu_diagnostics`. It collects OpenVINO/GenAI versions, device
properties, OpenCL platform/driver/features, and the build logs of tiny FP32 and
FP16 kernels. It uses isolated children with a 45-second limit per check and
loads no model, reads no chats/settings, changes no runtime/driver, and downloads
nothing. A timeout or missing driver is recorded in the report. OpenCL platform
indices are independent of OpenVINO GPU identifiers.

These small compilation probes cannot certify Qwen inference. Their logs are
not the original model's compiler log, which some OpenVINO release builds omit.
Missing OpenCL FP16/subgroup features are possible compatibility limitations,
even on hardware supporting those operations through CUDA. If all probes pass,
the failing model kernel still needs investigation. LazyRig does not silently
change precision, expand weights, raise memory budgets, or disable devices by
brand. Whole-model CPU fallback requires CPU selected and fallback explicitly allowed in Automatic; it is
not an RTX GPU fix.

References:
- [OpenVINO 2026 GPU enumeration](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
- [A separate NVIDIA OpenCL FP16 failure report](https://github.com/openvinotoolkit/openvino/issues/33600#issuecomment-3758837071) — a possible limitation, not evidence about this machine.

The **Device** menu in each model row has checkboxes. Select the devices
allowed to run that model, choose the **Primary device**, and click **Apply**.
The same picker is available in Settings. A new model starts with **one device**.
Registration/download defaults prefer a detected Intel discrete GPU, then an
Intel integrated GPU, then CPU. Within a GPU class, larger configured capacity
wins. Registration through the API with no device also screens known export
size. This is a conservative default, not a benchmark or a guarantee of fit.
Explicit selections, including experimental devices and MTP devices, are kept.

Choose an execution mode in the device picker:

| Mode | Load policy |
| --- | --- |
| Automatic | Primary first, then eligible selected GPU/GPU or GPU/CPU splits. |
| Single device | Exactly one selected device; no alternate target. |
| Split across selected devices | One HETERO load using all selected devices, followed by the inference probe. No whole-model fallback. |

**Allow whole-model fallback** is off by default, including saved selections
that do not specify the policy. Enable it explicitly in Automatic to permit the
whole model on another selected GPU or CPU after split attempts. Selecting CPU
only for the MTP head does not enable main-model CPU fallback. No unselected
device is added. A missing primary blocks loading unless alternate fallback is
allowed. An explicit split requires every selected device to be detected.

An integrated GPU and CPU share system RAM. Automatic can try them as alternative
final stages; an explicit split accepts at most one of them. Native operation
support and OpenVINO memory estimates determine the partition. **Planning budgets
are not forwarded to the partitioner and do not enforce per-device weight sizes.**
Selecting a split therefore requests native partitioning, not a guaranteed 20 GiB
allocation on one GPU with the exact remainder on another.

With MTP enabled, the picker explains that splitting is disabled. Saving Split
mode with MTP fails before settings change. MTP is never disabled silently.
Manual percentage sliders remain retired; saved manual allocations migrate to
Automatic with the same devices and no percentage targets.

A healthy worker is reused, and a successful target is remembered until load-time
settings or a selected device's budget change, or the server restarts. Load
plans say when the remembered target is used first. A budget edit keeps a healthy
worker; unload and reload to retry the primary. Cancellation, deadlines and failed
cleanup stop retries. All attempts share the original load and request deadlines.

**Recent activity → Load plan & attempts** shows each candidate as not attempted,
skipped before native loading, loading, failed or loaded. It distinguishes weight
screening, MTP restrictions, known GPU compilation failures, inference-probe
failures and RAM-guard stops. Native errors are retained (up to 4096 characters
per attempt); Copy load details includes them. Mixed architectures alone never
produce an incompatibility diagnosis. The dashboard displays a bounded excerpt.

Every managed multi-device text load runs a short two-token text probe before
being marked loaded. It checks prefill and decode without using chat history or
invoking tools. A failed probe may advance automatic allocation after cleanup;
a later generation failure is not silently replayed. The probe does not certify
vision, MTP, long contexts or sustained GPU memory use.

OpenVINO pipeline parallelism is a preview feature. Splitting need not improve
decode speed: stages and transfers run in sequence. Different GPU architectures
alone do not prove incompatibility, and selecting multiple devices does not
force an even split or establish measured VRAM usage. Native model, runtime and
driver limitations still apply. This control does not implement tensor parallelism.

**Known Windows split failure:** an operator-reported Arc Pro B60 + Iris Xe
HETERO attempt compiled but crashed during the first generation probe with
`0xC0000005` in `igdrcl64.dll` (driver `32.0.101.9150`, reported OpenVINO/GenAI
2026.4.0). No probe token was produced. The supplied memory readings did not
show the configured host-RAM or process-commit guard being reached. Different
GPU architectures or driver branches are possible factors, not a confirmed cause.
A later dashboard showing a newer runtime does not establish that splitting is fixed.

Use **Single device** for the basic setup. For a failed split, expand
**Recent activity → Load plan & attempts → Copy load details** to retain the
runtime versions, native exit code, bounded Python fault trace and probe stage.
These diagnostics are not a native crash dump, and the failure remains unresolved.

References:
- [OpenVINO Heterogeneous Execution](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html)
- [OpenVINO CPU properties](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)
- [OpenVINO GPU properties](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)

Older saved placement reports remain readable through
`python -m lazyrig.placement_report --job JOB_ID`. Historical calibration and
assignment reports are retained as evidence; they do not enable manual mode.

During native loading, LazyRig samples available host RAM and stops the worker
at or below a 10% margin, bounded to 1–8 GiB (about 6.4 GiB on a
64 GiB machine). It checks before spawning and before accepting a ready worker.
Automatic placement may continue only after cleanup succeeds. Windows uses available physical RAM;
Linux uses `MemAvailable` and visible cgroup v2 limits. Unknown readings do not
block loading and are explicitly reported. Sampling applies to loading, not
ongoing generation, and cannot prevent overshooting between checks.

On Windows, a fresh available-RAM reading minus that margin also sets a native
Job Object committed-memory limit before the model worker starts. Windows
rejects job allocations that would exceed it. This limit remains active for
the worker's lifetime, including generation; it does not change client token
budgets. It is a process commit limit, not a cap on VRAM, every driver-managed
allocation or total machine usage. Without a valid RAM reading no commit limit
is installed. Other applications can still consume RAM after the limit is set.
The terminal's `load memory guard v2` entries report the installed limit and
lowest sampled free RAM. Actual protection against the reported RTX shared-RAM
growth still requires native validation; these checks do not establish its
cause or guarantee that a split will load.

The full-GPU weight check reads only the exported language `.bin` file size,
not the tensor contents. It compares against the configured GPU budget. It is conservative screening,
not a peak-memory estimate: compilation, reorders, caches and runtime buffers
can need more memory than the download size.

Two explicit controls can permit a primary GPU loading attempt that may use
shared system RAM:

- Raise **Devices → Settings → Planning budget (GiB)**. Values above dedicated
  VRAM remain valid, including the existing 48 GiB budget workaround.
- Enable **Allow primary GPU shared-memory attempt above its planning budget**
  in the model's device picker. In Automatic this follows eligible splits, and
  only when stored-weight screening blocked the ordinary primary attempt. In
  Single device mode it allows that one GPU attempt directly.

The second control permits an attempt; it cannot force driver-managed RAM use.
It does not retry a GPU that already failed native loading, disable protection,
or change physical memory. It books the whole primary planning budget to avoid
co-residency while allowing the explicit over-budget attempt. The ring still
reports a scheduling reservation, not the actual memory consumed.

A full-GPU reservation below the known language-weight size is flagged and
raised to that floor for admission, without rewriting the saved setting. Weights
are only a lower bound: cache, buffers and compilation need headroom. The model's
Memory reservation does not raise a device's planning budget. If a model is
already on a fallback device, unload and reload after changing the budget.

Default device budgets remain conservative. Native allocation errors, the
host-RAM loading guard, Windows worker commit limits, cancellation, timeouts and
cleanup requirements still apply. Explicit native `OFFLOAD_RATIO` settings
bypass the stored-weight screening. Shared GPU memory in Windows is system RAM
used by the GPU; it does not establish CPU execution. OpenVINO's detected dGPU
memory property reports dedicated VRAM and is not the combined dedicated/shared
pool shown by Task Manager.

The Devices rings display configured reservations for resident workers, not
measured memory consumption. A full ring can mean one worker has reserved the
whole device budget even while it is idle. Live GPU usage is not collected by
this dashboard; use the operating system's GPU memory counters to inspect it.

CPU/GPU splits use GenAI's stateful SDPA path. Continuous-batching
cache settings stay saved but are inactive for those workers. MTP stays under
user control: disable it for multi-device splitting; with MTP enabled, Automatic
can try selected devices individually only when whole-model fallback is explicitly allowed, including full CPU execution if selected. Image pipelines and NPU
models currently use one device. This feature does not add CUDA or guarantee
that a discovered device can execute a particular export.

Split workers conservatively reserve the configured budget of every participating
device, because GenAI does not expose measured per-device peak memory here.
The scheduler leases complete device sets atomically; overlapping work queues,
while independent pinned placements can run concurrently. Initial automatic
loads reserve the full allowed fallback pool. Model handoffs and restoration
also reserve every device they may touch. Runtime placement properties supplied
manually in advanced JSON remain intact and disable the managed picker until
removed; there are not two competing sources of placement settings.

## CPU inference threads

Under **Devices → CPU**, choose **Auto** or a custom **CPU inference threads**
limit and save. Auto leaves thread scheduling to OpenVINO; it does not mean
one thread. A custom positive whole number limits CPU inference resources for
CPU-only models and the CPU portion of GPU/CPU placements. It is not a target
CPU-utilization percentage or a promise that every thread stays busy. More
threads do not necessarily improve single-request decode speed.

Explicit per-model CPU thread settings take precedence over this device
default. The existing **Models → Settings → OpenVINO runtime → CPU inference
threads** control remains available. Saving a device default requires an idle
server and unloads affected CPU workers when their effective setting changes;
GPU-only workers are retained. Load the model again to apply the new limit.

Thread count and inference streams are separate: a single stream can use
multiple threads. See [OpenVINO CPU thread scheduling](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html).

Runtime references: [OpenVINO HETERO execution](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html),
[2026.4.1 HETERO partitioning](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/src/plugins/hetero/src/plugin.cpp),
and [the approximate weight-boundary algorithm](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/src/inference/src/dev/iplugin.cpp).

## Memory reservation defaults

Add model and Imports fill **Memory reservation (GiB)** with the selected device's
usable memory budget. Selecting another device refills the number; **Default**
restores that device's budget after a manual edit. Model Settings provides the
same controls. Existing saved reservations are preserved until edited or the
model moves to another device. The Models table's device selector also applies
the new device's default.

New devices use 85% of detected GPU memory or physical CPU RAM, rounded down to
0.1 GiB. Saved device budgets are respected, including custom values. If memory
cannot be detected, the existing fallback is 8 GiB for CPU or 2 GiB otherwise;
review the device budget in that case. Changing a device budget does not rewrite
saved model reservations; use **Default** and save to adopt it.

Using the whole usable budget is a conservative starting reservation: one model
can remain resident on that device at a time, without requiring a guessed model
peak. It is not a measurement or cap on actual RAM/VRAM and does not guarantee
that an export fits. To keep multiple models loaded together under the budget
policy, enter measured peak reservations that cover weights, caches and runtime
buffers. Model size on disk alone does not establish peak runtime memory.

## Text sampling and Qwen3.8 presets

The default output token budget is **8,192** when neither the model nor the
request specifies one. It includes reasoning, tool calls and the visible answer.
A saved model budget or an explicit conversation/API value still takes precedence;
clear an override to inherit the default. The configured context limit still
covers input plus output, and context-overflow checks remain in effect.
On the first upgrade, the old 1,024-token model default saved by earlier Settings
forms becomes 8,192 when the configured context exceeds 8,192. Other saved
limits, smaller-context models and conversation-specific overrides are preserved.
After that upgrade, explicitly setting 1,024 again is respected across restarts.

Image models also have **Use as default image tool** beside the API checkbox.
New dashboard chats select that model automatically. Choosing another model
clears the previous preference. If no preferred model is enabled, new chats
use the first enabled image model; if none exists, the image tool is disabled.
Existing conversations retain their saved selection, including Disabled.
This preference does not enable server-managed tools on ordinary API requests.

The sampling controls support temperature, top-p, top-k, min-p, presence penalty, frequency penalty and repetition penalty. Request values override saved model settings. Unset values inherit the exported pipeline's generation configuration; temperature has LazyRig's existing fallback of 0.7, or 0 with MTP. Blank fields do not introduce the former settings-panel defaults of top-p 0.9 and top-k 40. Clearing a saved field removes its override.

Select a **Qwen3.8 sampling preset**, then save. These values come from the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Parameter | Thinking | Instruct / non-thinking |
|---|---|---|
| `temperature` | 1.0 | 0.7 |
| `top_p` | 0.95 | 0.80 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 0.0 | 1.5 |
| `repetition_penalty` | 1.0 | 1.0 |

The preset also sets `template_context.enable_thinking` to the matching mode and resets frequency penalty to its neutral value of 0. Presets are explicit saved defaults; changing thinking mode in a client does not automatically change the sampling settings. A client that selects another mode should send the matching sampling values. Existing settings are not migrated or overwritten during an upgrade, and the presets are not automatically applied to other models.

**MTP remains under your control.** The installed OpenVINO GenAI 2026.4 Qwen MTP implementation supports greedy decoding only. Selecting a preset leaves MTP as it was and explains the conflict if enabled. To use Qwen's recommended sampled temperatures, explicitly disable MTP before saving. To retain MTP, keep temperature 0; that is a different decoding configuration from these recommendations. This is a backend restriction, not an inherent property of an MTP layer.

Native responses include **`x_sampling`**, containing the effective values after model defaults and request overrides, plus `source: "openvino"` and `do_sample`. Greedy decoding reports effective temperature 0; unlimited top-k is normalized to 0. The same data is saved as `sampling` in the job and is available through its result endpoint. Jobs that have not run native inference have no native snapshot. These diagnostics help investigate repetition, but matching a preset does not establish or guarantee a quality fix.

## Image dimensions and diffusion scheduler

See [Image settings and official defaults](IMAGE-SETTINGS.md) for Qwen-Image-2.1 aspect-ratio presets, its 32-pixel alignment, the diffusion sampler and shift controls, and the Qwen official-default reset. The image `diffusion_scheduler` settings are separate from the text model's `scheduler_config` / KV cache.

## Image memory retention

**Memory after image requests → Image model retention → Automatic (default)**
keeps a healthy image worker loaded between consecutive requests, including
after it displaced a text model. A displaced model no longer consumes a
reservation unless it must be restored immediately. A later request for another
model evicts the image worker only when the device budget or exclusive policy
requires it. Explicit immediate text restoration and managed chat continuation
can still release the image worker after saving the result when both models
cannot coexist. Unrelated devices and unused registrations do not count.

| Option | After a successful generation or edit |
| --- | --- |
| Automatic (default) | Keep the worker unless models needed immediately cannot share the device |
| Unload after each request | Always close the image worker after saving the result |
| Keep loaded between requests | Retain the worker until another request needs its memory, an idle timeout applies, or it is explicitly unloaded |

The API setting is `keep_image_model_loaded`: `"auto"` (default), `false` (unload)
or `true` (keep). It is a model setting, not an inference-request parameter:

```json
{"model":"my-image-model","keep_image_model_loaded":"auto"}
```

Send this to `POST /api/settings`. Newly registered models and configurations
without the field use Automatic. Existing saved `true` or `false` selections are
preserved; select Automatic once for an older model to adopt the new behavior.

Newly detected devices default to **Keep within memory budget**. An existing
**Unload on switch** device policy still explicitly permits only one loaded
model. Change that device policy if you want Automatic to keep several models
together. Decisions use the configured device budget and model peak reservations,
not measured free RAM/VRAM. Include weights, cache and runtime working memory,
and leave room for the OS and other applications.

**Text model after images** remains a separate choice. **Load on next request**
is the default for text models that were displaced; models on another device or
still resident within budget stay loaded. **Restore previous text models**
reloads displaced models before the image job finishes. Restoration still obeys
the device budget and may evict the image worker. Managed chat loads its text
model when it needs to continue the conversation.

Stopping text generation requests cooperative cancellation. After the native
pipeline returns and acknowledges the stop, LazyRig keeps the worker loaded
for the next message. If no acknowledgement arrives within five seconds, or
the worker fails, LazyRig terminates it. Stopping during a long prefill or a
stalled native call may therefore still unload the model. A successful Stop
logs `generation stopped | model kept loaded`. Changing MTP mode/device or
other load-time settings still requires a rebuild; logs name the changed keys.

Failed or cancelled image work still closes the affected worker. Manual unload,
an operation switch between generation and editing, and an idle timeout can
also release it. A zero idle timeout disables timed unloading. Changing retention
alone does not recompile a healthy worker; the new choice applies to subsequent
requests.

Idle workers release the previous request's input/output tensors and random
generator references. A retained native runtime may cache compiled models and
working allocations. Process exit releases that worker's owned allocations;
saved input/output files remain available on disk.

On Windows, the actual model interpreter joins a server-owned Job Object before loading weights. Its descendants are included, and the server checks for zero active members during unload. The operating system also closes that ownership when the server exits. On POSIX, model workers have private process sessions and cleanup kills remaining members after stopping the leader. Process handles are closed. A failed cleanup remains visible as `cleanup-failed` and blocks reuse of that device until cleanup succeeds.

`GET /api/state` exposes `scheduler.workers` with model, device, native PID and operation; `loaded`/`unloaded` job events include `worker_pid`; unload events and terminal metrics also report the reason. This is ownership information, not live VRAM measurement. The main server, an intentionally resident LLM, and reclaimable OS file cache can remain in RAM after an image worker exits. Do not end every `python.exe`: other Python applications may be unrelated. If upgrading while old model processes remain, first stop the old server and use Task Manager's command-line column to identify only that installation's model workers.

## Image editing

Image models have an **Image editing** section showing capability detection and any missing components. A complete Qwen-Image-2.1 export must contain XML/BIN pairs for `vae_encoder`, `vision_encoder`, and `text_encoder_i2i`, alongside the generation components. The existing Download / convert workflow retains those components when included by the compatible exporter. If they are missing, obtain or convert a complete export; a settings switch cannot create them. Native runtime/device compatibility is checked when the edit pipeline loads.

Use the same alias at `/v1/images/generations` and `/v1/images/edits`. The server changes the native pipeline automatically, unloading the previous worker first. The **Load** button loads the generation pipeline; the first edit switches it to editing. Both modes use the selected device and plugin settings. Memory reservation must cover the larger of the generation/editing peaks, including editing's extra encoders and working memory. A text model displaced by an edit reloads when next requested. Enable **Text model after images → Restore previous text models** for eager restoration instead.

Qwen-Image-2.1 uses source-image conditioning and has no denoising-strength control. Compatible classic diffusion models expose **Default edit denoising strength**, saved as `generation.strength`, in `(0, 1]` (default 0.8). This is separate from guidance and inference steps. One source image is supported, without a mask. The **Chat → Edit image** task sends the selected image to the real edit API.

Omitted API `size` or `size: "auto"` uses the model's saved output dimensions. Hermes may request 1536×1024 or 1024×1536 for landscape/portrait output. To permit this, raise **Maximum width/height** and the memory reservation as appropriate, or ask Hermes for square 1024×1024 output. The server rejects requests beyond configured maxima instead of silently resizing them.

## Vision pipeline

Choose **VLM · text + images** for a complete vision-language OpenVINO export. The vision encoder, text embeddings, language model and processor configuration must be present; selecting VLM cannot add vision to a text-only export. Existing `vlm-text` settings migrate to `vlm`. The API advertises `input_modalities: ["text", "image"]` and accepts image parts in user messages and tool results. Use **Chat → Image attachment** to test the actual image path. See [API.md](API.md) for request examples and limits.

## Multi-token prediction (MTP)

The settings panel detects a bundled OpenVINO MTP head: both `openvino_mtp_model.xml` and `openvino_mtp_model.bin` must exist in the model directory. Detection makes the enable option available; the native runtime validates architecture/device compatibility during Load. A source architecture's MTP layer does not establish that an existing export contains it. Download a compatible export containing the head or export it using that model's supported OpenVINO procedure. The importer preserves an existing export's XML/BIN files; this setting does not convert missing MTP weights.

Enable **MTP decoding**, choose **MTP device**, then set **Draft tokens per step** (integer 1–64, default 2). **Same as model** is the default and follows the main model's resolved device, including automatic fallback. Choose a configured CPU or GPU to run the bundled head there explicitly. This does not change the main model's device or select another model. The console visibly changes the model temperature to **0** and prefix caching to **Disabled** before saving. Frontend requests must also use temperature 0; conflicting requests fail with an explanatory error. MTP and `prompt_lookup` cannot be combined. Enabling/disabling MTP or changing its device unloads the worker; **Load** or the next request applies the new configuration. Changing only the draft-token count reuses a healthy worker.

With MTP enabled, LazyRig currently loads the main language model on one device
and skips automatic multi-device splitting. Moving the MTP head does not offload
the main model's weights. If those weights exceed the primary GPU's configured budget,
raise that device's budget to permit a shared-memory loading attempt where
supported, choose a smaller model or a primary device with enough memory, or select CPU in
the **main model's device allocation** and enable whole-model fallback. Selecting
CPU only in **MTP device** does not permit that main-model fallback. Disable MTP
to try automatic model splitting. Runtime, cache and head memory still need
headroom beyond the main weights; increasing the configured budget does not
increase physical memory.

This follows the [official OpenVINO Qwen3.8-27B example](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov): `genai.draft_model(model_path, device)` is supplied to the LLM/VLM constructor, `SchedulerConfig.enable_prefix_caching` is false, and generation uses greedy decoding with `num_assistant_tokens`. The [native draft-model API](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.draft_model.html) accepts the head's selected device. Native architecture/device compatibility is still checked during loading and generation; CPU/GPU routing tests do not establish support or speed gains for every hardware pair. Managed multi-device SDPA splitting still requires MTP disabled.

LazyRig reserves both participating devices for the request. A separate head device reserves its full configured budget while resident, because the head also needs output-layer weights and runtime cache; its BIN file alone is not a peak-memory estimate. This is scheduler accounting, not an allocation of all that memory. On the same device, include head/cache overhead in the main model's reservation. A CPU head uses the Devices CPU inference-thread default unless explicitly overridden in the model's properties. A different-device head uses its own latency preference instead of inheriting GPU-only main-model properties. Loading logs and worker state report the selected MTP device.

For API configuration, send to `POST /api/settings`:

```json
{
  "model": "my-vlm",
  "mtp": {"enabled": true, "num_assistant_tokens": 2, "device": "CPU"},
  "generation": {"max_tokens": 8192, "temperature": 0},
  "scheduler_config": {"enable_prefix_caching": false}
}
```

These objects replace previous values, so include other settings you want to keep. Omit `mtp.device` or set it to `""` for **Same as model**. Per-request `num_assistant_tokens` overrides the saved count only when MTP is enabled; it survives retries and image-generation handoffs without changing model defaults. Responses report applied generation settings in `x_mtp`. The head's device is a saved load-time setting, not a request parameter. Set `mtp.enabled` to false to disable MTP; choose any desired temperature/prefix settings separately afterward.

## KV cache

| Control | OpenVINO property | Values / meaning |
|---|---|---|
| KV-cache precision | `plugin_config.KV_CACHE_PRECISION` | INT8 → `u8`, INT4 → `u4`; also `f16`, `bf16`, `f32`, `undefined` |
| Key / value precision override | `KEY_CACHE_PRECISION`, `VALUE_CACHE_PRECISION` | Same values; override the shared setting for each cache |
| Key / value group size | `KEY_CACHE_GROUP_SIZE`, `VALUE_CACHE_GROUP_SIZE` | Nonnegative integers; layout and head-dimension compatibility are checked by OpenVINO |
| Key quantization mode | `KEY_CACHE_QUANT_MODE` | `BY_CHANNEL` or `BY_TOKEN` |
| Initial attention cache | `scheduler_config.cache_size` | Whole GiB, including KV and linear-attention state; 0 or omitted allocates on demand. Not a hard RAM/VRAM cap |
| Reuse prompt prefixes | `enable_prefix_caching` | Boolean; omitted keeps the native default |
| Tokens per scheduler step | `max_num_batched_tokens` | Positive integer |
| Sequence limit | `max_num_seqs` | Positive integer; blank defaults to **1**, matching the one active request per model worker. Higher values do not enable concurrent requests in LazyRig |
| Hybrid checkpoint interval | `cache_interval_multiplier` | Optional nonnegative integer multiplier; 0 requires prefix caching disabled |
| Dynamic split / fuse | `dynamic_split_fuse` | Boolean |

**Device support:** these are real runtime properties, not guarantees that every device/model supports every value. Intel documents GPU INT4 KV compression with `u4` from OpenVINO 2026.2; support and precision defaults change with runtime, device and attention backend. The settings catalog leaves final device/model validation to OpenVINO during Load instead of silently replacing your choice. Separate key/value overrides can supersede the shared precision, so clear them when you want one precision for both. INT4 can affect output quality; compare it with INT8 for your model.

The supported Python GenAI binding takes an integer `cache_size`; fractional GiB are rejected before saving. Advanced JSON may use `num_kv_blocks` for a smaller explicit allocation, or `num_linear_attention_blocks` for supported hybrid models. Do not set both nonzero `cache_size` and `num_kv_blocks`. The hybrid checkpoint multiplier trades memory for prefix reuse granularity and has no effect on models without linear-attention state. Leave it blank to retain the runtime's adaptive value.

For hybrid models such as Qwen3.8, a fixed cache allocation with prefix caching
disabled reserves a full recurrent state per scheduler sequence. GenAI's bare
`SchedulerConfig` defaults to 256 sequences; inheriting that default can reject
even a single request with `Requested linear attention cache allocation exceeds
the configured cache size`. LazyRig now defaults this field to 1 whenever it
constructs a scheduler configuration, including MTP. Explicit sequence and block
counts are preserved. This changes neither the client output-token budget nor
the model's context limit. Native-default and SDPA paths are not switched to
continuous batching by this change.

If that error persists, set **Scheduler sequence limit** to 1 or blank and clear
an explicit `num_linear_attention_blocks` override if present. Leave **Initial
attention cache** blank or at 0 for allocation on demand, or choose an explicit
allocation that fits the state and available memory. The native error, supplied
cache settings and actionable guidance are retained in the saved job. This
configuration error does not trigger device fallback or an automatic increase
in memory allocation. GenAI 2026.4.1 can grow the KV pool beyond its initial
`cache_size`; neither a nonzero value nor automatic allocation is a hard memory
limit. See the [upstream cache-allocation report](https://github.com/openvinotoolkit/openvino.genai/issues/4481).

The memory reservation must include weights, KV memory and inference overhead. Cache size, the configured context-token limit, and measured memory are different quantities. LazyRig reports prompt/generation tokens with their count accuracy; visual input totals become available after native generation. It does not measure live KV occupancy or free VRAM. Prefix reuse lasts while the worker remains loaded. Unloading an LLM to make room for an image model destroys that worker's cache; the resumed LLM rebuilds it from saved history and input images.

KV precision affects inference state. Model **weight** INT4/INT8 compression is selected during **Download / convert**, not by this runtime control. No export files are rewritten when these settings change.

## Runtime controls

| Control | Property | Meaning |
|---|---|---|
| Performance preference | `PERFORMANCE_HINT` | Native latency / throughput preference |
| Compute precision hint | `INFERENCE_PRECISION_HINT` | FP16, BF16, FP32 or automatic; separate from weight and KV precision |
| CPU threads | `INFERENCE_NUM_THREADS` | Positive integer; applicable to CPU |
| Streams | `NUM_STREAMS` | Positive integer, `AUTO`, or CPU `NUMA`; device support varies |
| Compiled-model cache directory | `CACHE_DIR` | Server-side folder for compiled artifacts; separate from conversation KV state |
| Execution preference | `EXECUTION_MODE_HINT` | Performance or accuracy optimization preference; does not undo weight quantization |
| GPU compilation threads | `COMPILATION_NUM_THREADS` | Positive limit for GPU compilation; does not control decode threads |
| CPU core preference | `SCHEDULING_CORE_TYPE` | Any, performance or efficiency cores where supported |
| CPU hyper-threading | `ENABLE_HYPER_THREADING` | Use logical sibling threads for CPU inference where supported |
| Pin CPU inference threads | `ENABLE_CPU_PINNING` | Request CPU thread affinity where runtime/OS support it |

CPU scheduling controls are grouped under **CPU scheduling**. The new CPU-only
settings apply to the main model's CPU execution/offload, not the MTP head.
GPU compilation threads apply only to executing GPUs. In mixed plans LazyRig
routes these controls to the matching plugin; saved controls are inactive if
that device kind is not used. Existing per-device JSON overrides take precedence.
Blank controls retain runtime defaults. More threads or streams do not guarantee
faster decoding and can increase memory use; workers still run one request at a time.

These common controls are also shown for image models. Text-only scheduler controls are hidden for images. Advanced plugin JSON remains available for additional device properties; the server forwards them to OpenVINO. Invalid known values/types fail before settings are committed; unsupported native properties or combinations fail on model load with the native error.

## API example

Send to `POST /api/settings` with the configured API key:

```json
{
  "model": "my-chat-model",
  "plugin_config": {
    "KV_CACHE_PRECISION": "u4",
    "PERFORMANCE_HINT": "LATENCY"
  },
  "scheduler_config": {
    "cache_size": 4,
    "enable_prefix_caching": true,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 1
  }
}
```

This is a property example, not a measured memory recommendation. JSON objects replace previous objects, so retain any other properties you need. Send `{}` to remove all overrides in an object. The administration UI preserves unexposed plugin properties when you change individual controls.

## Keep multiple models loaded

Under **Devices → Model residency**, **Keep within memory budget** is the default
for newly detected devices. Set a usable RAM/VRAM budget and reserve each model's
peak memory, including weights, KV cache and inference buffers. Compatible LLM
and image workers stay loaded together when those reservations fit. Requests
remain one at a time per device; admission uses reservations, not live memory
measurements. **Unload on switch** remains available for an explicitly exclusive
device.

For images, **Automatic** is the default. It works with standalone generation,
edits and managed LLM → image → LLM handoffs. A healthy LLM that remained resident
resumes on the same worker. Under memory pressure, the scheduler evicts idle
workers by least recent use. Automatic evaluates resident models and those
required for immediate restoration or continuation, not models already evicted
and waiting for a future request. Changing output dimensions alone does not
reload a compatible worker. Explicit
**Keep loaded between requests** and **Unload after each request** remain
available. Manual unload, timeouts, failures and pipeline changes still apply.
