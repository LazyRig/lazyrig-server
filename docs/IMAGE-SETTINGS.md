# Image settings and official reference values

Checked against the [Qwen-Image-2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1), its [scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json), the [OpenVINO notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen-image-2.1), and the native GenAI implementation on September 28, 2026.

## Dimensions

Qwen-Image-2.1's native OpenVINO pipeline requires twice the VAE spatial scale (16), giving 32-pixel alignment. Values are forwarded unchanged, not rounded.

| Aspect label | Width | Height |
| --- | ---: | ---: |
| 1:1 | 2048 | 2048 |
| 4:3 | 2400 | 1792 |
| 3:4 | 1792 | 2400 |
| 3:2 | 2528 | 1696 |
| 2:3 | 1696 | 2528 |
| 16:9 | 2752 | 1536 |
| 9:16 | 1536 | 2752 |

These are named presets, not exact rational aspect ratios or an exhaustive allow-list. The model card does not establish 2752×2752 as a hard maximum. LazyRig's maxima are configurable server limits; there is no separate area cap. Actual feasibility depends on the runtime, export and available memory.

Existing registrations keep their maxima. Click **Allow all listed preset sizes**, then **Save**, to raise both caps to at least 2752. This button preserves your default output dimensions. Selecting an individual preset changes the default width/height and raises either cap when needed. New Qwen registrations start with 2752-pixel caps with 2048×2048 output by default. Minimum 256 pixels is a LazyRig policy. Non-Qwen adapters retain the prior 64-pixel policy.

## Diffusion scheduler

**AUTO / Export default** is the recommended starting point. OpenVINO reads `scheduler/scheduler_config.json`. The settings section shows that file's values and permits supported overrides through `diffusion_scheduler`, separate from the LLM batching/KV `scheduler_config`.

Qwen-Image-2.1's official scheduler configuration:

| Setting | Reference value | Meaning |
| --- | --- | --- |
| Sampler | FlowMatch Euler | Qwen's native path needs flow-matching scheduler methods |
| Dynamic shifting | true | Shift schedule according to output image sequence length |
| Base / maximum shift | 0.5 / 0.9 | Shift interpolation values |
| Base / maximum image sequence length | 256 / 8192 | Interpolation anchors, **not hard context or resolution limits** |
| Static shift | 1.0 | Used when dynamic shifting is disabled |
| Terminal shift | 0.02 | End of the sigma schedule; `null` removes this override target |
| Training timesteps | 1000 | Training time scale, distinct from inference steps |

Blank named fields inherit the exported value. Advanced override JSON supports the same eight fields. To explicitly disable terminal stretching, set `shift_terminal` to `null`; clearing a named control instead inherits the export. Changing scheduler settings unloads only the affected image worker; subsequent loading/generation/editing applies a temporary merged scheduler file. Exported files are never rewritten.

Qwen's sampler selector allows AUTO or FLOW_MATCH_EULER_DISCRETE. Compatible classic diffusion exports additionally offer the native DDIM/Euler/PNDM/LCM selections; loading and inference validate compatibility, and changing the sampler changes the output. LCM generally requires a matching model. Unknown export families retain AUTO.

The checked Qwen path directly computes `calculate_shift(image_seq_len)` and calls `set_timesteps_with_mu`, which applies exponential shifting. Its generic `time_shift_type` field is visible in the export snapshot but is not an effective switch on that path. The checked FlowMatch parser also does not consume the Diffusers `use_karras_sigmas`, `use_beta_sigmas`, `use_exponential_sigmas`, `invert_sigmas` or `stochastic_sampling` flags. These flags remain in the source configuration; LazyRig rejects them as override controls rather than presenting ineffective options.

## Reference generation and runtime settings

**Use Qwen official defaults** selects AUTO with no scheduler overrides and uses the official model-card dimensions and step count: 2048×2048 output and 40 steps. LazyRig uses seed -1 (fresh noise per job); the example seed 42 is reproducibility data, not a quality recommendation. Width/height admission caps reset to 2752. Guidance, negative prompt and text-encoder limit are omitted so the exported pipeline supplies their defaults. There is no notebook-specific optimized generation preset. Save applies the reset. Existing saved custom settings remain until explicitly changed.

| Control | Native option / behavior |
| --- | --- |
| Inference steps | `num_inference_steps` |
| Guidance scale | `guidance_scale`; blank inherits the export/native default |
| Negative prompt | `negative_prompt`; disabled means omitted, enabled may be empty |
| Text encoder token limit | `max_sequence_length`, Qwen native default 8192; blank keeps native default |
| Seed | `-1` resolves to a random seed once per job; nonnegative values remain exact. `TorchGenerator(seed)` for Qwen; native RNG for other exports |
| Default width/height | Explicit `width`/`height`, including edits |
| Memory after requests | Automatic retention by default; keeps a successful worker until required restoration or another request needs the memory, with explicit keep/unload overrides |
| Runtime tuning | Existing device, compute-precision hint, performance hint, streams, CPU threads and compiled-model cache controls |

Qwen's extra CFG pass requires a supplied negative prompt and guidance greater than 1. Merely raising guidance without a negative prompt does not enable that pass in the checked implementation. The official reset leaves the negative-prompt checkbox disabled. An enabled empty string is different from omission. The token limit controls prompt encoding; scheduler sequence-length anchors control image latent timesteps.

The checked notebook uses the chosen device with the pipeline's default plugin settings. It does not establish universal NUM_STREAMS, CPU-thread or KV-quantization values for every image model or Intel GPU. Leave runtime overrides blank unless validating a specific device setting. Image diffusion settings do not accept LLM SchedulerConfig or MTP controls.

## Supported capabilities and limits

The dimensions, sampler, shift controls, steps, guidance, seed, negative prompt, text-encoder limit, runtime properties and worker-retention settings are exposed. Several upstream features require more than settings:

- The current Qwen OpenVINO image-edit adapter accepts one reference image and no separate mask. The model card's up-to-ten-image and broader editing features are not all available through this adapter.
- Explicit transparency/background selection, multiple outputs per request, LoRA workflows and component-by-component device/offload configuration are not exposed/validated as Qwen controls in LazyRig. The checked PyTorch CPU-offload example is not an OpenVINO plugin property.
- TaylorSeer is a generic GenAI generation option, but the checked Qwen native pipeline does not consume it; no Qwen acceleration toggle is advertised.
- Native automatic edit sizing follows the source aspect ratio when dimensions are omitted. LazyRig continues using saved/explicit dimensions so that its configured output caps remain enforceable.

The 2026-10-22 [tutorial captures](GETTING-STARTED.md#5-generate-an-image) show a returned Qwen image and VLM continuation, followed by a nighttime edit and completed VLM continuation on the operator's Windows/Intel setup. They are workflow examples, not a performance benchmark or broad quality assessment. Image-edit fidelity depends on the model, export, settings and target hardware; a completed job alone does not establish visual correctness.

## Noisy or unchanged Qwen edits

The chat image tool requests a fresh seed for each invocation, including chained edits. Newly registered Qwen models and the official-defaults reset use seed `-1`. Existing saved fixed seeds are preserved: use `-1` in model settings or the direct image/edit request for fresh noise, or supply a different fixed edit seed. The chat sampling seed does not control the chat image tool.

A Qwen contributor traced a closely matching haloed, oversaturated near-copy to reusing the generation noise for editing: the same RNG, seed and output shape can reproduce the original noise. The reporter confirmed that changing only the edit seed resolved that case. LazyRig previously used 42 by default for both operations. This is a strong explanation to test, not confirmation that every failed edit has that cause. A separate report describes degradation despite different seeds.

For a controlled check, edit the **original** source again with the same prompt, 1280×960 dimensions, 40 steps and existing scheduler/guidance settings, changing only the seed to `12345` (provided the original seed differed). If it succeeds, the preset was not the deciding variable. If it still fails, compare another seed and then a smaller target such as 768×576; keep the source image unchanged. OpenVINO's Qwen implementation resizes the conditioning image independently, so changing LazyRig's output size is not equivalent to changing every Diffusers `output_resolution` behavior. Do not silently resize user requests or add a second editing pass.

The native path performs one denoising loop followed by VAE decoding. LazyRig publishes the PNG after `generate()` returns and atomic encoding finishes, while still holding the device lease. A callback-reported early return now fails the job instead of publishing an artifact. `x_inference` records the runtime, actual seed, effective guidance/negative prompt, requested steps, callback step count, actual output dimensions and normal return. If callbacks are unavailable, their total remains unknown. `x_seed` and artifact metadata expose the resolved seed; idempotent retries retain it.

A completed job confirms a saved file, not visual compliance. The dashboard's tool result distinguishes the requested prompt from evidence. Vision-model continuations also receive the output pixels in an ordinary user image message after the text tool results. Text-only models receive the file report without being instructed to invent its appearance. Neither path guarantees semantic correctness or starts an automatic quality retry.

Related troubleshooting reports: [Diffusers #14824, author correction](https://github.com/huggingface/diffusers/issues/14824#issuecomment-5770745487), [Qwen contributor's explanation](https://github.com/huggingface/diffusers/issues/14824#issuecomment-5763057767), [separate unresolved report #14858](https://github.com/huggingface/diffusers/issues/14858), [ComfyUI #16607](https://github.com/Comfy-Org/ComfyUI/issues/16607). Checked 2026-10-03. These reports describe specific environments and do not establish a universal fix.


## Automatic GPU error recovery

During image generation/editing, `CL_OUT_OF_RESOURCES` and `CL_MEM_OBJECT_ALLOCATION_FAILURE` trigger one retry in a fresh worker, after the failed process has been confirmed stopped. LazyRig keeps the same source, effective prompt, resolved seed, dimensions and other settings. The dashboard and terminal show the recovery, and Response JSON records its outcome and original error. Stop and the original request deadline still apply. A repeated error ends the job; no automatic resolution reduction or retry loop is used. Successful image workers remain reusable under the selected retention policy.

This handles a recoverable failure without proving its cause. On 2026-10-18 an Arc Pro B60 edit failed before its first denoising callback after two successful edits. A later fresh worker completed two further edits and a third reached step 5/40 in the supplied log. The accompanying screenshot showed 23.5/24.0 GB dedicated memory usage in Task Manager. That is limited headroom, not proof of a memory leak or a particular driver defect. Native validation of the new automatic recovery path is still pending.
