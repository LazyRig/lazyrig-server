'use strict';
window.Help = (() => {
const $$ = selector => [...document.querySelectorAll(selector)];
// Help is deliberately opt-in. Plain labels never generate their own explanations.
const fieldHelp = {
  budget_gib: 'Planning target for scheduling reservations and the stored-weight loading check. It is not passed to OpenVINO as a partition size. You can set a GPU budget above dedicated VRAM to permit shared-system-memory loading when the driver supports it. This is not measured usage or a hard allocation limit. Leave room for runtime caches, the operating system and other applications. After changing it, unload and reload a model already using a fallback device to retry the primary GPU.',
  policy: 'Keep within memory budget retains models while their reservations fit and evicts the least recently used model when needed. Requests still execute one at a time per device.',
  restore_text_after_image: 'Reloads text models displaced by this image request before releasing the device. Leave off to load them only when a client next requests text; consecutive image jobs then avoid unnecessary swaps.',
  keep_image_model_loaded: 'Automatic keeps a successful image model loaded for the next request. Displaced models do not count unless they must be restored immediately. It releases the image worker when required restoration or chat continuation cannot share the device. Unload always releases it after a request; Keep retains it until eviction or idle timeout. Device budget and Unload on switch rules still apply when another model is requested.',
  frequency_penalty: 'Discourages tokens according to how often they have appeared. 0 disables this penalty.',
  mtp_enabled: 'Requires an exported MTP head. The current runtime uses temperature 0 and disables prefix caching while MTP is active. Splitting the main model is unavailable with MTP; this choice is never changed automatically.',
  mtp_device: 'Runs the bundled MTP head on this CPU or GPU. This does not offload the main model, which must fit on one device while MTP is enabled. Same as model follows automatic fallback. CPU must be selected and whole-model fallback explicitly allowed for a full CPU fallback. A separate head device is reserved while loaded; native compatibility is checked during loading. CPU uses its configured thread setting.',
  mtp_num_assistant_tokens: 'How many tokens the MTP head proposes per decoding step. Speed depends on how many proposals the main model accepts.',
  max_width: 'Admission limit for image requests. Raising it does not change the model’s supported resolutions.',
  max_height: 'Admission limit for image requests. Raising it does not change the model’s supported resolutions.',
  image_negative_prompt_enabled: 'Sends the saved negative prompt, including an intentionally empty one. Requires guidance greater than 1.',
  max_sequence_length: 'Token limit for the image model’s text encoder. Leave blank to keep its native setting.',
  image_scheduler_type: 'AUTO uses the scheduler bundled with the model export. A different sampler can change image quality.',
  image_scheduler_json: 'Overrides are merged into a temporary copy of the exported scheduler configuration. Leave {} to preserve the exported values.',
  use_dynamic_shifting: 'Adapts the denoising schedule to image sequence length using the exported shift parameters.',
  shift: 'Schedule adjustment used when dynamic shifting is off. Start with the exported value.',
  base_shift: 'Shift value at the base image sequence length. The scheduler interpolates between the base and maximum anchors.',
  max_shift: 'Shift value at the maximum image sequence length. Keep the export default unless tuning the denoising schedule.',
  base_image_seq_len: 'Lower image-latent sequence anchor for dynamic shifting. It does not limit prompt length.',
  max_image_seq_len: 'Upper image-latent sequence anchor for dynamic shifting. It does not set output dimensions.',
  shift_terminal: 'Target sigma at the end of the denoising schedule. Leave blank to inherit the export.',
  num_train_timesteps: 'Resolution of the schedule used during training. Keep the exported value when choosing inference steps.',
  plugin_config: 'Device-specific OpenVINO options. Controls and JSON stay synchronized; model loading checks device support.',
  scheduler_config: 'Native scheduling and KV cache options. These do not change LazyRig’s one-request-at-a-time execution per device.',
  template_context: 'Variables passed to the model’s chat template, including enable_thinking. Sampling settings are configured separately.',
  KV_CACHE_PRECISION: 'Lower precision reduces conversation-cache memory. Availability depends on the device and model.',
  KEY_CACHE_PRECISION: 'Overrides precision for the key cache only. Leave blank to inherit the shared KV-cache setting.',
  VALUE_CACHE_PRECISION: 'Overrides precision for the value cache only. Leave blank to inherit the shared KV-cache setting.',
  KEY_CACHE_GROUP_SIZE: 'Number of key-cache values sharing quantization parameters. Leave blank for the runtime default.',
  VALUE_CACHE_GROUP_SIZE: 'Number of value-cache values sharing quantization parameters. Leave blank for the runtime default.',
  KEY_CACHE_QUANT_MODE: 'Chooses whether key-cache quantization parameters are grouped by channel or by token. Device support is checked during loading.',
  enable_prefix_caching: 'Reuses cached prefixes shared by requests. The cache is lost when its worker unloads.',
  max_num_batched_tokens: 'Maximum tokens processed in one native scheduler step. Smaller batches can reduce peak work but slow prompt processing.',
  max_num_seqs: 'Internal OpenVINO sequence limit. It does not increase LazyRig’s per-device request concurrency.',
  dynamic_split_fuse: 'Lets the native scheduler split prompt processing across steps and combine it with decoding work.',
  PERFORMANCE_HINT: 'Latency favors individual response time; throughput favors sustained processing. LazyRig still schedules one request per device.',
  INFERENCE_PRECISION_HINT: 'Requests a compute precision from the device plugin. It does not convert the saved model weights.',
  'sampling-preset': 'Fills Qwen’s recommended sampling values and thinking mode. Save to apply; MTP remains a separate choice.',
  'image-size-preset': 'Fills official Qwen dimensions. Save model settings to apply them.',

  host: '127.0.0.1 accepts connections from this computer. 0.0.0.0 also accepts connections through its network interfaces.',
  request_timeout_seconds: 'Maximum wait for an inference worker to finish. Slow image generations may need a longer limit.',
  browser_access: 'Allow listed browser origins, or inference from any frontend with the API key. Administrative requests still require a permitted origin.',
  allowed_origins: 'Use the address of each web frontend, including its scheme and port, for example http://localhost:3000.',
  conversion: 'Automatic uses an existing OpenVINO export when available, or converts the source weights. Exports only skips conversion.',
  precision: 'Lower precision can reduce exported model size and memory use, with a possible loss of quality. Conversion itself can still need substantial RAM.',
  id: 'The name clients send in the API model field. Keep it stable when replacing the model files.',
  memory_gib: 'Filled automatically from the selected device’s usable memory budget. This conservative default keeps one model resident on that device. Changing the device refills it; Default restores it. You can enter a measured peak (weights, cache and buffers) to allow models to share the device. This is a scheduling estimate, not measured usage or a memory limit. Full-GPU loads use at least the known language-weight size. Splits and explicit over-budget attempts book the participating device budgets conservatively; native usage can differ.',
  revision: 'Use a branch, tag or commit ID. A commit ID pins the download to an exact repository version.',
  subfolder: 'Import files from this folder within the repository. Leave blank to use the repository root.',
  profile: 'Select the conversion recipe for the model architecture. Keep automatic detection unless the model requires a specific recipe.',
  context_limit: 'Total allowance for the prompt, conversation history and response. Raising this value does not extend the model’s supported context.',
  pipeline: 'Choose the runtime that matches the exported model. Use Auto-detect unless detection selects the wrong one.',
  max_tokens: 'Includes reasoning and tool calls as well as the visible answer. Defaults to 8,192 unless the model or conversation overrides it. A small limit can cut off a response before it finishes.',
  default_image_tool: 'Use this image model for new dashboard chats. Choosing it replaces the previous default. If none is chosen or it is disabled, new chats use the first enabled image model. Existing conversations keep their saved choice.',
  thinking: 'Enables the model’s reasoning mode when supported. Reasoning uses the output token budget and can increase response time.',
  temperature: 'Higher values produce more varied responses; lower values make sampling more predictable. Blank uses the model default.',
  top_p: 'Narrows sampling to a set of likely tokens. Lower values reduce the choices; 1 disables this filter.',
  top_k: 'Considers only this many highest-probability tokens at each step. 0 disables this filter.',
  min_p: 'Filters out tokens that are much less likely than the leading candidate. 0 disables this filter.',
  presence_penalty: 'Positive values discourage tokens already used in the text and can encourage new topics. 0 disables this penalty.',
  repetition_penalty: 'Values above 1 discourage repeated tokens. 1 disables this penalty.',
  image_model: 'Enables generation and supported image edits for this conversation. Uploaded and previously generated images become editable sources. A text LLM can direct an edit; a vision model can also inspect the image.',
  size: 'Automatic lets the image tool or prompt enhancement choose dimensions, falling back to the image model default. A preset or Custom dimensions fixes the output size for generation and edits, overriding tool arguments and enhancement. Unsupported sizes are rejected, never silently resized.',
  enhance_prompt: 'Rewrites image requests using LazyRig\'s general generation or editing instructions before image inference. Adds an LLM pass; enhanced edits require a vision model. Explicit output dimensions take priority over suggested ratios.',
  enhancement_model: 'The LLM that rewrites the image prompt. Edit enhancement requires a vision model so the rewrite is grounded in the source image.',
  steps: 'More denoising steps take longer and do not always improve quality. Start with the model’s default.',
  guidance_scale: 'Controls how strongly generation follows the prompt. The useful range depends on the model; blank keeps the export default.',
  seed: 'A fixed seed repeats the random starting point. For images, -1 selects a new seed per job. When editing a Qwen-Image-2.1 result, avoid reusing the seed that created it: this can produce a noisy near-copy.',
  strength: 'Lower values preserve more of the source image; higher values allow larger changes. Available only on models that support this control.',
  negative_prompt: 'Describe content or visual traits to avoid. Only models with negative-prompt support can use this field.',
  idle_timeout_seconds: 'Unload after this much inactivity. 0 disables the idle timer; model switching and explicit unload still apply.'
};
const headingHelp = {
  'Planning budget (GiB)': 'Scheduling target, not a hard memory cap or a partition-size instruction. Leave room for runtime caches, the operating system and other applications.',
  'Model residency': 'Keep within memory budget is the default for newly detected devices and retains models together when their peak reservations fit. Unload on switch explicitly permits only one loaded model, even when more could fit.',
  'Pipeline': fieldHelp.pipeline,
  'Thinking': fieldHelp.thinking,
  'MTP decoding': 'Requires an export with multi-token prediction layers. The current runtime uses greedy decoding (temperature 0) when MTP is enabled.',
  'KV cache precision': 'Lower precision reduces memory used by the conversation cache. Available precisions depend on the model and device.',
  'Diffusion scheduler': 'Auto uses the scheduler and settings bundled with the export. Changing them can affect image quality.',
  'After image generation': 'Keeping the model loaded avoids loading it again for the next image. Device residency rules can still unload it when another model needs the memory.',
  'Overrides · JSON': 'Override individual values from the exported diffusion scheduler configuration. Leave {} to keep all exported values.',
  'Plugin properties · JSON': 'Device-specific OpenVINO options applied when loading the model. Leave {} to use the defaults.',
  'Scheduler properties · JSON': 'Advanced scheduling and KV cache options for the text runtime. Leave {} to use the defaults.'
};
const helpPopover = document.createElement('div');
helpPopover.id = 'setting-help';
helpPopover.setAttribute('popover', 'manual');
helpPopover.setAttribute('role', 'note');
helpPopover.setAttribute('aria-label', 'Setting explanation');
helpPopover.hidden = true;
const helpCopy = document.createElement('p');
helpCopy.id = 'setting-help-copy';
const helpClose = document.createElement('button');
helpClose.type = 'button';
helpClose.className = 'help-close';
helpClose.setAttribute('aria-label', 'Close explanation');
helpClose.textContent = '×';
helpPopover.append(helpCopy, helpClose);
document.body.append(helpPopover);
let activeHelp = null, fieldSequence = 0;

function hideHelp(restoreFocus = false) {
  const previous = activeHelp;
  if (previous) {
    previous.setAttribute('aria-expanded', 'false');
    previous.removeAttribute('aria-describedby');
  }
  activeHelp = null;
  if (helpPopover.matches(':popover-open')) helpPopover.hidePopover();
  helpPopover.hidden = true;
  if (restoreFocus && previous?.isConnected) previous.focus();
}

function positionHelp() {
  if (!activeHelp) return;
  if (!activeHelp.isConnected) { hideHelp(); return; }
  const anchor = activeHelp.closest('.field').getBoundingClientRect();
  const padding = 12;
  helpPopover.style.left = Math.max(padding, Math.min(anchor.left, innerWidth - helpPopover.offsetWidth - padding)) + 'px';
  const below = anchor.bottom + 7;
  helpPopover.style.top = (below + helpPopover.offsetHeight + padding <= innerHeight
    ? below : Math.max(padding, anchor.top - helpPopover.offsetHeight - 7)) + 'px';
}

function toggleHelp(button, description) {
  if (activeHelp === button) { hideHelp(); return; }
  hideHelp();
  activeHelp = button;
  helpCopy.textContent = description;
  helpPopover.hidden = false;
  if (typeof helpPopover.showPopover === 'function') helpPopover.showPopover();
  button.setAttribute('aria-expanded', 'true');
  button.setAttribute('aria-describedby', helpCopy.id);
  positionHelp();
}

function installHelp() {
  if (activeHelp && !activeHelp.isConnected) hideHelp();
  for (const original of $$('label:not(.field-label)')) {
    const control = original.querySelector('input, select, textarea');
    if (!control) continue;
    let controlRoot = control;
    while (controlRoot.parentElement !== original) controlRoot = controlRoot.parentElement;
    const titleNodes = [...original.childNodes].filter(node => node !== controlRoot);
    const title = titleNodes.map(node => node.textContent).join('').trim();
    const raw = control.dataset.runtimeKey || control.dataset.diffusionKey || control.name || control.id;
    const name = raw.replace(/^gen_/, '');
    const authored = original.dataset.help?.trim();
    const description = (authored && authored !== title ? authored : '') || fieldHelp[name] || fieldHelp[raw] || headingHelp[title];
    if (!control.id) control.id = 'setting-field-' + (++fieldSequence);

    // A separate label and help button preserve input focus and checkbox behavior.
    const field = document.createElement('div');
    field.className = ('field ' + original.className).trim();
    field.hidden = original.hidden;
    const caption = document.createElement('label');
    caption.className = 'field-label';
    caption.htmlFor = control.id;
    caption.append(...titleNodes);
    const heading = document.createElement('div');
    heading.className = 'field-heading';
    heading.append(caption);
    if (description) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'help-button';
      button.textContent = '?';
      button.setAttribute('aria-label', 'About ' + title);
      button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-controls', helpPopover.id);
      button.addEventListener('click', () => toggleHelp(button, description));
      heading.append(button);
    }
    if (original.classList.contains('checkbox')) field.append(controlRoot, heading);
    else field.append(heading, controlRoot);
    original.replaceWith(field);
  }
}

helpClose.addEventListener('click', () => hideHelp(true));
document.addEventListener('click', event => {
  if (!event.target.closest('.help-button') && !helpPopover.contains(event.target)) hideHelp();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && activeHelp) {
    event.preventDefault();
    event.stopPropagation();
    hideHelp(true);
  }
}, true);
document.addEventListener('focusin', event => {
  if (activeHelp && event.target !== activeHelp && !helpPopover.contains(event.target)) hideHelp();
});
document.addEventListener('scroll', positionHelp, true);
window.addEventListener('resize', positionHelp);
return {install: installHelp, hide: hideHelp};

})();
