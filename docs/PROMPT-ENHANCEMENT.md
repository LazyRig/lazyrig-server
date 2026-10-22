# Prompt enhancement and chat edits

## Dashboard

New chats automatically select the preferred enabled image model, or the first enabled image model. Set the preference with **Models → Settings → Use as default image tool**, or change/disable it for one conversation under **Chat → Conversation settings (sliders icon) → Image tool**. Attach a source and request an edit, or use **Use in chat** under a saved image. **Use in chat** selects a vision model when available and enables a compatible image-edit tool when one is registered. It reuses an image already attached to the current composer, including on repeated clicks. **Edit image** selects the direct editing task with that single source. The client declares `generate_image` and, when sources and an editing adapter are available, `edit_image`. Source IDs are scoped to that conversation request; the LLM cannot choose a filesystem path. Each edit accepts one source.

**Enhance image prompt** is off by default and saved per conversation. In chat, the selected chat LLM performs enhancement. Direct Generate/Edit tasks have an **Enhancement model** selector. Enhanced editing requires a VLM to inspect the source image. A text-only LLM may direct an ordinary edit without enhancement.

The dashboard displays enhancement separately from denoising. **Image prompt** shows the rewritten prompt and actual dimensions. **Response JSON** exports the recorded stage results. Original user messages remain intact.

## LazyRig enhancement defaults

The two bundled defaults are original LazyRig instructions, released under
[Apache License 2.0](../LICENSE). They combine the application's input/output
requirements with general composition, typography and image-editing practices.

| File | Purpose |
| --- | --- |
| `lazyrig/prompts/image_generation.txt` | Develop the requested image through composition, framing, lighting, materials, color and lettering |
| `lazyrig/prompts/image_editing.txt` | Ground each requested operation in the source, develop its visual result and protect relevant source features |

### What the expanded instructions do

Generation enhancement actively fills useful open visual decisions. A short scene
request can acquire a coherent viewpoint, subject placement, lighting, palette,
surface treatment and background. Explicit counts, actions, exclusions, styles
and literal text remain binding. Detail adapts to the medium: a logo keeps clear
shapes and spacing; a painting keeps its mark-making; a diagram does not acquire
invented data. A precise or minimalist brief does not need extra decoration.

Editing enhancement distinguishes a local operation, a global transformation and
a new composition using the source subject. It includes operation-specific advice
for recoloring, material changes, removal, replacement, relocation, pose,
lighting, style, restoration, text, cropping and canvas extension. Preservation
is selective: an edit should keep identity or geometry where appropriate while
allowing the shadows, silhouette or surroundings that the requested change needs.
The selected image model still determines which edits it can actually perform.

Both defaults use English descriptive prose unless the user explicitly requests
another rewrite language. Words to appear **inside the image** keep their requested
spelling and language. Editing does not translate existing signs or labels unless
asked. The prompt separates literal lettering from output dimensions and other
instructions that should not appear as printed content.

Suggested output lengths vary with the task: roughly 180–350 words for a developed
scene, 80–180 for a focused edit, and up to about 450 for a complex request. These
are flexible guides, not padding requirements. Both preserve the existing
1–8000-character validation limit and the enhancement output-token budget.
Generation can choose a ratio suited to the requested composition; editing normally
follows the source, using a new ratio only for an explicit size/shape or a clearly
requested new canvas. The application's exact-size settings remain authoritative.

The expanded system instructions consume more input context than the earlier
short defaults. Allow space for the full instructions, the request, source-image
tokens for an edit, and the reserved output. If preflight reports insufficient
context, raise the enhancement model's configured context limit within its supported
range, or use a suitable model with a larger context. Do not solve that error by
truncating the instruction or dropping the source image.

### Cases for evaluating a chosen model

These are manual acceptance cases, not claimed model results. Enable enhancement,
inspect **Image prompt** and the final image, and compare with enhancement disabled.
Keep the image model, seed and generation settings fixed for a useful comparison.

| Request and source | What to check |
| --- | --- |
| Generate a ceramic observatory on a windy cliff at dusk | Develop a readable composition, material, atmosphere and light without changing the central idea |
| Generate a flat, two-color emblem with exactly three leaves and no text | Keep the count and simple style; add no lettering, extra objects or photographic treatment |
| Generate a poster reading “NOCHE ABIERTA” and “Sábado 18”, at 1024×1536 | Preserve both strings and accents; establish hierarchy; retain the exact size through the workflow |
| Recolor only the left chair in a two-chair photograph | Identify the right target, preserve its form and shading, leave the other chair alone |
| Remove a foreground bag from a patterned floor | Continue the exposed pattern and remove the bag's obsolete shadow without rearranging the room |
| Replace one sign's text with “明日も営業” | Keep literal characters, surface perspective and surrounding text |
| Place a source product into a new workshop scene | Develop the new surroundings while retaining the product's geometry and identifying markings |
| Change a portrait to a bold ink illustration | Make the style change apparent while retaining identity; avoid clauses that demand unchanged photographic texture |
| Extend a landscape to 16:9 with more room on the right | State the new framing and coherent extension; preserve the subject instead of stretching it |

Review missing constraints, unintended additions, exact lettering, source fidelity
and edit strength separately from JSON validity. More words alone are not evidence
of better images.

## Enhancement integration

The dashboard fetches `/prompts/image_generation.txt` or
`/prompts/image_editing.txt` using the configured API authentication. It sends the
complete file as a system message in a separate `/v1/chat/completions` request.
An edit includes the selected source pixels. The user input contains the original
request and the image brief resolved from chat. The server's legacy enhancement
extension uses the same new defaults.

For client compatibility, the old `/prompts/system_prompt_t2i.txt` and
`/prompts/system_prompt_edit.txt` URLs resolve to the **new LazyRig defaults**.
They never read an old file left on disk. Restart the server and reload the
browser after upgrading. For an in-place upgrade, remove the two old prompt files
from `lazyrig/prompts/`; a fresh extraction contains only the new pair. The release
builder explicitly selects the new files, excluding legacy or private prompt files.

The request uses native JSON-schema output, disables thinking where supported,
and inherits saved model sampling. Its independent output budget is up to 2048
tokens; the chat's smaller output budget and stop strings are not inherited.
MTP retains its explicit configuration and runtime restriction.

The client validates the result before submitting it to the image endpoint:

- Generation returns string fields `rewritten_prompt` and `wh_ratio`.
- Editing also returns `ratio_follow`. It uses either a positive `W:H` ratio or
  `ratio_follow: "<image1>"` with an empty ratio, never both.
- A rewritten prompt contains 1–8000 characters; editing uses a single paragraph.
- Invalid or incomplete enhancement stops the workflow before image inference.
  Native JSON validation also rejects duplicate keys.

Structure checks cannot establish whether a model followed the requested language,
exact text or editing intent. Validate the new defaults with your selected LLM/VLM
and image model; transport tests do not establish visual quality.

## Bring your own client-side prompt

Enhancement stays **off by default**. Ordinary image generation, editing and VLM
image-tool calls work without this rewriting pass. To keep prompt preparation
entirely in your client:

1. Leave **Enhance image prompt** off in the dashboard, or omit `enhance_prompt`
   (or send `false`) when using the image API.
2. Prepare the final generation/edit instruction yourself or with your client's
   preferred LLM/VLM and system instructions.
3. Send that instruction as `prompt` to `/v1/images/generations` or
   `/v1/images/edits`, with your chosen size and other supported parameters.

LazyRig then forwards the request's prompt to the image pipeline without an
additional enhancement pass. Image tool callers can supply their final instruction
in the tool's `prompt` argument. Client-side instructions do not need the enhancement
JSON fields unless the client chooses to implement this optional protocol.

Keep any separately obtained model-specific instructions in your own client
configuration and observe their source terms. The dashboard has no custom
system-instruction editor for its enhancement pass; its regular **System prompt**
field applies to the conversation, not this separate pass.

## Dimensions

Both dashboard tools accept optional `size: "WIDTHxHEIGHT"` for exact output pixels and `aspect_ratio: "W:H"` for shape-only requests. For example, `size: "1376x928"` preserves those dimensions; `aspect_ratio: "1376:928"` only specifies proportions and may produce a much larger image. The tool descriptions instruct the LLM to use `size` whenever the user requests pixel dimensions.

**Conversation settings (sliders icon) → Image size → Custom dimensions** fixes the output independently of the LLM's interpretation. It is available in chat when an image tool is selected, as well as in direct Generate/Edit tasks. The priority order is:

1. Explicit Image size in Conversation settings.
2. Exact `size` in the tool arguments.
3. Tool-requested `aspect_ratio`.
4. Enhancement's ratio or source-following instruction.
5. Image model defaults when no size or ratio is supplied.

Exact sizes are checked against the selected adapter's minimum, alignment and configured width/height limits before enhancement or image submission. Unsupported values produce an error; they are never rounded, resized or replaced by a preset. A valid exact size remains unchanged through enhancement and editing, including `ratio_follow: "<image1>"`.

For ratio-only requests, the client selects a matching published preset or fits the image adapter's configured limits and dimension multiples. Arbitrary ratios may require rounding. Without an exact size or tool ratio, `ratio_follow: "<image1>"` uses the source dimensions when valid.

Recognized Qwen-Image-2.1 exports retain the seven official presets and initial 2752-pixel width/height limits. Other image models use their own capabilities and defaults. These image-adapter size rules do not change ordinary text inference.

## Client and server responsibilities

The dashboard submits separate public requests for chat routing, optional enhancement, image inference and LLM continuation. Each stage has its own persisted retry identity and result. It waits for the image job to reach successful completion before passing a `role: "tool"` result with the matching `tool_call_id` back to the LLM. A saved artifact alone does not start continuation while cleanup is still running.

The server owns each accepted job, progress, device lease and artifact. The browser owns the sequence between jobs. Closing the page does not cancel accepted background work; later stages wait for the browser to return. **Stop** explicitly cancels active work and prevents subsequent stages. Reopening a completed conversation never regenerates its images.

External clients can implement the same sequence with ordinary Chat Completions and Images APIs. The bundled defaults govern only explicitly enabled enhancement, not every incoming image prompt. The old explicit `auto_image` and `enhance_prompt` backend extensions remain available for existing integrations, but are deprecated and unused by the dashboard. Their combined job metadata remains documented in [API](API.md).
