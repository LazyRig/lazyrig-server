# LazyRig API

Base URL: `http://127.0.0.1:8765`. JSON request bodies use `Content-Type: application/json`. If authentication is configured, send `Authorization: Bearer YOUR_KEY` for every API and artifact request except `/api/health`. The owner `api_key` permits administration and inference. An optional `inference_api_key` permits inference and its recovery routes only; see **Credentials and readiness** below. Configured browser origins receive CORS headers.

## Credentials and readiness

`GET /api/health` is an unauthenticated liveness/version endpoint. It confirms the
HTTP service can respond; it does not certify that the native runtime or a model
can run inference. `GET /api/ready` requires the same authentication as inference
when keys are configured. It returns HTTP 200 with `ready: true`, or HTTP 503 with
`ready: false`, `reasons` and `Retry-After: 1`. Its checks include storage state and configured disk headroom,
runtime availability, maintenance/draining, failed worker cleanup and admission
capacity, including a saturated preparation pool. `?model=ALIAS` additionally checks that the requested model is registered
and enabled. This does not load the model or run a hardware inference probe.

Use `POST /api/server`, authorized with the owner key, to set an optional
`inference_api_key`. It must be at least 16 characters, differ from the owner key,
and requires an owner key of at least 16 characters. An empty string disables the
extra credential. Configure it through this API or the saved server configuration;
the existing dashboard key field continues to manage the owner key.

The inference key can list models, prepare context, submit inference, read job
status/results/events, cancel jobs and retrieve artifacts. It cannot read the
administrative state or change models, imports, runtime or server settings;
those operations return HTTP 403. The owner key still works for all routes.
Both credentials are stored in the local settings file and omitted from state
responses. These are shared service roles, **not per-user identities**: inference
key holders can access each other's jobs and artifacts. The API `user` request
field does not establish ownership.

## Admission, deadlines and transport limits

The beta reserves capacity before remote-image retrieval and tokenization.
Default limits are eight active/queued/reserved jobs (`max_jobs`) and two requests
preparing inputs (`max_preparing`). `/api/context` also uses bounded preparation
and the request budget; it cannot bypass capacity through preflight. Retries of the same idempotency key serialize
with each other without holding the global service lock during preparation.
Duplicate requests still recover the original job; they do not generate twice.

| Condition | Response |
| --- | --- |
| Job capacity exhausted | HTTP 503, `error.code: "queue_full"`, `x_retryable: true`, `Retry-After: 1` |
| Preparation capacity exhausted | HTTP 503, `error.code: "preparation_full"`, `x_retryable: true`, `Retry-After: 1` |
| Durable storage unavailable | HTTP 503, `error.code: "storage_unhealthy"`, `x_retryable: true`, `Retry-After: 1` |
| Disk headroom below configured minimum or unavailable | HTTP 503, `error.code: "disk_pressure"`, `x_retryable: true`, `Retry-After: 1` |
| Inference request deadline expired without saved output | HTTP 504, `request_timeout`, `x_retryable: false` |
| HTTP body receive deadline exceeded | HTTP 408 where a response can be sent; connection closes |

`X-Should-Retry` mirrors the explicit retryability flag on these JSON responses.
Retain the same idempotency key when retrying a transient refusal. A timed-out
accepted job remains the same failed job when its key is replayed; a new key is
an intentional new inference request. A fully saved image remains retrievable
when later cleanup or restoration exceeds the deadline; inspect status/warnings.

`request_timeout_seconds` now supplies one monotonic inference-request budget
(default 3600 seconds). It starts after the bounded HTTP body has been received
and includes idempotency-key waiting, input preparation, queue wait and subsequent
model stages. `load_timeout_seconds` remains a per-load cap within that budget.
Waiting for the shared tokenizer lock also honors cancellation and the remaining
deadline. An encode already executing remains cooperative at its call boundaries.
Cleanup may require additional time to terminate and verify a native worker.
Deadlines are checked at cooperative boundaries; a synchronous third-party
operation in the main process cannot be forcibly interrupted safely. Client and
proxy timeouts remain separate. Prefer async jobs for operations that must survive
a client disconnection, and explicitly cancel them when intended.

The HTTP listener defaults to 64 concurrent connections (`max_connections`) and
30-second HTTP receive/write limits (`http_timeout_seconds`). Header and body
receive have absolute deadlines, so trickling bytes does not keep a request open
indefinitely. These transport limits do not impose a 30-second inference or SSE
lifetime. Plain HTTP connection saturation returns a bodyless 503 with `Retry-After: 1`
and `X-Should-Retry: true`; an over-capacity TLS connection closes before
performing a handshake. Header/idle timeouts close the connection. SSE and
keep-alive clients consume connection slots. Retain reverse-proxy connection,
body-size and slow-client limits when serving over a network.

Unused browser connections and idle keep-alive connections expire silently.
`Incomplete HTTP request: receive deadline exceeded; connection closed` means
the client started sending a request but did not finish its request line or
headers in time. Incomplete bodies still receive HTTP 408 when possible. These
are transport receive limits, separate from an inference deadline.


## Chat completions

`POST /v1/chat/completions`

```json
{
  "model": "my-chat-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

Supported fields: `model`, `messages`, `stream`, `stream_options.include_usage`, `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`, `stop`, `tools`, `tool_choice`, `chat_template_kwargs`, `reasoning_effort`, `timings_per_token`, `response_format`, `num_assistant_tokens`, `n: 1`, and `user` (accepted but not used for identity or authorization). Messages support system, user, assistant and tool roles with text content or arrays of text parts; prior assistant function calls may have null content. VLMs also accept image parts in user messages and tool results. Set defaults per model; request values override them. The complete history must be supplied on every public chat request.

Text sampling values must be finite JSON numbers, not strings or Booleans: temperature 0–2, top-p `(0, 1]`, integer top-k 0–1,000,000, min-p 0–1, presence/frequency penalties -2–2, and positive repetition penalty. Top-k 0 and min-p 0 disable their filters. Explicit zeros are preserved. Saved `generation` settings use the same validation; omitted settings inherit the model/runtime, except temperature defaults to 0.7 (0 for MTP). MTP currently requires greedy temperature 0. The console provides explicit [Qwen3.8 presets](MODEL-SETTINGS.md#text-sampling-and-qwen38-presets); clients switching thinking mode must also choose their intended sampling values.

Output allowance precedence is `max_completion_tokens`, then `max_tokens`, then
the model's saved `generation.max_tokens`, then 8,192 tokens. The client may
choose a different allowance on every request without reloading a compatible
worker. Input plus the requested allowance must fit the configured context;
LazyRig rejects overflow instead of silently reducing the client's allowance.

Completed inference that fails tool-call validation returns its error without
dispatching any client tool. A malformed JSON response returns HTTP 502 with
`error.code: "invalid_response_format"` and `x_retryable: false`. A live worker
that completed its exchange remains available for a corrected client request;
the server does not automatically repeat client tool calls or raise their token
budget. Native failures, Stop, deadlines and incomplete exchanges still unload
the worker. Full errors and the original generated text remain available from
authenticated `GET /api/jobs/{id}`.

### Vision input

Register/select a compatible `vlm` export. Send OpenAI-style content parts, for example:

```json
{
  "model": "my-vlm",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {
        "url": "data:image/png;base64,BASE64_ENCODED_PNG",
        "detail": "auto"
      }}
    ]
  }],
  "max_tokens": 256,
  "auto_image": false
}
```

Replace the placeholder with actual base64 image bytes, or use a public HTTP(S) image URL. Local paths, `file:` URLs, URL credentials and private/local network URLs are rejected; upload a local image as a data URL. Redirects are checked as well. The server does not forward the client's API key to the image host. Remote URLs are fetched after reserving preparation/job capacity and before a job receipt is committed, so allow time for that transfer.

PNG, JPEG, WebP and nonanimated GIF are supported. There is no fixed image-count limit for vision chat, including user images and tool results across the entire request history. Size limits: 10 MiB encoded-file bytes per image, 24 MiB total image bytes, 20 megapixels per image, and 32 MiB for the JSON request body including base64 expansion. Model context and runtime capacity also apply. `detail` accepts `auto`, `low` or `high`; `low` reduces the image to fit 512×512 while preserving aspect ratio. `auto` and `high` leave resolution handling to the native model processor. Animations, audio and video parts are rejected.

Images are normalized, stored in `server-data/inputs/`, and represented by internal references in persisted jobs. They remain available after unloading/reloading a model. The server renders the model's chat template with ordered image placeholders and supplies the corresponding RGB tensors to `VLMPipeline.generate`. All historical images are supplied again for each stateless public chat request. Clients must continue sending the complete original conversation and images; internal `lazyrig-input:` references are not accepted from public callers. Plain LLM pipelines reject image parts with an instruction to select VLM.

The same content parts work with streaming chat, `/api/chat/jobs` and `/api/context`. Image preflight counts are estimates; see **Context preflight** below. `/v1/models` exposes `pipeline`, `input_modalities`, `output_modalities`, and `capabilities.vision` / `capabilities.image_input` for frontend discovery. A frontend may need its cached model list refreshed.

### Images returned by client tools

VLMs also accept the same content parts on `role: "tool"`, including Hermes `vision_analyze` and screenshot tools. Include the preceding assistant `tool_calls` in the conversation and retain the corresponding nonempty `tool_call_id` on each tool result:

```json
{
  "role": "tool",
  "name": "vision_analyze",
  "tool_call_id": "call_vision_1",
  "content": [
    {"type": "text", "text": "Inspect the attached image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_ENCODED_PNG"}}
  ]
}
```

This is one message within the complete `messages` array. The server preserves its tool role, name and call ID while mapping its image parts to ordered native tensors. Image-only tool results and multiple tool results are supported. Client tools remain client-executed; receiving a result does not rerun them. Images in user and tool messages share the same validation/size limits, persistence and context accounting. This applies to synchronous chat, SSE, asynchronous jobs and context preflight. System/assistant image parts remain unsupported.

### MTP request settings

When model settings enable a bundled MTP head, `num_assistant_tokens` optionally overrides its saved draft-token count (integer 1–64). The default is 2 unless changed in model settings. Temperature must be 0; omitting it uses the model's saved greedy default. Prefix caching is disabled when loading an MTP worker. Requests cannot activate MTP on a worker loaded without it. `max_tokens` still controls the generated output budget; the draft count is not extra output allowance. See [MODEL-SETTINGS.md](MODEL-SETTINGS.md) for setup and export requirements.

The bundled head's device is configured through `POST /api/settings`, for example `"mtp": {"enabled": true, "num_assistant_tokens": 2, "device": "CPU"}`. Use a configured physical CPU/GPU ID; omit `device` or use an empty string to follow the main model's resolved target. This is a load-time setting: changing it reloads the worker. A request-level draft-token override does not change device placement or require reloading. Active worker state includes `mtp_device` alongside the main `device`.

### Frontend template, reasoning and timing options

The same options work with synchronous chat, SSE, asynchronous chat jobs, and `POST /api/context`:

```json
{
  "model": "my-chat-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false},
  "reasoning_effort": "high",
  "timings_per_token": true,
  "stream": true
}
```

- `chat_template_kwargs`: a JSON object or null, merged over the model's saved **Chat template settings**. Values go into the original HF Jinja template before OpenVINO tokenization. `enable_thinking`, `preserve_thinking` and `add_vision_id`, when supplied, must be booleans. Additional JSON variables are passed through; they affect output only if the template uses them. Controlled fields `messages`, `tools`, `add_generation_prompt`, `chat_template`, `bos_token`, `eos_token` and `pad_token` cannot be overridden here.
- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `default` or null. A positive top-level effort overrides an effort inside `chat_template_kwargs`, and enables thinking unless these request kwargs explicitly set it false. `none` disables thinking even when the kwargs say true. Omission/null/`default` leaves the kwargs or saved model defaults in effect. This controls the prompt template; it is not a separate OpenVINO reasoning-budget parameter. Templates that do not implement these variables cannot gain reasoning behavior from them.
- The Qwen3.8 template's declared effort levels are `low`, `medium`, and `xhigh`. For that template, `minimal` maps to `low` and `high` maps to `xhigh`. Other templates retain the supplied level. `x_template_context` reports the merged, normalized variables; `x_template_notes` describes mappings. Template-internal defaults remain internal when no override is supplied.
- `timings_per_token`: boolean, default false. True includes a `timings` object on generated SSE text updates and completed JSON/saved results, with `prompt_n`, `predicted_n`, `prompt_ms`, `predicted_ms`, `prompt_per_token_ms`, `predicted_per_token_ms`, `prompt_per_second` and `predicted_per_second`. Measurements may be null until enough tokens exist. SSE output may batch tokens; this flag does not force one event per token.

Live timings use the worker's monotonic clock around `generate()`. **`prompt_ms` measures time to the first token callback, not isolated prompt/prefill compute time.** Consequently its per-token/rate values are estimates based on that interval. Decode timing runs from the first callback until the update/end; the rate excludes the first callback's token batch (`x_decode_tokens`). These timings include worker callback/IPC overhead, exclude queueing, model load/compile, server-side template preparation/tokenization, image generation and model switching, and describe the current text pass only. `x_source`, `x_scope` and `x_prompt_ms_basis` label this meaning.

At completion, available native OpenVINO metrics are attached separately under `timings.x_openvino_metrics`: `ttft_ms`, `tpot_ms`, `tokens_per_second`, `generate_ms`. Missing or non-finite native measurements are omitted. `x_timing_passes` retains initial, retry and continuation timing records; `usage` still totals tokens across all completed text passes. A new pass resets the live timer.

Overrides never modify the registered model or force a reload merely because a template flag changes. They survive image handoffs. The existing bounded retry for an incomplete image tool response temporarily disables thinking for that retry; the final continuation restores the request's settings. Context preflight must receive the same options and output budget as generation to describe the same rendered prompt.

### Reasoning and structured responses

Ordinary and tool-enabled chat replies both put the final answer in `message.content` and any reasoning in `message.reasoning_content`. SSE uses the corresponding `delta.content` and `delta.reasoning_content`. Native Qwen/DeepSeek-style `<think>` delimiters are retained during decoding, then removed from the public channels. A prompt that already opened the reasoning block is handled the same way. No heuristic guesses reasoning from prose. An unterminated block remains reasoning, sets `x_reasoning_incomplete: true`, and has empty answer content; reaching the output budget uses `finish_reason: "length"`. Raw output remains available in the administration job record's `text`; `content` and `reasoning_content` are also saved separately. Usage includes reasoning tokens.

`response_format` may be `{ "type": "text" }` (also the default), `{ "type": "json_object" }`, or a named JSON schema. For example, Hermes title generation can send:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "session_title",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"title": {"type": "string"}},
      "required": ["title"],
      "additionalProperties": false
    }
  }
}
```

OpenVINO's `StructuredOutputConfig` constrains token sampling to the supplied schema. This requires a compatible native runtime/backend/schema; unsupported native constraints fail explicitly. JSON requests disable thinking for that request, close any mandatory thinking prefix before generation, and count the format instructions in context preflight. A JSON constraint is cleared for the next plain request without reloading the model. `strict: false` still uses the native schema constraint; there is no unconstrained best-effort fallback.

Default automatic image routing is suppressed for JSON-format requests. Explicit `auto_image: true` or active client tools together with JSON output are rejected; use separate requests. `tool_choice: "none"` may disable supplied tool definitions. `x_json_valid` reports the final JSON syntax/object check (schema enforcement is native); a truncated JSON result may be partial with `x_json_valid: false` and `finish_reason: "length"`. Unsupported schema features or native pipeline limitations remain model/runtime dependent.

### Response

The result has `id`, `object: "chat.completion"`, `created`, `model`, `choices`, `usage`, plus:

| Extension | Meaning |
|---|---|
| `x_context` | Current/final inference context counters below |
| `x_job_id` | Persistent job ID; use for status, events or cancellation |
| `x_artifacts` | Saved images from server-managed tool calls |
| `x_mode` | `openvino` |
| `x_reasoning_incomplete` | Whether output ended inside a reasoning block |
| `x_response_format`, `x_json_valid` | Applied output format; JSON validity is included for JSON requests |
| `x_template_context`, `x_template_notes` | Applied template variables and effort mapping notes |
| `x_mtp` | Whether MTP is enabled and the applied `num_assistant_tokens` when enabled |
| `x_sampling` | Native resolved sampling values, `do_sample` and `source: "openvino"`; null without a native snapshot. Greedy temperature and unlimited top-k are reported as 0. Available in JSON, final SSE chunks and saved results; the job stores it as `sampling`. |
| `timings`, `x_timing_passes` | Optional timing data when `timings_per_token` is true |

```json
{
  "input_tokens": 89,
  "generated_tokens": 42,
  "used_tokens": 131,
  "context_limit": 32768,
  "output_budget": 256,
  "remaining_tokens": 32637,
  "remaining_after_budget": 32423,
  "fits": true,
  "accuracy": "exact",
  "cache_occupancy": null,
  "scope": "current inference",
  "prompt_sha256": "..."
}
```

`fits` checks input plus the requested output budget. `remaining_tokens` excludes already generated tokens; it does not reserve future output. `remaining_after_budget` reserves the complete requested output budget. Counts include the actual rendered chat template, tool definitions and any runtime-emitted reasoning tokens. The configured context limit is an operator setting, not automatically discovered physical memory. `usage` counts all completed LLM passes for the request, while `x_context` describes the latest pass.

Malformed requests fail with HTTP 400 before queuing. Synchronous non-streaming chat also checks context before queuing. Streaming and asynchronous chat prepare/tokenize inside the job, so context overflow is reported through its SSE error or saved job result. Model loading/generation failure returns HTTP 500 with `x_job_id`. Backend error messages are preserved for troubleshooting. An overflow after a server-managed image may fail the request while retaining its artifact in `x_artifacts` and the job record. The adapter does not silently truncate or compact. Automatic model placement can try the explicitly allowed device pool during loading, with each change recorded in job events; it does not replay text generation. Bounded inference retries cover incomplete server-managed tool output and the specific native image GPU resource errors described below.

### Streaming

With `stream: true`, content type is `text/event-stream` and each `data:` record is a `chat.completion.chunk`. The initial response header `X-Job-Id` identifies the job immediately. Content arrives in `choices[0].delta.content`, with `x_context` on text updates. A final chunk carries `finish_reason`, `x_context`, `x_artifacts` and `x_job_id`, followed by an optional usage chunk (`choices: []`) when requested, then `data: [DONE]`.

Errors after headers are sent appear as SSE error objects. Disconnecting the original synchronous `/v1` request cancels unfinished work, including server-managed image generation. The server also detects closure before response headers while waiting on a non-streaming request. A Stop/abort and a connection timeout look the same to the server. For durable long-running work, submit through `/api/chat/jobs`, `/api/image/jobs`, `/api/image/edits`, or `Prefer: respond-async`; disconnecting their receipt, status, result or event connections does not cancel the background job. Use authenticated `POST /api/jobs/ID/cancel` with `{}` to stop it explicitly. Transport failures are logged as `client_disconnected` events without marking successful inference failed or attempting a second HTTP response.

Client-tool requests stream ordinary text and reasoning incrementally while holding tool arguments until validation. Server-managed `auto_image` buffers the initial routing pass and streams the final continuation. During model/image work, periodic empty deltas carry `x_status` and saved `x_artifacts`. Use the native job event stream for detailed lifecycle and denoising progress.

### Client-managed tools

Image tools declared by a client should distinguish exact pixel `size` (for example `"1376x928"`) from a shape-only `aspect_ratio` (for example `"3:2"`). Pass exact `size` unchanged to the Images endpoint. The dashboard declares both parameters for generation and editing; unsupported exact sizes fail validation instead of being scaled. Other clients own their tool schema and must pass the requested size themselves. Natural-language dimensions inside an image prompt do not override the API's `size` field.

`tools` accepts function definitions. `tool_choice` supports `auto` and `none`; forced/named calls are not implemented. Complete JSON `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks and Qwen XML `<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>` blocks become assistant `tool_calls`. String parameters preserve their XML text; typed non-string parameters use JSON values with the limited XML compatibility rules below. A natural-language preamble becomes `message.content`. Completed reasoning, including a `<think>` prefix opened by the prompt, is separated as `reasoning_content` for both ordinary replies and tool calls. Inline code or balanced quotes in the preamble do not invalidate a real tool call. Fenced, inline-code, blockquoted, indented or labelled examples remain text and are not executed. Incomplete client calls at the token limit remain text with `finish_reason: "length"`. The client executes its tools and submits resulting messages in a new request; the server does not execute arbitrary client functions.

Complete bare JSON envelopes are also recognized when they comprise the whole response: `{"name":"DECLARED_NAME","arguments":{...}}`, `{"function":"DECLARED_NAME","args":{...}}`, or an OpenAI-shaped `{"type":"function","function":{"name":"DECLARED_NAME","arguments":"{...}"}}`. These are output adapter formats, not special function-name aliases. A fenced or prose-embedded JSON example is not a call. Potential JSON envelopes are buffered until validated; normal prose still streams incrementally.

Malformed model tool output makes the job `failed` with an `error_code` (for example `undeclared_tool`). Synchronous and saved-result responses return HTTP 502 with `x_retryable: false` and `X-Should-Retry: false`; SSE emits the same error. The input request was accepted, but no executable tool batch is published. Clients may deliberately ask the model to repair its output; transport retries should not regenerate it. Forced/named `tool_choice`, `parallel_tool_calls`, Responses, audio and full JSON Schema argument validation are not implemented.

Qwen XML Boolean parameters accept `true`/`false` in any letter case, optionally surrounded by one matching pair of single or double quotes. For example, `<parameter=background>False</parameter>` becomes `"background": false`. Normalization requires an explicit Boolean type, or a simple `type`/`anyOf`/`oneOf` union that includes Boolean and excludes string. A Boolean-or-array parameter still accepts an array; a nullable Boolean also accepts JSON `null`. String values, unknown types and ambiguous string/Boolean unions are not subject to Boolean coercion. Numbers `0`/`1`, `yes`/`no`, comments, code fences, unfinished literals and explanatory suffixes are not Boolean aliases.

XML parameters explicitly typed as an array or object also accept complete data literals using single-quoted strings, nested Python `True`/`False`/`None`, and a trailing comma after a value. For example, `<parameter=questions>[{'question': 'Which background?', 'choices': ['Blue', 'White'], 'multi_select': False}]</parameter>` becomes a real JSON array in the returned arguments. One complete Markdown fence (unlabelled, `json` or `python`) and one JSON-string wrapper around the collection may be unwrapped. Simple type unions work when they include array/object and exclude string. Quoted text inside a collection retains its value; string parameters, unknown schemas and unions including string are not subject to this normalization.

This is a data parser: expressions, function calls, comments, bare object keys, tuples/sets, duplicate keys, non-finite numbers, incomplete containers and extra text still fail. It does not infer missing entries or wrap a scalar/object in an array. The non-JSON literal fallback is bounded to 1,048,576 characters, 100,000 tokens and 64 container levels. Other primitive values and JSON-form tool blocks still require valid JSON. This decoding is not full JSON Schema validation; the client remains responsible for its tool schema constraints. A malformed call anywhere in a response rejects the whole batch before any client `tool_calls` are returned, including SSE and saved-result retrieval.

### Legacy server-managed images (deprecated)

Retained for existing integrations. The dashboard uses client-managed tools and separate public image requests. New clients should use that ordinary contract; this extension is not required for model switching.

```json
{
  "model": "my-chat-model",
  "messages": [{"role": "user", "content": "Generate an image of a quiet mountain lake"}],
  "max_tokens": 2048,
  "auto_image": true,
  "image_model": "my-image-model",
  "width": 512,
  "height": 512,
  "steps": 20,
  "image_seed": 42
}
```

These extension fields opt into `generate_image(prompt)` and, when the selected model supports editing and sources are available, `edit_image(prompt, image_id)`. If `image_model` is omitted, the first enabled image model is selected; provide the alias for predictable selection when several image models are registered. Dimensions, steps, seed, guidance and negative prompt inherit the selected image model's settings unless overridden. Do not combine explicit `auto_image: true` with client `tools` or `tool_choice: "none"`. The LLM must actually request a tool; enabling it alone does not force image inference.

For an edit, pass sources in standard VLM image message parts or use the `image_sources` extension. Text-only LLMs can select an edit source through `image_sources` but do not receive its pixels. Each source needs a `url` and an optional `description` of at most 512 characters. Supported URLs follow the image-edit endpoint rules below. Sources are listed oldest to newest with server-issued IDs; the model must select one of those IDs and cannot supply a path. Saved artifacts in server-side session history are also available. Stateless API clients must include the relevant sources on each request.

```json
{
  "model": "my-vision-model",
  "messages": [{"role": "user", "content": "Make the bicycle blue and preserve everything else."}],
  "auto_image": true,
  "image_model": "my-image-model",
  "image_sources": [{"url": "/artifacts/PREVIOUS_JOB_ID.png", "description": "The bicycle to edit"}],
  "enhance_prompt": true
}
```

`enhance_prompt` is a boolean, default `false`. When enabled, the chat model runs the bundled generation or edit enhancement instructions in a separate pass before image inference. Enhanced edits require a VLM and pass it the selected source pixels. Plain edits can use a text-only LLM. The user's original message stays unchanged. Explicit `width`/`height` take precedence; otherwise the validated enhancement ratio is fitted to the image model's supported dimensions. This extension and nonempty `image_sources` require `auto_image: true`. See [prompt enhancement](PROMPT-ENHANCEMENT.md).

`auto_image` defaults to `false` and must be enabled on each request. Plain OpenAI-compatible chat and client-owned tools never activate server-managed image generation. The dashboard does not use this extension. Context preflight follows the same rule. The initial routing pass is buffered only for opted-in server-managed image requests; the final continuation streams. The retired `default_auto_image` and `default_image_model` configuration keys are ignored when restoring older settings and rejected by `/api/server`.

Model swapping is independent of this extension. Direct chat, image-generation and image-edit endpoints all acquire the required device before loading a model. A chat request arriving during an image job waits; it does not interrupt the image worker. `GET /api/jobs/{id}` and `/events` expose server-owned status without running an LLM. Standard chat responses do not substitute progress messages for model output. A synchronous client must allow sufficient time; background submission and polling avoid tying job completion to that client connection.

The scheduler unloads competing models when needed, saves the image, applies the image retention policy, and reloads/resumes the LLM using history plus the tool result. It returns final text with artifact metadata. Automatic image retention is the default and can keep both workers loaded when the request's working set fits the device policies; explicit keep/unload settings remain available. The final tool result contains the output image URL; generated output pixels are not automatically passed to the LLM. Calls inside completed reasoning and quoted examples are not executed. Duplicated/multiple image calls, undeclared functions and text after the final tool block are rejected. Only one managed image operation is executed in a round trip; a further call is not recursively executed. Current edits use one source per operation.

`x_prompt_enhancement` contains the rewrite, actual dimensions, model and status when enhancement ran, otherwise `null`. Saved jobs expose the same record as `prompt_enhancement`. Job events include `enhancement_started`, `enhancement_progress`, `prompt_enhanced` and `enhancement_failed`; the raw JSON rewrite is not streamed as assistant text. `usage` includes enhancement tokens alongside routing and continuation. Stop cancels the whole job, including enhancement; enhancement failure does not start image inference.

For an incomplete tool block or unfinished reasoning that reaches `max_tokens`, the server retries once with thinking disabled and a concise instruction, preserving the same output budget. No partial arguments are executed. Failed retries return a clear error with `x_tool_diagnostics`, `x_context`, `x_job_id` and any retained `x_artifacts`. Successful responses include `x_tool_attempts`. `usage` sums all text passes, including this retry; `x_context` describes the last pass. Full diagnostics are saved in the job's `tool_attempts`.

## Image generation

`POST /v1/images/generations`

```json
{
  "model": "my-image-model",
  "prompt": "A quiet mountain lake",
  "size": "512x512",
  "steps": 20,
  "seed": 42,
  "response_format": "b64_json"
}
```

Fields: `model`, `prompt`, `n: 1`, `size`, `response_format` (`url` or `b64_json`), `steps`/`num_inference_steps`, `seed`/`rng_seed`, `guidance_scale`, `negative_prompt`, `max_sequence_length` (Qwen-Image-2.1), `enhance_prompt`, `enhancement_model`, and `user` (accepted metadata only). Defaults come from the model. With enhancement off, omitted `size` or `"auto"` uses saved model dimensions; it does not derive an aspect ratio from an input image. Width/height must be multiples of 32 for Qwen-Image-2.1 (64 for other adapters), at least 256, and within the model's configured maxima. Maximum steps: 100. Image seed: -1 for fresh noise, or 0–2147483647 for a fixed seed. If neither request nor model supplies a seed, fresh noise is selected. Existing saved fixed seeds are honored. Prompt: 1–8000 characters.

Random seeds are resolved once per accepted image job and exposed as `x_seed` and in artifact metadata. Idempotent retries reuse that job and seed. For Qwen-Image-2.1 edits of its own outputs, use a different seed from the original generation; repeating the same noise at the same dimensions can produce a noisy near-copy. The dashboard's chat image tools request `seed: -1` for every invocation, independently of the LLM's chat seed.

Native results also include `x_inference`: runtime/version, actual seed and output size, effective guidance and negative prompt, requested steps, callback-reported denoising progress, and `returned_normally`. The callback total may differ from requested steps for classic img2img strength. If a callback total exists and the pipeline returns early, no PNG is published. Without callbacks, the total is unknown. `visual_verification: "not_performed"` means the file has not been checked for semantic correctness; a completed job alone does not prove the edit matches the instruction.

To enhance a direct generation or edit, send `enhance_prompt: true` and a registered text model alias in `enhancement_model`. Editing requires a VLM. These fields work with synchronous and asynchronous image endpoints; multipart uses the literal `true` or `false` for the toggle. Explicit `size` overrides an enhanced ratio. The response includes `x_prompt_enhancement`, and `revised_prompt` reports the prompt actually sent to the image model. The same job ID, artifact URL, idempotency, progress and cancellation contract covers both stages. Enabling this extension adds LLM inference time.

Returns `created`, `data: [{url, revised_prompt}]` (plus `b64_json` when requested), `x_job_id`, `x_mode`, `x_warning`, `x_media_type`. Inference produces PNG. Saved images survive failure to reload a previous LLM; in that case the response includes the image and an `x_warning`. If image generation itself fails, HTTP 500 includes the job ID.

Generation defaults to URL output: the saved image is directly available at `data[0].url` and `artifact_url`. Edits continue to default to base64. An explicit `response_format: "b64_json"` includes the bytes **and** the URL; clients can save the bytes without another authenticated download. When authentication is enabled, artifact URLs require the owner or inference bearer token. With `public_base_url` set, they use that canonical origin; otherwise they use the request Host and listener's HTTP/HTTPS scheme. For TLS proxy setup, see [CONNECTING.md](CONNECTING.md#display-generated-images).

The response becomes available as soon as the image is saved. `x_result_ready: true` confirms the image exists; `x_status` and `x_phase` describe the whole job, which can still be running during cleanup/restoration. `status_url` and `result_url` allow later checks. Image retention defaults to `keep_image_model_loaded: "auto"`: a healthy worker stays loaded between requests, including after displacing another model. It considers current residents and models required for immediate restoration/continuation; already-evicted models waiting for a future request do not count against its budget. A later competing request applies normal device eviction. Displaced text models load on their next request unless `restore_text_after_image: true` explicitly enables eager restoration. That device remains reserved throughout this sequence; another LLM request queues normally. Set `keep_image_model_loaded: true` to explicitly retain a healthy image worker, or `false` to always unload after each request, through registration or `/api/settings`; eviction, manual unload and idle timeout still apply. A cleanup/restoration error after delivery appears in subsequent status/result reads as `x_warning`; it does not remove the saved image.

Image generation and editing automatically retry **once** when native inference on a GPU reports `CL_OUT_OF_RESOURCES` or `CL_MEM_OBJECT_ALLOCATION_FAILURE`, provided no result/file has been produced. The failed worker must terminate successfully before its replacement starts. The retry stays in the same job and device lease, preserves the effective prompt, captured source image, resolved seed and all inference settings, and uses the original end-to-end deadline. Prompt enhancement and tool dispatch are not repeated. Healthy workers still follow the normal retention policy. Load failures, unrelated inference errors, cancellation, expired deadlines, failed cleanup and errors after output publication are not automatically replayed.

The job's `image_recovery` and responses' `x_image_recovery` record `retry_count`, `max_retries`, `reason`, `first_error` and `status` (`retrying`, `succeeded`, `failed` or `cancelled`); they are absent/null when unused. SSE emits `image_retry` (with `message` and `recovery`) and `image_recovered`. Progress resets for the new attempt, and the dashboard/terminal show recovery. If the retry encounters another recognized GPU resource error, the job fails with `gpu_resource_error`, HTTP 502, `x_retryable: false` and `X-Should-Retry: false`. A new client submission must not be used to bypass the retry limit. This is bounded recovery, not a diagnosis or guarantee against native GPU/driver faults.

## Image editing

`POST /v1/images/edits` accepts **multipart/form-data** or **application/json**. Use the same registered image model alias as generation. Complete Qwen-Image-2.1, Stable Diffusion, SDXL and Latent Consistency exports are detected; runtime/device compatibility is checked on native load. `/v1/models` exposes `capabilities.image_editing` and an `image_editing` object with availability, required-component notes and limits. The alias remains registered as an image pipeline.

Hermes's OpenAI image provider uses multipart `image` or `image[]` uploads. Windows Command Prompt example (replace the alias, key and file path):

```bat
curl.exe http://127.0.0.1:8765/v1/images/edits ^
  -H "Authorization: Bearer YOUR_API_KEY" ^
  -F "model=my-image-model" ^
  -F "prompt=Add a red scarf and preserve the subject and composition." ^
  -F "image=@C:\Pictures\source.png" ^
  -F "size=1024x1024" ^
  -F "n=1"
```

Do not manually set a multipart boundary when using an SDK or curl. A Python client can call `client.images.edit(model="my-image-model", prompt="Add a red scarf.", image=open("source.png", "rb"), size="1024x1024")` with the configured base URL and bearer key.

JSON alternative:

```json
{
  "model": "my-image-model",
  "prompt": "Add a red scarf and preserve the subject and composition.",
  "image": "data:image/png;base64,BASE64_ENCODED_PNG",
  "size": "auto",
  "steps": 40,
  "seed": 42
}
```

Exactly **one source image** and `n=1` are supported. Use one of `image`, `image_url`, `images`, or `image[]`; an array must have one item. JSON accepts a string or `{ "url": "..." }`, including a public HTTP(S) image URL or base64 data URL. A saved `/artifacts/ID.png` path, or its absolute URL on the configured public origin (request origin when unset), is read directly from this server's artifact store. No arbitrary server/client file paths are accepted. Other private-network URLs are rejected; upload the bytes instead. The bearer key is never forwarded to external image hosts.

Source files may be PNG, JPEG, WebP or nonanimated GIF, at most 10 MiB and 20 million decoded pixels. The total HTTP request limit is 32 MiB. Input is normalized to RGB PNG and retained with the job; base64 and multipart bytes are not stored in its JSON. With enhancement off, `size: "auto"` means saved output dimensions. With enhancement on, a validated `ratio_follow` follows the selected source dimensions within adapter limits, or `wh_ratio` sets the suggested aspect ratio. Explicit dimensions obey the same limits as generation and take precedence.

Qwen-Image-2.1 uses the official image-conditioning path: `Image2ImagePipeline` receives one RGB tensor, the prompt, inference options and `TorchGenerator(seed)`. It needs the export's `vae_encoder`, `vision_encoder`, and `text_encoder_i2i` XML/BIN components. **Omit `strength` for Qwen**; its conditioning is not denoising initialization. Classic diffusion exports accept `strength` in `(0, 1]` (default 0.8 or the model's saved setting). Masks and multiple source images are explicitly rejected; describe the desired edit in the prompt.

Other fields are the generation fields above, plus `output_format: "png"` and `background: "auto"|"opaque"`. Recognized `quality` and `input_fidelity` values are accepted as client hints, with `x_parameter_notes` explaining that there is no direct OpenVINO equivalent; inference uses the saved settings and explicit steps. Quality values: `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `standard`, `hd`; fidelity: `low`, `high`. These hints do not change inference quality settings. Transparency and other output formats are not implemented.

Edits return `data: [{b64_json, url, revised_prompt}]` **by default**, so clients do not need an authenticated artifact fetch. `response_format: "url"` is also available. Responses include `x_operation: "edit"`, `x_artifacts`, `x_job_id`, `x_parameter_notes`, `x_prompt_enhancement`, `x_mode`, `x_media_type`, and `x_warning`. Output is PNG. Generation responses identify `x_operation: "generate"`.

Switching generation/editing on the same alias closes the old worker before constructing the new pipeline. Repeated edits reuse a compatible resident worker under the default `keep_image_model_loaded: "auto"`, unless immediate restoration/continuation needs its device capacity, or under explicit `true`. Set `false` to unload after every request. Device leases and LLM eviction/restoration follow the generation lifecycle, including saved artifacts if cleanup or restoration fails. Include the extra editing encoders and working memory in the model reservation. Client-owned tools such as Hermes can use this endpoint directly; the optional server-managed `edit_image` chat tool shares its adapter and lifecycle.

Use `POST /api/image/edits` with either body format, or send `Prefer: respond-async` to `/v1/images/edits`, to receive a job ID immediately and retrieve the saved result later. Authentication and browser-origin rules are the same as the generation endpoint.

## Context preflight

`POST /api/context`

```json
{
  "model": "my-chat-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "auto_image": false
}
```

Returns the context object directly. Supply the same messages, output budget, tools, `image_sources` and `enhance_prompt` settings as the upcoming request. `tools` includes client function definitions; `auto_image: true` includes available server generation/edit tools and source IDs. This preflight describes the initial chat pass; a subsequent enhancement pass has its own context check. Text preflight can report `fits: false` without running inference. LLM tokenizers run on CPU independently of GPU model residency.

For image input, preflight returns `accuracy: "estimated-multimodal"`, `image_count`, and `visual_tokens_included: false`. If text/placeholders plus the output budget already exceed the configured context limit, `fits` is `false` and inference is rejected. Otherwise `fits` remains `null` because visual-token expansion is not yet known. Numeric input/remaining counts at this stage cover text and placeholders, not expanded vision embeddings. The image request cannot be guaranteed to fit before native preprocessing, which the pipeline performs within generation. Final native performance metrics reconcile `input_tokens`, set `accuracy: "exact"` and `visual_tokens_included: true`, and calculate `fits` from that total. If native input metrics are unavailable, the response retains its estimated label. The prompt digest includes image content, so changing pixels changes `prompt_sha256`. Image downloads/storage also occur during context checks and are deduplicated locally.

## Model registry and lifecycle

| Method | Endpoint | Body or result |
|---|---|---|
| GET | `/v1/models` | Enabled aliases with `task`, `pipeline`, `device`, `context_length`, modalities and capabilities |
| POST | `/api/models/inspect` | `{ "path": "C:/models/export" }` → structural inspection |
| POST | `/api/models` | Register as below; HTTP 201 |
| POST | `/api/models/ALIAS/load` | `{}` → asynchronous job, HTTP 202 |
| POST | `/api/models/ALIAS/unload` | `{}` → asynchronous job, HTTP 202 |
| POST | `/api/models/ALIAS/remove` | `{}` → remove registry entry, preserve disk files |
| POST | `/api/settings` | `{ "model": "ALIAS", ...settings }` or `{ "device": "ID", ...settings }` |

```json
{
  "id": "my-chat-model",
  "name": "Local chat model",
  "path": "C:/models/my-openvino-export",
  "device": "CPU",
  "memory_gib": 4,
  "context_limit": 8192,
  "generation": {"max_tokens": 256, "temperature": 0.7}
}
```

Task/pipeline are inspected; `pipeline` may override the detected choice (`llm`, `vlm`, `image`) subject to task consistency. Legacy `vlm-text` settings migrate to `vlm`. Registration accepts `generation`, `plugin_config`, `scheduler_config`, `template_context`, `mtp`, `idle_timeout_seconds`, and image-only `keep_image_model_loaded` (`"auto"`, `true` or `false`; default `"auto"`). Memory is a configured peak reservation including working/KV and MTP memory, not just disk weight size.

The administration state includes `scheduler.workers: [{model,device,pid,operation}]`. `pid` is the actual native worker PID (null for test fixtures); operation is `text`, `generate` or `edit`. `loaded` and `unloaded` events also include `worker_pid`. An unload failure retains the model in state `cleanup-failed` with an `unload_failed` event. Admission on that device first retries cleanup and fails if it is still unconfirmed; the server does not falsely report released memory. These fields report process ownership, not process RSS or live free VRAM.

Model settings also accept `name`, `device`, `path`, `pipeline`, `memory_gib`, `context_limit`, `enabled`, `max_width`, `max_height`. JSON settings objects replace their previous objects; include values you want to retain. Unsupported runtime properties fail during Load. Supported scheduler keys: `cache_size` (whole GiB), `max_num_batched_tokens`, `max_num_seqs`, `enable_prefix_caching`, `dynamic_split_fuse`, `cache_interval_multiplier`, `num_kv_blocks`, and `num_linear_attention_blocks`. See [MODEL-SETTINGS.md](MODEL-SETTINGS.md) for the controls, validation, and native-device limitations. `/api/state.runtime_controls` exposes the control definitions; it is a settings catalog, not a detected hardware capability list. `idle_timeout_seconds: 0` retains residency until eviction/manual unload.

When LazyRig constructs a native scheduler configuration, omitted `max_num_seqs`
defaults to 1 because each model worker processes one request at a time. Explicit
overrides are retained. `cache_size` includes linear-attention state on hybrid
models and is an initial allocation, not a hard RAM/VRAM cap; 0 allocates on
demand. A native linear-state/cache mismatch is saved with
`error_code: attention_cache_budget_exceeded` and `cache_settings`, without
silently increasing memory or retrying other devices.

OpenVINO GPU exceptions containing `CL_BUILD_PROGRAM_FAILURE` are saved as
`error_code: gpu_kernel_build_failed`, with phase `GPU kernel compilation failed`
and the original exception in the guidance. This identifies compilation failure,
not its cause or an out-of-memory condition. Automatic allocation may continue
to the next selected placement after confirmed cleanup. Fallback events retain this specific reason code.

Registration at `POST /api/models` also defaults omitted `memory_gib` to the selected device's budget. Changing a model's device through `POST /api/settings` applies the new device's budget unless the same request supplies `memory_gib`. Keeping the same device or editing unrelated settings preserves the saved number. Explicit numeric reservations remain supported. See [Memory reservation defaults](MODEL-SETTINGS.md#memory-reservation-defaults).

Device settings accept `budget_gib` and `policy` (`exclusive` or `budget`). Model/device settings changes require an idle server. Compatible workers remain loaded; changes to load-time settings or residency constraints evict only affected workers. Independent device jobs may run concurrently; work sharing a device is FIFO serialized. The queue accepts at most eight jobs.

## Downloads, folders and server settings

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/folders?path=...` | Directory listing on the server, parent and roots |
| POST | `/api/downloads` | Download and optionally convert/register; HTTP 202 |
| POST | `/api/devices/refresh` | Rescan real OpenVINO hardware while idle |
| GET | `/api/downloads` | Persistent download statuses and inspection results |
| POST | `/api/downloads/ID/cancel` | `{}` to cancel |
| POST | `/api/server` | Persist server settings |
| GET | `/api/state` | Models, runtime detection, devices, jobs, downloads, sanitized server settings |
| GET | `/api/health` | Process health, not model readiness; no key required |

```json
{
  "url": "https://huggingface.co/OWNER/REPOSITORY",
  "destination": "C:/models/new-model",
  "revision": "main",
  "subfolder": "",
  "conversion": "auto",
  "precision": "fp16",
  "profile": "auto",
  "task": "auto",
  "register": true,
  "device": "CPU",
  "id": "my-model"
}
```

`conversion` is `auto` (default) or `never`; `precision` is `fp16` (default), `int8`, or `int4`. Existing IR exports are not requantized. `profile` is `auto`, `standard`, or `qwen-image-21`; the Qwen-Image-2.1 architecture always selects its matching profile. `task` may override detection with `text-generation-with-past`, `image-text-to-text`, or `text-to-image`. Compatible VLM exports serve text and image input. An MTP source layer must be exported as a supported OpenVINO head before MTP can be enabled; arbitrary source conversions are not guaranteed to include it.

`register` defaults to false for API callers and true in the console. With registration enabled, `device` must be a configured device; `id` defaults to the destination folder name. Omitting `memory_gib` uses the selected device's configured usable memory budget; a numeric value overrides it. This conservative default permits one resident model on the device and is not measured peak memory or a hard allocation limit. Successful imports include `model_id`; an auto-registration error appears in `registration_error` and does not remove the export. Unsupported source architectures fail visibly.

Job phases: `queued`, `downloading`, `preparing-runtime`, `waiting-converter`, `converting`, `validating`, then `completed`, `failed`, `cancelled` or `interrupted`. Conversion progress is a phase and rolling `log` rather than a fabricated percentage. `bytes` and `total_bytes` describe model-file download progress only. `converted`, `profile`, `revision`, `inspection` and provenance describe the result. Cancel applies to downloads, setup and conversion.

Downloads support public Hugging Face repositories and HTTP(S) ZIPs. Destination must be new/empty. URL credentials are rejected. HF files are pinned to the resolved commit. Cancellation and status survive as job records; unfinished downloads do not resume across restarts.

Server fields: `host`, `port`, `models_dir`, `download_max_gib`, `load_timeout_seconds`, `request_timeout_seconds`, `terminal_metrics`, `api_key`, `inference_api_key`, `allowed_origins`, `browser_access`, `tls_cert`, `tls_key`, `public_base_url`, `max_connections`, `http_timeout_seconds`, `max_jobs`, `max_preparing`, `min_free_disk_mib`, `event_retention_days`. Restart after changing listener/TLS/transport settings; CORS and credentials apply to subsequent requests. Secret key values are omitted from API state responses. The local settings file contains them.

`max_jobs` accepts integers 1–128 (default 8); `max_preparing` 1–32 (default 2);
`max_connections` 1–1024 (default 64); `http_timeout_seconds` 0.1–300 (default 30).
`min_free_disk_mib` accepts finite numbers 0–1,048,576 (default 256 MiB);
`event_retention_days` accepts finite numbers 0–3650 (default 0). Both apply live.
Zero disables the respective disk-admission check or old-text compaction.
`public_base_url` is empty by default, or a canonical HTTP(S) **origin** such as
`https://lazyrig.example`: paths, query strings, fragments and credentials are
not accepted. When set, it determines absolute artifact URLs and explicitly
allows that host/origin through the local listener's checks. Job receipt status/result/event/cancel links remain relative for compatibility.
Forwarded headers are not trusted implicitly. See [CONNECTING.md](CONNECTING.md) for proxy setup.

The `request_timeout_seconds` setting covers the inference request budget as
described above; it no longer restarts independently for every inference phase.

Allowed frontend URLs normalize to origins (scheme, hostname and port); paths and default ports are removed. `OPTIONS` does not require API authentication. Allowed origins receive CORS headers on JSON, errors, SSE and artifacts; preflight permits requested header names, including SDK headers, and handles the legacy private-network request header. Disallowed origins remain blocked and are recorded in `state.server.connection_errors` (last 20 since startup). See [CONNECTING.md](CONNECTING.md) for mixed-content, local-network permissions and certificate setup.

## Asynchronous inference and saved results

- `POST /api/chat/jobs` accepts the same JSON as `/v1/chat/completions`.
- `POST /api/image/jobs` accepts the same JSON as `/v1/images/generations`.
- `POST /api/image/edits` accepts the same JSON or multipart form as `/v1/images/edits`.
- `GET /api/image/jobs` lists up to 200 recent direct image jobs, optionally filtered by `?model=ALIAS`. Each entry includes prompt, creation time, status, phase, `x_result_ready`, `x_warning`, `x_artifacts`, expected `artifact_url`, saved `data[].url`, and result/status URLs. Filtering happens before the limit, so chat activity does not hide image history. This is a read-only recovery route, not a generation request.
- Alternatively send `Prefer: respond-async` to any of these `/v1` inference endpoints. Ordinary requests keep their existing synchronous/SSE response format.

Submission returns HTTP 202 with `object: "inference.job"`, `id`, `x_job_id`, `status_url`, `result_url`, phase, context and any artifacts. Image receipts also include absolute `artifact_url` and `artifacts_url` links immediately; `x_result_ready` distinguishes a future output URL from a saved image. Headers include `X-Job-Id`, `Location` (the result URL) and `Retry-After: 1`. Chat tokenization runs in the queued job, so a slow tokenizer does not hold the submission request open.

Poll `GET /api/jobs/ID/result` using the same bearer key:

| State | HTTP | Body |
|---|---|---|
| Queued/running without a ready image | 202 | Job receipt with current phase, context and saved artifacts |
| Completed chat | 200 | Same OpenAI-shaped chat completion, final text, usage and artifacts |
| Image saved, including during LLM restoration | 200 | Image response, original response format, `x_result_ready: true`, current `x_status` / `x_phase` |
| Cancelled without a saved result | 410 | `request_cancelled`, `x_retryable: false`, `x_status: "cancelled"`, job ID and retained artifacts; `X-Should-Retry: false` header |
| Request deadline expired without a saved result | 504 | `request_timeout`, `x_retryable: false`, retained job identity and artifacts |
| Invalid model tool output | 502 | Saved error code, no executable calls, `x_retryable: false`; `X-Should-Retry: false` |
| Failed/interrupted | 500 | Error, `x_status`, job ID, context and retained artifacts |

A direct image job that saved its image before restoration failed retains the established image response with `x_warning`. Client tool-call IDs are stable across repeated result retrievals. Reading a result never queues inference. Saved chat/image jobs can also be retrieved after an upgrade; load/unload control jobs use the plain status route.

### Safe inference retries

Chat completions, image generation and editing endpoints, synchronous or asynchronous, accept an optional `Idempotency-Key` header (1–128 visible ASCII characters, no spaces). Choose one key per intended inference request and persist it **before** the first POST. Repeating the same operation and request body with that key returns the original job/result; it never replays inference, including for failed, cancelled or interrupted jobs. A cancelled request without an image returns HTTP 410 on resubmission, including async endpoints; clients must stop polling or retrying it. Identities persist in the server database across restart. An interrupted job is not resumed automatically.

Reusing a key with a different body or operation returns HTTP 409 with `x_job_id`, `status_url` and `result_url` for the original request. JSON field order and multipart boundaries/filenames are ignored; parsed values and image bytes participate in identity. Keep the same request fields on retry, including response format. Validation failures before admission do not reserve the key. Keys are retained with the database without automatic expiry. Lookup keys are hashed; a 32-character lowercase hexadecimal key also appears publicly as its new job ID and, for an image request, its artifact filename. They are not authentication credentials; all requests still require the server's bearer key. A deliberate new generation needs a new key. Unkeyed requests, even with identical prompts, create independent jobs.

For a new image request with a 32-character lowercase hexadecimal `Idempotency-Key`, the exact PNG URL is `/artifacts/KEY.png` and the result route is `/api/jobs/KEY/result`, known before submission. Existing keys keep their original IDs. Other supported key strings continue to receive server-generated job IDs. Conflicts with an existing job ID or artifact file return 409 without overwriting it. Use `Prefer: respond-async` for an immediate receipt. Before acceptance, the output route is 404; during generation it is 202 JSON with `Retry-After: 1`; once saved it is 200 image bytes; a cancelled job without an image returns 410, and a failed image job returns 500 with its error.

`GET /api/artifacts` lists up to 200 existing PNG/SVG artifacts, newest modified first. Each item has an absolute `url`, filename, byte size, modification time and available job/model/prompt metadata. Files without job records remain discoverable, with null job metadata. In-progress files not yet recorded as saved are excluded. It lists only supported artifact names inside the artifact directory. The job list additionally reports pending jobs; neither listing starts inference. `/api/health` advertises both routes under `links`. All artifact and job routes use the configured bearer key.

No extra client software is required. See [image results and artifact discovery](CONNECTING.md#image-results-and-artifact-discovery) for the plain HTTP contract. `tools/image_job.py` is optional. Preserve the server data directory when upgrading.

In the console, **View result / View progress** opens a saved job. The chat workspace saves each stage and key in IndexedDB before submission. A lost receipt triggers an identical keyed retry, never a fresh generation. Accepted jobs continue independently; subsequent client stages resume when the browser returns. Server restart retains completed results; interrupted jobs are not resumed automatically.

`browser_access` defaults to `"origins"` (the exact allowlist). Set it to `"api-key"` with an API key of at least 16 characters to accept arbitrary frontend origins, including `Origin: null`, for inference, model listing, context, job/status/result/cancel/events and artifact routes. Actual protected requests still require the bearer key; preflight does not. This mode does not grant arbitrary origins access to administration, downloads, file browsing or settings. Explicitly listed origins keep their existing access. Missing keys become readable 401 responses for these browser clients. CORS cannot override mixed-content, certificate, CSP or browser local-network permissions.

## Native job protocol

Stream persistence stores compact text updates internally. The stored-event
reader accepts untouched legacy snapshots and partially upgraded metadata.
Startup does not rewrite historical event bodies or build new secondary event
indexes over a populated legacy table. Old database space is retained; explicit
historical compaction can still be I/O intensive. The external event
contract remains cumulative text/content/reasoning with ordered sequence cursors;
clients should keep their existing replay assembly logic. Legacy event rows remain
readable. Optional `event_retention_days` (default 0, disabled) compacts repeated
text events for terminal jobs older than that many days. Idle maintenance checks
about once per minute and handles up to 20 eligible jobs per pass. It retains the
last cumulative text checkpoint at its original sequence/time and every non-text
lifecycle/artifact event. A replay cursor may therefore skip removed intermediate
text sequences and receive the retained cumulative checkpoint.

Compaction preserves all jobs, messages, idempotency keys, inputs and artifacts.
It frees SQLite pages for reuse; it does not run VACUUM or promise an immediate
file-size reduction. Overall storage remains operator-managed.
`min_free_disk_mib` defaults to 256 MiB of free space on the data-directory volume.
Admission/readiness return `disk_pressure` when free space is below that threshold
or cannot be measured. This is admission backpressure, not a hard quota or a
promise that a running request's output fits. Zero disables the check. If terminal snapshots cannot be
written, new durable work pauses. Receipts/results expose `x_durable: false` and
`x_storage_warning` for a terminal snapshot retained only in the server process.
When no earlier error exists, `error_code: "storage_failure"` identifies the failed
final save. A background write probe runs about once per second and pending
terminal snapshots must be flushed before storage health clears. A process-local
snapshot is used consistently by status/result and recent-job listings while the
server stays alive. It is not a promise that the final record will survive a restart: recovery
may mark an uncommitted terminal job interrupted. Already published image files
remain available; inspect both the result and durability fields. Cleanup still
attempts worker termination if writing a cleanup event fails. If the initial job
and retry key were saved but the queued event fails before inference starts,
the same identity is finalized as failed rather than left queued without a worker.


- `GET /api/jobs/ID`: saved status, context, result text, error, request settings and artifact references.
- `GET /api/jobs/ID/events`: ordered SSE lifecycle, text, context, progress and completion events. Resume using `Last-Event-ID` or `?after=SEQUENCE`. Event stream ends with `event: end`.
- `POST /api/jobs/ID/cancel` with `{}`: idempotent cancellation, using the same bearer key as submission. Returns `ok`, `id`, `status`, `cancel_requested` and `status_url`. The acknowledgement means the stop was accepted; wait for terminal `status: "cancelled"` or the `finished` event to confirm worker cleanup. Completed jobs are unchanged. Queued jobs are removed before inference; running work releases its worker and device lease. Images saved before cancellation remain downloadable. A persisted cancellation stays cancelled across server restart.
- `GET /artifacts/ID.png`: saved PNG image.

Legacy native session routes remain available: `POST /api/sessions` creates a session; `GET /api/sessions/ID` returns its messages; `POST /api/jobs` submits `{session_id,model,prompt,max_tokens,tools_enabled,image_model,width,height,steps,seed,temperature}`. Disable `tools_enabled` for a native text-only job. This protocol appends history on the server; `/v1/chat/completions` expects full client-provided history instead.

SQLite stores native sessions and per-request API sessions. Active jobs become interrupted after server restart. No automatic regeneration occurs.

## Generate an API key

`POST /api/server/api-key` with `{}` returns `{"api_key":"lr_...","saved":false}`. The key contains 32 cryptographically random bytes encoded as URL-safe text after the `lr_` prefix. The response is not cached. This is an administration route: existing bearer authentication and administration origin restrictions apply, including when arbitrary browser origins are enabled for inference.

Generation does not save or activate the key. Apply it with `POST /api/server` containing `{"api_key":"the generated key"}`, authorized with the currently active key if one exists. The previous key stops working after a successful save. `/api/state` returns only `api_key_set`, never the key. In the console use **Generate key → Copy key → Save server settings**; Save updates the console's own authentication. Copy uses the unsaved field, or the current browser connection key when that field is empty. This creates a credential value, not a multi-user identity. The optional `inference_api_key` is a separate shared role described above; generation alone does not activate either role.

## Image settings

Qwen-Image-2.1 uses width/height multiples of 32, including all seven sizes listed on its official model card. Other image adapters retain the existing 64-pixel policy. Configured maxima still apply, and violations identify the requested dimensions and current server limits. LazyRig imposes no separate maximum-area rule. Neither the largest preset side (2752) nor the scheduler's `max_image_seq_len` is a model-resolution cap.

`GET /v1/models` and `/api/state` include `image_size_constraints` (multiple, minimum, configured maxima and presets) and `diffusion_scheduler_info` (export configuration, merged configuration, selected type and available controls). New Qwen registrations have 2752×2752 configured maxima; existing registrations retain their saved limits.

Set per-model diffusion settings through `POST /api/settings`:

```json
{
  "model": "my-image-model",
  "max_width": 2752,
  "max_height": 2752,
  "diffusion_scheduler": {
    "type": "FLOW_MATCH_EULER_DISCRETE",
    "config": {
      "use_dynamic_shifting": true,
      "base_shift": 0.5,
      "max_shift": 0.9,
      "base_image_seq_len": 256,
      "max_image_seq_len": 8192,
      "shift": 1.0,
      "shift_terminal": 0.02,
      "num_train_timesteps": 1000
    }
  }
}
```

`diffusion_scheduler: {"type":"AUTO","config":{}}` restores the export's scheduler without editing its files. An override is saved with the registered model and applied on worker creation for both generation and editing. It is not an arbitrary per-request plugin object. Incompatible samplers and unsupported override keys fail before replacing settings.

Image generation/edit requests accept the optional `max_sequence_length` integer for Qwen-Image-2.1, in JSON or multipart edits. It overrides the saved `generation.max_sequence_length`. A missing saved/request field preserves the native text-encoder default. `negative_prompt` remains a string (including an intentional empty string) and requires guidance greater than 1 when guidance is explicit. Both options pass through asynchronous jobs and saved settings used by LLM image handoffs. Disable/remove a saved negative prompt to omit it entirely; an empty prompt can still activate Qwen's extra CFG pass. No dependency reinstall is required.

## Model device allocation

`POST /api/settings` accepts a per-model `placement` object. `device` remains the
primary and must match the first entry. For example:

```json
{
  "model": "my-text-model",
  "device": "GPU.1",
  "placement": {
    "mode": "auto",
    "devices": ["GPU.1", "GPU.0"],
    "shares": {},
    "allow_fallback": false,
    "shared_memory": "budget"
  }
}
```

`mode` accepts `auto`, `single` or `split`. Single requires exactly one device;
Split requires at least two and attempts precisely that combination, without
whole-model fallback. It rejects enabled MTP, missing devices at load time and
more than one shared-memory stage (integrated GPU or CPU). CPU stays last in a
mixed selection. Image and NPU targets remain single-device. The split runtime
chooses partition sizes; `budget_gib` is not forwarded as a partition limit.

Auto tries the primary, then supported selected GPU groups / GPU-CPU combinations.
`allow_fallback` defaults to false, including when omitted in older settings.
Set true explicitly to allow whole-model selected GPU / CPU attempts after splits.
MTP disables split candidates; it is never switched off automatically.

`shared_memory` is `budget` by default, preserving ordinary stored-weight
screening against the configured budget, including budgets above dedicated VRAM.
`allow` adds a primary-GPU native attempt when that screen rejects its weight
size. In Auto it follows splits; in Single it is the sole native attempt. This
books the full primary planning budget, not measured usage, and cannot force
driver support. An already attempted primary is never retried by this setting.

Manual percentages remain retired. Saved manual configuration migrates to Auto
with the same devices; new manual requests fail atomically. Advanced native
placement JSON stays a separate expert path and disables the managed picker.

`GET /api/state` exposes `placement_info` (including stored language-weight size,
reservation-floor warning and split eligibility), `effective_placement`, and
hardware support metadata. Labels do not establish native model compatibility.
A full-GPU reservation is at least the known language-weight size unless explicit
native weight offload is configured. Split reservations book each device budget.

New `placement_plan` and `placement_attempt` events describe ordered candidates
and statuses (`not_attempted`, `skipped`, `loading`, `failed`, `succeeded`,
`cancelled`). Jobs retain the latest load plan and its attempts; their event
history retains earlier loads in the same job. Original native errors are bounded
to 4096 characters per attempt; dashboard summaries include 600-character excerpts.
Compilation failures and inference-probe failures have distinct reason codes.
Existing `placement_fallback`, `placement_selected` and weight-screen events remain.
No architecture incompatibility is inferred solely from a device combination.

Full failed jobs and failed load attempts can include `worker_diagnostics`:
installed `runtime` versions, `fault_handler_enabled`, `validation` checkpoints
and reported token counts, and a bounded `fault_log` (up to 8192 bytes of text).
The fault handler records Python stack locations, not prompt text or tensor data;
its output is not a native crash dump or proof of a particular driver defect.
An empty/unavailable trace is explicit. Buffered token reporting may combine
prefill and decode, so the last checkpoint is not an exact native kernel stage.
Use the dashboard's **Copy load details** for the full evidence; compact job
summaries omit traces. Natural process exits are observed for up to two seconds
after IPC closes, within cancellation and request/load deadlines. Cleanup's own
termination code never replaces an unknown original exit status.

All attempts keep the original load/request deadlines. Cancellation, timeout,
failed cleanup and unrelated failures stop retries; generation is not replayed.

`load_memory` events report each worker load's `initial`, `ready` or `failed`
stage, model and attempted device. Jobs retain the latest 32 entries in
`load_memory_events`; the event history retains the emitted events. Fields
include `status` (`available` or `unavailable`), `source`, `total_gib`,
`available_gib`, `headroom_gib`, `minimum_available_gib`, `sample_count`,
`job_commit_limit_bytes`, `job_peak_commit_bytes` and `job_commit_limit_scope`.
The Windows peak is measured committed memory for the worker job, not VRAM,
physical RAM usage or proof that a failed allocation exceeded the limit.
Unknown readings and an
absent installed limit are represented by null values. The minimum is sampled,
not a measured peak allocation. On Windows the installed commit limit lasts
for the worker's lifetime; it is not a per-device VRAM cap. A native load-time
`MemoryError` is eligible for automatic placement fallback only after cleanup;
generation errors do not trigger placement retries.

## Terminal feedback and model residency

For a CPU device, `POST /api/settings` accepts
`{"device":"CPU","inference_threads":8}` to set its default thread limit.
Use `"inference_threads":null` to restore Auto; omit it to keep the current
setting. Explicit values must be positive whole numbers, and the setting is
rejected for non-CPU devices. It applies only to CPU inference, including CPU
subgraphs in HETERO placements. Explicit model thread properties take precedence.
This is a thread limit per compiled CPU model, not a global process thread cap.

`POST /api/server` accepts `{"terminal_metrics": true}` (default). It prints throttled prompt-processing speed, decode speed, TTFT, context counts and model activity without requiring the client to request timing fields. Prompt speed uses `prompt_per_second` and is labelled `TTFT estimate`: input tokens divided by the worker interval to the first token callback, rather than isolated prefill-compute throughput. It becomes available after the first token and can reflect prefix-cache reuse; visual token counts may remain estimated until final native usage. Missing rates show `n/a`. Set `terminal_metrics` to false to disable this feedback. Client `timings_per_token` behavior is unchanged. Context counts are token usage, not measured KV-memory occupancy.

`POST /api/settings` accepts `{"device": "GPU.0", "policy": "budget", "budget_gib": 22}`. Newly detected devices default to `budget`; existing saved policies remain unchanged. Explicit `exclusive` unloads other workers before admission. `budget` keeps them when the sum of model `memory_gib` reservations fits; otherwise it unloads the least recently used workers. Image retention defaults to `"auto"` and keeps a successful worker unless models required immediately cannot share the device. Save `{"model": "YOUR_IMAGE_ALIAS", "keep_image_model_loaded": "auto"}` to adopt this for an older model. Existing saved booleans remain valid: `true` explicitly retains between requests, while `false` always unloads. Automatic considers current residents and models required for immediate restoration or managed continuation. Displaced text models or prompt enhancers waiting for a later request do not consume reservations; unrelated devices and unused registrations are ignored. Failures/cancellation still release the affected image worker. Device requests remain serialized. Reservations must cover weights, KV cache and runtime buffers; they are not a measurement of available memory.


`restore_text_after_image` is a per-image-model boolean, default `false`. Direct image jobs release displaced text models and let the next text request load its requested model. Set it to `true` through model settings to restore previous text models before job completion. This can add an unnecessary swap between consecutive image requests; it is independent of image retention. Legacy managed chat still loads its LLM for its required continuation.

## Receipt fields, dashboard summaries and streaming

Asynchronous receipts include `events_url`, `cancel_url` and `x_handoff_complete`. Image responses also include `x_handoff_complete`, true only when the job completed successfully. `x_result_ready` can be true while restoration is still running or after restoration failed; use `x_status` / `status_url` to distinguish these cases. All retrieval URLs use the same bearer authentication as submission.

`GET /api/state?summary=1` returns dashboard-sized job records with prompt previews and no prepared token arrays or full conversation payloads. The full job record remains at `/api/jobs/{id}`. Ordinary `GET /api/state` retains its existing response shape.

Text SSE sends incremental content/reasoning for client-tool requests. Tool-call arguments are sent only after complete validation; malformed calls produce an error and no executable partial call. Text before a possible `<tool_call>` marker can stream, with trailing whitespace held until the final parser determines whether it is a tool preamble. The initial server-managed image-routing pass remains buffered; the final continuation streams. Final response retrieval remains authoritative. SSE responses include `X-Accel-Buffering: no` and `Cache-Control: no-cache, no-transform`.

## Stop and disconnect semantics

Image and chat job receipts include `disconnect_policy` (`continue` for background jobs, `cancel` for the original synchronous request) and `x_cancel_requested`. Replaying a chat or image key does not change the original job’s lifetime policy. Closing a GET observer never cancels a job. Existing records without a policy retain their background behavior.

Clients implementing **Stop** for background jobs must POST `{}` to the receipt’s `cancel_url`; closing the progress stream or stopping local polling conveys no cancellation intent. Stop response retries are safe. Keep the same image request key for recovery; a fresh key or an unkeyed POST is a separate generation, even when the prompt is identical. A client’s automatic retries must not create fresh requests after an intentional stop. If a proxy keeps the upstream connection open after its downstream client disconnects, use explicit cancellation or configure the proxy to propagate request aborts.

The dashboard persists stop intent before delivery, including clicks before the submission receipt arrives. If delivery fails, it reconnects and retries only cancellation. An unconfirmed submission may be repeated with the identical saved key to recover and cancel that same job; it never creates a new generation. HTTP 410 cancelled errors are distinct from HTTP 500 inference failures; streamed chat emits the same `request_cancelled` error before `[DONE]` when the stream is still connected. A fully saved image is returned with HTTP 200 even if cancellation stops its later restoration; inspect `x_status` and `x_handoff_complete`.

## Dashboard instruction files

Authenticated `GET /prompts/image_generation.txt` and `/prompts/image_editing.txt`
return LazyRig's independently authored Apache-2.0 instructions. The dashboard
uses them as complete system messages in separate structured-output chat requests.
The former `/prompts/system_prompt_t2i.txt` and `/prompts/system_prompt_edit.txt`
URLs resolve to the same LazyRig instruction files. These defaults apply only to
explicitly enabled enhancement. Clients
can prepare their own final prompts and leave enhancement disabled. See
[prompt enhancement](PROMPT-ENHANCEMENT.md) for the workflow and output contract.

## Runtime maintenance (dashboard administration)

These routes follow the dashboard's API-key and exact-origin checks. They are
not inference-client endpoints.

| Method and path | Purpose |
| --- | --- |
| `POST /api/runtime/check` with `{}` | Start a stable-release check; returns status and a `check_id` |
| `GET /api/runtime/update` | Read the check status, installed/target versions and last update result |
| `POST /api/runtime/update` with `{"check_id":"…"}` | Install the freshly checked target and restart the server |

Installation requires a successful check no older than 15 minutes, an unchanged
managed environment and no active requests, queued jobs or imports. The target
and restart command cannot be supplied by API clients. Once installation is
accepted, new POST work receives HTTP 503 until shutdown; the server is offline
during installation. Poll the status route after restart to read `last_update`.
See [Runtime updates](RUNTIME-UPDATES.md).
