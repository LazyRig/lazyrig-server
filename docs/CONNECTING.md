# Connect a frontend to LazyRig

The current API serves language, vision-language and image models. Use the
documented [API subset](API.md) when connecting a chat client, an image tool or
your own application. Client setup uses the same API whether LazyRig is hosted
on Windows or Linux.

## Configure the server

1. Open the administration console at `http://127.0.0.1:8765` on the server machine. Check that the displayed version matches your installation.
2. Open the settings gear in **API & server**. For straightforward frontend access, use **Generate key**, click **Copy key**, choose **API & server → Browser access → Any frontend with API key**, save, and enter that key in the frontend. This also accepts local-file/sandbox origins for inference. Alternatively keep **Listed frontend origins**, paste a URL such as `http://localhost:5173` or `https://your-frontend.example` into **Allowed browser origins**, and save. Use the frontend address here; the server address belongs in the frontend's endpoint setting. Paths are removed automatically, and CORS changes apply immediately.
3. Set the frontend's OpenAI-compatible base URL to `http://127.0.0.1:8765/v1` for a local HTTP connection. A field asking for the full chat endpoint instead needs `http://127.0.0.1:8765/v1/chat/completions`. Use a registered model alias, as returned by `/v1/models`.
4. If authentication is configured, give the frontend the owner key for your own trusted client, or configure the optional inference-only key described below. It must send `Authorization: Bearer YOUR_KEY`. Do not put the key in the URL.
5. Configure the frontend's image provider with the same server base URL and a registered image-model alias. Its tool implementation calls `/v1/images/generations` or `/v1/images/edits`, then supplies the result to the chat model. The LLM and its export template must support the client's declared tools. There is no global image-routing toggle. In the dashboard, select **Chat → Conversation settings (sliders icon) → Image tool**.

Models load automatically. [Model residency](MODEL-SETTINGS.md#keep-multiple-models-loaded) determines which workers can remain loaded. On a shared device, exclusive residency unloads competing workers; budget residency retains them when their configured reservations fit. **Automatic** image retention is the default and keeps the image worker when it fits the request's working set and device policy. Displaced text models normally load again on their next request; **Text model after images** can explicitly restore them earlier. Client timeouts must cover queueing, loading and inference. For a tool-using thinking model, allow enough output tokens for both reasoning and a complete tool call.

Client tool calls and their follow-up messages reuse the compatible loaded chat
model. Changing `max_tokens`, `max_completion_tokens`, sampling or prompt history
does not reload it. Explicit client output limits take precedence; 8,192 is the
fallback when neither the client nor the model supplies a limit. A rejected tool
call or invalid JSON response still fails the request, but keeps the live text
worker if inference completed normally. Native failures, Stop and deadlines
still retire it. `generation complete` in the terminal describes inference only;
wait for `job completed` to confirm that response validation also passed. Failed
jobs show a safe failure category; the full error is in Recent activity and the
authenticated `/api/jobs/{id}` response.

For a frontend on another computer, select listen address `0.0.0.0`, set an API key of at least 16 characters, save and restart. Use the server computer's reachable address in the frontend; `localhost` always refers to the computer running the browser. The operating system firewall must permit the chosen port.

For a temporary command-line origin setting:

```bat
start-server.bat --allow-origin "http://localhost:5173"
```

## Inference-only client credential

The existing dashboard **Client API key** connects with the owner credential and
needs administration access. For another inference client, configure an optional
`inference_api_key` through authenticated `POST /api/server`. Generate a separate
strong key using the existing key-generation route, then save it in that field;
do not replace the owner's `api_key`. Both keys must be at least 16 characters
and must differ. Empty `inference_api_key` disables the extra credential.

The inference key supports models, chat/images, context, job status/events/results,
cancellation and artifact downloads; administration returns 403. This is suitable
for separating a trusted client's inference access from settings access. It does
not isolate clients from one another's jobs or artifacts. Keep the owner key in
the dashboard. Full examples and readiness are in [API.md](API.md).

## HTTPS frontends

An origin allowlist cannot override browser mixed-content checks, certificate validation, local-network permissions, or the frontend's own Content Security Policy. Use an HTTPS endpoint or route the API through the frontend's same-origin HTTPS proxy.

LazyRig can serve HTTPS directly. Under **API & server → HTTPS endpoint**, enter the PEM certificate and private-key file paths and restart. The certificate must be trusted by the browser and cover the hostname used in the endpoint. A self-signed certificate is not automatically trusted. Leave both fields empty for HTTP. The server does not obtain certificates or alter trust stores.

The equivalent launcher options are:

```bat
start-server.bat --tls-cert "C:\certs\localhost.pem" --tls-key "C:\certs\localhost-key.pem" --allow-origin "https://your-frontend.example"
```

For this example, configure the frontend's base URL as `https://localhost:8765/v1` and use a certificate valid for `localhost`. If it connects by IP, the certificate must also cover that IP. Certificate/key errors stop startup with an actionable message.

Browsers may ask the frontend for local-network access permission. Grant that permission when connecting it to your local server. If the frontend is in a restricted embedded frame, its hosting page may need to delegate that permission. A frontend CSP must also permit the API address in `connect-src`. These settings belong to the frontend/browser, so server CORS headers cannot change them.

## Diagnose “Failed to fetch”

- Open `/api/health` at the exact server address from the same browser. If it fails, check the server process, listener address, port, certificate and firewall.
- In the terminal or **Recent rejected browser origins**, check the explicit reason, `Origin` and `Host`. Add an HTTP(S) origin with **Add latest HTTP(S) origin to form**, then save; or select the key-protected mode. For a direct local connection, use localhost; for a same-host reverse proxy, configure its canonical `public_base_url`; direct network access requires listen address `0.0.0.0` and restart. Rejections are recorded only when a request reaches the server. If the browser blocks a request before sending it, check the browser's Network/Console panels for mixed-content, certificate, CSP or local-network errors.
- Successful preflights return HTTP 204 without requiring a bearer key. They allow custom SDK headers as well as `Content-Type`, `Authorization` and `Last-Event-ID`. Authentication failures for an allowed origin are readable HTTP 401 responses, rather than opaque CORS failures.
- A `file://` page or sandboxed frame may have origin `null`. It is rejected in the default origin-list mode. **Any frontend with API key** supports this origin for authenticated inference. Serve a local frontend over HTTP, for example run `python -m http.server 5173 --bind 127.0.0.1` inside its folder, then open `http://127.0.0.1:5173` and add that origin.

Chrome may request `/.well-known/appspecific/com.chrome.devtools.json` when its
developer tools open. This optional workspace-discovery request is unrelated to
inference or API credentials. LazyRig returns 404 without logging the expected
probe. A 401 on `/api/state` or an inference route still requires the configured
API key; `ERR_CONNECTION_REFUSED` means the server could not be reached.

## Display generated images

A standard image response contains `data[].url` or `data[].b64_json`. The client displays the image and sends a matching tool result to the LLM. The dashboard implements that sequence. The deprecated server-managed chat extension returns final assistant text plus `x_artifacts`. Streaming returns the artifacts in the final completion chunk. Frontends must read this extension to show the images; ordinary text-only chat renderers may ignore it. The included chat workspace displays them.

Artifact URLs are relative to the API server and require the same bearer key when authentication is enabled. Fetch the image with the authorization header, convert the response to a Blob URL, then set that URL on the image element. Do not resolve `/artifacts/...` against the frontend's host or put API keys into image URLs. Direct `/v1/images/generations` requests can use `response_format: "b64_json"` instead.

For an external TLS reverse proxy, set `public_base_url` through `POST /api/server`
to the canonical public origin, for example `https://lazyrig.example`. It controls
absolute image URLs and explicitly permits that host/origin on
the loopback listener. Configure the proxy to preserve that Host header and
forward authorization, stream SSE without response buffering, and impose its
own connection/body limits. Forwarded headers are not implicitly trusted.
Job receipt status/result/event/cancel links remain relative; resolve them against
the API server origin. Do not include a URL path prefix in `public_base_url`; subpath hosting is not
implemented. Keep backend loopback binding when the proxy is on the same host.
Use a strong owner key and the separate inference key as appropriate.

With no public origin configured, URLs retain the listener's HTTP/HTTPS scheme
and the request Host. `response_format: "b64_json"` remains an option for clients
that cannot fetch authenticated image URLs. `/api/ready` provides authenticated
readiness; `/api/health` alone only demonstrates liveness.

Failed tool handoffs include `x_tool_diagnostics`, the raw LLM response and context usage. An image generated before a later LLM failure is retained in `x_artifacts`. Recent server activity also exposes the attempted tool responses.

References: [MDN CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), [MDN local-network access](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Local_network_access).

## Image exists, but the final LLM reply was not received

A traceback in `wfile.write` / `socket.sendall` means response delivery failed. It does not by itself establish whether a client timeout, proxy, network change or local security software closed the connection. Check the saved job before resubmitting.

Aborted/reset/broken connections are caught without attempting a second response. Background jobs keep running, while synchronous requests cancel unfinished inference when their connection closes. Already saved images and final results are preserved. To recover an earlier job, open **Recent server activity → View result**. This reads saved text and artifacts without generating again. A failed model restoration is displayed as a job error with its retained image; it is not reported as a successful LLM continuation.

The chat workspace now returns a job ID immediately, streams progress, displays the saved image while the LLM resumes, and displays final assistant text above the response data. If the connection drops or the page reloads, it resumes tracking the same job.

For external frontends, enable `stream: true` and allow sufficient time for model swaps. Streaming sends periodic JSON progress chunks, including the image before final text. A frontend's hard total timeout still needs increasing; server timeouts do not override it. For clients you can modify, use `/api/chat/jobs` (or `Prefer: respond-async`) and poll the returned `result_url`. Every API/result/artifact request uses the same bearer key. Never automatically resubmit an uncertain request before checking whether the first job already exists.

## Frontend template extension errors

LazyRig accepts `chat_template_kwargs`, `reasoning_effort` and `timings_per_token` on chat requests. You may keep the frontend's template options and reasoning selector enabled. Template options apply per request; `none` disables thinking. If the frontend sends both a positive effort and `chat_template_kwargs: {"enable_thinking": false}`, thinking remains disabled. Qwen3.8 maps frontend `minimal`/`high` to its native `low`/`xhigh` levels.

If the same unsupported-fields message remains after upgrading, stop the old server process, replace the application files, start it again and refresh the administration page. Check the version reported by `/api/health` at the exact endpoint configured in the frontend matches your installed release. No setup reinstall is needed for this patch. Invalid field types and unsupported fields beyond the documented API still receive validation errors.

## Hermes response channels and titles

Reasoning is returned separately as `reasoning_content` in both ordinary and streamed responses. `response_format` now accepts Hermes's `json_schema` title request; leave that request enabled. JSON generation temporarily disables thinking without changing saved model preferences. Valid tool calls with inline code in their preamble are accepted; quoted examples stay text. Keep a sufficient token budget: a model that stops before finishing reasoning or a tool call cannot provide a complete executable call.

## Hermes and other vision clients

For image analysis, choose **Models → Settings → VLM · text + images**, save, and Load the complete vision export. Existing VLM registrations migrate automatically. Refresh the frontend's model list or reconnect it: `/v1/models` now advertises image input on VLM aliases. If the frontend has its own vision-capability override, enable it for that alias. LazyRig cannot change a frontend's independent model allowlist.

Send images as user-message or tool-result `image_url` content parts, using a base64 data URL or a public HTTP(S) image URL. A local path or private URL cannot be fetched as an image; use upload/base64. The included **Chat → Image attachment** exercises the same API path. Vision input is independent of the server's image-generation tool switch. Requests with images work in both streaming and ordinary responses, and input images survive an image-generation model swap. Full payload examples and size limits are in [API.md](API.md).

If you enable MTP in model settings, set frontend temperature to **0** or omit it to use the model default. Nonzero temperature is rejected with an explicit MTP message. The server disables prefix caching for MTP and exposes the applied draft count in `x_mtp`.

## Hermes vision tool results

A VLM accepts image parts in user messages and client tool results, including Hermes's `vision_analyze` output. Keep the preceding assistant tool call and matching `tool_call_id`. LazyRig preserves the role and pairing while supplying the pixels to the native VLM. Historical tool images survive model reloads.

Use a VLM alias and a nonempty `tool_call_id` on each image-bearing tool result. Check that `/api/health` at the client's configured address matches the release you installed.

## Hermes image editing

The edit endpoint is **`POST http://127.0.0.1:8765/v1/images/edits`**. It uses the same image-model alias and API key as `/v1/images/generations`. Keep the image provider's base URL at `http://127.0.0.1:8765/v1`; the SDK appends `/images/edits` itself. Use your reachable server address if Hermes runs elsewhere.

In Hermes, select its **OpenAI-compatible image provider**, set its model to your registered Qwen image alias, and point its image-provider connection to LazyRig. This image connection is separate from the chat-model selection. The inspected Hermes provider supports `image_gen.openai.base_url` and `image_gen.openai.key_env`, with `OPENAI_BASE_URL` and `OPENAI_API_KEY` as fallbacks; `OPENAI_IMAGE_MODEL` selects a custom image alias. For an otherwise unconfigured provider, Windows Command Prompt can supply those fallbacks before launching Hermes:

```bat
set "OPENAI_BASE_URL=http://127.0.0.1:8765/v1"
set "OPENAI_API_KEY=YOUR_LAZYRIG_KEY"
set "OPENAI_IMAGE_MODEL=my-image-model"
hermes
```

Existing explicit provider settings can take precedence over environment fallbacks. Keep credentials in your existing key/environment configuration. No OpenAI cloud account or cloud model is involved when the configured URL points to LazyRig.

Hermes's edit path calls `client.images.edit(...)` with a named source file. LazyRig accepts that multipart format (`image` or a single `image[]`), plus JSON for direct callers. Send **one source image**, an edit prompt, `model`, and optionally `size` and `n: 1`. The default result contains `b64_json`, which Hermes can save without a second authenticated request. Examples are in [API.md](API.md#image-editing). A local file available to Hermes can be uploaded; a client-side path in a JSON request cannot be opened by the server. Hermes may reject a private artifact URL before contacting the server; use its saved local output as the source, or make a direct JSON edit request with LazyRig's artifact path.

Check **Models → Settings → Image editing** before testing. A complete Qwen-Image-2.1 export needs `vae_encoder`, `vision_encoder`, and `text_encoder_i2i` XML/BIN pairs. Use **Chat → Edit image** to verify the upload/API route. The server switches between generation and editing automatically; manual unloading or a second model registration is unnecessary.

Ask for square 1024×1024 output initially. Hermes may select 1536×1024 or 1024×1536 for other aspect ratios; raise the image model's maximum dimensions and memory reservation if you want those sizes. Qwen edits do not use `strength`. Masks, multiple reference images and transparent output are not supported by this adapter.

Keep the client's timeout long enough for native model loading and editing; a subsequent chat request may also wait for LLM restoration. Direct integrations may use `Prefer: respond-async` and poll the returned result URL; Hermes's standard SDK image call is synchronous. Closing a synchronous request cancels unfinished work. A timeout also closes the connection; use async submission if work must survive it. Background jobs require an explicit POST to `cancel_url` to stop them. Use the recovery workflow below before repeating an uncertain edit. Server-managed `generate_image` chat routing remains text-to-image; Hermes's own edit tool should call `/v1/images/edits`.

## Image results and artifact discovery

No extra script or Hermes-specific integration is required. The normal `POST /v1/images/generations` response returns the saved image in the standard field:

```json
{"data": [{"url": "http://SERVER:8765/artifacts/0123456789abcdef0123456789abcdef.png"}]}
```

The response also carries `artifact_url`, `artifacts_url`, `x_job_id`, `x_result_ready`, and status/result links. The URL is returned once the image is saved, independently of LLM restoration. The frontend receives it as the image tool's result and can pass that result to its agent normally. LazyRig owns model unloading, generation and LLM restoration; the LLM does not need to locate files or coordinate the swap. Download the URL with the same bearer key used for inference. Clients unable to attach authentication to an image download can request `response_format: "b64_json"`; that response includes both the bytes and the URL.

For images already generated, use **`GET /api/artifacts`** with that key. It lists up to 200 existing images, newest modified first, with full URLs and available job metadata. It also finds older files without a database job record. `GET /api/image/jobs` lists recent direct image jobs, including pending jobs, prompts, expected `artifact_url` and saved `data[].url`. Neither GET endpoint generates anything. `/api/health` advertises both discovery routes under `links`.

### Know the output URL before submission

A client can choose a fresh 32-character lowercase hexadecimal UUID and send it as `Idempotency-Key`. For a new key, that value becomes the image job ID and filename. The client therefore knows both URLs before sending:

```text
Image:  http://SERVER:8765/artifacts/0123456789abcdef0123456789abcdef.png
Result: http://SERVER:8765/api/jobs/0123456789abcdef0123456789abcdef/result
```

Use a new random ID for each intentionally new image; the value above is only an example. Submit through the same image endpoint:

```http
POST /v1/images/generations HTTP/1.1
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
Idempotency-Key: 0123456789abcdef0123456789abcdef
Prefer: respond-async

{"model": "my-image-model", "prompt": "A quiet subway platform", "size": "auto"}
```

`Prefer: respond-async` returns HTTP 202 immediately with the job ID and absolute `artifact_url`. The alternative `/api/image/jobs` route always uses that asynchronous behavior. A client that lets the server choose its ID still receives the same URL in its receipt. Synchronous calls continue to return the normal image response when ready.

Before submission, the reserved-by-convention URL returns 404 because the job does not exist. After acceptance, GET of that same artifact URL returns 202 JSON with `x_result_ready: false` and `Retry-After: 1` while pending; it returns 200 image bytes when ready, including during cleanup or optional LLM restoration. A failed job with no image returns an error, never a placeholder image. Read the result URL for JSON metadata and any cleanup/restoration warning. New generations use `.png`; older saved SVG artifacts remain readable for compatibility.

Repeated submissions with the same key and identical request retrieve the existing job; they do not generate a second image. A different body or an ID belonging to another job/file returns 409. Keys and output identities survive restart. Previously assigned keys retain their original job IDs; do not assume the filename of a key first used before this feature. Preserve the server data directory when upgrading; use a separate copied directory for this beta test as described in [BETA-TEST.md](BETA-TEST.md).

This is an HTTP contract usable by any frontend. It does not require a local state file, helper script, shared filesystem or client-side model management. `tools/image_job.py` remains an optional command-line convenience, not a requirement.


## RAM after image edits

Image generation/edit requests apply the configured retention policy before whole-job completion. With **Automatic**, the worker can remain loaded when the request's working set fits the device policy. A saved image can become available while cleanup or optional restoration is still running; check job status before assessing idle memory. Request tensors are scoped to one inference call, and Windows process ownership covers the native interpreter and descendants.

In **Models → Settings → Memory after image requests**, use **Unload after each request** to always release the image worker. **Automatic** is the default and retains it on a dedicated device or when the shared working set fits the configured budget. Old workers from a separate server process are not adopted by a new server; check the owning LazyRig installation when diagnosing a process left after shutdown.

Allow time for a model reload on every image request. A retained or explicitly restored LLM and the main server continue to use RAM. Reclaimable file cache is also distinct from memory owned by a running image worker. **Keep loaded between requests** is an explicit performance option and retains model/native-runtime memory. The API state and job events expose native worker PIDs for diagnosis; a failed unload is reported and blocks that device instead of allowing another load on top of it.


## Hermes `background` tool argument

`Invalid JSON value for parameter background; no tool was executed` is a rejection of a model-generated tool argument. It can occur before an image-edit request is submitted. Hermes's `terminal` tool declares `background` as a Boolean; inspect the complete model output and declared schema to identify the invalid value.

The XML tool parser accepts complete true/false literals in any case, optionally with matching single or double quotes, for explicitly Boolean parameters. It sends proper JSON Booleans back to Hermes in normal responses, SSE and saved results. This is schema-based, not a special case for the word `background`. Boolean unions such as Hermes's `notify` also work; arrays remain arrays. Prompts, commands and string parameters retain their text. Ambiguous values such as `yes`, `no`, `0`, `1`, code snippets or `False because...` still fail without returning an executable tool call. JSON-form tool blocks still require valid JSON.

If an invalid-value error remains, inspect the saved job's raw `text` through authenticated `GET /api/jobs/{job_id}` and compare the complete tool block with its declared parameter type. The separate HTTP image-edit option `background` is a string (`auto` or `opaque`); it does not enable transparent image output.

## Hermes `questions` array error

`Invalid JSON value for parameter questions (expected array); no tool was executed` is a LazyRig tool-parser rejection of the LLM's response. It happens before the corresponding client tool is dispatched. Hermes's “temporarily unavailable” wrapper does not identify an image-model outage here. The upstream `clarify` tool uses a `questions` array; the raw call is needed to establish which tool and value your model actually emitted.

The XML tool parser accepts complete array/object values in Qwen XML when they use single quotes, nested Python `True`/`False`/`None`, trailing commas, a complete JSON/Python code fence, or one extra JSON-string wrapper. It returns proper JSON arguments to any compatible client. This applies through the declared parameter schema, not a special case for Hermes or the name `questions`. Truncated or ambiguous calls still fail instead of guessing the intended question or starting an edit.

Obtain the job ID from `X-Job-Id` (or the SSE error's `x_job_id`) and read saved raw `text` through authenticated `GET /api/jobs/{job_id}`. Include the complete tool block and the client's tool schema when diagnosing a rejection; the short error alone does not expose the malformed value. Reading the job does not execute a tool or generate another image.

## Repeated LLM text and sampling diagnostics

Native chat results report `x_sampling`; the saved job exposes the same information as `sampling`. It includes effective temperature, top-p/top-k/min-p, penalties and whether sampling was enabled. Compare the saved job's `text` with what Hermes received to distinguish an inference-side repetition from a delivery/assembly problem. A client's repetition warning does not identify the cause by itself.

Choose a Qwen3.8 preset under Models → Settings if appropriate for your model and thinking mode. Hermes's explicit request settings take precedence. MTP stays under user control and currently requires temperature 0 in OpenVINO; disable it explicitly to test Qwen's recommended sampled presets. Preserve the failing job and compare one configuration at a time when diagnosing a reproducible loop.

## Provider-discovery 404 messages

Hermes probes several backends even when using a custom OpenAI-compatible endpoint:

| Probe | Backend being detected |
| --- | --- |
| `/api/v1/models` | LM Studio |
| `/api/tags` | Ollama |
| `/v1/props`, `/props` | llama.cpp |
| `/version` | vLLM |

These routes correctly return 404 on LazyRig and use the ordinary HTTP request log format. Its model list is `/v1/models` and health/version information is `/api/health`. Other unexpected 404s remain visible. Do not configure LazyRig as one of those native backends to suppress the probes. See the [Hermes detection implementation](https://github.com/NousResearch/hermes-agent/blob/main/agent/model_metadata.py).


## Dashboard workspace and streaming

The included Chat workspace uses persistent browser conversations and replayable job events for text, vision, generation and editing. Generated images can be downloaded, edited, or attached to a vision conversation. Results remain on the server if the browser disconnects. A chat can be exported from History.

For Hermes or another agent, prefer one tool operation that submits and waits programmatically for its recorded job. Do not use a new LLM turn to infer completion from elapsed time. The standard `/v1/images/*` routes support `Prefer: respond-async`, so a client need not use a different URL family if it can set that header. Use receipt links, retain the idempotency key, and distinguish image readiness from completed restoration. See the [API reference](API.md).

If text appears in sentence-sized batches, verify `stream: true`, SSE flushing by any reverse proxy, and incremental rendering in the client. The server avoids whole-response buffering for client-tool chat, streams the post-image LLM continuation, and uses 20 ms native text batching. Tool arguments are still deliberately held until validated.

## Client Stop and cancellation

Synchronous `/v1/chat/completions`, `/v1/images/generations` and `/v1/images/edits` now monitor client disconnects, even while the model is loading or generation has not produced response headers. An abort requests worker termination; the job becomes cancelled after cleanup. Timeout and intentional abort are indistinguishable at this transport layer.

For Hermes or another client using the asynchronous image workflow, Stop must send `POST /api/jobs/{id}/cancel` with `{}` and the bearer key. Merely closing SSE, leaving the page, or stopping polling leaves background work running. The receipt advertises `cancel_url`. Retry a lost cancellation acknowledgement safely, then read status until `cancelled`. Do not retry cancelled generation with a fresh request key.

Cancelled results without a saved image use HTTP 410 (`request_cancelled`) and `X-Should-Retry: false`, rather than a retryable server error. Reusing the original image key returns that cancelled job’s error without starting another generation. A new key explicitly requests new work. Already saved images remain retrievable if cancellation occurs during restoration. If an intermediate proxy swallows client aborts, use the cancel endpoint rather than relying on its upstream connection closing.
