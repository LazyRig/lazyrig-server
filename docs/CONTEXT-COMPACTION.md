# Automatic context compaction

LazyRig v0.41 can summarize older conversation exchanges before the model runs out of context. It works for the dashboard and OpenAI-compatible clients. It uses the current local model; no external service, API key or additional package is required.

Open **API & server → Server settings → Automatic context compaction** to configure it. Changes apply to new jobs; accepted jobs retain their settings snapshot.

| Setting | Default | Meaning |
| --- | --- | --- |
| Enabled | Yes | Apply the safeguard to new chat requests. |
| Start at | 75% | Prompt occupancy that triggers a compaction attempt. It may start earlier when the requested output needs more space. |
| Recent context target | 20% | Preferred recent-tail token target as a fraction of the trigger budget, not a promise to retain exactly 20% of the full context. |
| Preferred recent messages | 20 | Keep this many recent messages when they fit. This preference can shrink to reserve output and summary space. |
| Protected first messages | 3 | Preserve the first messages, counting the server's implicit system prompt when used. Tool groups are kept whole. |
| Summary tokens | 2,048 | Maximum summary output, reduced for small context windows or limited remaining room. |
| Compaction timeout | 1,800 seconds | Maximum total compaction time, within the existing overall request deadline. |

All system instructions, the latest user message and everything following it, image-bearing messages, unresolved tool calls and orphan tool results stay protected. Completed tool calls and their results are retained or summarized as complete groups. The server counts the actual rendered text template and tool definitions, reserves the client's output allowance, and leaves a small additional margin.

Older eligible exchanges are summarized with thinking and tools disabled. An oversized transcript is processed in bounded chunks that each fit the model window; the previous summary is carried into the next chunk. The implementation permits at most 16 summary passes in one compaction attempt. It does not sample or silently discard sections of the selected transcript. Summaries are lossy: details can be omitted or misstated by the model. The original browser and server histories remain available.

The resulting prompt is tokenized again. An empty, incomplete, ineffective or oversized summary is not committed. If protected input alone cannot fit, reduce the latest input, tool definitions or output allowance, or configure a larger context supported by the export and runtime. A cancelled or failed compaction does not erase the original transcript.

Client requests remain stateless. A small in-memory cache reuses a summary only when the entire original prefix, model configuration, template, tools, compaction settings and output allowance match. Changed history invalidates reuse. Cache eviction or a server restart can require summarizing again. No client-side history is overwritten.

Compaction appears as **Compacting context** during generation and in the answer's context metrics afterward. `x_compaction` exposes before/after input tokens, summary calls, summary token usage, elapsed time and reuse status. Summary work is excluded from answer-pass token speeds and ordinary answer `usage`; it is included in end-to-end request time.

## Exact-prompt benchmarks and client controls

Clients can disable compaction for an individual request:

```json
{
  "model": "my-chat-model",
  "messages": [{"role": "user", "content": "The exact benchmark prompt"}],
  "max_tokens": 4096,
  "auto_compact": false
}
```

In the dashboard, choose **Conversation settings → Automatic context compaction → Disabled for this conversation**. The server-wide switch takes precedence: a client cannot enable compaction when the owner has disabled it.

An isolated single-message quality task is never summarized: it is the latest protected request. Compaction also does not detect reasoning loops or increase the output budget. If a client sends `max_tokens: 4096`, the model still stops at that limit even when the model's saved output default is 65,536 and its context is 262,144. In LazyRig Bench v1.3.0, Quality has its own **Evaluation protocol → Output budget** setting, defaulting to 4,096. Start a new run to change its saved protocol.

## Vision and native overflow recovery

Vision preflight cannot count all expanded image tokens; its estimate remains explicitly labelled. If the native worker rejects a prompt with a recognized context-length error before producing any answer text, the server may attempt one stronger compaction and retry the text pass once. It does not retry after partial output, execute a tool twice, or hide an unrelated native error. Images and the latest tool results remain protected, so some oversized inputs must still be rejected. The configured context limit must reflect the export and runtime's supported window.

## Configuration

```json
"context_compaction": {
  "enabled": true,
  "threshold": 0.75,
  "target_ratio": 0.20,
  "protect_first_n": 3,
  "protect_last_n": 20,
  "summary_max_tokens": 2048,
  "timeout_seconds": 1800
}
```

This object can be saved through `/api/server`. `/api/context` only inspects the supplied prompt and returns compaction eligibility/pressure flags; it never loads the model or generates a summary.
