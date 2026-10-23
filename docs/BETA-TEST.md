# Testing LazyRig Server v0.41 beta

This is the first public beta of LazyRig Server. Start with the
[illustrated setup guide](GETTING-STARTED.md). This document covers beta limits,
existing installations, rollback and more detailed operational checks.

Intel GPUs are the supported graphics target. CPU support follows the installed
OpenVINO CPU plugin, regardless of vendor. Other detected accelerators are
experimental. Compatibility depends on the model export, runtime, driver and
operating system; a detected device is not proof of successful inference.

## Current limits

- **Image editing:** one source image per edit; no inpainting mask or multi-source
  editing. The tutorial includes a returned nighttime edit, but one example does
  not establish general editing fidelity.
- **Multiple devices:** splitting is experimental and does not guarantee combined
  usable VRAM or a performance improvement. A reported Arc Pro B60 + Iris Xe split
  failed during its first inference probe after compilation. Start with one device;
  see [device allocation](MODEL-SETTINGS.md#device-allocation-and-offload).
- **Concurrency and recovery:** work sharing devices runs serially; independent
  devices can run concurrently. The default admission limits are eight
  active/queued/reserved inference jobs and two requests preparing inputs.
  Accepted asynchronous jobs can survive a browser disconnect while the server
  remains running; later browser-managed tool stages wait for reconnection.
  A server restart interrupts unfinished inference rather than resuming generation.
- **Storage:** completed results remain available. Input/artifact retention is
  manual and storage quotas are not implemented. Optional historical text-event
  compaction and the free-disk admission check are described in [API](API.md).
- **Access:** intended for trusted local use. The owner key grants administration
  and inference access. An optional inference-only key restricts administrative
  operations but does not isolate users' jobs or artifacts. Follow
  [Connections](CONNECTING.md) for network access, keys, CORS and TLS.
- **API/features:** the OpenAI API implementation is a documented subset. Responses
  API, audio, continuous multi-client batching, remote worker hosting and MCP are
  not implemented in this release.

## Evidence and scope

The 2026-10-22 tutorial captures show model registration, text and vision replies,
image generation through a VLM tool call, a nighttime image edit and the VLM's
continuation after both image operations on the operator's Windows/Intel setup.
They are workflow examples, not performance benchmarks or cross-device certification.
Broader compatibility and endurance remain under evaluation.

The source repository's maintainer records preserve earlier checks and known
failures. Sections below describe the operational changes developed throughout
v0.41 and how to evaluate or upgrade an existing installation.

## Efficient generation writes

Text streaming, image steps and prompt-enhancement progress now persist a compact
event and a small progress record in one transaction. Full job records are saved
at lifecycle boundaries, including cancellation and final results, rather than
for every streamed update. Status endpoints and dashboard metrics still reflect
the last committed update; stream delivery is not delayed by a save timer.

An interrupted process recovers the latest committed text before marking its job
interrupted (or cancelled when a stop was already recorded). SQLite durability
and automatic checkpoint defaults are unchanged. This revision creates only a
small progress table; it does not compact, rewrite or delete historical events.
Apply it by stopping the server normally, replacing the application files and
restarting. No runtime/dependency reinstall or manual database operation is needed.

## Startup correction included in this archive

The first v0.41 package scanned and updated all historical stream events before
printing its ready message. Large databases could therefore cause sustained disk
activity with an empty terminal. This corrected package leaves historical event
payloads unchanged and reads them on demand; it also prints startup stages before
application imports, database work and device detection.

If you already installed v0.41, stop that beta with Ctrl+C and wait for its Python
process to exit. Replace the supplied application files in the **beta** folder,
keeping its existing `.venv`, `.runtime`, models and whole data directory. No
dependency reinstall is required for this correction. Start `start-server.bat`
again. You should immediately see `Starting LazyRig Server...`, then the version
and startup stages. Do not run two copies against the same data directory.

Copy the **contents** of the ZIP's `lazyrig-server` folder directly into your
existing beta application folder and accept replacement of the supplied files.
Do not create a nested second `lazyrig-server` folder and then launch the old
files. After restart and a browser refresh, the dashboard should show
**v0.41 beta**. From that application folder, this read-only command also confirms
the corrected timeout handler is present:

```bat
findstr /n /c:"Incomplete HTTP request: receive deadline exceeded" lazyrig\server.py
```

Keep `state.sqlite3`, `state.sqlite3-wal` and `state.sqlite3-shm` together; do not
delete journal files to force startup or reclaim space. The corrected reader
accepts both an untouched v0.4 database and metadata left by an interrupted first
beta upgrade. SQLite recovery after an interrupted write can still take time;
the terminal now identifies database opening before that work begins.

A large `-wal` after the original migration can remain allocated for reuse even
after a checkpoint. A normal shutdown usually checkpoints and removes `-wal`
and `-shm` when the last database connection closes; let Python finish that work.
If they remain after shutdown, keep them with the database rather than deleting
them manually. Their size alone does not establish whether changes are pending.

The existing database will **not automatically shrink**. It may contain old
cumulative stream records and reusable pages. Deliberate archive/compaction and
space reclamation are separate maintenance work requiring a coherent backup.
`event_retention_days` remains off by default; opting into historical compaction
can itself require substantial I/O on a large legacy database.

Earlier builds printed `Request timed out: ReceiveTimeout(...)` or `HTTP receive
deadline reached; closing idle or incomplete connection` for unused browser
connections reaching the default 30-second receive limit. Idle expiry is now
silent. If a client starts a request but does not finish its request line or
headers in time, the console reports `Incomplete HTTP request: receive deadline
exceeded; connection closed`. Incomplete uploads still return HTTP 408 when
possible. Receive guards remain enabled; inference has its own deadlines.

`GET /api/state?summary=1 ... 401` means the dashboard did not send an accepted
API key. Use **Client API key** and connect with the server's existing owner key.
The client keeps that key in tab-scoped session storage; a fresh tab, a different
origin, or an outdated key may need authentication again. A visible unauthenticated
tab continues polling every 2.5 seconds, so repeated 401 lines can come from an old
tab even when another tab is connected. An inference-only key cannot administer
the dashboard. No new key is generated or required by this startup correction.

## Fresh installation validation

For a clean installation test, extract the user ZIP into a new writable folder
and run `start-server.bat`. Let it create its own environment and empty data
directory. Do not copy old settings, databases or Python environments into this
first-run test. Model weights remain in their existing external folders and can
be registered after startup succeeds.

Stop the other LazyRig instance before testing the same hardware. A fresh
installation defaults to `http://127.0.0.1:8765`; open the address printed by the
launcher. An empty model registry is expected. Initial Python/dependency downloads
can take time; retain the setup log if installation fails. This validates a fresh
installation, separately from the upgrade procedure below.

## Safe setup with an existing installation

1. Stop your current LazyRig server normally and let workers exit. Keep its
   original application folder and data directory as the rollback copy.
2. Extract the beta ZIP into a **new folder**, for example
   `C:\LazyRig-beta\lazyrig-server`. Do not overwrite the working installation.
3. While the original server is stopped, copy its **whole** `server-data` folder
   into the new beta folder. Copying only the SQLite file can omit data in its
   journal or WAL files. If you use a custom `--data-dir`, copy that entire
   directory and point the beta only at the copy.
4. Keep model paths pointing to your existing model exports. Model files do not
   need to be duplicated. Do not copy `.venv` or `.runtime` environments between
   installation folders; setup creates the application environment at its new
   location. Experimental runtime profiles can be prepared again when needed.
5. Start the beta with its `start-server.bat` or `start-server.sh`. Only the beta
   should use the old host/port while it is running. Verify the dashboard version
   says **v0.41 beta** and your model/device settings are present.
6. Reload the dashboard. Keep the same browser, scheme, hostname and port to
   access the existing browser chat history. A different port is a different
   browser origin and will show separate history.

The beta enforces one server owner per canonical data directory. A second
instance using that directory should fail clearly rather than marking the first
instance's active jobs interrupted. Separate directories still do not isolate
physical GPU memory: avoid concurrent stable/beta inference during comparison.

Browser conversations live in IndexedDB, separately from `server-data`. Export
important conversations from History before testing. No browser database migration
is intended in this release. Using the same origin accesses the same browser
conversation store, so an installation copy is not a browser-history backup.

## Rollback

Stop the beta and wait for its workers to exit. Restart the original application
from the original folder with its **original, untouched data directory**. Do not
point v0.4 at the beta's modified database. Keep beta results in the beta folder;
download images or export conversations explicitly if you want to keep them.

Do not delete a lock file to make concurrent use work. An OS-owned lock is
released when the owning process exits; a file remaining on disk does not mean
the lock is held. If a start is refused, identify the running instance first.

## Changed operating contracts

| Area | Beta behaviour |
| --- | --- |
| Admission | Default eight active/queued/reserved jobs and two input preparations; overload returns 503 with retry guidance before expensive preparation |
| Request deadline | `request_timeout_seconds` covers preparation, device queueing and model stages; it is no longer a fresh allowance for each phase |
| Storage failure | In-memory job ownership is released, admission pauses when durable storage is unhealthy, and readiness reports the condition |
| Stream storage | Compact events and small progress records commit together; full jobs are saved at lifecycle boundaries. Replay stays cumulative. Historical compaction stays off by default |
| Disk headroom | Default 256 MiB minimum free space for new admission; `disk_pressure` refuses work when below the threshold |
| Readiness | Authenticated `/api/ready` distinguishes ability to admit work from `/api/health` liveness; it does not run a model probe |
| HTTP limits | Default 64 connections, 30-second receive/write limits; inference and SSE do not inherit a 30-second lifetime |
| Credentials | Optional `inference_api_key` separates inference from administration; all inference callers still share jobs/artifacts |
| Reverse proxy | Optional `public_base_url` sets the canonical origin for returned URLs; forwarded headers are not trusted automatically |

See [API.md](API.md) for exact configuration names, valid ranges and error bodies.
Some new operator fields use `POST /api/server` or the saved configuration rather
than new dashboard controls. Existing configurations receive default limits.
Input/artifact retention remains manual and storage quotas are not added.
`min_free_disk_mib` defaults to 256; zero disables that admission check. It does
not reserve enough space for every running request. Optional
`event_retention_days` defaults to zero (off). When enabled, idle maintenance
compacts repeated text updates on older terminal jobs while retaining their final
text and every non-text lifecycle/artifact event. It does not delete jobs, keys,
messages, inputs or image files. Freed SQLite pages are reusable; the database
file may not immediately shrink. Keep monitoring free space and preserve a
coherent stopped-server backup.

The deadline bounds checks and worker cancellation; it cannot safely force an
arbitrary synchronous third-party call in the main process to return immediately.
Worker termination and cleanup may add time beyond the configured request budget.

## What to test on your machine

Use your usual model exports and settings first. Change one variable at a time
when comparing with the baseline. Record the server/runtime version, device,
driver, model path/revision, input, sampling settings and job ID for a failure.

| Check | Expected result |
| --- | --- |
| Startup and copied configuration | Correct beta version; existing aliases, device policies and settings available |
| Duplicate data-directory owner | Second instance refuses to start; first instance keeps its job state |
| Text and vision | Existing text/VLM requests finish; a text prompt already exceeding the context budget is rejected even when it includes an image |
| Long streamed reply and reconnect | Same final text and reasoning after reconnection; no duplicated output |
| Admission saturation | Capacity rejection is explicit and retryable; unrelated cancellation stays responsive |
| Stop during load, text, image steps and restoration | Cancellation is acknowledged; worker cleanup completes; already saved images remain available |
| Queue timeout | A waiting job expires within its configured deadline and never starts later |
| Direct generation and editing | One correct artifact per accepted request; retries with the same key do not create another |
| LLM/image handoff | Existing retention/restoration choices work; saved image is retrievable before restoration ends |
| Separate devices | Independent-device requests can progress concurrently; verify memory and timing on actual devices |
| Restart and recovery | Completed jobs/artifacts remain accessible; unfinished jobs are interrupted, not silently regenerated |
| Browser history | Conversation selection, folders, draft preservation, scrolling and ordinary multiple-tab use remain functional |

Start with a short smoke run, then repeat a representative sequence of text,
vision, image generation, editing, cancellation and reconnection for sustained
use. Check process count and memory after the chosen idle/unload interval; retained
workers and OS file cache are expected to differ from a fully unloaded state.

## Parallelism scope

Independent devices retain concurrent execution. Inference on a shared device
remains serial in this beta. The memory **budget** policy permits compatible
workers to stay loaded; it does not turn the native pipeline into a multi-client
batching server. This release does not implement a new continuous-batching
worker or concurrent image pipelines, and does not claim a throughput increase.

Multi-reference image editing remains a separate future integration. Image edits
still accept one source image. There is no change to image enhancement prompts,
model quality or image generation settings in this hardening release.

## Evidence and limits

Check the hosted Actions result for the exact source revision.
The CI workflow includes a Windows Python 3.13 lane in addition to Linux Python
3.12/3.13. A configured workflow is not a completed CI run and does not establish
device compatibility. Hosted fixture tests do not measure inference throughput.

Use the [API reference](API.md) for changed request/error contracts and
[client connection guide](CONNECTING.md) for authentication and deployment.
