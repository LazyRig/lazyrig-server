# Changelog

## v0.41 beta — 2026-10-07

Priority production-readiness hardening on the v0.4 beta baseline. Install into a
fresh folder with a separate copy of stopped `server-data`; see
[Beta testing and rollback](docs/BETA-TEST.md). Keep the original application and
original data together for rollback. No public release is created automatically.

- Publication documentation, 2026-10-22: simplify the README and first-release
  post, replace the illustrative tutorial with real v0.41 screenshots, and add
  the text, vision, image-tool generation and nighttime-edit walkthroughs.
  Keep advanced configuration in the detailed guides. Include a graphical
  first-upload guide and preserve the known beta limitations.

- Resource number formatting, 2026-10-22: display memory reservations, capacities,
  planning budgets, model sizes and memory warnings with one decimal place.
  Preserve exact editable settings, scheduling calculations and diagnostic data.

- Split failure evidence, 2026-10-22: wait up to two seconds for a natural worker
  exit after IPC closes, in cancellable steps within the original deadline.
  Preserve the real exit code before cleanup; never label our own termination
  as a native crash. Capture a bounded Python fault-handler trace independently
  of IPC, installed runtime versions and split-probe progress in saved load
  details. Query the Windows worker's peak committed memory separately from
  available host RAM and its commit limit. Keep validation, placement choices,
  native settings and guards unchanged. This is a diagnostics update, not a
  confirmed fix for the reported HETERO inference failure.

- Resource policy and code-copy styling, 2026-10-22: add Automatic, Single device
  and explicit selected-device Split modes. Auto attempts splits before optional
  whole-model fallback, which is now explicitly opt-in. Add a primary GPU
  shared-memory attempt option; preserve higher-than-VRAM planning budgets and
  all native cleanup, timeout and RAM protection. Correct underestimated full-GPU
  reservations to known language-weight size without rewriting saved values.
  Show planned/skipped/failed/successful targets and bounded original native
  reasons in activity, including compilation versus inference-probe failures.
  Explain the MTP conflict before loading and never silently disable MTP.
  Label scheduling estimates and planning budgets distinctly from measured usage.
  Restyle code-copy controls as rounded charcoal/orange rectangles with icons,
  visible keyboard focus and copied feedback. Existing chat geometry is retained.

- Shared-memory loading budget, 2026-10-21: honor an explicit GPU Memory budget
  above detected dedicated VRAM instead of silently clamping the stored-weight
  precheck. This permits the native loader to try driver-managed shared memory,
  including with MTP, without promising that the runtime will fit the model.
  Keep conservative defaults, host-RAM/commit guards, native failure handling,
  deadlines and cleanup. A selected-device budget edit clears the remembered
  fallback preference for the next fresh load while keeping healthy residents.
  Explain the existing budget control in errors/help and distinguish it from
  model reservations and actual GPU usage. No model/runtime settings are migrated.
- Approved chat layout, revised 2026-10-21: retain the normal dashboard navigation
  and equal menu spacing. Use rounded history and a compact rounded chat header,
  with the conversation itself on the page background. Align the header and
  composer in one default-width column. Replace the large new-conversation
  button with a subtle plus beside Add folder; remove the reading-width toggle.
  Add clearance between history highlights and the slim neutral scrollbar, with
  an extra right inset and a thin vertical guide for conversations inside expanded
  folders. Folder hover and selection cover the same complete row; selected
  rows use orange action dots. Remove the horizontal divider and its extra spacing above conversations
  outside folders, matching the 8 px gap between adjacent folder rows. Keep the small-screen history drawer, circular icon controls
  and rounded rectangular text actions.
  Match the composer curve to the 34 px send button using equal right/bottom
  inset and concentric radii. Preserve drafts, folders, autosaved settings, image
  tools, streaming and Stop through the existing controllers and server API;
  no data migration is needed. Audit intermittent response loops without changing
  inference behavior: no cause was confirmed by the available fixture tests.
- Approved dashboard redesign, 2026-10-20: apply the charcoal/orange visual design
  to the real server with rounded controls, circular icon actions, model search
  and task filters, and responsive navigation. Move model settings into a tabbed
  dialog with a blurred backdrop; device, server and conversation settings also
  use dialogs. Preserve actual settings, folder browsing, image tools, history
  and server API handlers. Conversation settings continue to autosave. Device
  rings show reservations, not measured memory use. Keep CPU support labels
  vendor-neutral while retaining experimental labels for other accelerators.
- Intel support scope and automatic allocation, 2026-10-20: label Intel CPU/GPU
  devices as supported targets and other detected devices as Experimental,
  preserving explicit selections and MTP choices. Default new models to one
  suitable device. Retire manual percentage controls and migrate saved manual
  selections to automatic; reject new manual settings atomically. Try selected
  GPUs individually before splitting and keep iGPU/CPU as alternative host-memory
  stages. Add execution preference, GPU compilation threads and CPU scheduling
  controls with device-specific routing. Native GPU validation is still required.
- GPU compiler diagnostics, 2026-10-20: classify OpenCL kernel build failures
  separately from allocation errors, retain the original exception and identify
  the reason during Automatic fallback. Add diagnose-gpu.bat for isolated,
  bounded runtime/driver discovery and tiny FP32/FP16 compilation probes without
  loading model weights. Detection does not imply model compatibility; OpenCL
  probe logs do not identify the original failing kernel. No NVIDIA kernel,
  precision, device-selection or runtime replacement is claimed by this update.
- Attention-cache sequence default, 2026-10-20: default constructed GenAI
  scheduler configurations to one sequence, matching LazyRig's serial model
  workers. The inherited 256-sequence default could reserve excessive linear
  attention state and reject fixed cache allocations on hybrid Qwen models.
  Preserve explicit settings, client token limits, native-default pipelines and
  SDPA splitting. Show the one-sequence default in settings, describe cache_size
  as an initial allocation, and report this specific failure with supplied
  settings and guidance instead of retrying devices or increasing memory.
- Explicit manual splits and cooperative Stop, 2026-10-19: replace production
  manual ratio calibration with state-aware operation affinities for GenAI
  2026.4.x. Preserve the five-percentage-point acceptance limit, original
  exports and hardlinked weights. Report assigned graph bytes without claiming
  measured VRAM. Run a short prefill/decode probe before accepting multi-device
  loads; Automatic may fall back after a failed probe and can try the primary
  GPU with CPU before full CPU fallback. Manual remains one explicit attempt.
  Retain text workers after acknowledged cancellation; terminate unresponsive
  workers after five seconds. Name load-time settings that trigger a rebuild.
  Native CPU tests and graph metadata checks pass; Intel GPU/Windows behavior
  still needs hardware validation. These changes do not claim to repair native
  multi-GPU driver failures or enable MTP with manual SDPA splitting.
- Worker exit diagnostics and MTP budget messages, 2026-10-19: capture the
  process exit code before cleanup and show it in saved jobs and terminal
  output, including Windows hexadecimal status. Distinguish a closed worker
  connection with unknown status from an observed process exit. Preserve
  cancellation, deadlines and native exceptions; do not retry an unexplained
  process exit automatically. Explain that MTP currently disables main-model
  splitting and that moving its head does not offload the main weights.
  Main-model CPU fallback still requires CPU selected in its device allocation.
  These diagnostics do not fix or identify the reported native multi-GPU exit.
- MTP device selection, 2026-10-19: select Same as model, CPU, or a configured
  GPU for the bundled MTP head. Pass the selection to GenAI, reserve both
  devices, apply CPU thread settings to a CPU head, and reload when its device
  changes. Preserve the existing same-device default and per-request draft
  count. Desktop/mobile save-and-reopen checks pass; native cross-device MTP
  generation remains unverified. Add a read-only saved-job placement-report
  command for diagnosis without loading the model again. The latest Arc/CPU
  and Arc/Iris manual 50/50 tests still fail preflight; this update does not
  change splitting or relax its acceptance checks.
- Stateful split ordering, 2026-10-19: order independent language-model cache
  updates by data dependency before manual partitioning. Unrelated sink order
  could leave state pairs open across most of the graph and make very different
  sliders produce the same large primary-device share. Query and compile one
  temporary normalized view; keep weights, connections and original exports
  unchanged, with hardlinks instead of weight copies. Preserve the five-point
  acceptance guard and bounded calibration, including a nearby-ratio check for
  nonmonotonic empty GPU queries. Native small-model CPU state/output equivalence
  and query behavior are tested; Windows Arc/Iris generation still needs validation.
- Manual split calibration and VLM GPU handoff, 2026-10-19: measure exported
  language-weight shares while adjusting the runtime query ratios, with bounded
  attempts. Reject a split that remains more than five percentage points from
  any requested share instead of accepting 88.2/11.8 as 50/50. For verified
  manual multi-GPU VLM plans, place text embeddings on the language input's
  queried device to avoid forwarding a remote tensor from a different GPU.
  Native mixed-GPU execution still requires hardware validation. Show only
  "Stopped." when a cancelled response has no saved images, and remove the
  redundant CPU-secondary-device hint from the picker. Offer compatible available
  GPUs in the primary selector even when unchecked, selecting a newly chosen
  GPU automatically. Keep GPU priority for mixed GPU/CPU allocations and CPU
  primary for CPU-only use.
- VLM split routing and CPU threads, 2026-10-19: preserve physical-device
  properties through GenAI's VLM component routing. Previously a manual query
  could pass while the language-model compile lost its GPU weight ratio.
  Log requested and queried language-weight shares separately from VRAM usage.
  Add an Auto/custom CPU inference-thread default under Devices, applying it
  to CPU execution and the CPU portion of mixed placements while preserving
  explicit model overrides. Native Arc split/memory acceptance remains open.
- Manual device aliases and Windows memory limits, 2026-10-19: resolve a
  selected GPU alias through native device properties before checking HETERO
  assignments such as GPU.0. Keep rejecting unselected or ambiguous devices.
  Install a Windows worker-job committed-memory limit before loading, based on
  fresh available RAM minus system headroom. Retain sampled loading checks and
  report actual limits, available RAM and the lowest sampled value. The limit
  remains active during generation but does not cap VRAM or all driver-managed
  allocations. Continued Auto RAM exhaustion on RTX remains unverified.
- Loading-memory and manual-partition fixes, 2026-10-19: accept valid boundary
  Parameters/Results added by OpenVINO HETERO queries while still checking every
  original operation and selected-device assignment. Automatic loading skips
  a full-GPU attempt when stored language weights already exceed its usable
  budget. Monitor available host RAM during native loading, preserve 10%
  headroom (bounded to 1–8 GiB), and confirm worker cleanup before any automatic
  fallback. Manual settings are never silently changed. This sampled safeguard
  is not a hard memory cap; native GPU validation remains required.
- Device allocation, 2026-10-19: replace model-row device selection with a
  checkbox menu and primary selector. Add Advanced distribution with automatic
  primary → GPU split → GPU/CPU split → CPU load fallback, and approximate
  manual weight-share sliders. Use native HETERO partitioning, capability and
  graph-preflight checks; keep image/NPU models on one device. Whole-placement
  leases, actual-target pinning, remembered successful targets and refreshed
  reservations preserve safe handoffs and independent-device concurrency.
  Sliders do not claim exact layer counts or hard memory limits. Native GPU
  validation remains pending; see Model settings for SDPA/MTP restrictions.
- Folder context, 2026-10-18: a selected folder, or the parent of the selected
  conversation, receives New chat whether the folder is open or closed.
  Selecting an outside conversation clears the folder destination. Selecting a
  folder removes the highlight from conversations outside it while preserving
  the displayed chat. Creating inside a closed folder leaves it closed. This
  supersedes the earlier open-folder requirement; no extra controls are added.
- Folder selection, 2026-10-18: make New chat follow the current selection.
  Create inside a selected expanded folder, or the expanded parent of the
  selected conversation. Selecting a conversation outside folders, or moving
  the active conversation outside, clears the destination and button label.
  Open folders alone do not select a destination. This supersedes the earlier
  remembered-folder behavior; the compact cue and F5 section restore remain.
- Chat context, 2026-10-18: show the destination folder icon and a compact,
  truncated folder name inside New chat when the destination folder exists and
  is expanded, including while History is hidden. Keep the full name in the
  tooltip and accessible label. Restore the current dashboard section on F5
  within the same tab; fresh tab sessions default to Models.
- Client tool worker retention, 2026-10-18: keep a live, completed text worker
  after rejecting malformed tool calls or invalid JSON output. Keep the request
  failed and return its validation error; never execute a partial call or retry
  client tools on the server. Native failures, incomplete exchanges, Stop and
  deadlines still trigger cleanup. Terminal output distinguishes generation
  completion from job completion and reports safe failure categories without
  printing prompts or tool arguments. Explicit client output budgets retain
  precedence over model defaults; changing budgets does not reload the model.
- Chat defaults and image actions, 2026-10-18: raise the inherited output budget
  to 8,192 tokens in chat, context preflight and model settings. Upgrade the old
  saved 1,024-token model default once when the configured context exceeds 8,192;
  preserve other saved limits, subsequent custom values and conversation overrides. Consolidate Edit in chat into
  Use in chat, selecting a vision model and enabling an editing tool when available.
  Keep Edit image for direct editing. Reuse attached images on repeated clicks,
  merge concurrent file reads without duplicates or lost attachments, and discard
  pending reads when the conversation or task changes. New chats automatically
  enable the preferred available image tool; add Use as default image tool beside
  the image model API checkbox, with one saved preference and a first-enabled
  fallback. Existing chat selections and ordinary API tool behavior are preserved.
- Automatic GPU image recovery, 2026-10-18: retry native image generation/editing
  once in a fresh worker for CL_OUT_OF_RESOURCES or CL_MEM_OBJECT_ALLOCATION_FAILURE.
  Verify cleanup first; keep the same job, device lease, source, prompt, seed,
  settings and deadline. Preserve normal healthy-worker retention. Show recovery
  in the dashboard, terminal and response metadata, reset attempt progress, and
  never repeat enhancement/tool dispatch or replay already-produced output.
  Stop, expired deadlines and cleanup failures prevent further attempts.
  Failed dashboard Response JSON now includes status, error and recovery details.
- Automatic image retention, 2026-10-18: retain successful generation/edit
  workers after displacing other models when text restoration is deferred.
  Count current residents and models needed immediately, not already-evicted
  workers awaiting a future request. Consecutive image requests and resolution
  changes reuse the compatible worker. Preserve eager restoration, managed
  LLM continuation, budget/exclusive eviction and explicit unload settings.
- Activity/history recovery, 2026-10-18: View result and View progress reopen
  the saved conversation for that job, including earlier image-tool stages,
  instead of creating a copy. Preserve its image attachments, draft, settings
  and recency; reveal its folder when needed. Serialize repeated opens and wait
  for saved history to load. Jobs without a conversation create one with the
  correct title, task and model. Prefer the oldest matching conversation when
  an earlier build has already made a copy; existing history is not deleted.
- Memory defaults, 2026-10-18: fill numeric model reservations from the selected
  device's usable budget instead of a fixed 8 GiB. Refill on device changes in
  Add model, Imports and Settings; offer Default and preserve manual edits.
  Apply the same defaults to API registration/import and device switching.
  Detect physical CPU RAM for new device budgets; preserve saved device budgets
  and existing model reservations. Defaults conservatively reserve the full
  device budget, rather than claiming to measure a model's peak memory.
- Form spacing, 2026-10-18: separate Inspect folder, its result and the fields
  below; add gaps inside expanded form options and between adjacent settings
  grids. Separate import errors and the server restart note from nearby controls.
- History readability, 2026-10-18: use the orange tint for the selected folder
  and a subtle neutral background with orange text for the selected conversation.
  Clear the folder highlight when selecting or moving the active conversation
  outside it. Folder counts inherit the selected orange text color.
  Add a small gap between folder headings and their conversations so selected
  backgrounds do not touch.
- New conversation folders, 2026-10-18: create chats inside the chosen folder
  while it is open; closing it makes New chat create outside without reopening
  it. Preserve the chosen folder across unfiled-conversation selection and refresh.
  Keep the open/closed folder icons and persist new memberships.
- Startup output, 2026-10-18: shorten database opening and schema-check messages
  by removing the explanatory event-history sentences.
- Conversation deletion, 2026-10-18: remove deleted conversations from History
  immediately, including the selected conversation, and refresh folder counts.
  Preserve the next conversation's draft; deleting the last conversation opens
  an empty chat without copying the deleted draft.
- History controls, 2026-10-15: replace the separate Add folder row and close
  icon with a compact folder-plus button beside Conversations. Keep History and
  Escape as panel-close controls. Inset conversation and folder action menus
  from the right edge.
- HTTP logging, 2026-10-15: close unused browser connections and expired idle
  keep-alive connections silently. Continue reporting incomplete request
  timeouts, HTTP errors and inference failures; preserve receive deadlines.
- Dashboard dialogs, 2026-10-10: keep notifications inside the active dialog,
  including nested folder pickers, so errors remain readable above the backdrop.
  Align folder action buttons and wrap long root paths on narrow screens. Use
  "Choose a model folder" for model selection, with purpose-specific titles for
  imports and server settings. Reject empty inspection paths with "Choose a model
  folder first." instead of scanning the application directory.
- Windows folder browser, 2026-10-09: skip drive probes that fail with access
  denied or device-unavailable errors, so an unrelated protected drive cannot
  disable the folder picker. An inaccessible selected folder returns an
  actionable HTTP 400 instead of an internal error. Windows permissions remain
  unchanged; the server account must be able to read the selected model folder.
- Generation write efficiency, 2026-10-08: commit each compact stream event and
  its small progress record together, without rewriting the full job for every
  text chunk, image step or enhancement progress update. Status, metrics and
  replay remain current; restart recovery retains the last committed text.
  Full snapshots remain at lifecycle boundaries and finalization. Preserve
  SQLite durability/checkpoint defaults and leave historical events unchanged.
- Dashboard scrollbars, 2026-10-08: use the message area's existing subtle 6 px
  scrollbar across pages and nested panels, including conversation history,
  settings, logs, dialogs and text fields. Keep transparent tracks, rounded dark
  thumbs and no arrow buttons; horizontal scrollbars use the same treatment.
- Chat sender labels, 2026-10-08: place YOU and the responding model above their
  message cards, using the device label's muted color and regular 11 px type.
  Keep message content, reasoning, images, metrics and actions inside the cards.
- Startup correction, 2026-10-07: leave legacy event payloads unchanged instead
  of backfilling every historical event and building large secondary indexes
  before serving. Read legacy/partially migrated events on demand. Print the
  launcher, import, database and device-detection stages immediately. Existing
  database space is retained; no automatic deletion or vacuum is performed.
- Clarify HTTP receive-timeout diagnostics: idle connections can expire without
  any failed inference request. Avoid a logging exception when a connection
  expires before sending its first request; keep slow-header/body protections.
- Acquire an OS-owned exclusive data-directory lock before SQLite recovery.
  Concurrent use of the same canonical directory fails without changing live jobs.
- Finalization releases in-memory job ownership even when persistence fails.
  Storage health is explicit; admission pauses when durable writes are unavailable.
  Process-local terminal snapshots are marked non-durable and shown consistently
  across recovery/listing routes. Cleanup still runs when its event write fails.
- Reserve capacity before expensive preparation. Matching idempotency keys use
  short independent coordination instead of holding the global service lock across
  downloads/tokenization. Overload has explicit HTTP 503/Retry-After semantics.
  A failed queued-event write after saving a retry identity finalizes that same
  job instead of leaving an unstarted job permanently queued.
- Carry one monotonic request deadline through preparation, device queueing and
  model stages. Timed-out accepted jobs retain their identity; saved images remain
  recoverable when later cleanup/restoration fails or expires. Waiting for the
  tokenizer lock observes the deadline; native encoding itself remains cooperative.
- Store streamed text updates compactly while retaining external cumulative event
  replay and legacy rows. Optional old terminal text compaction is off by default.
  Admission checks 256 MiB free disk by default; this is backpressure, not a hard
  quota. No automatic artifact/input deletion is added.
- Preserve known context overflow when vision-token cost is still estimated: adding
  an image no longer hides text that already exceeds the configured budget.
- Add authenticated `/api/ready` independently of liveness, bounded HTTP admission
  and receive deadlines, explicit proxy `public_base_url`, and an optional
  inference-only credential. Shared client jobs are not tenant-isolated.
- Correct a pre-existing Send-focus scroll adjustment that could stop an untouched
  view following streamed output. Focus the prompt without scrolling before Send
  hides; preserve manual reading position and the workspace layout. Regression
  coverage includes actual scrollbar-thumb dragging and completion.
- Support prerelease archive names and include a safe beta-testing guide. Extend
  the CI definition with Windows Python 3.13 alongside Linux Python 3.12/3.13.

Shared-device inference remains serial; independent-device overlap remains
supported. No continuous-batching worker, multi-reference image edit, model setting
or image prompt change is introduced. The creator kit contains
executed checks and outstanding native Windows/GPU validation in
`maintainer/BETA-TEST-VALIDATION.md`.

## v0.4 beta

- Switching conversations no longer rewrites complete message/image records or
  replaces the history buttons. Saved input images are assigned directly to image
  elements instead of reparsing their full data URLs as HTML. Drafts and settings
  remain preserved. Slight pointer movement ending on the clicked conversation
  selects it, and the grabbing cursor appears only during an actual drag.
- Removed the No folder heading; the regular history list remains a drop target
  for moving conversations out of folders, including when that list is empty.
- Drag conversations onto History folders to move them directly, or into
  the list below folders to return them to the main list. Drop targets highlight, collapsed
  destinations open, and the current conversation, draft and recency are retained.
- Blocked chat startup shows an amber warning with an icon directly below the
  chat controls: “Chat blocked — close other LazyRig tabs.” Recovery instructions
  remain visible until startup resumes, replacing the muted footer warning.

- A database open queued behind an older tab's unfinished upgrade now displays
  a persistent recovery message, even when the browser emits no blocked event.
  Close all older LazyRig tabs/windows once and reopen the same address to release
  that existing queue; this does not require clearing history or changing keys.
- Successful dashboard authentication clears its obsolete API-key warning.

- Fixed chat startup hanging when an older dashboard tab held conversation
  storage open. Folders now use separate storage, preserving existing chats,
  drafts and folders without requiring a conversation database upgrade.
- Simplified History: Add folder is a single row, and each conversation has an
  actions menu for moving, exporting and deleting that conversation. Opening its
  menu leaves the current conversation selected. Disabled Send no longer shows
  a waiting cursor.
- Chat's model status aligns with the selected task/model values and includes
  the full hardware device name alongside its device ID.
- Streaming replies stop following as soon as the user scrolls up. Returning
  to the bottom or choosing Latest resumes following. Completed responses keep
  existing images mounted so reloading placeholders cannot shift the reading view.
- Conversation history no longer changes order when opening a chat, editing a
  draft or changing settings. New conversations and submitted messages determine
  recency. History offers optional flat folders through Add folder, with rename,
  collapse and Move controls. Removing a folder keeps its conversations.
- Model runtime setup checks compatible existing environments before installing
  a new one. A requirements-file hash change no longer automatically duplicates
  the Qwen runtime. Validated environments keep their original paths and package
  versions; missing readiness markers can be recovered after read-only checks.
  Setup messages identify the runtime's use for both conversion and model loading.
- Runtime lookup checks all saved validation results before probing incomplete
  installations, avoiding repeated compatibility checks of an unused folder on
  later model loads.
- Added space between chat text and images. The conversation scrollbar has a
  muted thumb, transparent track, no arrow buttons and clearance from cards.
- Image validation errors now name the detected unsupported format and identify
  the failing image/message in vision requests, including earlier attachments
  resent with conversation history.
- Failed Chat turns remain visible but their prompts and attachments are excluded
  from later requests and image-tool source choices, including after reloading
  an existing conversation. Earlier successful context is retained.
- Direct editing reports its one-source limit when multiple files are dropped,
  instead of silently retaining one. Switching to Edit preserves attachments
  until the user removes extras. Image-tool instructions distinguish this limit
  from vision chat's support for multiple images.
- Refined Chat within the existing dashboard design: messages are individual
  cards on the section background, with comfortable margins and a composer that
  grows with the draft. The original SERVER CONTROL header and expanded sidebar
  are retained. Selected navigation and conversations use a borderless tint;
  Parameters and History buttons stay orange while their panels are open, without
  decorative arrows. Click again or press Escape to close a panel. History has
  separate Close and Delete controls. Short windows keep the composer reachable.
- Chrome's optional DevTools workspace probe now returns a quiet 404 instead of
  an authentication error. API authentication and host/origin checks are retained.
- Rewrote the project introduction and publication copy around local multimodal
  model serving. The docs distinguish current language, vision-language and
  image support from the broader project direction, and give Windows and Linux
  setup equal visibility.
- Fixed startup on existing environments without `packaging` or pip. Runtime
  update parsing is now loaded on demand, with a read-only local fallback.

- Added stable runtime updates from `update-runtime.bat` and the dashboard, with
  idle checks, coordinated runtime dependencies, restart and recovery handling.
- Automatic image-model retention is the default for new settings. Dedicated
  devices keep their models loaded; shared devices keep them when the configured
  working set fits the memory budget. Explicit keep/unload choices remain available.
- New devices default to memory-budget residency. Existing device policies and
  saved retention choices are preserved.
- One-click Windows launch prepares a missing application environment. The
  illustrated tutorial covers installation, a single VLM for text and vision,
  VLM-driven image generation/editing, device memory and client connections.
- Optional prompt enhancement uses LazyRig's own Apache-2.0 instruction files.
  Generation and editing have separate detailed defaults. Prompts can also be
  prepared entirely in the client with enhancement disabled.
- Model imports use a neutral URL hint and retain recognized upstream licenses,
  notices and model cards, including through supported conversions.
- The dashboard identifies the OpenVINO™ runtime and links to its official
  documentation. Model-license guidance explains the separate terms for downloads.
- The end-user package contains the application, installers, user guides,
  screenshots and licenses, with obsolete package files removed.

This version follows the earlier `v4.0alpha4.x` development packages. The public
version is **v0.4 beta**, with `0.4beta` used in the API and archive filename.
Follow [Data and upgrades](README.md#data-and-upgrades) when migrating an installation.

Compatibility depends on the model export, runtime and device. See
[Current limits](README.md#current-limits). No model weights are bundled, and this
release makes no new image-quality or performance claim.

OpenVINO is a trademark of Intel Corporation or its subsidiaries.
