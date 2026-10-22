# Update the OpenVINO runtime

LazyRig can check for a newer **stable** OpenVINO GenAI release on PyPI and update
OpenVINO, GenAI and Tokenizers together in its own `.venv` environment. Use either
the dashboard or the Windows BAT file; both use the same updater.

## From the dashboard

1. Select **Update runtime** beside the version under **RUNTIME**.
2. Review the installed versions and latest compatible release. Checking does
   not change the environment and is safe while models are loaded.
3. Finish active requests and model imports, then select **Install and restart**.
4. Keep the updater console open. LazyRig stops, applies and verifies the update,
   and starts again with the same address and launch arguments. The dashboard
   reconnects automatically. Models load again when requested.

The installer refuses to start while requests, queued jobs or imports are active;
finish them and retry. There is no scheduled update and no update on startup.
On Windows, a new console takes over the running server after the update. The
old launcher may show its usual “press any key” message and can be closed.

## From Windows

1. Stop LazyRig with **Ctrl+C** and close any other tools using its environment.
2. Double-click **`update-runtime.bat`** in the application folder.
3. Review the check and enter **y** to install an available update.
4. When finished, start LazyRig with **`start-server.bat`**.

To check without installing, run `update-runtime.bat --check` from a terminal.
The BAT file requires an existing installation; use `start-server.bat` first
on a new installation.

On Linux, stop the server and run from the application folder:

```sh
.venv/bin/python -m lazyrig.runtime_update
./start-server.sh
```

Add `--check` for a check only. Custom environments can check for releases but
must be updated with their own package-management workflow.

## Compatibility and recovery

Qwen-Image uses a separate environment under `.runtime/converters` for both model
loading and conversion. Updating application files over the existing installation
preserves that folder. If a profile's requirements file changes, LazyRig first
checks older environments for that same profile: installed versions, exact pinned
source URLs, imports and `pip check` must pass before reuse. These checks do not
install packages. A compatible environment stays in its original location and is
remembered for later loads. A missing readiness marker can be recovered in the
same way. Saved validation results are checked before any runtime is probed again,
so an incomplete newer folder does not delay reuse of a validated older one.
Incompatible older environments are left untouched; setup runs when no
compatible installation is found. Model-load profile setup and validation details
are written to `.runtime/profile-setup.log`.

Existing installations do not need a dependency reinstall just to start LazyRig.
The update parser is loaded only when checking or installing an update. If the
standalone `packaging` helper is absent, the updater can use pip's parser or the
local pip wheel bundled with standard CPython. A check does not install packages.
If none of these parser sources is available, only update checks are unavailable;
the server continues to run.

- The target is the newest non-yanked stable GenAI wheel compatible with your
  Python, platform and this LazyRig version's declared runtime range. A newer
  release outside that range is shown separately. Pip also checks installed
  package constraints before making changes; conflicting dependencies stop the
  update rather than forcing an incompatible installation.
- GenAI selects the matching Runtime and Tokenizers dependencies. Existing
  non-runtime package versions are preserved; required new dependencies may be
  added. Runtime packages are never intentionally downgraded.
- **Nightly, experimental and custom runtime installations are preserved.**
  Update these with their original model-specific installation profile. A stable
  release is not proof that it supports every feature of an experimental build.
- Wheels are downloaded before installation, including the previous runtime
  versions for recovery. A download or dependency-resolution failure leaves the
  runtime unchanged. If installation or validation fails, the updater attempts
  to restore the previous three runtime packages and verifies them again.
- This updates Python runtime packages only. It does not update LazyRig itself,
  GPU drivers, model exports, prompts, settings or saved conversations.

Progress is saved in **`logs/runtime-update.log`**; the last outcome is in
**`logs/runtime-update.json`** and the dashboard update dialog. A temporary
disconnection during an update is expected. If recovery also fails, the updater
leaves the server stopped: run `setup-windows.bat` (or `setup.sh`) to repair a
stable environment, or use the original profile for a custom environment.

Validation checks package consistency, runtime imports and device discovery.
Test your usual models after an update; this is not a full inference-quality test.

See the official [OpenVINO GenAI dependency guidance](https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/genai-dependencies.html)
for why these runtime packages need compatible versions.
