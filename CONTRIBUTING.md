# Contributing

Use standard 64-bit Python 3.12 or 3.13. Application code lives in `lazyrig/`;
the dashboard is in `lazyrig/web/` and has no build step.

## Development checks

In an isolated Python environment:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The GitHub Actions workflow defines Linux Python 3.12/3.13 and Windows Python
3.13 lanes, including JavaScript syntax and release assembly. Configuration is
not evidence that a hosted run passed; record the actual result before release.

Tests use temporary directories and deterministic workers from `tests/support.py`.
These fixtures are developer tools, not a production backend or selectable server
mode. The SDK contract test uses real HTTP and the OpenAI Python client.
Native probes require explicitly supplied local exports.

For browser checks:

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/browser_workspace.py
python tests/browser_image_tools.py
python tests/browser_image_sizes.py
python tests/browser_client_recovery.py
python tests/browser_cancellation.py
```

Evidence is written to `test-results/`, excluded from releases.
Use the Actions run for the exact revision to check hosted results.
Keep local validation notes and publishing handoffs outside the repository.

## Refresh tutorial screenshots

The public guide uses actual v0.41 captures in `docs/images/tutorial/`.
Review every image for credentials and personal information before publication.
Update the relevant instructions and captions together.
Never present a fixture response as native model validation.

The legacy illustrative capture utility is available for development. With
Playwright and Chromium installed, direct its output outside the public guide:

```sh
python tests/capture_tutorial.py --output-dir test-results/illustrative-tutorial
```

An existing Chromium executable can be supplied with `--browser /path/to/chromium`.
The utility uses temporary model/device registrations and deterministic responses,
not model weights. Review its UI assumptions before reusing it after a redesign.
Its output is not a replacement for the operator's real-model tutorial captures.
The release builder includes PNGs placed in `docs/images/tutorial/`.

## Behavior to preserve

- Own the canonical data directory before recovery; never bypass the owner lock.
- Reserve capacity before expensive preparation and release permits on failure.
  Preserve same-key job identity while keeping cancellation responsive.
- Release in-memory terminal ownership even when storage fails. Do not claim an
  unflushed terminal snapshot is durable or re-open admission before reconciliation.
- Keep cumulative external event replay unchanged when compacting internal rows.
- Keep ordinary chat and client-owned tools separate from deprecated, explicitly
  selected server orchestration extensions. Tool names have no special meaning
  to the ordinary chat endpoint.
- Preserve device ownership through inference, atomic artifact publication and
  required cleanup. Model residency and conversation identity are separate.
- Recover a keyed request as the same job. Never regenerate failed or cancelled
  work automatically under a new key.
- Publish complete, validated tool arguments. Do not expose partial artifacts.
- Keep prompt enhancement opt-in. Its two defaults in `lazyrig/prompts/` are
  independently authored LazyRig material; maintain their JSON output contract.
  Do not reintroduce upstream prompt text without a documented compatible grant.
- Keep model-specific behavior in capability/export adapters. Document actual
  runtime restrictions instead of exposing controls that have no effect.

Report reproducible failures with the version, model/export, device, client,
minimal request and relevant logs. Separate fixture results from native inference
and full-model quality observations.

## Build a release

```sh
python tools/build_release.py
```

The default build creates `dist/LazyRig_Server_v0.41beta.zip` for end users: the
runnable application, setup files, user guides, screenshots and licenses. It
excludes tests, development tools, CI configuration and maintainer documents.

To also build an archive of the public source and development files:

```sh
python tools/build_release.py --include-source
```

Both archives are deterministic for identical source bytes and include their own
`MANIFEST.sha256.json`. The builder verifies each archive and its file hashes.
The manifest is generated for each archive and is not tracked as source.

Models, local settings, keys, runtime environments, logs and test results are
excluded. Keep the source tree free of private data; the release allowlist is not
a substitute for reviewing files you add. Update `lazyrig/__init__.py`, the
README, changelog and release notes together. Dashboard version text comes from
the server.

Personal handoffs, internal validation records and announcement drafts belong
outside the repository. Both archive selectors exclude those files.

Contributions to LazyRig are under the [Apache License 2.0](LICENSE),
unless explicitly stated otherwise. Preserve upstream notices and do not add
model weights or third-party code without checking its terms.
