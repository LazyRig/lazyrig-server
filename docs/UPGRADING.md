# Upgrade to LazyRig Server v0.41

1. Stop the server and wait for its model workers to exit.
2. Back up the application folder and the complete `server-data` directory, including any SQLite companion files. Export important browser conversations separately; they are stored by the browser origin.
3. Extract `LazyRig_Server_v0.41.zip`. Copy the contents of its `lazyrig-server` folder over the existing application folder, replacing supplied application files. Keep `.venv`, `.runtime`, `server-data`, model directories and your custom configuration files. The release archive contains none of those local data directories.
4. Start with the existing `start-server.bat` or `start-server.sh`. Run setup again only if the environment is missing or reports a missing/incompatible dependency.
5. Confirm the dashboard and `/api/state` report **0.41**. Reload the browser tab to receive the updated dashboard assets.

The update adds default compaction settings to existing configurations. Review them under **API & server → Server settings**. Existing conversations and jobs remain available. Previously recovered client jobs are not automatically moved out of your folders; new recoveries use **Client requests**, and reopening an existing conversation preserves its saved location.

The switch inside **Manage folder → Remove folder** is enabled by default. Removing the folder also deletes its browser conversations after confirmation. Switch it off to keep the conversations in the main history list. Active conversations must finish or be stopped before they can be deleted; server jobs and generated files remain available.

Loaded models now reload automatically when saved settings require a new pipeline. Sampling, template flags and context/output budgets apply without reloading a compatible worker. A failed reload is recorded in Recent activity with its error; the configuration remains saved for correction.

The output allowance supplied by a client still takes precedence over model defaults. The server never silently increases it. Automatic context compaction manages input history and cannot fix an answer that runs out of output tokens or repeats indefinitely.

To roll back, stop v0.41 and restore the backed-up application and complete data directory together. Keep any newer generated artifacts separately before restoring the backup. The older [testing guide](BETA-TEST.md) contains additional platform, device and operational checks.
