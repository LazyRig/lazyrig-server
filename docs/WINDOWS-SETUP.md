# Windows installation: one-click setup and start

**For a normal installation, extract the release and double-click
`start-server.bat`.** It runs setup if the application environment is missing,
then starts LazyRig. You do not need to install Python manually.

> **Testing v0.41:** read [safe beta setup and rollback](BETA-TEST.md) before using an existing installation.

## First run

1. Use the supplied `LazyRig_Server_v0.41beta.zip`.
2. Right-click the ZIP and choose **Extract All**. Use a writable location such
   as `C:\LazyRig`; open the extracted `lazyrig-server` folder. Keep the package
   together rather than moving a launcher out of it.
3. Double-click **`start-server.bat`**. If extensions are hidden, look for
   **start-server** with the type **Windows Batch File**. Use your usual Windows
   account; there is no need to switch users or choose **Run as administrator**.
   The account running the server needs access to the application and model folders.
4. Keep the console open. On the first run, it may download Python and runtime
   packages before checking detected OpenVINO devices. This can take several
   minutes depending on your connection. Model weights are not downloaded here.
5. Wait for the server address, then open **http://127.0.0.1:8765** in your browser.
   The dashboard does not open automatically. **Server online** confirms the
   browser is connected; an empty model list is normal on a fresh installation.

Continue with the [illustrated getting-started tutorial](GETTING-STARTED.md#2-add-your-first-model).

## Windows publisher warning

Windows may show **Open File — Security Warning** for `start-server.bat`, with
**Unknown publisher** and a notice that the file has no valid digital signature.
This beta is distributed as a source ZIP with batch launchers; it does not include
a signed Windows installer. Windows can use a downloaded file's origin metadata
(Mark of the Web) to decide whether to display a warning, so the prompt can appear
even when the launcher contents have not changed.

Check that the file is the expected `start-server.bat` in your extracted release
and that the download came from the intended source. For a release you have
verified and trust, **Run** continues the launch. You do not need to disable
Windows security settings or run as administrator. A checksum can confirm that
your download matches the published archive; it is not a publisher signature.

See [Microsoft's explanation of Attachment Manager](https://support.microsoft.com/en-gb/windows/security/information-about-the-attachment-manager-in-microsoft-windows).

## What the automatic installer does

- Downloads a pinned, SHA-256-checked `uv` dependency installer into `.runtime/`.
- Reuses a compatible application environment when possible. For a new environment,
  it looks for standard 64-bit CPython 3.13 on your computer; otherwise it downloads
  a private Python 3.13 runtime into the application folder.
- Creates `.venv/` and installs the packages in `requirements-openvino.txt` there.
- Checks dependency consistency and OpenVINO's detected devices.
- Records setup output in `logs/setup-*.log`.

It does not install packages into your global Python, change global PATH or Python
registry settings, or require administrator access. It does not install GPU
drivers or model weights. First setup needs internet; conversion profiles can
require further downloads when you import a source model later.

The ZIP is a source package with setup tools, not a bundle containing Python,
OpenVINO runtimes and models. “One-click” describes the launcher handling setup
and server startup after extraction.

## Which file should I run?

| File | Purpose | What happens afterward |
| --- | --- | --- |
| **`start-server.bat`** | First run and everyday launch | Sets up a missing environment, then runs the server |
| `setup-windows.bat` | Install or repair without starting the server | When setup succeeds, run `start-server.bat` |
| `setup-portable.bat` | Force a private Python runtime for installation or repair | When setup succeeds, run `start-server.bat` |
| `setup-windows.ps1` | Installer called by the launchers | Leave it beside the BAT files |

The launchers invoke PowerShell with a policy override for that process; they do
not permanently change the computer's PowerShell execution policy.

## Everyday use

Double-click **`start-server.bat`**, then open the dashboard address. The existing
Python environment is reused. Keep one server window open for that installation.

To stop, press **Ctrl+C** in the server window and let cleanup finish. The BAT
launcher may then ask you to press a key before closing. Closing only the browser
does not stop the server. Accepted background jobs can continue while the server
runs; later conversation/tool stages need the browser again.

## Repair an interrupted or moved installation

Stop the server first, then run **`setup-windows.bat`** again. If you want an
independent Python runtime, use **`setup-portable.bat`** instead.

The start launcher checks whether its Python executable exists. If that executable
exists but a dependency download was interrupted, use a setup file to repair the
environment; launching again alone does not reinstall all dependencies.

Setup can rename an unusable environment to `.venv-backup-...` before rebuilding.
It preserves model folders and server data. Read the final error and the latest
setup log if installation fails; retry after fixing the reported cause.

After moving the folder, rerun setup because Python environments can contain
absolute paths. Use `setup-portable.bat` if the installation uses a private runtime.

## Upgrade without losing your work

Stop the original server, extract this beta into a fresh folder and preserve the
original installation and original data untouched. Reuse external model paths and
copy the whole stopped `server-data/` directory into the beta installation. If
using `--data-dir`, point it at that separate copy. Run setup in the new location.
Never run two servers against the same data directory. To roll back, use the
original application with the original data, not the beta-modified copy. Follow
[the beta setup procedure](BETA-TEST.md).

The browser stores chat history separately in IndexedDB. Keep the same browser,
host and port, and use **History → Export** for an additional conversation copy.
See [Data and upgrades](../README.md#data-and-upgrades).

## Quick troubleshooting

| What you see | What to check |
| --- | --- |
| No `start-server.bat` | Extract the complete release; check inside its `lazyrig-server` folder and enable filename extensions if needed |
| Setup failed | Read the final console message and `logs/setup-*.log`; rerun setup after correcting it |
| Missing Python module when starting | Run `setup-windows.bat` to repair dependencies |
| Address already in use | Stop the previous LazyRig instance; do not start a second copy on the same port |
| Browser cannot connect | Keep the server console open and use the address printed there; the default is `http://127.0.0.1:8765` |
| GPU missing | Install/check the appropriate drivers, use **Refresh devices**, and inspect the console's device list |
| Access denied to a model folder | Use a folder readable by the Windows account running LazyRig. Switching users can change access to model folders and drives. The folder picker skips drive probes that fail; it does not change Windows permissions. |

## Manual PowerShell entry point

<details>
<summary>For users who prefer a terminal</summary>

Run inside the extracted folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup-windows.ps1
.\.venv\Scripts\python.exe -m lazyrig
```

Add `-Portable` to the installer command to force its private-runtime route.

</details>

## Setup notes and references

The instructions follow the packaged launcher and installer code. Windows setup
has not been executed in the Linux validation environment. If installation fails,
include the console error and your Windows version in a bug report.

- [uv Python installation](https://docs.astral.sh/uv/guides/install-python/)
- [uv local storage](https://docs.astral.sh/uv/reference/storage/)
- [uv command reference](https://docs.astral.sh/uv/reference/cli/)
- [Pinned uv release](https://github.com/astral-sh/uv/releases/tag/0.12.19)

## Runtime updates

Use **Update runtime** beside the dashboard runtime version, or stop LazyRig
and double-click **`update-runtime.bat`**. The script checks stable releases and
asks before installing; start LazyRig again afterwards. Dashboard installation
restarts the server automatically. See [Runtime updates](RUNTIME-UPDATES.md).
