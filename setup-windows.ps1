# LazyRig Windows bootstrap. Compatible with Windows PowerShell 5.1 and PowerShell 7.
param([switch]$Portable)
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
$RuntimeRoot = Join-Path $Root '.runtime'
$VenvRoot = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$UvVersion = '0.12.19'
$UvArchiveSha256 = '6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0'
$UvDirectory = Join-Path $RuntimeRoot ('uv-' + $UvVersion)
$UvExe = Join-Path $UvDirectory 'uv.exe'
$TranscriptStarted = $false
$SetupExitCode = 1

function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArgs, [string]$Step)
    & $Executable @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit $LASTEXITCODE). See the output above."
    }
}

function Get-RuntimeInfo {
    param([string]$Executable)
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $Probe = "import sys,struct,sysconfig,json; print(json.dumps(dict(version=list(sys.version_info[:3]),bits=struct.calcsize('P')*8,base=sys.base_prefix,gil_disabled=bool(sysconfig.get_config_var('Py_GIL_DISABLED')),implementation=sys.implementation.name)))"
        $OutputLines = @(& $Executable -I -c $Probe 2>$null)
        if ($LASTEXITCODE -ne 0 -or $OutputLines.Count -eq 0) { return $null }
        $Info = $OutputLines[-1] | ConvertFrom-Json
        if ($Info.implementation -ne 'cpython' -or $Info.bits -ne 64 -or $Info.gil_disabled) { return $null }
        if ($Info.version[0] -ne 3 -or $Info.version[1] -lt 12 -or $Info.version[1] -gt 13) { return $null }
        return $Info
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $PreviousPreference
    }
}

function Install-LocalUv {
    if (Test-Path -LiteralPath $UvExe -PathType Leaf) { return }
    $Stage = Join-Path $RuntimeRoot ('download-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $Stage -Force | Out-Null
    try {
        Write-Host "Downloading local dependency installer (uv $UvVersion)..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $Archive = Join-Path $Stage 'uv.zip'
        $Url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
        Invoke-WebRequest -Uri $Url -OutFile $Archive -UseBasicParsing
        $ActualHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $UvArchiveSha256) { throw 'Downloaded uv archive failed SHA-256 verification. Run setup again.' }
        Expand-Archive -LiteralPath $Archive -DestinationPath (Join-Path $Stage 'unpacked') -Force
        $UvCandidates = @(Get-ChildItem -LiteralPath (Join-Path $Stage 'unpacked') -Filter 'uv.exe' -File -Recurse)
        if ($UvCandidates.Count -ne 1) { throw 'The uv archive did not contain exactly one uv.exe.' }
        New-Item -ItemType Directory -Path $UvDirectory -Force | Out-Null
        Copy-Item -LiteralPath $UvCandidates[0].FullName -Destination $UvExe
    } finally {
        if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
    }
}

function Find-SystemPython313 {
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $OutputLines = @(& $UvExe python find 3.13 --no-managed-python --no-python-downloads --no-config 2>$null)
        if ($LASTEXITCODE -eq 0 -and $OutputLines.Count -gt 0) {
            $Candidate = [string]$OutputLines[-1]
            if ($null -ne (Get-RuntimeInfo $Candidate)) { return $Candidate }
        }
    } finally {
        $ErrorActionPreference = $PreviousPreference
    }
    return $null
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This installer requires 64-bit x86 Windows (Intel or AMD).'
    }
    New-Item -ItemType Directory -Path (Join-Path $Root 'logs') -Force | Out-Null
    $LogFile = Join-Path $Root ('logs\setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    Start-Transcript -LiteralPath $LogFile -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host 'LazyRig Server setup'
    Write-Host "Application folder: $Root"
    Write-Host 'Internet access is needed to download the runtime and dependencies.'
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    # Process-local configuration: no global PATH or Python registry modifications.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $RuntimeRoot 'python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $RuntimeRoot 'bin'
    $env:UV_CACHE_DIR = Join-Path $RuntimeRoot 'cache'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'
    $env:UV_PYTHON_INSTALL_BIN = '0'
    $env:UV_LINK_MODE = 'copy'
    Install-LocalUv

    $ExistingInfo = Get-RuntimeInfo $VenvPython
    $Reuse = $null -ne $ExistingInfo
    if ($Portable -and $Reuse) {
        $PrivatePrefix = [IO.Path]::GetFullPath($env:UV_PYTHON_INSTALL_DIR).TrimEnd('\') + '\'
        $Reuse = [IO.Path]::GetFullPath([string]$ExistingInfo.base).StartsWith($PrivatePrefix, [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $Reuse) {
        if (Test-Path -LiteralPath $VenvRoot) {
            $Backup = '.venv-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
            Rename-Item -LiteralPath $VenvRoot -NewName $Backup
            Write-Host "Previous environment kept in $Backup. Settings, models and outputs are unchanged."
        }
        $SelectedPython = $null
        if (-not $Portable) { $SelectedPython = Find-SystemPython313 }
        if ($null -eq $SelectedPython) {
            Write-Host 'Installing a private Python 3.13 runtime in the application folder...'
            Invoke-Checked $UvExe @('python', 'install', '3.13', '--no-bin', '--no-registry', '--no-config') 'Private Python download'
            Invoke-Checked $UvExe @('venv', $VenvRoot, '--python', '3.13', '--managed-python', '--no-config') 'Environment creation'
        } else {
            Write-Host "Using installed Python: $SelectedPython"
            Invoke-Checked $UvExe @('venv', $VenvRoot, '--python', $SelectedPython, '--no-python-downloads', '--no-config') 'Environment creation'
        }
    } else {
        Write-Host 'Reusing the existing compatible application environment.'
    }
    $SelectedInfo = Get-RuntimeInfo $VenvPython
    if ($null -eq $SelectedInfo) { throw 'The application Python environment could not be started. Rerun setup to repair it.' }
    Write-Host ('Application Python: ' + ($SelectedInfo.version -join '.') + ' (64-bit)')
    Write-Host 'Installing OpenVINO GenAI, OpenVINO tokenizers, Pillow and dependencies...'
    Invoke-Checked $UvExe @('pip', 'install', '--python', $VenvPython, '--requirement', (Join-Path $Root 'requirements-openvino.txt'), '--only-binary', ':all:', '--default-index', 'https://pypi.org/simple', '--no-config') 'Dependency installation'
    Invoke-Checked $UvExe @('pip', 'check', '--python', $VenvPython, '--no-config') 'Dependency verification'
    Write-Host 'Checking OpenVINO and detected devices...'
    Invoke-Checked $VenvPython @('-m', 'lazyrig.probe') 'OpenVINO device check'
    Write-Host ''
    Write-Host 'Setup complete. Run start-server.bat and open http://127.0.0.1:8765'
    Write-Host "Setup log: $LogFile"
    $SetupExitCode = 0
} catch {
    Write-Host ''
    Write-Host ('Setup failed: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Close any running LazyRig server before retrying.'
    Write-Host 'You can rerun setup-portable.bat for a private Python runtime.'
    if ($TranscriptStarted) { Write-Host "Setup log: $LogFile" }
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
exit $SetupExitCode
