<#
.SYNOPSIS
  One-shot local setup for GardenerBDX on Windows 11 (RTX 4070).

.DESCRIPTION
  Installs everything the project runbook (docs/GETTING_STARTED_GPU.md on the
  claude/magical-hamilton-q8uu91 branch) asks for, in order:

    1. Claude Code CLI            (npm, skipped if already installed)
    2. Repo clone + branch        ($HOME\BDX-code-and-simulator)
    3. Python 3.11 venv           ($HOME\isaac  - Isaac Sim requires 3.11)
    4. Isaac Sim (pip) + Isaac Lab (source, $HOME\IsaacLab)   [skippable]
    5. This project, editable, into the same venv
    6. scripts\preflight.py readiness check

  Safe to re-run: every step detects existing installs and skips or updates.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\install_local_windows.ps1

.EXAMPLE
  # Light install first (kinematic/MuJoCo only, no multi-GB Isaac download):
  powershell -ExecutionPolicy Bypass -File .\install_local_windows.ps1 -SkipIsaac

.EXAMPLE
  # Accept the NVIDIA EULA up front so headless/scripted Isaac runs don't prompt:
  powershell -ExecutionPolicy Bypass -File .\install_local_windows.ps1 -AcceptEula
#>
[CmdletBinding()]
param(
    [switch]$SkipIsaac,
    [switch]$AcceptEula,
    [string]$RepoDir     = "$HOME\BDX-code-and-simulator",
    [string]$VenvDir     = "$HOME\isaac",
    [string]$IsaacLabDir = "$HOME\IsaacLab",
    [string]$Branch      = 'claude/magical-hamilton-q8uu91'
)

$ErrorActionPreference = 'Stop'
$RepoUrl     = 'https://github.com/JakeSimeroth/BDX-code-and-simulator.git'
$IsaacLabUrl = 'https://github.com/isaac-sim/IsaacLab.git'

function Step([string]$Msg)  { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Note([string]$Msg)  { Write-Host "    $Msg" -ForegroundColor DarkGray }
function Ok([string]$Msg)    { Write-Host "    OK: $Msg" -ForegroundColor Green }
function Fail([string]$Msg) {
    Write-Host "`nFAILED: $Msg" -ForegroundColor Red
    Write-Host 'Fix the issue above and re-run this script; completed steps are skipped.' -ForegroundColor Red
    exit 1
}
# Native commands don't throw on failure in PowerShell - check exit codes explicitly.
function Run([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { Fail "'$Exe $($ArgList -join ' ')' exited with code $LASTEXITCODE" }
}

# ---------------------------------------------------------------- prerequisites
Step 'Checking prerequisites (git, Node 18+, Python 3.11)'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail 'git not found. Install it: winget install Git.Git   (then reopen PowerShell)'
}
Ok 'git found'

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail 'Node.js not found (needed for Claude Code). Install: winget install OpenJS.NodeJS.LTS'
}
$nodeMajor = [int]((node --version).TrimStart('v').Split('.')[0])
if ($nodeMajor -lt 18) { Fail "Node $nodeMajor found, but Claude Code needs Node 18+. Update Node.js." }
Ok "Node $(node --version)"

# Isaac Sim pip wheels exist for Python 3.11 ONLY - the venv must be 3.11 exactly.
$py311 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) { $py311 = $true }
}
if (-not $py311) {
    Fail 'Python 3.11 not found (Isaac Sim requires 3.11 exactly). Install: winget install Python.Python.3.11'
}
Ok "Python 3.11 found ($(py -3.11 --version))"

# ---------------------------------------------------------------- 1. Claude Code
Step 'Claude Code CLI'
if (Get-Command claude -ErrorAction SilentlyContinue) {
    Ok "already installed ($(claude --version))"
} else {
    Run 'npm' @('install', '-g', '@anthropic-ai/claude-code')
    Ok 'installed'
}

# ---------------------------------------------------------------- 2. repo + branch
Step "Repository -> $RepoDir (branch: $Branch)"
if (Test-Path (Join-Path $RepoDir '.git')) {
    Note 'repo already cloned; fetching latest'
    Run 'git' @('-C', $RepoDir, 'fetch', 'origin', $Branch)
} else {
    Run 'git' @('clone', $RepoUrl, $RepoDir)
}
Run 'git' @('-C', $RepoDir, 'checkout', $Branch)
Run 'git' @('-C', $RepoDir, 'pull', '--ff-only', 'origin', $Branch)
Ok "on branch $Branch"

# ---------------------------------------------------------------- 3. Python venv
Step "Python 3.11 venv -> $VenvDir"
$activate = Join-Path $VenvDir 'Scripts\Activate.ps1'
if (Test-Path $activate) {
    Note 'venv already exists'
} else {
    Run 'py' @('-3.11', '-m', 'venv', $VenvDir)
}
. $activate
$venvPy = (Get-Command python).Source
if ($venvPy -notlike "$VenvDir*") { Fail "venv activation failed: python resolves to $venvPy" }
Ok "venv active ($venvPy)"
Run 'python' @('-m', 'pip', 'install', '--upgrade', 'pip')

# ---------------------------------------------------------------- 4. Isaac Sim + Isaac Lab
if ($SkipIsaac) {
    Step 'Isaac Sim + Isaac Lab: SKIPPED (-SkipIsaac)'
    Note 'You can still run the kinematic twin and MuJoCo physics.'
    Note 'Re-run without -SkipIsaac when ready for USD conversion and gait training.'
} else {
    Step 'Isaac Sim via pip (several GB - grab a coffee on first run)'
    Run 'pip' @('install', 'isaacsim[all,extscache]', '--extra-index-url', 'https://pypi.nvidia.com')
    Ok 'isaacsim installed'

    Step "Isaac Lab from source -> $IsaacLabDir"
    if (Test-Path (Join-Path $IsaacLabDir 'isaaclab.bat')) {
        Note 'Isaac Lab already cloned'
    } else {
        Run 'git' @('clone', $IsaacLabUrl, $IsaacLabDir)
    }
    Push-Location $IsaacLabDir
    try {
        # isaaclab.bat installs Isaac Lab + rsl_rl into the ACTIVE venv.
        Run (Join-Path $IsaacLabDir 'isaaclab.bat') @('--install')
    } finally {
        Pop-Location
    }
    Ok 'Isaac Lab installed into the venv'
}

if ($AcceptEula) {
    # Persist for future sessions so scripted/headless Isaac runs never prompt.
    [Environment]::SetEnvironmentVariable('OMNI_KIT_ACCEPT_EULA', 'YES', 'User')
    $env:OMNI_KIT_ACCEPT_EULA = 'YES'
    Ok 'NVIDIA EULA accepted via OMNI_KIT_ACCEPT_EULA=YES (persisted for your user)'
}

# ---------------------------------------------------------------- 5. this project
Step 'GardenerBDX project install (editable, same venv)'
Push-Location $RepoDir
try {
    if ($SkipIsaac) {
        Run 'pip' @('install', '-e', '.[sim,viz]')
    } else {
        Run 'pip' @('install', '-e', '.[sim,vla,train,viz]')
    }
    Ok 'project installed'

    # ------------------------------------------------------------ 6. preflight
    Step 'Preflight readiness check'
    & python 'scripts\preflight.py'
    if ($LASTEXITCODE -ne 0) {
        Note 'preflight reported problems - read its checklist above for the exact fix hints.'
    }
} finally {
    Pop-Location
}

# ---------------------------------------------------------------- done
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host ' Setup complete. To start working:' -ForegroundColor Green
Write-Host "   & `"$activate`""
Write-Host "   cd `"$RepoDir`""
Write-Host '   claude'
Write-Host ' First message to Claude:  read CLAUDE.md and run preflight' -ForegroundColor Green
Write-Host ' Task runner vocabulary:   python scripts\dev.py list' -ForegroundColor Green
if (-not $AcceptEula -and -not $SkipIsaac) {
    Write-Host ' Note: first Isaac boot asks you to accept the NVIDIA EULA and' -ForegroundColor Yellow
    Write-Host ' compiles RTX shaders (looks hung for a few minutes - it is not).' -ForegroundColor Yellow
}
Write-Host '============================================================' -ForegroundColor Green
