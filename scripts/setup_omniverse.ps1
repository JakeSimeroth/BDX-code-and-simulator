<#
  Bootstrap NVIDIA Isaac Sim / Isaac Lab for GardenerBDX on a Windows 11 RTX box.
  Native-Windows counterpart of setup_omniverse.sh (uses isaaclab.bat).

  PREREQS: NVIDIA RTX GPU + driver 580.65+, Python 3.11, ~50 GB disk.
  Run from a Python 3.11 venv that already has Isaac Sim + Isaac Lab installed.

  Usage (PowerShell):
    $env:ISAACLAB_PATH = "C:\IsaacLab"
    .\scripts\setup_omniverse.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Repo = (Get-Location).Path
$IsaacLabPath = if ($env:ISAACLAB_PATH) { $env:ISAACLAB_PATH } else { Join-Path $HOME "IsaacLab" }

Write-Host "==> GardenerBDX Omniverse setup (Windows)"
Write-Host "    repo:     $Repo"
Write-Host "    IsaacLab: $IsaacLabPath"

# 1) GPU sanity
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "nvidia-smi not found. Isaac Sim needs an NVIDIA RTX GPU + driver."; exit 1
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2) Isaac Lab presence
if (-not (Test-Path $IsaacLabPath)) {
    Write-Host @"
!! Isaac Lab not found at $IsaacLabPath.
   In a Python 3.11 venv (no legacy Omniverse Launcher needed):
     pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com
     git clone https://github.com/isaac-sim/IsaacLab.git "$IsaacLabPath"
     cd "$IsaacLabPath"; .\isaaclab.bat --install
   Docs: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
   Then re-run this script.
"@
    exit 1
}
$IsaacLab = Join-Path $IsaacLabPath "isaaclab.bat"

# 3) install this package into Isaac's python
Write-Host "==> installing gardener-bdx into Isaac Lab's python"
& $IsaacLab -p -m pip install -e $Repo

# 4) convert the robot to USD
Write-Host "==> converting URDF -> USD"
& $IsaacLab -p "$Repo\scripts\convert_to_usd.py" --input "$Repo\models\robot\gardener_bdx.urdf"

# 5) smoke-train a few iterations
Write-Host "==> smoke test: 5 training iterations on 64 envs"
& $IsaacLab -p -m gardener_bdx.training.train_isaaclab --num_envs 64 --max_iterations 5 --headless

Write-Host @"

==> Done. Full run:
    $IsaacLab -p -m gardener_bdx.training.train_isaaclab --num_envs 4096 --headless --max_iterations 1500
Then watch it in the interactive Isaac Sim window:
    $IsaacLab -p scripts\run_isaac.py --policy models\policies\locomotion.npz
"@
