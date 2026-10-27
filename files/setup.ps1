# Arcade Apprentice - first-run setup. Downloads a private copy of Python and the AI libraries.
# Nothing is installed system-wide, and UNINSTALL.bat removes all of it again. Signed: Claude, 9 Oct 2026.
#   setup.ps1            full setup (Python + PyTorch + emulator)
#   setup.ps1 -FixTorch  only swap PyTorch for a build that suits this graphics card (START HERE does this itself)
param([switch]$FixTorch)
# "Continue": pip and nvidia-smi write harmless warnings to stderr, which "Stop" turns into fatal errors on
# Windows PowerShell 5.1. Downloads use -ErrorAction Stop explicitly instead.
$ErrorActionPreference = "Continue"
trap { Write-Host "  Setup stopped: $_" -ForegroundColor Red; exit 1 }
$ProgressPreference = "SilentlyContinue"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pathFile = Join-Path $here "python-path.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $here "logs") | Out-Null
try { Start-Transcript -Path (Join-Path $here "logs\setup.log") -Append | Out-Null } catch {}

# Where the private Python lives. Normally files\runtime. PyTorch has very long file names, so if this folder is deep
# and Windows "long paths" are off, install to a short folder in the user's AppData instead (remembered in python-path.txt).
$rt = Join-Path $here "runtime"
$long = 0
try { $long = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -ErrorAction Stop).LongPathsEnabled } catch {}
if ($long -ne 1 -and ($rt.Length + 180) -gt 255) {
  $tag = [BitConverter]::ToString((New-Object Security.Cryptography.MD5CryptoServiceProvider).ComputeHash([Text.Encoding]::UTF8.GetBytes($here))).Replace("-", "").Substring(0, 8)
  $rt = Join-Path $env:LOCALAPPDATA "PortableAIArcade\rt-$tag"
}
if ($FixTorch -and (Test-Path $pathFile)) { $rt = Split-Path -Parent (Split-Path -Parent ((Get-Content $pathFile -TotalCount 1).Trim())) }
$py = Join-Path $rt "python"
$pyexe = Join-Path $py "python.exe"
# pip keeps a copy of everything it downloads (PyTorch alone is 2-3 GB), normally in the user's AppData. Keep that copy
# inside our own folder instead, and empty it when setup finishes, so nothing gets left behind.
$env:PIP_CACHE_DIR = Join-Path $rt "pip-cache"
function Clear-PipCache { if (Test-Path $env:PIP_CACHE_DIR) { Remove-Item -Recurse -Force $env:PIP_CACHE_DIR -ErrorAction SilentlyContinue } }

Write-Host ""
if ($FixTorch) { Write-Host "  ARCADE APPRENTICE - repairing PyTorch for your graphics card (about 3 GB)" -ForegroundColor Yellow }
else {
  Write-Host "  ARCADE APPRENTICE - first-time setup" -ForegroundColor Yellow
  Write-Host "  This downloads about 4 GB (Python + the AI libraries, 5-15 minutes). Only needed once." -ForegroundColor Gray
  $drive = Get-PSDrive -Name ($rt.Substring(0, 1)) -ErrorAction SilentlyContinue
  if ($drive -and $drive.Free -lt 6GB) { Write-Host ("  WARNING: only {0:N1} GB free on drive {1}: - it needs about 6 GB." -f ($drive.Free / 1GB), $drive.Name) -ForegroundColor Red }
  if ($rt -ne (Join-Path $here "runtime")) { Write-Host "  (This folder is deep inside others, so the AI libraries go in $rt)" -ForegroundColor Gray }
}
Write-Host ""

if (-not $FixTorch) {
  New-Item -ItemType Directory -Force -Path $py | Out-Null
  if (-not (Test-Path $pyexe)) {
    Write-Host "  [1/4] Downloading Python 3.12..."
    $zip = Join-Path $rt "python.zip"
    Invoke-WebRequest "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" -OutFile $zip -UseBasicParsing -ErrorAction Stop
    Expand-Archive $zip -DestinationPath $py -Force -ErrorAction Stop
    Remove-Item $zip
    $pth = Get-ChildItem $py -Filter "python3*._pth" | Select-Object -First 1
    (Get-Content $pth.FullName) -replace "^#\s*import site", "import site" | Set-Content $pth.FullName
    Add-Content $pth.FullName "Lib\site-packages"
  }
  Write-Host "  [2/4] Setting up the package installer..."
  Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile (Join-Path $rt "get-pip.py") -UseBasicParsing -ErrorAction Stop
  & $pyexe (Join-Path $rt "get-pip.py") --no-warn-script-location -q
  # Microsoft C++ runtime DLLs, so PyTorch loads even on PCs that never installed the "VC++ Redistributable"
  & $pyexe -m pip install --no-warn-script-location -q msvc-runtime
}

# Which PyTorch build? It must work with THIS graphics card - every route is self-tested before it's kept.
#   NVIDIA: RTX 50-series (compute 12.x) needs a CUDA 13 build, older cards the CUDA 12.6 build.
#   AMD Radeon: AMD's official ROCm build for Windows (RX 7000/9000 class, recent Adrenalin driver),
#               else Microsoft DirectML (older Radeons too).   Intel Arc/Iris: DirectML.   Otherwise: CPU.
$ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
$ROCM_SDK = @("$ROCM/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "$ROCM/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
              "$ROCM/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "$ROCM/rocm-7.2.1.tar.gz")
$ROCM_TORCH = "$ROCM/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
$vendor = (& $pyexe (Join-Path $here "device.py") --vendor 2>$null | Select-Object -Last 1)
if (-not $vendor) { $vendor = "none" }
function Test-Gpu { $null = & $pyexe (Join-Path $here "gpucheck.py") --test 2>&1; return [bool]($LASTEXITCODE -eq 0) }
$torchOk = $false
if ($vendor -eq "nvidia") {
  $cc = 0.0
  try { $q = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $q) { $cc = [double](($q | Select-Object -First 1).Trim()) } } catch {}
  if ($cc -ge 10) { $builds = @("cu130", "cu132", "cu128") } elseif ($cc -ge 7.5 -or $cc -eq 0) { $builds = @("cu126", "cu130") } else { $builds = @("cu126") }
  foreach ($b in $builds) {
    Write-Host "  [3/4] NVIDIA graphics card found (compute $cc) - installing GPU PyTorch ($b, big download)..." -ForegroundColor Green
    & $pyexe -m pip install --no-warn-script-location -q --force-reinstall --no-deps torch --index-url "https://download.pytorch.org/whl/$b"
    & $pyexe -m pip install --no-warn-script-location -q torch --index-url "https://download.pytorch.org/whl/$b"
    if (Test-Gpu) { $torchOk = $true; break }
    Write-Host "  That build doesn't suit this card - trying the next one..." -ForegroundColor Yellow
    Clear-PipCache
  }
}
if ($vendor -eq "amd") {
  Write-Host "  [3/4] AMD Radeon found - trying AMD's ROCm PyTorch (big download)..." -ForegroundColor Green
  & $pyexe -m pip install --no-warn-script-location -q --no-cache-dir @ROCM_SDK
  & $pyexe -m pip install --no-warn-script-location -q --no-cache-dir --force-reinstall $ROCM_TORCH
  if (Test-Gpu) { $torchOk = $true }
  else { Write-Host "  ROCm doesn't support this Radeon (or the driver is old) - trying DirectML instead..." -ForegroundColor Yellow }
}
if (-not $torchOk -and ($vendor -eq "amd" -or $vendor -eq "intel")) {
  Write-Host "  [3/4] Installing the DirectML graphics route (works with most AMD and Intel cards)..." -ForegroundColor Green
  & $pyexe -m pip install --no-warn-script-location -q torch-directml
  if (Test-Gpu) { $torchOk = $true }
  else { & $pyexe -m pip uninstall -y -q torch-directml 2>$null }
}
if (-not $torchOk) {
  if ($vendor -ne "none") { Write-Host "  Couldn't get the graphics card working - using the CPU instead (slower, still learns)." -ForegroundColor Yellow }
  else { Write-Host "  [3/4] No usable graphics card found - installing the CPU version of PyTorch..." -ForegroundColor Yellow }
  & $pyexe -m pip install --no-warn-script-location -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
}
# A route that didn't work can leave gigabytes of graphics libraries behind (AMD's ROCm alone is several GB):
# remove any the working PyTorch doesn't use.
& $pyexe (Join-Path $here "tidy_packages.py")
if ($FixTorch) { & $pyexe (Join-Path $here "device.py") --record; Clear-PipCache; Write-Host "  PyTorch repaired." -ForegroundColor Green; exit 0 }

Write-Host "  [4/4] Installing the game emulators and the rest..."
& $pyexe -m pip install --no-warn-script-location -q "ale-py>=0.11" gymnasium numpy pillow pygame psutil
& $pyexe -c "import torch, ale_py, pygame; print('  Ready. PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Write-Host "  Setup check failed - see the messages above." -ForegroundColor Red; exit 1 }
& $pyexe (Join-Path $here "device.py") --record
if ($rt -ne (Join-Path $here "runtime")) { Set-Content -Path $pathFile -Value $pyexe -Encoding ASCII }
Clear-PipCache
Write-Host ""
Write-Host "  Setup finished." -ForegroundColor Green
