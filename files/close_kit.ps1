# Arcade Apprentice - CLOSE THE KIT. Run it with "CLOSE THE KIT.bat" (next to START HERE.bat).
# Does what the page's "Close the kit" button does, for when the page is already shut: every AI that's training is
# told to stop (each one saves first), the Library Robot stops after its current game, then everything else the kit
# has open (the page's server, the report card, Watch and Find-the-score windows) is closed.
param([string]$Kit, [int]$WaitMinutes = 10, [switch]$NoPause)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
function Say([string]$Text, [string]$Colour = "Gray") { Write-Host "  $Text" -ForegroundColor $Colour }
function Finish([int]$Code) { if (-not $NoPause) { Write-Host ""; Read-Host "  Press Enter to close" | Out-Null }; exit $Code }

if (-not $Kit) { $Kit = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$Kit = (Resolve-Path -LiteralPath $Kit).ProviderPath.TrimEnd('\')
$files = Join-Path $Kit "files"

# The kit's private Python: in the kit normally, in AppData when the kit sits deep inside other folders
$runtimes = @(Join-Path $files "runtime")
$pathFile = Join-Path $files "python-path.txt"
if (Test-Path -LiteralPath $pathFile) {
  try { $runtimes += Split-Path -Parent (Split-Path -Parent ((Get-Content -LiteralPath $pathFile -TotalCount 1).Trim())) } catch {}
}

function Get-KitProcesses {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue | Where-Object {
    $exe = $_.ExecutablePath
    $exe -and @($runtimes | Where-Object { $exe.StartsWith($_ + "\", [StringComparison]::OrdinalIgnoreCase) }).Count
  })
}
function Get-Learners {     # the ones that need to save before they stop
  @(Get-KitProcesses | Where-Object { $_.CommandLine -match '(trainer|evolution|goexplore|teacher|autopack)\.py' })
}

Write-Host ""
Write-Host "  ARCADE APPRENTICE - CLOSE THE KIT" -ForegroundColor Yellow
Write-Host ""
if (-not @(Get-KitProcesses).Count) { Say "The kit isn't running. Nothing to close." Green; Finish 0 }

# 1. Ask nicely. If the page's server is up it does it all properly; if not, leave the STOP notes it would have left.
$asked = $false
try { $null = Invoke-WebRequest "http://127.0.0.1:8795/api/quit" -UseBasicParsing -TimeoutSec 5; $asked = $true } catch {}
if (-not $asked) {
  $cmds = (@(Get-Learners) | ForEach-Object { $_.CommandLine }) -join "`n"
  Get-ChildItem -LiteralPath (Join-Path $Kit "My AIs") -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $run = Join-Path $_.FullName "files\run"
    if ($cmds.IndexOf($_.FullName, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and (Test-Path -LiteralPath $run)) {
      Set-Content -LiteralPath (Join-Path $run "STOP") "stop"           # only the AIs that are actually training
    }
  }
  if ($cmds -match 'autopack\.py') { Set-Content -LiteralPath (Join-Path $files "autopack\STOP") "stop" }
}

# 2. Give anything that's learning time to save
$learners = @(Get-Learners)
if ($learners.Count) {
  Say "Telling $($learners.Count) AI(s) to stop. Each one saves first, which can take a minute or two..."
  $deadline = (Get-Date).AddMinutes($WaitMinutes)
  while (@(Get-Learners).Count -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 3; Write-Host "." -NoNewline }
  Write-Host ""
  if (@(Get-Learners).Count) { Say "Still going after $WaitMinutes minutes, so closing them anyway. They keep what they last saved." Yellow }
  else { Say "All saved." Green }
} elseif ($asked) {
  Start-Sleep -Seconds 6       # let the page's server close itself the polite way
}

# 3. Close whatever's left: the page's server, the report card, Watch and Find-the-score windows
foreach ($p in @(Get-KitProcesses)) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1
if (@(Get-KitProcesses).Count) { Say "Something wouldn't close. Restart the PC to be sure." Red; Finish 1 }
Say "The kit's closed." Green
Finish 0
