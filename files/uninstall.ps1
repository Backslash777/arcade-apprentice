# Arcade Apprentice - UNINSTALL. Run it with UNINSTALL.bat (next to START HERE.bat).
# Removes everything the kit put on this PC:
#   - its private Python and AI libraries: files\runtime, or the short folder in AppData\Local\PortableAIArcade that
#     setup uses when the kit sits deep inside other folders
#   - the scratch folders its arcade windows use in Windows' Temp folder
# and, if you choose, the kit itself - your trained AIs and the games you put in it included.
# For testing:  uninstall.ps1 -Kit <kit folder> -Choice 1|2 -Force     (no questions asked)
param([string]$Kit, [string]$Choice = "", [switch]$Force)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

function Say([string]$Text, [string]$Colour = "Gray") { Write-Host "  $Text" -ForegroundColor $Colour }

function Done([int]$Code) {
  if (-not $Force) { Write-Host ""; Read-Host "  Press Enter to close" | Out-Null }
  exit $Code
}

function Get-SizeGB([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return 0.0 }
  $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  return [double]$sum / 1GB
}

function Remove-Tree([string]$Path) {
  # PyTorch has very long file names, which trip up ordinary deletes: use the long-path form, then robocopy if needed
  if (-not (Test-Path -LiteralPath $Path)) { return $true }
  $long = if ($Path.StartsWith("\\")) { $Path } else { "\\?\" + $Path }
  & cmd.exe /d /c rd /s /q "$long" 2>$null
  if (Test-Path -LiteralPath $Path) {
    $empty = Join-Path $env:TEMP ("aa-empty-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    New-Item -ItemType Directory -Path $empty -Force | Out-Null
    & robocopy.exe $empty $Path /MIR /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    Remove-Item -LiteralPath $empty -Force -ErrorAction SilentlyContinue
    & cmd.exe /d /c rd /s /q "$long" 2>$null
  }
  return -not (Test-Path -LiteralPath $Path)
}

function Test-SpecialFolder([string]$Path) {
  # Never remove a folder like the Desktop itself, even if the kit was unzipped straight onto it and it's now empty
  $p = $Path.TrimEnd('\')
  if ($p.Length -le 2) { return $true }
  $special = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("MyDocuments"),
               [Environment]::GetFolderPath("UserProfile"), (Join-Path $env:USERPROFILE "Downloads"),
               [Environment]::GetFolderPath("CommonDesktopDirectory"), $env:OneDrive, $env:TEMP)
  foreach ($s in $special) { if ($s -and ($p -ieq $s.TrimEnd('\'))) { return $true } }
  return $false
}

# ---- Which kit? ----
if (-not $Kit) { $Kit = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
try { $Kit = (Resolve-Path -LiteralPath $Kit -ErrorAction Stop).ProviderPath } catch { Say "Can't find the kit's folder: $Kit" Red; Done 1 }
if ($Kit.Length -gt 3) { $Kit = $Kit.TrimEnd('\') }
$files = Join-Path $Kit "files"
if (-not ((Test-Path -LiteralPath (Join-Path $files "setup.ps1")) -and (Test-Path -LiteralPath (Join-Path $Kit "START HERE.bat")))) {
  Say "That doesn't look like an Arcade Apprentice folder, so nothing was touched: $Kit" Red
  Done 1
}

Write-Host ""
Write-Host "  ARCADE APPRENTICE - UNINSTALL" -ForegroundColor Yellow
Write-Host ""
Say "Having a look what's here..."

# ---- Where its private Python lives: in the kit normally, in AppData when the kit sits deep inside other folders ----
$appRoot = Join-Path $env:LOCALAPPDATA "PortableAIArcade"
$md5 = New-Object Security.Cryptography.MD5CryptoServiceProvider
$tag = [BitConverter]::ToString($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($files))).Replace("-", "").Substring(0, 8)
$runtimes = @((Join-Path $files "runtime"), (Join-Path $appRoot "rt-$tag"))
$pathFile = Join-Path $files "python-path.txt"
if (Test-Path -LiteralPath $pathFile) {
  try {
    $rt = Split-Path -Parent (Split-Path -Parent ((Get-Content -LiteralPath $pathFile -TotalCount 1).Trim()))
    if ($rt -like "$appRoot\rt-*") { $runtimes += $rt }        # only ever a folder setup made, nowhere else
  } catch {}
}
$runtimes = @($runtimes | Select-Object -Unique | Where-Object { Test-Path -LiteralPath $_ })

$libsGB = 0.0
foreach ($rt in $runtimes) { $libsGB += Get-SizeGB $rt }
$scratch = Join-Path $env:TEMP "PortableAIArcade"
$ais = @(Get-ChildItem -LiteralPath (Join-Path $Kit "My AIs") -Directory -Force -ErrorAction SilentlyContinue).Count
$roms = @(Get-ChildItem -LiteralPath (Join-Path $Kit "ROMs - put your games here") -Recurse -File -Force -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -ne "desktop.ini" }).Count

Write-Host ""
Say "The kit:  $Kit"
if ($runtimes.Count) { Say ("Downloaded AI libraries:  {0:N1} GB" -f $libsGB) } else { Say "Downloaded AI libraries:  none (it hasn't been set up on this PC)" }
Say "Your trained AIs:  $ais      Your game files:  $roms"
Write-Host ""

# ---- What to remove ----
if ($Choice -notin @("1", "2", "3")) {
  Write-Host "  What would you like to remove?" -ForegroundColor White
  Write-Host "    1   Just the downloaded AI libraries. Keeps the kit, your trained AIs and your games."
  Write-Host "        (START HERE fetches them again if you ever want to use it again.)"
  Write-Host "    2   Everything: the libraries AND the kit, including your trained AIs and your games."
  Write-Host "    3   Nothing. Leave it all as it is."
  Write-Host ""
  do { $Choice = ([string](Read-Host "  Type 1, 2 or 3 and press Enter")).Trim() } until ($Choice -in @("1", "2", "3"))
}
if ($Choice -eq "3") { Say "Nothing removed." Green; Done 0 }
if ($Choice -eq "2" -and -not $Force -and ($ais -gt 0 -or $roms -gt 0)) {
  Write-Host ""
  Say "That deletes your $ais trained AI(s) and $roms game file(s) as well." Yellow
  Say "If you want to keep any of them, copy them somewhere else first." Yellow
  $sure = ([string](Read-Host "  Type YES to delete the lot, or just press Enter to cancel")).Trim()
  if ($sure -ne "YES") { Say "Cancelled. Nothing removed." Green; Done 0 }
}
Write-Host ""

# ---- Close the kit if it's running (the page's server, training, Watch windows) ----
if ($runtimes.Count) {
  $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $exe = $null
    try { $exe = $_.Path } catch {}
    if (-not $exe) { return $false }
    foreach ($r in $runtimes) { if ($exe.StartsWith($r + "\", [StringComparison]::OrdinalIgnoreCase)) { return $true } }
    return $false
  })
  if ($running.Count) {
    Say "Closing the kit first (it was still running)..."
    $running | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
}

$allGood = $true

# ---- The downloaded AI libraries ----
foreach ($rt in $runtimes) {
  Say "Removing the AI libraries in $rt ..."
  if (-not (Remove-Tree $rt)) { $allGood = $false; Say "Couldn't remove all of $rt. Restart the PC and run UNINSTALL again." Red }
}
foreach ($name in @("python-path.txt", "gpu-setup.json")) { Remove-Item -LiteralPath (Join-Path $files $name) -Force -ErrorAction SilentlyContinue }
if ((Test-Path -LiteralPath $appRoot) -and -not (Get-ChildItem -LiteralPath $appRoot -Force -ErrorAction SilentlyContinue)) {
  Remove-Item -LiteralPath $appRoot -Force -ErrorAction SilentlyContinue
}

# ---- Scratch folders the arcade windows used in Windows' Temp folder (only ones whose window has closed) ----
$swept = 0
$candidates = @(Get-ChildItem -LiteralPath $scratch -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^[a-z]+-\d+$' }) +
              @(Get-ChildItem -LiteralPath $env:TEMP -Directory -Force -Filter "PAA_*" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^PAA_(?:chk|compat)_\d+$' })
foreach ($d in $candidates) {
  $null = $d.Name -match '(\d+)$'
  try { $procId = [int]$Matches[1] } catch { continue }
  if (Get-Process -Id $procId -ErrorAction SilentlyContinue) { continue }      # that window is still open
  if (Remove-Tree $d.FullName) { $swept++ }
}
if ((Test-Path -LiteralPath $scratch) -and -not (Get-ChildItem -LiteralPath $scratch -Force -ErrorAction SilentlyContinue)) {
  Remove-Item -LiteralPath $scratch -Force -ErrorAction SilentlyContinue
}
if ($swept) { Say "Cleared $swept scratch folder(s) out of Windows' Temp folder." }

# ---- Choice 2: the kit itself. Only the kit's own files go, so if it was unzipped straight onto the Desktop
#      (no folder of its own), nothing else there is touched. ----
$selfBat = $null
$rest = @()
if ($Choice -eq "2") {
  Say "Removing the kit..."
  $own = @("files", "My AIs", "ROMs - put your games here", "README.txt", "START HERE.bat", "START HERE - with window.bat",
           "MAKE PROBLEM REPORT.bat", "CLOSE THE KIT.bat", "LIBRARY ROBOT - READ ME.txt")
  $own += @(Get-ChildItem -LiteralPath $Kit -File -Force -Filter "Problem report - *.txt" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
  foreach ($name in $own) {
    $p = Join-Path $Kit $name
    if (-not (Test-Path -LiteralPath $p)) { continue }
    if ((Get-Item -LiteralPath $p -Force).PSIsContainer) { $gone = Remove-Tree $p }
    else { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue; $gone = -not (Test-Path -LiteralPath $p) }
    if (-not $gone) { $allGood = $false; Say "Couldn't remove $p" Red }
  }
  $selfBat = Join-Path $Kit "UNINSTALL.bat"          # still running: removed a moment after this window closes
  $rest = @(Get-ChildItem -LiteralPath $Kit -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "UNINSTALL.bat" })
  if (-not (Test-SpecialFolder $Kit) -and $rest.Count -eq 1 -and $rest[0].Name -eq "desktop.ini") {
    Remove-Item -LiteralPath $rest[0].FullName -Force -ErrorAction SilentlyContinue       # Google Drive's folder file
    $rest = @()
  }
}

Write-Host ""
if ($allGood) { Say "All done." Green } else { Say "Nearly done - see the red lines above." Yellow }
if ($Choice -eq "1") { Say "The kit's still here with your trained AIs and games. Double-click START HERE to set it up again." }
$leftAlone = @($rest | Where-Object { $_.Name -ne "desktop.ini" } | ForEach-Object { $_.Name })
if ($leftAlone.Count) { Say ("These aren't part of the kit, so they were left alone: " + ($leftAlone -join ", ")) }
if (-not $Force) { Write-Host ""; Read-Host "  Press Enter to close" | Out-Null }

if ($selfBat -and (Test-Path -LiteralPath $selfBat)) {
  # UNINSTALL.bat can't delete itself while it's running, so a hidden helper does it two seconds after this window
  # closes - and removes the kit's folder too if that's now empty (rd on its own never deletes anything that isn't).
  $tail = if ($rest.Count -or (Test-SpecialFolder $Kit)) { "" } else { " & rd `"$Kit`"" }
  Start-Process -FilePath "cmd.exe" -ArgumentList "/d /c ping -n 3 127.0.0.1 >nul & del /f /q `"$selfBat`"$tail" -WindowStyle Hidden -WorkingDirectory $env:TEMP
}
exit 0
