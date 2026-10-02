# Start-menu and Desktop shortcuts for jev photos (run from the unpacked folder).
#   powershell -ExecutionPolicy Bypass -File install-desktop.ps1           add
#   powershell -ExecutionPolicy Bypass -File install-desktop.ps1 -Remove   remove
param([switch]$Remove)
$exe = Join-Path $PSScriptRoot 'jev-photos.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path (Split-Path $PSScriptRoot) 'jev-photos.exe' }
$links = @(
  (Join-Path ([Environment]::GetFolderPath('Programs')) 'jev photos.lnk'),
  (Join-Path ([Environment]::GetFolderPath('Desktop')) 'jev photos.lnk')
)
if ($Remove) { $links | ForEach-Object { Remove-Item $_ -ErrorAction SilentlyContinue }; 'removed'; exit 0 }
if (-not (Test-Path $exe)) { Write-Error "jev-photos.exe not found next to this script"; exit 1 }
$shell = New-Object -ComObject WScript.Shell
foreach ($l in $links) {
  $s = $shell.CreateShortcut($l)
  $s.TargetPath = $exe
  $s.WorkingDirectory = Split-Path $exe
  $s.IconLocation = "$exe,0"
  $s.Description = 'Organize, tag and search photos'
  $s.Save()
}
'shortcuts added: Start menu and Desktop'
