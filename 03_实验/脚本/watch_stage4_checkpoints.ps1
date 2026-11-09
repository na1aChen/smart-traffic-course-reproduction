$ErrorActionPreference = 'Stop'

$projectRoot = '<PROJECT_ROOT>'
$source = Join-Path $projectRoot '03_实验\P2-001\train\weights\last.pt'
$backupRoot = Join-Path $projectRoot '03_实验\P2-001\train\weights\backup'
$logPath = Join-Path $projectRoot '03_实验\logs\P2-001\checkpoint_backup.log'

New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null

$pendingStamp = $null
$backedUpStamp = $null

function Copy-StableCheckpoint {
    param([datetime]$Stamp)
    if ($null -eq $Stamp -or $Stamp -eq $backedUpStamp) {
        return
    }
    if (-not (Test-Path -LiteralPath $source)) {
        return
    }
    $name = 'last_{0}.pt' -f $Stamp.ToString('yyyyMMdd_HHmmss')
    $destination = Join-Path $backupRoot $name
    Copy-Item -LiteralPath $source -Destination $destination -Force
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash
    $line = '{0} {1} {2}' -f (Get-Date).ToString('o'), $name, $hash
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

while ($true) {
    $training = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match 'stage4_runner\.py train|stage4_runner\.py resume' }

    if (-not (Test-Path -LiteralPath $source)) {
        Start-Sleep -Seconds 30
        continue
    }

    $stamp = (Get-Item -LiteralPath $source).LastWriteTimeUtc
    if ($stamp -eq $pendingStamp) {
        Copy-StableCheckpoint -Stamp $stamp
        $backedUpStamp = $stamp
    }
    else {
        $pendingStamp = $stamp
    }

    if (-not $training) {
        Copy-StableCheckpoint -Stamp $pendingStamp
        break
    }

    Start-Sleep -Seconds 30
}
