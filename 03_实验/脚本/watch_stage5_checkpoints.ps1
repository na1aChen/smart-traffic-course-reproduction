$ErrorActionPreference = 'Stop'

$projectRoot = '<PROJECT_ROOT>'
$runRoot = Join-Path $projectRoot '03_实验\robustness'
$logPath = Join-Path $projectRoot '03_实验\logs\STAGE5\checkpoint_backup.log'
$seen = @{}
$copied = @{}

New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null

while ($true) {
    $active = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match 'run_stage5_queue|run_stage5_resume_queue|stage5_runner\.py train|stage5_runner\.py resume' }

    $checkpoints = Get-ChildItem -LiteralPath $runRoot -Recurse -Filter last.pt -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\train\\weights\\last\.pt$' }

    foreach ($checkpoint in $checkpoints) {
        $key = $checkpoint.FullName
        $signature = '{0}|{1}' -f $checkpoint.LastWriteTimeUtc.Ticks, $checkpoint.Length
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $signature
            continue
        }
        if ($seen[$key] -ne $signature) {
            $seen[$key] = $signature
            $copied[$key] = $null
            continue
        }
        if ($copied.ContainsKey($key) -and $copied[$key] -eq $signature) {
            continue
        }

        $experiment = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $checkpoint.FullName))
        $backupRoot = Join-Path $checkpoint.DirectoryName 'backup'
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
        $name = '{0}_{1}.pt' -f $experiment, $checkpoint.LastWriteTimeUtc.ToString('yyyyMMdd_HHmmss')
        $destination = Join-Path $backupRoot $name
        Copy-Item -LiteralPath $checkpoint.FullName -Destination $destination -Force
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash
        Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
            '{0} {1} {2}' -f (Get-Date).ToString('o'), $destination, $hash
        )
        $copied[$key] = $signature

        Get-ChildItem -LiteralPath $backupRoot -Filter '*.pt' |
            Sort-Object LastWriteTime -Descending |
            Select-Object -Skip 4 |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    }

    if (-not $active -and $checkpoints.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 30
}
