$ErrorActionPreference = 'Stop'

$projectRoot = '<PROJECT_ROOT>'
$python = Join-Path $projectRoot '02_代码\环境\venv\Scripts\python.exe'
$runner = Join-Path $projectRoot '02_代码\阶段5\stage5_runner.py'
$logRoot = Join-Path $projectRoot '03_实验\logs\STAGE5'
$queueStatusPath = Join-Path $logRoot 'queue_status.json'
$watcherScript = Join-Path $projectRoot '03_实验\脚本\watch_stage5_checkpoints.ps1'
$experiments = @(
    'BASELINE-002-S2027',
    'P2-002-S2027',
    'BASELINE-BUDGET48-S2026'
)

$watcher = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'watch_stage5_checkpoints\.ps1' -and $_.ProcessId -ne $PID }
if (-not $watcher) {
    Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $watcherScript) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot 'checkpoint_watcher_resume_stdout.log') `
        -RedirectStandardError (Join-Path $logRoot 'checkpoint_watcher_resume_stderr.log') | Out-Null
}

if (-not (Test-Path -LiteralPath $queueStatusPath)) {
    throw "Stage 5 queue status not found: $queueStatusPath"
}

$queue = Get-Content -LiteralPath $queueStatusPath -Raw | ConvertFrom-Json
$current = $queue.current_experiment
if (-not $current) {
    throw "Stage 5 queue has no current experiment; status=$($queue.status)"
}

$startIndex = [Array]::IndexOf($experiments, $current)
if ($startIndex -lt 0) {
    throw "Unknown current experiment in queue: $current"
}

for ($index = $startIndex; $index -lt $experiments.Count; $index++) {
    $experiment = $experiments[$index]
    $trainSummary = Join-Path $projectRoot "01_立项\阶段5_证据\training\$experiment.json"
    $evalSummary = Join-Path $projectRoot "01_立项\阶段5_证据\evaluation\$experiment.json"
    $expRoot = Join-Path $projectRoot "03_实验\robustness\$experiment"
    $lastCheckpoint = Join-Path $expRoot 'train\weights\last.pt'
    $backupCheckpoint = Get-ChildItem -LiteralPath (Join-Path $expRoot 'train\weights\backup') -Filter '*.pt' -ErrorAction SilentlyContinue |
        Select-Object -First 1

    $queue.status = 'running'
    $queue.current_experiment = $experiment
    $queue.updated_at = (Get-Date).ToString('o')
    $queue | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $queueStatusPath -Encoding UTF8

    $trainPassed = $false
    if (Test-Path -LiteralPath $trainSummary) {
        $trainPassed = ((Get-Content -LiteralPath $trainSummary -Raw | ConvertFrom-Json).status -eq 'passed')
    }
    if (-not $trainPassed) {
        if ((Test-Path -LiteralPath $lastCheckpoint) -or $null -ne $backupCheckpoint) {
            & $python $runner resume --experiment $experiment 2>&1 |
                Tee-Object -FilePath (Join-Path $logRoot "$experiment`_resume.log")
            if ($LASTEXITCODE -ne 0) {
                throw "$experiment resume failed with exit code $LASTEXITCODE"
            }
        }
        else {
            & $python $runner train --experiment $experiment 2>&1 |
                Tee-Object -FilePath (Join-Path $logRoot "$experiment`_train.log")
            if ($LASTEXITCODE -ne 0) {
                throw "$experiment training failed with exit code $LASTEXITCODE"
            }
        }
    }

    if (-not (Test-Path -LiteralPath $evalSummary)) {
        & $python $runner evaluate --experiment $experiment 2>&1 |
            Tee-Object -FilePath (Join-Path $logRoot "$experiment`_test.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$experiment evaluation failed with exit code $LASTEXITCODE"
        }
    }

    if ($queue.completed_experiments -notcontains $experiment) {
        $queue.completed_experiments += $experiment
    }
}

$queue.status = 'passed'
$queue.current_experiment = $null
$queue.completed_count = @($queue.completed_experiments).Count
$queue.updated_at = (Get-Date).ToString('o')
$queue | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $queueStatusPath -Encoding UTF8
