$ErrorActionPreference = 'Stop'

$projectRoot = '<PROJECT_ROOT>'
$python = Join-Path $projectRoot '02_代码\环境\venv\Scripts\python.exe'
$runner = Join-Path $projectRoot '02_代码\阶段5\stage5_runner.py'
$logRoot = Join-Path $projectRoot '03_实验\logs\STAGE5'
$queueStatus = Join-Path $logRoot 'queue_status.json'
$experiments = @(
    'BASELINE-002-S2027',
    'P2-002-S2027',
    'BASELINE-BUDGET48-S2026'
)

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$startedAt = (Get-Date).ToString('o')
$completed = @()

try {
    for ($index = 0; $index -lt $experiments.Count; $index++) {
        $experiment = $experiments[$index]
        $queue = [ordered]@{
            started_at = $startedAt
            updated_at = (Get-Date).ToString('o')
            status = 'running'
            current_experiment = $experiment
            completed_experiments = $completed
            total_experiments = $experiments.Count
            completed_count = $completed.Count
        }
        $queue | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath $queueStatus -Encoding UTF8

        & $python $runner train --experiment $experiment 2>&1 |
            Tee-Object -FilePath (Join-Path $logRoot "$experiment`_train.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$experiment training failed with exit code $LASTEXITCODE"
        }

        & $python $runner evaluate --experiment $experiment 2>&1 |
            Tee-Object -FilePath (Join-Path $logRoot "$experiment`_test.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$experiment evaluation failed with exit code $LASTEXITCODE"
        }

        $completed += $experiment
    }
}
catch {
    $queue = [ordered]@{
        started_at = $startedAt
        updated_at = (Get-Date).ToString('o')
        status = 'failed'
        current_experiment = $experiment
        completed_experiments = $completed
        total_experiments = $experiments.Count
        completed_count = $completed.Count
        error = $_.Exception.Message
    }
    $queue | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $queueStatus -Encoding UTF8
    throw
}

$finalQueue = [ordered]@{
    started_at = $startedAt
    updated_at = (Get-Date).ToString('o')
    status = 'passed'
    current_experiment = $null
    completed_experiments = $completed
    total_experiments = $experiments.Count
    completed_count = $completed.Count
}
$finalQueue | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $queueStatus -Encoding UTF8
