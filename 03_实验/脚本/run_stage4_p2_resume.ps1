$ErrorActionPreference = 'Stop'

$projectRoot = '<PROJECT_ROOT>'
$python = Join-Path $projectRoot '02_代码\环境\venv\Scripts\python.exe'
$runner = Join-Path $projectRoot '02_代码\阶段4\stage4_runner.py'
$logRoot = Join-Path $projectRoot '03_实验\logs\P2-001'
$evidenceRoot = Join-Path $projectRoot '01_立项\阶段4_证据'

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null

$startedAt = (Get-Date).ToString('o')
$resumeExit = $null
$evalExit = $null
$status = 'failed'
$errorText = $null

try {
    & $python $runner resume --device 0 --workers 0 --seed 2026 2>&1 |
        Tee-Object -FilePath (Join-Path $logRoot 'resume_stdout.log')

    $resumeExit = $LASTEXITCODE
    if ($resumeExit -ne 0) {
        throw "P2 resume failed with exit code $resumeExit"
    }

    & $python $runner evaluate `
        --batch 8 `
        --imgsz 640 `
        --device 0 `
        --workers 0 `
        --seed 2026 2>&1 |
        Tee-Object -FilePath (Join-Path $logRoot 'test_stdout.log')

    $evalExit = $LASTEXITCODE
    if ($evalExit -ne 0) {
        throw "P2 test evaluation failed with exit code $evalExit"
    }

    $status = 'passed'
}
catch {
    $errorText = $_.Exception.Message
}
finally {
    $payload = [ordered]@{
        experiment_id = 'P2-001'
        started_at = $startedAt
        finished_at = (Get-Date).ToString('o')
        status = $status
        resume_exit_code = $resumeExit
        evaluation_exit_code = $evalExit
        training_summary = (Join-Path $evidenceRoot 'stage4_training_summary.json')
        evaluation_summary = (Join-Path $evidenceRoot 'stage4_test_evaluation.json')
        error = $errorText
    }
    $payload |
        ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $logRoot 'resume_wrapper_status.json') -Encoding UTF8
}
