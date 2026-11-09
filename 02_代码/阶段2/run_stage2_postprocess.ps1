param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,

    [string]$ProjectRoot = "<PROJECT_ROOT>"
)

$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $ProjectRoot "02_代码\环境\venv\Scripts\python.exe"
$runnerPath = Join-Path $ProjectRoot "02_代码\阶段2\stage2_runner.py"
$auditPath = Join-Path $ProjectRoot "02_代码\阶段2\audit_stage2.py"
$summaryBuilderPath = Join-Path $ProjectRoot "02_代码\阶段2\write_stage2_summary.py"
$logsRoot = Join-Path $ProjectRoot "03_实验\logs"
$statusPath = Join-Path $logsRoot "BASELINE-001\postprocess_status.json"
$postprocessLog = Join-Path $logsRoot "BASELINE-001\postprocess.log"
$trainingSummaryPath = Join-Path $ProjectRoot "01_立项\阶段2_证据\stage2_training_summary.json"
$evaluationSummaryPath = Join-Path $ProjectRoot "01_立项\阶段2_证据\stage2_test_evaluation.json"
$auditJsonPath = Join-Path $ProjectRoot "01_立项\阶段2_证据\阶段2最终审计.json"
$testEvalDir = Join-Path $ProjectRoot "03_实验\baseline_yolov10n\test_eval"

New-Item -ItemType Directory -Path (Split-Path $statusPath) -Force | Out-Null

function Write-PostprocessStatus {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message,
        [Nullable[int]]$EvaluationExitCode = $null,
        [Nullable[int]]$AuditExitCode = $null
    )

    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
        status = $Status
        stage = $Stage
        message = $Message
        training_process_id = $TrainingProcessId
        training_summary = $trainingSummaryPath
        evaluation_summary = $evaluationSummaryPath
        audit_json = $auditJsonPath
        evaluation_exit_code = $EvaluationExitCode
        audit_exit_code = $AuditExitCode
        log = $postprocessLog
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    $payload | ConvertTo-Json -Depth 4 | Tee-Object -FilePath $postprocessLog -Append
}

Write-PostprocessStatus -Status "waiting" -Stage "training" -Message "等待正式训练进程结束。"

$deadline = (Get-Date).AddHours(8)
while (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
    if ((Get-Date) -gt $deadline) {
        Write-PostprocessStatus -Status "blocked" -Stage "training" -Message "训练超过 8 小时仍未结束，未启动测试评估。"
        exit 2
    }
    Start-Sleep -Seconds 30
}

$summaryDeadline = (Get-Date).AddMinutes(2)
while (-not (Test-Path -LiteralPath $trainingSummaryPath)) {
    if ((Get-Date) -gt $summaryDeadline) {
        Write-PostprocessStatus -Status "blocked" -Stage "training" -Message "训练进程已结束，但在 2 分钟内没有找到训练汇总。"
        exit 3
    }
    Start-Sleep -Seconds 5
}

$trainingSummary = Get-Content -Raw -LiteralPath $trainingSummaryPath | ConvertFrom-Json
if ($trainingSummary.status -ne "passed") {
    Write-PostprocessStatus -Status "blocked" -Stage "training" -Message "正式训练未通过；根据职责隔离规则，不启动测试集评估。"
    exit 4
}

if (Test-Path -LiteralPath $testEvalDir) {
    Write-PostprocessStatus -Status "blocked" -Stage "evaluation" -Message "测试评估目录已存在；为防止重复暴露测试集，不自动重跑。"
    exit 5
}

Write-PostprocessStatus -Status "running" -Stage "evaluation" -Message "训练已通过，开始唯一一次测试集评估。"
& $pythonPath $runnerPath evaluate --batch 8 --imgsz 640 --device 0 --workers 0 --seed 2026 *>> $postprocessLog
$evaluationExitCode = $LASTEXITCODE
if ($evaluationExitCode -ne 0) {
    Write-PostprocessStatus -Status "failed" -Stage "evaluation" -Message "测试评估命令失败；保留现场，不自动重试。" -EvaluationExitCode $evaluationExitCode
    exit 6
}

$evaluationSummary = Get-Content -Raw -LiteralPath $evaluationSummaryPath | ConvertFrom-Json
if ($evaluationSummary.status -ne "passed") {
    Write-PostprocessStatus -Status "failed" -Stage "evaluation" -Message "测试评估汇总未通过；保留现场，不自动重试。" -EvaluationExitCode $evaluationExitCode
    exit 7
}

Write-PostprocessStatus -Status "running" -Stage "audit" -Message "测试评估完成，开始阶段 2 最终审计。" -EvaluationExitCode $evaluationExitCode
& $pythonPath $auditPath *>> $postprocessLog
$auditExitCode = $LASTEXITCODE

Write-PostprocessStatus -Status "running" -Stage "summary" -Message "审计完成，生成阶段 2 中文 Markdown 总结。" -EvaluationExitCode $evaluationExitCode -AuditExitCode $auditExitCode
& $pythonPath $summaryBuilderPath *>> $postprocessLog
$summaryExitCode = $LASTEXITCODE

if ($auditExitCode -eq 0 -and $summaryExitCode -eq 0) {
    Write-PostprocessStatus -Status "passed" -Stage "complete" -Message "训练、唯一一次测试评估、最终审计和阶段总结均通过。" -EvaluationExitCode $evaluationExitCode -AuditExitCode $auditExitCode
    exit 0
}

Write-PostprocessStatus -Status "failed" -Stage "summary" -Message "最终审计或阶段总结生成失败；保留现场供人工核验。" -EvaluationExitCode $evaluationExitCode -AuditExitCode $auditExitCode
exit 8
