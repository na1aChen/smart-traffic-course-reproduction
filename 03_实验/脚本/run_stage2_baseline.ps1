param(
    [int]$Epochs = 40,
    [int]$Batch = 8,
    [int]$Imgsz = 640
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "<PROJECT_ROOT>"
$Python = Join-Path $ProjectRoot "02_代码\环境\venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "02_代码\阶段2\stage2_runner.py"
$RunRoot = Join-Path $ProjectRoot "03_实验\baseline_yolov10n"
$LogRoot = Join-Path $ProjectRoot "03_实验\logs\BASELINE-001"
$StatusPath = Join-Path $LogRoot "run_status.json"
$TrainDirectory = Join-Path $RunRoot "train"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
if (Test-Path -LiteralPath $TrainDirectory) {
    throw "Refusing to overwrite existing training directory: $TrainDirectory"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "02_代码\源码\yolov10"
$env:PYTHONIOENCODING = "utf-8"
$StartedAt = (Get-Date).ToString("o")
$Status = [ordered]@{
    experiment_id = "BASELINE-001"
    started_at = $StartedAt
    status = "running"
    epochs = $Epochs
    batch = $Batch
    imgsz = $Imgsz
    workers = 0
}

try {
    & $Python $Runner train `
        --epochs $Epochs `
        --batch $Batch `
        --imgsz $Imgsz `
        --optimizer SGD `
        --patience 10 `
        --close-mosaic 5 `
        --save-period 5 `
        --device 0 `
        --workers 0 `
        --seed 2026 2>&1 | Tee-Object -FilePath (Join-Path $LogRoot "train_stdout.log")
    $TrainExit = $LASTEXITCODE
    if ($TrainExit -ne 0) {
        throw "Training runner failed with exit code $TrainExit"
    }

    & $Python $Runner evaluate `
        --batch 4 `
        --imgsz $Imgsz `
        --device 0 `
        --workers 0 `
        --seed 2026 2>&1 | Tee-Object -FilePath (Join-Path $LogRoot "evaluate_stdout.log")
    $EvalExit = $LASTEXITCODE
    if ($EvalExit -ne 0) {
        throw "Evaluation runner failed with exit code $EvalExit"
    }

    $Status.status = "passed"
}
catch {
    $Status.status = "failed"
    $Status.error = $_.Exception.Message
    throw
}
finally {
    $Status.finished_at = (Get-Date).ToString("o")
    $Status |
        ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $StatusPath -Encoding utf8
}

