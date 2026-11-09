$ErrorActionPreference = "Stop"

$EnvRoot = $PSScriptRoot
$ProjectRoot = Split-Path $EnvRoot -Parent
$CodeRoot = Split-Path $ProjectRoot -Parent
$SrcRoot = Join-Path $ProjectRoot "源码\yolov10"
$VenvRoot = Join-Path $EnvRoot "venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"
$WheelRoot = Join-Path $EnvRoot "wheels"
$EvidenceRoot = Join-Path $EnvRoot "证据"

New-Item -ItemType Directory -Force -Path $WheelRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    py -3.10 -m venv $VenvRoot
}

function Ensure-Wheel {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$ExpectedSize,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        curl.exe -L --fail --retry 20 --retry-delay 5 --retry-all-errors -C - -o $Path $Url
    }

    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $ExpectedSize) {
        throw "Wheel size mismatch: $Path size=$($item.Length), expected=$ExpectedSize"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
    if ($actual -ne $ExpectedSha256.ToUpperInvariant()) {
        throw "Wheel SHA-256 mismatch: $Path actual=$actual, expected=$ExpectedSha256"
    }
}

$TorchUrl = "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl"
$TorchPath = Join-Path $WheelRoot "torch-2.0.1+cu118-cp310-cp310-win_amd64.whl"
Ensure-Wheel `
    -Url $TorchUrl `
    -Path $TorchPath `
    -ExpectedSize 2619146901 `
    -ExpectedSha256 "F58D75619BC96E4322343C030B893613701CAA2D6DB8017155DA226C14171335"

$TorchvisionUrl = "https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-win_amd64.whl"
$TorchvisionPath = Join-Path $WheelRoot "torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl"
Ensure-Wheel `
    -Url $TorchvisionUrl `
    -Path $TorchvisionPath `
    -ExpectedSize 4947556 `
    -ExpectedSha256 "BFD2435D681418BEA8DACDE2B2CB6E5DD40A0E0243D3631E2B71C10BF9831F39"

& $Python -m pip install --upgrade pip
& $Python -m pip install $TorchPath
& $Python -m pip install "numpy==1.26.4"
& $Python -m pip install $TorchvisionPath --no-deps
& $Python -m pip install -r (Join-Path $SrcRoot "requirements.txt")

& $Python -m pip check | Tee-Object -FilePath (Join-Path $EvidenceRoot "setup_pip_check.txt")
& $Python -m pip freeze | Set-Content -LiteralPath (Join-Path $EvidenceRoot "setup_pip_freeze.txt") -Encoding utf8

$SmokeCode = @"
import platform
import torch
import torchvision

print("platform=" + platform.platform())
print("torch=" + torch.__version__)
print("torchvision=" + torchvision.__version__)
print("cuda_build=" + str(torch.version.cuda))
print("cuda_available=" + str(torch.cuda.is_available()))
print("device_name=" + torch.cuda.get_device_name(0))
x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("gpu_matmul=" + str(tuple(y.shape)))
"@

& $Python -c $SmokeCode | Tee-Object -FilePath (Join-Path $EvidenceRoot "setup_cuda_smoke_test.txt")

