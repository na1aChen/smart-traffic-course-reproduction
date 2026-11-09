param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$ToolRoot = $PSScriptRoot
$CodeRoot = Split-Path $ToolRoot -Parent
$ProjectRoot = Split-Path $CodeRoot -Parent
$Aria2 = Join-Path $CodeRoot "环境\tools\aria2\aria2-1.37.0-win-64bit-build1\aria2c.exe"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "03_实验\datasets\TT100K\raw"
}

$Url = "https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip"
$ExpectedSize = 19152969603L
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$Archive = Join-Path $OutputDirectory "data.zip"
$EvidenceRoot = Join-Path $ProjectRoot "03_实验\datasets\TT100K\download_evidence"

if (-not (Test-Path -LiteralPath $Aria2)) {
    throw "aria2c not found: $Aria2"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

$activeWriters = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "aria2c.exe" -and
        (
            $_.CommandLine -like "*$Url*" -or
            (
                $_.CommandLine -like "*--out=data.zip*" -and
                $_.CommandLine -like "*$OutputDirectory*"
            )
        )
    }
if ($activeWriters) {
    throw "An aria2c process is already downloading the official TT100K archive. Wait for it to exit before running this script."
}

if (Test-Path -LiteralPath $Archive) {
    $currentSize = (Get-Item -LiteralPath $Archive).Length
    if ($currentSize -gt $ExpectedSize) {
        throw "Existing archive is larger than the official size: actual=$currentSize expected=$ExpectedSize"
    }
}

& $Aria2 `
    --continue=true `
    --max-connection-per-server=16 `
    --split=16 `
    --min-split-size=1M `
    --file-allocation=none `
    --auto-file-renaming=false `
    --allow-overwrite=false `
    --retry-wait=10 `
    --max-tries=0 `
    --summary-interval=60 `
    --console-log-level=notice `
    "--dir=$OutputDirectory" `
    "--out=data.zip" `
    $Url

if ($LASTEXITCODE -ne 0) {
    throw "aria2c failed with exit code $LASTEXITCODE"
}

$item = Get-Item -LiteralPath $Archive
if ($item.Length -ne $ExpectedSize) {
    throw "Archive size mismatch: actual=$($item.Length) expected=$ExpectedSize"
}

$metadataPath = Join-Path $EvidenceRoot "server_headers.txt"
Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing -TimeoutSec 60 |
    Select-Object StatusCode, Headers |
    Format-List |
    Out-File -LiteralPath $metadataPath -Encoding utf8

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToUpperInvariant()
$record = [ordered]@{
    recorded_at = (Get-Date).ToString("o")
    source_url = $Url
    archive = $Archive
    expected_size_bytes = $ExpectedSize
    actual_size_bytes = $item.Length
    sha256 = $hash
    checksum_scope = "Locally computed provenance lock after download; publisher checksum was not listed"
}
$record |
    ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $EvidenceRoot "download_record.json") -Encoding utf8

Write-Output "TT100K download complete and size-verified: $Archive"
Write-Output "SHA-256: $hash"
