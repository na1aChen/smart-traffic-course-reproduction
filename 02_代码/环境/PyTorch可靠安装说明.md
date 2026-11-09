# PyTorch 可靠安装说明

记录日期：2026-11-07。本文对应 YOLOv10 官方仓库锁定的 Python、Torch 和 Torchvision 版本。

## 一、为什么不能只写 `pip install torch`

YOLOv10 官方 `requirements.txt` 固定了：

```text
torch==2.0.1
torchvision==0.15.2
```

如果直接执行不带索引地址的 `pip install torch`，pip 可能在安装依赖时重新选择版本，导致训练代码、Torch 和 Torchvision 不匹配。正式环境采用“官方索引确认版本与哈希 → 本地 wheel 安装 → smoke test”的流程。

## 二、官方 wheel

### 2.1 Torch

| 字段 | 值 |
| --- | --- |
| 官方索引 | `https://download.pytorch.org/whl/cu118/torch/` |
| 文件 | `torch-2.0.1+cu118-cp310-cp310-win_amd64.whl` |
| 下载地址 | `https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl` |
| 大小 | 2,619,146,901 字节 |
| SHA-256 | `F58D75619BC96E4322343C030B893613701CAA2D6DB8017155DA226C14171335` |

### 2.2 Torchvision

| 字段 | 值 |
| --- | --- |
| 官方索引 | `https://download.pytorch.org/whl/cu118/torchvision/` |
| 文件 | `torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl` |
| 下载地址 | `https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-win_amd64.whl` |
| 大小 | 4,947,556 字节 |
| SHA-256 | `BFD2435D681418BEA8DACDE2B2CB6E5DD40A0E0243D3631E2B71C10BF9831F39` |

两个哈希均已在 PyTorch 官方 `cu118` wheel 索引页核对，并与本地文件一致。

## 三、自动重装

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File `
  ".\大作业2\项目\02_代码\环境\setup_gpu_env.ps1"
```

脚本会复用已存在的完整 wheel；若 wheel 缺失，则用 `curl -C -` 断点续传。安装前逐项核对精确大小和 SHA-256，校验失败立即停止。

## 四、手动下载与安装

如果希望完全手动执行，可按以下顺序操作。

### 4.1 创建独立环境

```powershell
py -3.10 -m venv `
  "<PROJECT_ROOT>\02_代码\环境\venv"
```

### 4.2 下载 wheel

```powershell
$wheelDir = "<PROJECT_ROOT>\02_代码\环境\wheels"
New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null

curl.exe -L --fail --retry 20 --retry-delay 5 --retry-all-errors -C - `
  -o "$wheelDir\torch-2.0.1+cu118-cp310-cp310-win_amd64.whl" `
  "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-win_amd64.whl"

curl.exe -L --fail --retry 20 --retry-delay 5 --retry-all-errors -C - `
  -o "$wheelDir\torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl" `
  "https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-win_amd64.whl"
```

### 4.3 先校验，再安装

```powershell
Get-Item `
  "$wheelDir\torch-2.0.1+cu118-cp310-cp310-win_amd64.whl", `
  "$wheelDir\torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl"

Get-FileHash -Algorithm SHA256 `
  "$wheelDir\torch-2.0.1+cu118-cp310-cp310-win_amd64.whl", `
  "$wheelDir\torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl"
```

只有大小和 SHA-256 均匹配后，才执行：

```powershell
$python = "<PROJECT_ROOT>\02_代码\环境\venv\Scripts\python.exe"

& $python -m pip install `
  "$wheelDir\torch-2.0.1+cu118-cp310-cp310-win_amd64.whl"
& $python -m pip install "numpy==1.26.4"
& $python -m pip install `
  "$wheelDir\torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl" `
  --no-deps
& $python -m pip install -r `
  "<PROJECT_ROOT>\02_代码\源码\yolov10\requirements.txt"
```

`torchvision --no-deps` 的作用是阻止 pip 重新解析并替换已经锁定的 Torch。

## 五、安装后验收

```powershell
& $python -c "import torch,torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
& $python -m pip check
```

本次成功结果：

```text
Python 3.10.6
torch 2.0.1+cu118
torchvision 0.15.2+cu118
CUDA build 11.8
cuda_available=True
GPU NVIDIA GeForce RTX 4060 Laptop GPU
pip check: No broken requirements found
```

还通过了：

1. CUDA 矩阵乘。
2. FP16 autocast、反向传播和 AdamW 参数更新。
3. 官方 YOLOv10-N `model.safetensors` 加载并迁移到 `cuda:0`。
4. 官方示例图端到端推理。

原始输出位于 `证据` 子目录。
