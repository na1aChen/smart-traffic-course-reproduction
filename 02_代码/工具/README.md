# TT100K 数据工具

本目录中的脚本只负责可复现的数据准备，不训练模型，不改变官方源码。

## 下载

```powershell
powershell -ExecutionPolicy Bypass -File .\download_tt100k.ps1
```

下载器使用 aria2c 16 连接和断点续传，校验服务器 Content-Length、最终大小和 SHA-256。真实进度可由 `tt100k_download_progress.py` 从 `.aria2` 控制文件读取。

## 下载

```powershell
.\download_tt100k.ps1
```

- 固定使用清华 TT100K 官方页面的 `data.zip` 地址。
- 使用 aria2 多连接、无限重试和断点续传；重复运行会先检查是否已有写入进程。
- 官方页面未公布校验和，因此下载完成后核对服务端给出的精确字节数，再计算本地 SHA-256 作为来源锁版值。
- 下载记录写入 `03_实验\datasets\TT100K\download_evidence\download_record.json`。
- 若网络不稳定，脚本可再次运行；已有的 `.aria2` 控制文件会用于续传。
- 2026-11-07 首次负向测试发现，旧版守卫只匹配完整归档路径，未识别 aria2 的 `--dir/--out` 形式；该版本已被停止。现行脚本改为同时匹配官方 URL 与输出目录，并通过“已有下载进程时必须拒绝启动”的测试。

查询断点续传的真实进度：

```powershell
& "..\环境\venv\Scripts\python.exe" .\tt100k_download_progress.py `
  --archive "..\..\03_实验\datasets\TT100K\raw\data.zip"
```

aria2 多连接下载会提前把文件长度扩展到接近总量，不能把文件长度当作完成度。该脚本解析 `.aria2` 控制文件中的完成位图，输出已完成和剩余 GiB。

## 解压

```powershell
& "..\环境\venv\Scripts\python.exe" .\extract_tt100k.py `
  --archive "..\..\03_实验\datasets\TT100K\raw\data.zip" `
  --output-root "..\..\03_实验\datasets\TT100K\extracted"
```

- 先运行 ZIP 完整性检查。
- 拒绝绝对路径和 `..` 路径穿越。
- 已存在且大小一致的文件会跳过，支持中断后继续。
- `.part` 文件只在单文件写入完成后替换为正式文件。

## 转换为确定性 YOLO 布局

```powershell
& "..\环境\venv\Scripts\python.exe" .\prepare_tt100k.py `
  --data-root "..\..\03_实验\datasets\TT100K\extracted\data" `
  --output-root "..\..\03_实验\datasets\TT100K\yolo" `
  --archive "..\..\03_实验\datasets\TT100K\raw\data.zip" `
  --archive-sha256 "0D9F199A222417A849C108DAE8FE991DC0C9384B26FDC61953C798455F7774DA" `
  --val-fraction 0.10 `
  --seed 2026 `
  --link-mode hardlink
```

处理规则：

- 类别固定为官方教程的 45 类 `type45`，训练前不根据验证集或测试集结果调整。
- 官方 `train/ids.txt` 用于训练候选池，官方 `test/ids.txt` 只进入测试。
- 从官方训练池按 `sha256("2026:"+image_id)` 排序，前 10% 作为 validation。
- 原始测试集不参与类别选择、超参数选择或早停。
- 图片默认使用 NTFS 硬链接，避免复制约 18 GiB 图像。
- 每张图像生成一份 YOLO 标签文件；没有选中类别的图像生成空标签。
- 输出 `tt100k.yaml`、`dataset_manifest.json`、`dataset_statistics.csv`、`class_selection.json`。

## 训练前验证

```powershell
& "..\环境\venv\Scripts\python.exe" .\validate_tt100k_yolo.py `
  --dataset-root "..\..\03_实验\datasets\TT100K\yolo"
```

验证器检查图像/标签一一对应、split 无重复、标签字段和归一化坐标、类别 ID、统计 CSV 与原始标签计数、清单哈希。

## 可视化抽查

```powershell
& "..\环境\venv\Scripts\python.exe" .\visualize_tt100k_samples.py `
  --dataset-root "..\..\03_实验\datasets\TT100K\yolo" `
  --output "..\..\03_实验\datasets\TT100K\qa\train_samples.jpg" `
  --split train --count 8 --seed 2026
```

生成固定抽样的图像、类别和边界框叠加图，供训练前人工检查标签坐标。

## 测试

```powershell
& "..\环境\venv\Scripts\python.exe" .\test_prepare_tt100k.py
```

当前小样本测试已通过。

## 转换后严格验收

```powershell
& "..\环境\venv\Scripts\python.exe" .\validate_tt100k_yolo.py `
  --dataset-root "..\..\03_实验\datasets\TT100K\yolo"
```

检查项：

- 45 个类别 ID 连续且不少于 10 类。
- 每个划分的图像和标签一一对应。
- 不同划分的图像 ID 不重复。
- 每行标签均为五列，类别合法，坐标在 `[0, 1]` 内且宽高为正。
- `dataset_statistics.csv` 的声明计数与标签实际计数一致。
- 输出 split manifest 和标签树 SHA-256，作为训练前数据锁版值。
