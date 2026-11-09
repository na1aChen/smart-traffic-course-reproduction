# TT100K 数据目录

| 路径 | 用途 |
| --- | --- |
| `raw/` | 官方原始压缩包与校验信息 |
| `extracted/` | 官方压缩包安全解压结果 |
| `yolo/` | 固定 45 类的 YOLO 数据布局 |

当前数据包正在从清华官方页面下载：

`https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip`

下载完成后，阶段 1 还必须完成：

1. 文件大小与 SHA-256。
2. ZIP 完整性检查。
3. 官方 train/test ID 计数。
4. 45 类每类图像数和实例数。
5. train/val/test 固定划分与 SHA-256。
6. 图像和标签一一对应检查。

在这些检查完成前，不得开始基线训练。

