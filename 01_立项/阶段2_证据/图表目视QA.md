# 阶段 2 图表目视 QA

检查日期：2026-11-08。

## 检查范围

| 文件 | 检查结果 |
| --- | --- |
| `03_实验\baseline_yolov10n\train\results.png` | 通过 |
| `03_实验\baseline_yolov10n\train\confusion_matrix.png` | 通过 |
| `03_实验\baseline_yolov10n\train\val_batch0_pred.jpg` | 通过 |
| `03_实验\baseline_yolov10n\test_eval\PR_curve.png` | 通过 |

## 观察

- `results.png` 包含 40 轮训练损失、验证指标和 recall/mAP 面板；曲线不是空图，mAP50 与 mAP50-95 总体上升。
- 混淆矩阵包含 45 类与 background，主对角线可见，不是空白或渲染失败。
- 验证预测图包含 16 张道路场景，检测框和类别标签落在交通标志上，没有整体错位。
- 测试 PR 曲线正常显示 45 类细线、总体粗线及 `all classes 0.501 mAP@0.5` 图例。

## 结论

阶段 2 要求的训练曲线、混淆矩阵、验证预测图和测试 PR 曲线均已生成并可人工读取。类别间强弱、混淆关系和典型错误的具体诊断留在阶段 3，本文件只检查图表是否存在、可读且没有明显渲染故障。
