# BounceNet 训练指南

**Training Guide for BounceNet Event Classifier**

本文档详细介绍如何使用 `train_bouncenet.py` 训练 BounceNet 事件分类网络。

---

## 📋 目录

- [前置要求](#前置要求)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [训练流程详解](#训练流程详解)
- [输出文件](#输出文件)
- [使用训练好的模型](#使用训练好的模型)
- [训练技巧](#训练技巧)
- [常见问题](#常见问题)

---

## 前置要求

### 环境依赖

```bash
pip install torch numpy pandas matplotlib opencv-python
```

### 硬件要求

| 模式 | 最低显存 | 推荐显存 | CPU 训练 |
|------|---------|---------|---------|
| `trajectory_only` | 2 GB | 4 GB | ✅ 可行 |
| `visual_only` | 4 GB | 8 GB | ⚠️ 较慢 |
| `fusion` | 6 GB | 8 GB | ⚠️ 较慢 |

---

## 数据准备

### 1. 使用标注工具创建训练数据

```bash
# 标注单个文件
python labeling_launcher.py --csv data/train/match1/csv/1_01_01_ball.csv

# 批量标注整个目录
python labeling_launcher.py --match_dir data/train/match1
```

### 2. 标注文件格式

标注工具会自动生成 JSON 文件，格式如下：

```json
{
  "csv_path": "data/train/match1/csv/1_01_01_ball.csv",
  "video_path": "data/train/match1/video/1_01_01.mp4",
  "labeled_at": "2026-01-21T15:30:00",
  "events": [
    {
      "frame": 45,
      "event_type": "hit",
      "confirmed": true,
      "x": 256.5,
      "y": 180.2,
      "rule": "vy_reversal",
      "original_confidence": 0.8
    },
    {
      "frame": 120,
      "event_type": "landing",
      "confirmed": true,
      "x": 320.1,
      "y": 250.8,
      "rule": "visibility_drop",
      "original_confidence": 0.85
    }
  ],
  "statistics": {
    "total_events": 15,
    "confirmed": 12,
    "deleted": 3,
    "by_type": {
      "landing": 4,
      "hit": 8
    }
  }
}
```

### 3. 推荐目录结构

```
labels/                          # --label_dir 指向这里
├── match1/
│   ├── 1_01_01_ball_labels.json
│   ├── 1_01_02_ball_labels.json
│   └── ...
├── match2/
│   ├── 1_02_01_ball_labels.json
│   └── ...
└── match3/
    └── ...

data/                           # CSV 和视频文件
├── train/
│   ├── match1/
│   │   ├── csv/
│   │   │   ├── 1_01_01_ball.csv
│   │   │   └── ...
│   │   └── video/
│   │       ├── 1_01_01.mp4
│   │       └── ...
│   └── ...
```

### 4. 数据量建议

| 数据量 | 预期效果 |
|--------|---------|
| < 50 事件 | 可能过拟合，仅用于测试 |
| 100-300 事件 | 基本可用 |
| 300-500 事件 | 良好效果 |
| 500+ 事件 | 推荐，效果稳定 |

---

## 快速开始

### 最简命令

```bash
python train_bouncenet.py --label_dir labels/
```

### 推荐配置

```bash
python train_bouncenet.py \
    --label_dir labels/ \
    --mode trajectory_only \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --early_stopping 15 \
    --save_dir ckpts/bouncenet
```

### 使用 GPU

```bash
# 自动检测 CUDA
python train_bouncenet.py --label_dir labels/ --device cuda

# 强制使用 CPU
python train_bouncenet.py --label_dir labels/ --device cpu
```

---

## 命令行参数

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--label_dir` | str | **必填** | 标注 JSON 文件目录 |
| `--csv_dir` | str | None | CSV 文件目录（可选，通常自动查找） |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `trajectory_only` | 特征模式：`trajectory_only`, `visual_only`, `fusion` |
| `--window_size` | int | 5 | 特征窗口大小（单侧），总序列长度 = 2×window + 1 |
| `--traj_hidden_dim` | int | 64 | 轨迹编码器 LSTM 隐藏维度 |
| `--traj_feature_dim` | int | 64 | 轨迹特征输出维度 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch_size` | int | 32 | 批次大小 |
| `--epochs` | int | 100 | 最大训练轮数 |
| `--lr` | float | 1e-3 | 初始学习率 |
| `--weight_decay` | float | 1e-4 | AdamW 权重衰减 |
| `--early_stopping` | int | 15 | 验证集无提升时的早停轮数 |
| `--val_split` | float | 0.2 | 验证集比例 (0-1) |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_dir` | str | `ckpts/bouncenet` | 模型保存目录 |
| `--device` | str | `cuda` | 计算设备：`cuda` 或 `cpu` |
| `--num_workers` | int | 4 | 数据加载线程数 |
| `--seed` | int | 42 | 随机种子（确保可复现） |

---

## 训练流程详解

### 1. 数据加载

```
1. 扫描 --label_dir 下所有 JSON 文件
2. 为每个 JSON 查找对应的 CSV 轨迹文件
3. 加载已确认的事件作为正样本
4. 自动生成负样本（非事件帧）
5. 按 --val_split 划分训练集/验证集
```

### 2. 样本生成

对于每个标注事件：

```python
# 提取以事件帧为中心的特征窗口
window_size = 5  # 默认
seq_len = 2 * window_size + 1  # = 11 帧

# 轨迹特征 (11, 5)
# 每帧包含: [x_norm, y_norm, vx_norm, vy_norm, visibility]
```

### 3. 训练过程

```
每个 Epoch:
├── 训练阶段
│   ├── 前向传播
│   ├── 计算 Focal Loss + Confidence Loss
│   ├── 反向传播
│   └── 梯度裁剪 (max_norm=1.0)
├── 验证阶段
│   ├── 计算验证集 Loss 和 Accuracy
│   └── 计算每类准确率
├── 学习率调度
│   └── 如果 val_loss 不下降，降低学习率
└── 检查点保存
    └── 如果 val_acc 提升，保存 best.pt
```

### 4. 早停机制

```
如果连续 --early_stopping 个 epoch 验证集准确率不提升：
  → 停止训练
  → 保存 last.pt
  → 保存训练历史
```

---

## 输出文件

训练完成后，`--save_dir` 目录下会生成：

```
ckpts/bouncenet/
├── best.pt          # 验证集最佳模型
├── last.pt          # 最后一轮模型
└── history.json     # 训练历史
```

### best.pt / last.pt 内容

```python
{
    'epoch': 85,                    # 保存时的轮数
    'model_state_dict': {...},      # 模型权重
    'optimizer_state_dict': {...},  # 优化器状态
    'val_acc': 0.9312,              # 验证集准确率
    'config': {
        'mode': 'trajectory_only',
        'num_classes': 3,
        'traj_seq_len': 11
    }
}
```

### history.json 内容

```json
{
  "train_loss": [0.85, 0.65, 0.45, ...],
  "train_acc": [0.65, 0.78, 0.85, ...],
  "val_loss": [0.71, 0.55, 0.40, ...],
  "val_acc": [0.72, 0.82, 0.88, ...]
}
```

---

## 使用训练好的模型

### 方法 1: 通过 BounceDetector

```python
from bounce_detection import BounceDetector

# 加载带 BounceNet 的检测器
detector = BounceDetector(bouncenet_ckpt='ckpts/bouncenet/best.pt')

# 检测事件（自动过滤误检）
events = detector.detect_from_csv('trajectory.csv')
```

### 方法 2: 直接使用 BounceNetPredictor

```python
from bounce_detection import BounceNetPredictor

# 加载预测器
predictor = BounceNetPredictor.from_checkpoint(
    'ckpts/bouncenet/best.pt',
    device='cuda'
)

# 对候选事件分类
classified = predictor.classify_candidates(x, y, visibility, candidates)
```

### 方法 3: 手动加载模型

```python
import torch
from bounce_detection import BounceNet

# 加载检查点
ckpt = torch.load('ckpts/bouncenet/best.pt')

# 创建模型
model = BounceNet(
    mode=ckpt['config']['mode'],
    traj_seq_len=ckpt['config']['traj_seq_len']
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

---

## 训练技巧

### 1. 数据不足时

```bash
# 使用更小的模型
python train_bouncenet.py \
    --label_dir labels/ \
    --traj_hidden_dim 32 \
    --traj_feature_dim 32 \
    --window_size 3
```

### 2. 过拟合时

```bash
# 增加正则化
python train_bouncenet.py \
    --label_dir labels/ \
    --weight_decay 1e-3 \
    --batch_size 16
```

### 3. 欠拟合时

```bash
# 增加模型容量
python train_bouncenet.py \
    --label_dir labels/ \
    --traj_hidden_dim 128 \
    --traj_feature_dim 128 \
    --lr 5e-4 \
    --epochs 200
```

### 4. 类别不平衡

训练脚本默认使用 Focal Loss 处理类别不平衡，无需额外调整。

### 5. 可视化训练曲线

```python
import json
import matplotlib.pyplot as plt

# 加载历史
with open('ckpts/bouncenet/history.json') as f:
    history = json.load(f)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss Curve')

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Accuracy Curve')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

---

## 常见问题

### Q1: "No labeled data found!"

**原因**: 找不到标注文件或对应的 CSV 文件

**解决方案**:
1. 检查 `--label_dir` 路径是否正确
2. 确保 JSON 文件命名格式为 `*_labels.json`
3. 确保对应的 CSV 文件存在

### Q2: "CUDA out of memory"

**解决方案**:
```bash
# 减小批次大小
python train_bouncenet.py --label_dir labels/ --batch_size 16

# 或使用 CPU
python train_bouncenet.py --label_dir labels/ --device cpu
```

### Q3: 验证集准确率不提升

**可能原因**:
1. 数据量不足
2. 学习率过大或过小
3. 标注数据质量问题

**解决方案**:
```bash
# 尝试不同学习率
python train_bouncenet.py --label_dir labels/ --lr 5e-4

# 或增加早停耐心值
python train_bouncenet.py --label_dir labels/ --early_stopping 20
```

### Q4: 训练速度慢

**解决方案**:
```bash
# 增加数据加载线程
python train_bouncenet.py --label_dir labels/ --num_workers 8

# 使用仅轨迹模式（最快）
python train_bouncenet.py --label_dir labels/ --mode trajectory_only
```

### Q5: 如何继续训练？

目前版本不支持断点续训，但可以：
1. 加载 `last.pt` 的权重
2. 手动修改代码实现

---

## 完整示例

### 示例 1: 标准训练流程

```bash
# 1. 准备数据（使用标注工具）
python labeling_launcher.py --match_dir data/train/match1
python labeling_launcher.py --match_dir data/train/match2

# 2. 训练模型
python train_bouncenet.py \
    --label_dir labels/ \
    --mode trajectory_only \
    --epochs 100 \
    --batch_size 32 \
    --save_dir ckpts/bouncenet_v1

# 3. 测试模型
python test_bouncenet.py
```

### 示例 2: 融合模式训练

```bash
python train_bouncenet.py \
    --label_dir labels/ \
    --mode fusion \
    --epochs 150 \
    --batch_size 16 \
    --lr 5e-4 \
    --window_size 5 \
    --save_dir ckpts/bouncenet_fusion
```

### 示例 3: 轻量级模型

```bash
python train_bouncenet.py \
    --label_dir labels/ \
    --mode trajectory_only \
    --traj_hidden_dim 32 \
    --traj_feature_dim 32 \
    --window_size 3 \
    --save_dir ckpts/bouncenet_lite
```

---

## 训练输出示例

```
Looking for label files...
Found 25 labeled files

Creating dataloaders...
Loaded 342 samples
Class distribution: {'landing': 98, 'hit': 156, 'none': 88}
Train samples: 273
Val samples: 69

Creating model...
Model parameters: 196,420

Starting training...
============================================================
Epoch 1/100
  Train Loss: 0.8234, Acc: 0.6520
  Val Loss: 0.7012, Acc: 0.7101
  Class Acc: {'none': 0.65, 'landing': 0.72, 'hit': 0.74}
  -> New best model saved!

Epoch 2/100
  Train Loss: 0.5678, Acc: 0.7823
  Val Loss: 0.5234, Acc: 0.8116
  Class Acc: {'none': 0.78, 'landing': 0.82, 'hit': 0.83}
  -> New best model saved!

...

Epoch 85/100
  Train Loss: 0.1523, Acc: 0.9456
  Val Loss: 0.2234, Acc: 0.9312
  Class Acc: {'none': 0.90, 'landing': 0.94, 'hit': 0.95}
  -> New best model saved!

Early stopping at epoch 100
============================================================
Training complete! Best val acc: 0.9312
Model saved to: ckpts/bouncenet
```

---

## 相关文档

- [模块总览](bounce_detection/README.md)
- [Phase 1 规则检测](bounce_detection/README_phase1_rules.md)
- [Phase 2 BounceNet](bounce_detection/README_phase2_bouncenet.md)
- [标注工具](bounce_detection/labeling/README_labeling_tool.md)
