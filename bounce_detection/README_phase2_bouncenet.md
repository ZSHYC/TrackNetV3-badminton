# BounceNet 深度学习分类器 (Phase 2)

**BounceNet - Deep Learning Event Classifier for Badminton**

对 Phase 1 规则检测生成的候选事件进行分类，过滤误检并修正事件类型。

---

## 📋 目录

- [概述](#概述)
- [网络架构](#网络架构)
- [特征提取](#特征提取)
- [训练流程](#训练流程)
- [推理使用](#推理使用)
- [API 参考](#api-参考)
- [实验结果](#实验结果)

---

## 概述

### 动机

Phase 1 规则检测追求**高召回率**，会产生一些误检。BounceNet 的目标是：

1. **过滤误检**：识别并移除非事件候选
2. **修正分类**：更准确地区分 landing/hit 类型
3. **提供置信度**：为每个预测给出可靠性评估

### 三种工作模式

| 模式 | 输入特征 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| `trajectory_only` | 轨迹序列 | ~196K | 快速推理，无需视频 |
| `visual_only` | 视频 patch | ~350K | 视觉线索为主 |
| `fusion` | 轨迹 + 视频 | ~460K | 最高精度 |

### 分类目标

| 类别 | Label | 描述 |
|------|-------|------|
| `none` | 0 | 非事件（误检） |
| `landing` | 1 | 落地点 |
| `hit` | 2 | 击球点 |

---

## 网络架构

### 整体架构

```
输入特征
    │
    ├─────────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌─────────────────────┐           ┌─────────────────────┐
│  Trajectory Encoder │           │   Visual Encoder    │
│  ─────────────────  │           │   ───────────────   │
│  • 1D Conv (预处理)  │           │  • Patch Encoder    │
│  • Bi-LSTM (时序)    │           │  • Temporal Pooling │
│  • FC (投影)         │           │  • Attention        │
│        ↓             │           │         ↓           │
│   (N, 64)           │           │    (N, 128)         │
└─────────────────────┘           └─────────────────────┘
    │                                         │
    └───────────────┬─────────────────────────┘
                    │ Concatenate
                    ▼
            ┌───────────────┐
            │   Classifier  │
            │  ───────────  │
            │  • FC + BN    │
            │  • Dropout    │
            │  • FC → 3     │
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Output      │
            │  ───────────  │
            │  • logits (3) │
            │  • confidence │
            └───────────────┘
```

### TrajectoryEncoder

```python
class TrajectoryEncoder(nn.Module):
    """
    输入: (N, seq_len, 5)  # [x, y, v_x, v_y, visibility]
    输出: (N, feature_dim)
    
    结构:
    - 1D Conv: 预处理 + 局部特征
    - Bi-LSTM: 时序建模
    - FC: 特征投影
    """
```

**输入特征** (5维):
| 通道 | 特征 | 归一化 |
|------|------|--------|
| 0 | x 坐标 | / img_width |
| 1 | y 坐标 | / img_height |
| 2 | x 方向速度 | / 对角线长度 |
| 3 | y 方向速度 | / 对角线长度 |
| 4 | 可见性 | 0 或 1 |

### VisualEncoder

```python
class TemporalPatchEncoder(nn.Module):
    """
    输入: (N, seq_len, 3, 64, 64)  # 时序 RGB patches
    输出: (N, feature_dim)
    
    结构:
    - PatchEncoder: 每帧独立编码 (共享权重)
    - 1D Conv: 时序卷积
    - Attention Pooling: 加权聚合
    """
```

---

## 特征提取

### VisualFeatureExtractor

从视频帧中提取候选事件周围的视觉特征。

```python
from bounce_detection import VisualFeatureExtractor

extractor = VisualFeatureExtractor(
    patch_size=64,         # patch 大小
    temporal_window=5,     # 时序窗口（单侧）
    use_motion_history=True,
    img_size=(512, 288)
)

# 提取单个候选的特征
features = extractor.extract_features(
    frames,           # 视频帧列表
    x_coords,         # x 坐标序列
    y_coords,         # y 坐标序列
    center_frame_idx, # 候选帧索引
    visibility        # 可见性序列
)

# 返回:
# - center_patch: (64, 64, 3) 中心帧 patch
# - temporal_patches: (11, 64, 64, 3) 时序 patches
# - mhi_patch: (64, 64) 运动历史图
# - center_coords: (x_norm, y_norm) 归一化坐标
```

### 特征类型

| 特征 | 形状 | 描述 |
|------|------|------|
| `center_patch` | (H, W, 3) | 候选帧处以球为中心的 RGB patch |
| `temporal_patches` | (T, H, W, 3) | 连续 T 帧的 patch 序列 |
| `mhi_patch` | (H, W) | 运动历史图，编码近期运动 |

### 运动历史图 (MHI)

```
MHI 编码原理:
- 像素值表示该位置最近发生运动的时间
- 越亮 = 运动越近期
- 捕捉球的运动轨迹和速度信息

        时间 →
帧 t-4  ░░░░░░░░  (暗)
帧 t-3  ░░░░░░░░  
帧 t-2  ▒▒▒▒▒▒▒▒  (中等)
帧 t-1  ▓▓▓▓▓▓▓▓  
帧 t    ████████  (亮)
```

---

## 训练流程

### 数据准备

1. **使用标注工具生成训练数据**:
```bash
# 启动标注工具
python labeling_launcher.py --csv data/train/match1/csv/1_01_01_ball.csv

# 标注后会生成 JSON 文件:
# labels/match1/1_01_01_ball_labels.json
```

2. **JSON 标注格式**:
```json
{
  "csv_path": "data/train/match1/csv/1_01_01_ball.csv",
  "video_path": "data/train/match1/video/1_01_01.mp4",
  "events": [
    {
      "frame": 45,
      "event_type": "hit",
      "confirmed": true,
      "x": 256.5,
      "y": 180.2
    },
    {
      "frame": 120,
      "event_type": "landing",
      "confirmed": true,
      "x": 320.1,
      "y": 250.8
    }
  ]
}
```

### 训练命令

```bash
# 基础训练（仅轨迹特征）
python train_bouncenet.py \
    --label_dir labels/ \
    --mode trajectory_only \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --save_dir ckpts/bouncenet

# 融合模式（需要视频）
python train_bouncenet.py \
    --label_dir labels/ \
    --mode fusion \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--label_dir` | 必填 | 标注文件目录 |
| `--mode` | `trajectory_only` | 特征模式 |
| `--window_size` | 5 | 特征窗口（单侧） |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--early_stopping` | 15 | 早停轮数 |
| `--val_split` | 0.2 | 验证集比例 |

### 损失函数

```python
class BounceNetLoss:
    """
    组合损失:
    1. Focal Loss: 处理类别不平衡
       L_focal = -α(1-p)^γ * log(p)
    
    2. Confidence Loss: 鼓励正确预测有高置信度
       L_conf = BCE(confidence, is_correct)
    
    Total = L_focal + λ * L_conf
    """
```

### 数据增强

训练时自动应用:
- **水平翻转**: 50% 概率翻转 x 坐标和速度
- **噪声注入**: 坐标添加高斯噪声
- **时间偏移**: 轻微的时序平移

---

## 推理使用

### 方法 1: 通过 BounceDetector

```python
from bounce_detection import BounceDetector

# 加载带 BounceNet 的检测器
detector = BounceDetector(
    bouncenet_ckpt='ckpts/bouncenet/best.pt',
    use_visual_features=False  # 仅用轨迹特征
)

# 检测（自动使用 BounceNet 过滤）
events = detector.detect_from_csv('trajectory.csv')

# 或显式控制
events_rule_only = detector.detect(x, y, visibility, use_bouncenet=False)
events_filtered = detector.detect(x, y, visibility, use_bouncenet=True)
```

### 方法 2: 直接使用 BounceNetPredictor

```python
from bounce_detection import BounceNet, BounceNetPredictor

# 加载模型
predictor = BounceNetPredictor.from_checkpoint(
    'ckpts/bouncenet/best.pt',
    device='cuda'
)

# 准备候选（来自 Phase 1）
candidates = [
    {'frame': 45, 'event_type': 'hit', ...},
    {'frame': 120, 'event_type': 'landing', ...},
]

# 分类
classified = predictor.classify_candidates(
    x, y, visibility, candidates, frames=None
)

# 结果
for c in classified:
    print(f"Frame {c['frame']}: "
          f"original={c['event_type']}, "
          f"predicted={c['predicted_type']}, "
          f"confidence={c['prediction_confidence']:.2f}, "
          f"is_fp={c['is_false_positive']}")
```

### 输出格式

分类后的候选事件包含额外字段:

```python
{
    # 原有字段
    'frame': 45,
    'event_type': 'hit',      # 原始规则检测类型
    'rule': 'vy_reversal',
    'confidence': 0.80,
    
    # BounceNet 添加的字段
    'predicted_type': 'hit',   # BounceNet 预测类型
    'prediction_probs': {
        'none': 0.05,
        'landing': 0.15,
        'hit': 0.80
    },
    'prediction_confidence': 0.85,
    'is_false_positive': False  # 是否为误检
}
```

---

## API 参考

### BounceNet

```python
class BounceNet(nn.Module):
    def __init__(self,
                 mode: str = 'trajectory_only',
                 # 轨迹编码器
                 traj_input_dim: int = 5,
                 traj_hidden_dim: int = 64,
                 traj_feature_dim: int = 64,
                 traj_seq_len: int = 11,
                 # 视觉编码器
                 patch_size: int = 64,
                 visual_feature_dim: int = 128,
                 use_temporal_patches: bool = True,
                 # 分类器
                 num_classes: int = 3,
                 dropout: float = 0.3):
        ...
    
    def forward(self, traj_features, visual_features=None, 
                return_confidence=False):
        """
        Args:
            traj_features: (N, seq_len, 5) 轨迹特征
            visual_features: (N, seq_len, C, H, W) 视觉特征
            return_confidence: 是否返回置信度
        
        Returns:
            logits: (N, 3) 分类 logits
            confidence: (N, 1) 置信度（可选）
        """
    
    def predict(self, traj_features, visual_features=None, threshold=0.5):
        """
        Returns:
            {
                'labels': (N,) 预测标签,
                'probs': (N, 3) 类别概率,
                'confidence': (N,) 置信度
            }
        """
```

### VisualFeatureExtractor

```python
class VisualFeatureExtractor:
    def __init__(self,
                 patch_size: int = 64,
                 temporal_window: int = 5,
                 use_motion_history: bool = True,
                 img_size: Tuple[int, int] = (512, 288)):
        ...
    
    def extract_features(self, frames, x_coords, y_coords, 
                        center_frame_idx, visibility) -> Dict:
        """提取单个候选的特征"""
    
    def extract_batch_features(self, frames, x_coords, y_coords,
                              candidate_frames, visibility) -> Dict:
        """批量提取多个候选的特征"""
    
    def extract_patch(self, frame, x, y, patch_size=None) -> np.ndarray:
        """提取单个 patch"""
    
    def compute_motion_history_image(self, frames, center_frame_idx,
                                    threshold=25) -> np.ndarray:
        """计算运动历史图"""
```

### BounceNetPredictor

```python
class BounceNetPredictor:
    def __init__(self, model, feature_extractor=None,
                 kinematics_calculator=None, device='cuda'):
        ...
    
    @classmethod
    def from_checkpoint(cls, ckpt_path, mode='trajectory_only', device='cuda'):
        """从检查点加载"""
    
    def prepare_trajectory_features(self, x, y, visibility, 
                                   candidate_frames, window_size=5,
                                   img_size=(512, 288)) -> torch.Tensor:
        """准备轨迹特征"""
    
    def classify_candidates(self, x, y, visibility, candidates,
                           frames=None, threshold=0.5) -> List[Dict]:
        """对候选事件进行分类"""
```

---

## 实验结果

### 模型对比

| 模式 | 参数量 | 准确率 | 推理速度 |
|------|--------|--------|---------|
| trajectory_only | 196K | ~92% | 0.5ms/sample |
| visual_only | 350K | ~88% | 15ms/sample |
| fusion | 460K | ~94% | 16ms/sample |

*注：结果基于内部测试集，实际性能取决于训练数据质量*

### 各类性能

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| none | 0.85 | 0.80 | 0.82 |
| landing | 0.90 | 0.92 | 0.91 |
| hit | 0.93 | 0.95 | 0.94 |

### 训练曲线示例

```
Epoch 1/100
  Train Loss: 0.8542, Acc: 0.6521
  Val Loss: 0.7123, Acc: 0.7234
  -> New best model saved!

Epoch 50/100
  Train Loss: 0.2134, Acc: 0.9245
  Val Loss: 0.2567, Acc: 0.9123

Epoch 85/100
  Train Loss: 0.1523, Acc: 0.9456
  Val Loss: 0.2234, Acc: 0.9312
  -> New best model saved!

Early stopping at epoch 100
Training complete! Best val acc: 0.9312
```

---

## 常见问题

### Q: 应该选择哪种模式？

- **trajectory_only**: 推荐首选，速度快，无需视频
- **fusion**: 追求最高精度时使用
- **visual_only**: 轨迹数据噪声大时考虑

### Q: 训练数据量需要多少？

- 最低：~100 个标注事件
- 建议：500+ 标注事件
- 理想：1000+ 标注事件，覆盖多种比赛场景

### Q: 如何处理类别不平衡？

默认使用 Focal Loss，自动处理。也可以调整:
```python
criterion = BounceNetLoss(
    class_weights=[0.3, 1.0, 1.0],  # none 权重降低
    use_focal_loss=True,
    focal_gamma=2.0
)
```

### Q: 推理时 GPU 内存不足？

```python
# 使用 CPU
predictor = BounceNetPredictor.from_checkpoint(ckpt_path, device='cpu')

# 或减小批次
# 在 classify_candidates 中会自动处理
```

---

## 参考文献

- TrackNetV3: [GitHub](https://github.com/alenzenx/TracknetV3)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Motion History Image: Davis & Bobick, "The Representation and Recognition of Action Using Temporal Templates", CVPR 1997
