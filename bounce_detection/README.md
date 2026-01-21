# 羽毛球事件检测模块 (Phase 1)

**Badminton Event Detection Module - Rule-Based Approach**

基于运动学特征的羽毛球落地点（Landing）和击球点（Hit）检测模块，作为 TrackNetV3 的后处理扩展。

---

## 📋 目录

- [概述](#概述)
- [模块架构](#模块架构)
- [安装与依赖](#安装与依赖)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
  - [KinematicsCalculator](#1-kinematicscalculator-运动学计算器)
  - [BounceCandidateGenerator](#2-bouncecandidategenerator-候选事件生成器)
  - [BounceDetector](#3-bouncedetector-事件检测器)
- [检测规则详解](#检测规则详解)
- [参数配置](#参数配置)
- [输出格式](#输出格式)
- [可视化](#可视化)
- [测试与评估](#测试与评估)
- [设计决策](#设计决策)
- [后续规划 (Phase 2)](#后续规划-phase-2)

---

## 概述

### 背景

TrackNetV3 提供了逐帧的羽毛球位置检测 `(x, y, visibility)`，但无法直接判断球的事件类型（落地、击球等）。本模块通过分析轨迹的**运动学特征**，自动检测以下事件：

| 事件类型 | 标识 | 描述 |
|---------|------|------|
| **落地点** | `landing` | 球落地后停止或消失 |
| **击球点** | `hit` | 球被球拍击中，方向反转 |
| **出画面** | `out_of_frame` | 球飞出画面边缘 |

### 设计原则

1. **纯运动学分析**：不依赖画面位置判断落点，因为落点可能出现在场地任何位置
2. **无需额外平滑**：假设输入轨迹已由 InpaintNet 修复，无需重复预处理
3. **规则可解释**：每个检测结果都带有触发规则和特征值，便于调试
4. **Phase 2 兼容**：预留 BounceNet 接口，可无缝升级到机器学习方法

---

## 模块架构

```
bounce_detection/
├── __init__.py              # 模块入口
├── kinematics.py            # 运动学特征计算
├── candidate_generator.py   # 规则检测 + 候选生成
├── detector.py              # 主检测器 + 可视化
└── README.md                # 本文档
```

### 数据流

```
TrackNetV3 输出 (CSV/Dict)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  BounceDetector.detect()                                    │
│  ┌──────────────────┐    ┌───────────────────────────────┐  │
│  │ KinematicsCalc   │───▶│ BounceCandidateGenerator     │  │
│  │ ---------------  │    │ -------------------------    │  │
│  │ • v_x, v_y       │    │ • 规则1: visibility_drop     │  │
│  │ • a_x, a_y       │    │ • 规则2: speed_drop          │  │
│  │ • speed          │    │ • 规则3: vy_reversal         │  │
│  │ • direction      │    │ • 规则4: vx_reversal         │  │
│  │ • curvature      │    │ • 规则5: trajectory_end      │  │
│  └──────────────────┘    └───────────────────────────────┘  │
│                                     │                        │
│                                     ▼                        │
│                          ┌───────────────────┐               │
│                          │  候选合并 & 排序  │               │
│                          └───────────────────┘               │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  检测结果 List[Dict]
  • landing 落地点
  • hit 击球点  
  • out_of_frame 出画面
```

---

## 安装与依赖

### 依赖项

```txt
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0   # 仅用于可视化
```

### 安装

本模块已集成在 TrackNetV3 项目中，无需单独安装：

```python
from bounce_detection import BounceDetector, visualize_candidates
```

---

## 快速开始

### 基本用法

```python
from bounce_detection.detector import BounceDetector, visualize_candidates
import pandas as pd

# 初始化检测器
detector = BounceDetector()

# 从 CSV 文件检测
events = detector.detect_from_csv('data/test/match1/csv/1_05_02_ball.csv')

# 分类输出
for event in events:
    print(f"Frame {event['frame']:4d} | "
          f"Type: {event['event_type']:12s} | "
          f"Rule: {event['rule']}")
```

### 从 TrackNetV3 预测结果检测

```python
# 假设 pred_dict 是 TrackNetV3 的输出
pred_dict = {
    'Frame': [0, 1, 2, ...],
    'X': [100.5, 102.3, ...],
    'Y': [200.1, 198.7, ...],
    'Visibility': [1, 1, 1, ...]
}

events = detector.detect_from_prediction(pred_dict)
```

### 直接使用数组

```python
import numpy as np

x = np.array([100, 105, 110, 115, 110, 105])  # X 坐标
y = np.array([200, 195, 190, 200, 210, 220])  # Y 坐标
visibility = np.array([1, 1, 1, 1, 1, 1])     # 可见性

events = detector.detect(x, y, visibility)
```

---

## 核心组件

### 1. KinematicsCalculator (运动学计算器)

**文件**: `kinematics.py`

计算轨迹的运动学特征，作为事件检测的基础。

#### 计算特征

| 特征 | 符号 | 单位 | 描述 |
|-----|------|------|------|
| X速度 | `v_x` | pixels/frame | 使用 `np.gradient` 计算 |
| Y速度 | `v_y` | pixels/frame | 正值表示向下（图像坐标系） |
| X加速度 | `a_x` | pixels/frame² | 速度的一阶导数 |
| Y加速度 | `a_y` | pixels/frame² | 速度的一阶导数 |
| 速度大小 | `speed` | pixels/frame | $\sqrt{v_x^2 + v_y^2}$ |
| 运动方向 | `direction` | radians | $\arctan2(v_y, v_x)$ |
| 曲率 | `curvature` | 1/pixels | $\kappa = \frac{|v_x a_y - v_y a_x|}{(v_x^2 + v_y^2)^{3/2}}$ |

#### API

```python
class KinematicsCalculator:
    def __init__(self, fps: int = 30):
        """
        Args:
            fps: 视频帧率（用于可选的时间单位转换）
        """
    
    def compute(self, x, y, visibility) -> Dict[str, np.ndarray]:
        """计算完整轨迹的运动学特征"""
    
    def compute_at_frame(self, x, y, visibility, frame_idx, window_size=5) -> Dict[str, float]:
        """计算单帧的运动学特征（使用局部窗口）"""
    
    def get_trajectory_window(self, x, y, visibility, center_frame, 
                              window_size=5, normalize=True) -> np.ndarray:
        """提取轨迹特征窗口（用于 Phase 2 BounceNet）"""
```

---

### 2. BounceCandidateGenerator (候选事件生成器)

**文件**: `candidate_generator.py`

基于规则的事件检测核心类。

#### 检测策略

```
Landing (落地点) 检测策略:
├── visibility_drop: 运动中突然消失（场内）
├── speed_drop: 速度急剧下降 + 之后保持低速
└── trajectory_end: 回合结束时的最后可见帧

Hit (击球点) 检测策略:
├── vy_reversal: Y方向速度反转 + 速度保持较高
└── vx_reversal: X方向速度反转 + 速度保持较高

Out-of-Frame (出画面) 检测策略:
└── visibility_drop_edge: 在画面边缘消失
```

#### API

```python
class BounceCandidateGenerator:
    def __init__(self,
                 # 落地点检测参数
                 speed_drop_ratio: float = 0.3,
                 min_speed_before_landing: float = 8.0,
                 max_speed_after_landing: float = 5.0,
                 # 击球点检测参数
                 min_speed_at_hit: float = 5.0,
                 vy_reversal_threshold: float = 3.0,
                 # 通用参数
                 min_visible_before: int = 3,
                 merge_window: int = 3):
        """参数说明见下文"""
    
    def generate(self, x, y, visibility, kinematics=None, img_height=288) -> List[Dict]:
        """生成候选事件列表"""
    
    def generate_from_csv(self, csv_path, img_height=288) -> List[Dict]:
        """从 CSV 文件生成候选"""
```

---

### 3. BounceDetector (事件检测器)

**文件**: `detector.py`

整合运动学计算和候选生成的主类，提供简洁的 API。

#### API

```python
class BounceDetector:
    def __init__(self, fps=30, **kwargs):
        """初始化检测器（参数透传给 BounceCandidateGenerator）"""
    
    def detect(self, x, y, visibility, frames=None, img_height=None) -> List[Dict]:
        """检测事件"""
    
    def detect_from_csv(self, csv_path, img_height=None) -> List[Dict]:
        """从 CSV 文件检测"""
    
    def detect_from_prediction(self, pred_dict, img_height=None) -> List[Dict]:
        """从 TrackNetV3 输出检测"""
    
    def analyze_trajectory(self, x, y, visibility) -> Dict:
        """分析轨迹特性（调试用）"""
```

---

## 检测规则详解

### 规则 1: visibility_drop (可见性消失)

**触发条件**:
- 当前帧可见 (`visibility[t] == 1`)
- 下一帧不可见 (`visibility[t+1] == 0`)
- 之前有运动 (`recent_speed >= min_speed_before_landing`)

**分类逻辑**:
```python
if position_at_edge:
    event_type = 'out_of_frame'   # 边缘消失 → 出画面
    confidence = 0.50
else:
    event_type = 'landing'        # 场内消失 → 落地
    confidence = 0.85
```

**边缘判定**:
```python
is_edge = (x < 20 or x > img_width - 20 or 
           y < 20 or y > img_height - 20)
```

---

### 规则 2: speed_drop (速度骤降)

**触发条件**:
- 之前速度较高: `speed_before >= min_speed_before_landing` (默认 8.0)
- 速度急剧下降: `speed_current / speed_before < speed_drop_ratio` (默认 0.3)
- 之后保持低速: `speed_after <= max_speed_after_landing` (默认 5.0)

**物理意义**: 球落地后因摩擦/碰撞迅速减速至接近静止。

**输出**:
```python
{
    'event_type': 'landing',
    'rule': 'speed_drop',
    'confidence': 0.75,
    'features': {
        'speed_ratio': 0.15,
        'speed_before': 45.2,
        'speed_current': 6.8,
        'speed_after': 2.1
    }
}
```

---

### 规则 3: vy_reversal (Y速度反转)

**触发条件**:
- Y速度明显反转:
  ```python
  (vy_before > threshold and vy_after < -threshold) or
  (vy_before < -threshold and vy_after > threshold)
  ```
  其中 `threshold = vy_reversal_threshold` (默认 3.0)
- 反转后速度保持较高: `speed_after >= min_speed_at_hit` (默认 5.0)

**物理意义**: 击球使球的垂直运动方向反转（上升→下降 或 下降→上升）。

**输出**:
```python
{
    'event_type': 'hit',
    'rule': 'vy_reversal',
    'confidence': 0.80,
    'features': {
        'vy_before': 15.5,
        'vy_after': -12.3,
        'speed_after': 28.7
    }
}
```

---

### 规则 4: vx_reversal (X速度反转)

**触发条件**: 与 `vy_reversal` 类似，但检测水平方向的反转。

**物理意义**: 水平击球（如平抽）导致水平运动方向反转。

**优先级**: 低于 `vy_reversal`，避免重复检测同一帧。

**输出**:
```python
{
    'event_type': 'hit',
    'rule': 'vx_reversal',
    'confidence': 0.75,  # 略低于 vy_reversal
    'features': {
        'vx_before': 8.2,
        'vx_after': -15.6,
        'speed_after': 32.1
    }
}
```

---

### 规则 5: trajectory_end (轨迹结束)

**触发条件**:
- 是最后一个可见帧
- 结束前有运动: `recent_speed >= min_speed_before_landing * 0.5`
- 未被其他规则检测

**物理意义**: 回合结束时的落地点（视频切断或回合结束）。

**输出**:
```python
{
    'event_type': 'landing',
    'rule': 'trajectory_end',
    'confidence': 0.70,
    'features': {
        'speed': 12.3,
        'recent_avg_speed': 8.5
    }
}
```

---

## 参数配置

### 默认参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `speed_drop_ratio` | 0.3 | 速度下降到原来的 30% 以下视为骤降 |
| `min_speed_before_landing` | 8.0 | 落地前最小速度（过滤静止球） |
| `max_speed_after_landing` | 5.0 | 落地后最大速度（确认已停止） |
| `min_speed_at_hit` | 5.0 | 击球后最小速度（与落地区分） |
| `vy_reversal_threshold` | 3.0 | 速度反转检测阈值 |
| `min_visible_before` | 3 | 事件前最少可见帧数 |
| `merge_window` | 3 | 合并相邻候选的窗口大小 |

### 参数调优建议

```python
# 高召回率配置（宁多勿漏）
detector = BounceDetector(
    speed_drop_ratio=0.5,           # 放宽速度下降阈值
    min_speed_before_landing=5.0,   # 降低最小速度要求
    vy_reversal_threshold=2.0,      # 降低反转检测阈值
)

# 高精确率配置（减少误检）
detector = BounceDetector(
    speed_drop_ratio=0.2,           # 严格速度下降阈值
    min_speed_before_landing=12.0,  # 提高最小速度要求
    min_speed_at_hit=10.0,          # 提高击球后速度要求
)
```

---

## 输出格式

### 单个事件

```python
{
    'frame': 156,                    # 帧索引（0-based）
    'x': 423.5,                      # X 坐标
    'y': 287.2,                      # Y 坐标
    'event_type': 'hit',             # 事件类型: 'landing' | 'hit' | 'out_of_frame'
    'rule': 'vy_reversal',           # 触发规则
    'confidence': 0.80,              # 置信度 (0-1)
    'features': {                    # 附加特征（规则相关）
        'vy_before': 15.5,
        'vy_after': -12.3,
        'speed_after': 28.7
    }
}
```

### 事件类型说明

| event_type | 含义 | 典型规则 |
|------------|------|---------|
| `landing` | 球落地 | `visibility_drop`, `speed_drop`, `trajectory_end` |
| `hit` | 球被击中 | `vy_reversal`, `vx_reversal` |
| `out_of_frame` | 球出画面 | `visibility_drop_edge` |

---

## 可视化

### 使用 visualize_candidates

```python
from bounce_detection.detector import BounceDetector, visualize_candidates
import pandas as pd

# 加载数据
df = pd.read_csv('data/test/match1/csv/1_05_02_ball.csv')
x, y, vis = df['X'].values, df['Y'].values, df['Visibility'].values

# 检测
detector = BounceDetector()
events = detector.detect(x, y, vis)

# 可视化（保存到文件）
visualize_candidates(x, y, vis, events, 
                     save_path='analysis.png',
                     title='Event Detection: 1_05_02')

# 或直接显示
visualize_candidates(x, y, vis, events)
```

### 可视化输出

生成 2x2 子图：

1. **轨迹图 (X-Y)**：空间轨迹，事件点用不同颜色/符号标记
   - 🔴 红色星号 (*): 落地点
   - 🔵 蓝色圆圈 (○): 击球点
   - ⬜ 灰色叉号 (×): 出画面

2. **Y坐标-时间图**：Y坐标随帧变化，竖线标记事件

3. **速度-时间图**：速度大小随帧变化

4. **Y速度-时间图**：Y方向速度，填充区分上升/下降

---

## 测试与评估

### 运行测试脚本

```bash
# 测试单个文件
python test_bounce_detection.py --csv data/test/match1/csv/1_05_02_ball.csv --visualize

# 测试整个 match 目录
python test_bounce_detection.py --match_dir data/test/match1/csv --visualize
```

### 测试输出示例

```
============================================================
Testing: data/test/match1/csv/1_05_02_ball.csv
============================================================
Total frames: 517
Visible frames: 489 (94.6%)

=== Detected 1 LANDING candidate(s) ===
  [1] Frame  516 | Position (652, 698) | Rule: trajectory_end       | Confidence: 0.70

=== Detected 10 HIT candidate(s) ===
  [1] Frame   30 | Position (481, 385) | Rule: vy_reversal          | Confidence: 0.80
  [2] Frame   72 | Position (718, 436) | Rule: vy_reversal          | Confidence: 0.80
  ...

=== Detected 2 OUT_OF_FRAME candidate(s) ===
  [1] Frame   91 | Position (804, 0) | Rule: visibility_drop_edge | Confidence: 0.50
  [2] Frame  334 | Position (676, 0) | Rule: visibility_drop_edge | Confidence: 0.50
```

### Match1 测试集统计 (11 回合)

| 指标 | 数值 |
|-----|------|
| 总事件数 | 101 |
| 平均事件/回合 | 9.18 |
| 击球点 (HIT) | 75 |
| 落地点 (LANDING) | 7 |
| 出画面 | 19 |

---

## 设计决策

### 1. 为什么不使用画面位置过滤？

**原因**：羽毛球落点可能出现在场地任何位置（网前、中场、底线），无法简单通过 Y 坐标判断。

**替代方案**：使用速度变化（骤降 + 静止）判断落地。

### 2. 为什么不预处理轨迹（平滑/插值）？

**原因**：TrackNetV3 的 InpaintNet 已经完成轨迹修复，额外平滑可能破坏真实的速度突变特征。

### 3. 为什么使用 np.gradient 而非简单差分？

**原因**：`np.gradient` 使用中心差分，边界处理更平滑，减少端点噪声。

### 4. 候选合并策略

相邻帧可能触发多个规则，使用 `merge_window` 合并后保留：
1. 优先级最高的规则
2. 相同优先级下置信度最高的

---

## 后续规划 (Phase 2)

### BounceNet 分类网络

Phase 2 将引入深度学习方法，对规则生成的候选进行精筛：

```
Phase 1 候选
     │
     ▼
┌─────────────────────────────────────┐
│           BounceNet                 │
│  ┌──────────────┐ ┌──────────────┐  │
│  │ Trajectory   │ │   Visual     │  │
│  │ Encoder      │ │   Encoder    │  │
│  │ (LSTM/GRU)   │ │   (CNN)      │  │
│  └──────┬───────┘ └──────┬───────┘  │
│         │                │          │
│         └───────┬────────┘          │
│                 ▼                   │
│         ┌──────────────┐            │
│         │  Classifier  │            │
│         └──────────────┘            │
└─────────────────────────────────────┘
     │
     ▼
  精确分类: landing / hit / false_positive
```

### 预留接口

```python
# kinematics.py 中已实现
calc.get_trajectory_window(x, y, vis, frame, window_size=5, normalize=True)
# 返回 (11, 4) 数组: [x, y, v_x, v_y] × 11帧

# detector.py 中预留
detector = BounceDetector(
    bouncenet_ckpt='ckpts/BounceNet_best.pt',
    use_visual_features=True
)
```

### 数据标注工具

Phase 2 需要人工标注训练数据，将开发半自动标注工具：
- 基于 Phase 1 候选预填充
- 人工确认/修正/补充
- 导出为训练格式

---

## 许可证

本模块遵循 TrackNetV3 项目的 MIT 许可证。

---

## 更新日志

### v0.1.0 (2026-01-21)

- ✅ 实现 KinematicsCalculator 运动学计算
- ✅ 实现 BounceCandidateGenerator 规则检测
- ✅ 支持 landing / hit / out_of_frame 三类事件
- ✅ 添加可视化工具
- ✅ 完成 Match1 测试集验证
