# Bounce Detection Module for Badminton
# 羽毛球落点检测模块

from .kinematics import KinematicsCalculator
from .candidate_generator import BounceCandidateGenerator
from .detector import BounceDetector, visualize_candidates
from .bouncenet import (
    BounceNet, 
    BounceNetLoss, 
    BounceNetPredictor,
    EVENT_LABELS, 
    LABEL_TO_EVENT,
    create_bouncenet
)
from .visual_features import VisualFeatureExtractor, PatchEncoder, TemporalPatchEncoder

# ==================== 统一的事件类型定义 ====================
# 用于整个项目的事件类型常量，避免在各模块中重复定义

EVENT_TYPES = ['landing', 'hit', 'out_of_frame', 'other']
"""
事件类型列表:
- landing: 落地点（球落在场地上）
- hit: 击球点（球拍击打球的位置）
- out_of_frame: 出画（球离开视野）
- other: 其他事件
"""

# 主规则及其基础置信度
PRIMARY_RULES = {
    'vy_reversal': 0.85,        # vy 方向反转 → hit
    'vx_reversal': 0.80,        # vx 方向反转 → hit
    'acceleration_peak': 0.75,  # 加速度峰值 → hit
    'visibility_drop': 0.85,    # 可见性骤降（非边缘）→ hit
    'speed_drop': 0.75,         # 速度骤降 → landing
    'visibility_drop_edge': 0.50, # 边缘可见性骤降 → out_of_frame
    'trajectory_end': 0.70,     # 轨迹结束 → landing
}
"""主规则：决定事件类型，互斥关系"""

# 辅助规则及其置信度提升值
AUXILIARY_RULES = {
    'y_local_max': 0.05,        # Y坐标局部极值
    'speed_local_max': 0.05,    # 速度局部极值
}
"""辅助规则：提供额外证据，可叠加提升置信度（最高 0.95）"""

__all__ = [
    # Phase 1: Rule-based detection
    'KinematicsCalculator', 
    'BounceCandidateGenerator',
    'BounceDetector',
    'visualize_candidates',
    
    # Phase 2: BounceNet classification
    'BounceNet',
    'BounceNetLoss',
    'BounceNetPredictor',
    'EVENT_LABELS',
    'LABEL_TO_EVENT',
    'create_bouncenet',
    
    # Visual features
    'VisualFeatureExtractor',
    'PatchEncoder',
    'TemporalPatchEncoder',
    
    # Event type constants
    'EVENT_TYPES',
    'PRIMARY_RULES',
    'AUXILIARY_RULES',
]
