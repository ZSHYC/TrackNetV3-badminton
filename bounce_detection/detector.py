"""
落点检测器主类
Bounce Detector - 整合所有模块

提供简单的接口进行落点检测
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from .kinematics import KinematicsCalculator
from .candidate_generator import BounceCandidateGenerator, RallyBounceDetector


class BounceDetector:
    """
    羽毛球事件检测器
    
    检测两种事件:
    - 落地点 (landing): 球落地后停止/消失
    - 击球点 (hit): 球被球拍击中
    
    Phase 1: 仅使用规则方法
    Phase 2+: 加入 BounceNet 分类网络
    """
    
    def __init__(self,
                 # 运动学参数
                 fps: int = 30,
                 # 落地点检测参数
                 speed_drop_ratio: float = 0.3,
                 min_speed_before_landing: float = 8.0,
                 max_speed_after_landing: float = 5.0,
                 # 击球点检测参数
                 min_speed_at_hit: float = 5.0,
                 vy_reversal_threshold: float = 3.0,
                 # 通用参数
                 min_visible_before: int = 3,
                 merge_window: int = 3,
                 # BounceNet 参数 (Phase 2)
                 bouncenet_ckpt: Optional[str] = None,
                 use_visual_features: bool = False):
        """
        初始化事件检测器
        """
        self.fps = fps
        self.img_size = (512, 288)  # 默认 TrackNet 输入尺寸
        
        # 初始化运动学计算器
        self.kinematics_calculator = KinematicsCalculator(fps=fps)
        
        # 初始化候选生成器
        self.candidate_generator = BounceCandidateGenerator(
            speed_drop_ratio=speed_drop_ratio,
            min_speed_before_landing=min_speed_before_landing,
            max_speed_after_landing=max_speed_after_landing,
            min_speed_at_hit=min_speed_at_hit,
            vy_reversal_threshold=vy_reversal_threshold,
            min_visible_before=min_visible_before,
            merge_window=merge_window
        )

        # Phase 2: BounceNet
        self.bouncenet = None
        self.bouncenet_predictor = None
        self.use_visual_features = use_visual_features
        if bouncenet_ckpt is not None:
            self._load_bouncenet(bouncenet_ckpt)
    
    def _load_bouncenet(self, ckpt_path: str):
        """加载 BounceNet 模型"""
        import torch
        from .bouncenet import BounceNet, BounceNetPredictor
        from .visual_features import VisualFeatureExtractor
        
        print(f"Loading BounceNet from {ckpt_path}...")
        
        # 加载检查点
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # 创建模型
        mode = config.get('mode', 'trajectory_only')
        self.bouncenet = BounceNet(
            mode=mode,
            traj_seq_len=config.get('traj_seq_len', 11),
            traj_hidden_dim=config.get('traj_hidden_dim', 64),
            traj_feature_dim=config.get('traj_feature_dim', 64),
            num_classes=config.get('num_classes', 3)
        )
        self.bouncenet.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建视觉特征提取器（如果需要）
        feature_extractor = None
        if self.use_visual_features and mode in ['visual_only', 'fusion']:
            feature_extractor = VisualFeatureExtractor(
                patch_size=config.get('patch_size', 64),
                temporal_window=config.get('traj_seq_len', 11) // 2,
                img_size=self.img_size
            )
        
        # 创建预测器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bouncenet_predictor = BounceNetPredictor(
            model=self.bouncenet,
            feature_extractor=feature_extractor,
            kinematics_calculator=self.kinematics_calculator,
            device=device
        )
        
        print(f"BounceNet loaded successfully (mode: {mode})")
    
    def detect(self,
               x: np.ndarray,
               y: np.ndarray,
               visibility: np.ndarray,
               frames: Optional[List[np.ndarray]] = None,
               img_height: Optional[int] = None,
               use_bouncenet: bool = True) -> List[Dict]:
        """
        检测事件（落地点 + 击球点）
        
        Args:
            x: x 坐标序列
            y: y 坐标序列  
            visibility: 可见性序列
            frames: 视频帧列表 (Phase 2, 用于视觉特征)
            img_height: 图像高度 (保留兼容性)
            use_bouncenet: 是否使用 BounceNet 进行精筛
        
        Returns:
            events: 检测到的事件列表（包含落地点和击球点）
        """
        if img_height is None:
            img_height = self.img_size[1]
        
        # Step 1: 计算运动学特征
        kinematics = self.kinematics_calculator.compute(x, y, visibility)
        
        # Step 2: 生成候选事件（规则方法）
        candidates = self.candidate_generator.generate(
            x, y, visibility, 
            kinematics=kinematics,
            img_height=img_height
        )
        
        # Phase 1: 直接返回候选
        if self.bouncenet_predictor is None or not use_bouncenet:
            return candidates
        
        # Phase 2: 使用 BounceNet 进行精筛
        classified = self.bouncenet_predictor.classify_candidates(
            x, y, visibility, candidates, frames
        )
        
        # 过滤掉预测为 'none' 的候选
        filtered = []
        for cand in classified:
            if not cand.get('is_false_positive', False):
                # 使用 BounceNet 的预测类型
                if cand.get('predicted_type', cand['event_type']) != 'none':
                    cand['event_type'] = cand.get('predicted_type', cand['event_type'])
                    filtered.append(cand)
        
        return filtered
    
    def detect_from_csv(self, 
                        csv_path: str,
                        img_height: Optional[int] = None) -> List[Dict]:
        """
        从 CSV 文件检测落点
        
        Args:
            csv_path: CSV 文件路径
            img_height: 图像高度
        
        Returns:
            bounces: 检测到的落点列表
        """
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='Frame').fillna(0)
        
        x = df['X'].values
        y = df['Y'].values
        visibility = df['Visibility'].values
        
        return self.detect(x, y, visibility, img_height=img_height)
    
    def detect_from_prediction(self,
                               pred_dict: Dict,
                               img_height: Optional[int] = None) -> List[Dict]:
        """
        从 TrackNetV3 预测结果检测落点
        
        Args:
            pred_dict: TrackNetV3 输出 {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
            img_height: 图像高度
        
        Returns:
            bounces: 检测到的落点列表
        """
        x = np.array(pred_dict['X'])
        y = np.array(pred_dict['Y'])
        visibility = np.array(pred_dict['Visibility'])
        
        return self.detect(x, y, visibility, img_height=img_height)
    
    def analyze_trajectory(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           visibility: np.ndarray) -> Dict:
        """
        分析轨迹特性（用于调试和可视化）
        
        Returns:
            analysis: 轨迹分析结果
        """
        kinematics = self.kinematics_calculator.compute(x, y, visibility)
        
        # 基本统计
        visible_mask = visibility == 1
        
        return {
            'total_frames': len(x),
            'visible_frames': int(np.sum(visibility)),
            'visibility_ratio': float(np.mean(visibility)),
            'y_range': (float(np.min(y[visible_mask])) if any(visible_mask) else 0,
                       float(np.max(y[visible_mask])) if any(visible_mask) else 0),
            'x_range': (float(np.min(x[visible_mask])) if any(visible_mask) else 0,
                       float(np.max(x[visible_mask])) if any(visible_mask) else 0),
            'avg_speed': float(np.mean(kinematics['speed'][visible_mask])) if any(visible_mask) else 0,
            'max_speed': float(np.max(kinematics['speed'][visible_mask])) if any(visible_mask) else 0,
            'kinematics': kinematics
        }


def visualize_candidates(x: np.ndarray,
                         y: np.ndarray,
                         visibility: np.ndarray,
                         candidates: List[Dict],
                         save_path: Optional[str] = None,
                         title: str = "Event Detection"):
    """
    可视化轨迹和检测到的事件
    
    Args:
        x, y, visibility: 轨迹数据
        candidates: 候选事件列表
        save_path: 保存路径 (如果为 None, 显示图像)
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 事件类型颜色
    event_colors = {
        'landing': 'red',
        'hit': 'blue',
        'out_of_frame': 'gray'
    }
    event_markers = {
        'landing': '*',
        'hit': 'o',
        'out_of_frame': 'x'
    }
    
    # 1. 轨迹图 (x-y)
    ax1 = axes[0, 0]
    visible_mask = visibility == 1
    ax1.scatter(x[visible_mask], y[visible_mask], c=np.arange(len(x))[visible_mask], 
                cmap='viridis', s=10, alpha=0.6, label='Trajectory')
    ax1.plot(x[visible_mask], y[visible_mask], 'g-', alpha=0.3, linewidth=0.5)
    
    # 标记事件
    for cand in candidates:
        event_type = cand.get('event_type', 'unknown')
        color = event_colors.get(event_type, 'black')
        marker = event_markers.get(event_type, 's')
        ax1.scatter(cand['x'], cand['y'], c=color, s=200, marker=marker, 
                   edgecolors='black', linewidths=1, zorder=5,
                   label=f"{event_type}: frame {cand['frame']}")
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trajectory with Detected Events')
    ax1.invert_yaxis()
    
    # 创建图例（避免重复）
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # 2. Y 坐标随时间变化
    ax2 = axes[0, 1]
    frames = np.arange(len(y))
    ax2.plot(frames[visible_mask], y[visible_mask], 'b-', linewidth=1, label='Y coordinate')
    ax2.scatter(frames[~visible_mask], y[~visible_mask], c='gray', s=5, alpha=0.3, label='Invisible')
    
    for cand in candidates:
        event_type = cand.get('event_type', 'unknown')
        color = event_colors.get(event_type, 'black')
        ax2.axvline(x=cand['frame'], color=color, linestyle='--', alpha=0.7, linewidth=1)
        ax2.scatter(cand['frame'], cand['y'], c=color, s=100, marker='*', zorder=5)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y')
    ax2.set_title('Y Coordinate over Time')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度随时间变化
    ax3 = axes[1, 0]
    calc = KinematicsCalculator()
    kinematics = calc.compute(x, y, visibility)
    speed = kinematics['speed']
    
    ax3.plot(frames, speed, 'g-', linewidth=1, label='Speed')
    ax3.fill_between(frames, 0, speed, alpha=0.3)
    
    for cand in candidates:
        event_type = cand.get('event_type', 'unknown')
        color = event_colors.get(event_type, 'black')
        ax3.axvline(x=cand['frame'], color=color, linestyle='--', alpha=0.7, linewidth=1)
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Speed (pixels/frame)')
    ax3.set_title('Speed over Time (red=landing, blue=hit, gray=out)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Y 方向速度
    ax4 = axes[1, 1]
    v_y = kinematics['v_y']
    
    ax4.plot(frames, v_y, 'purple', linewidth=1, label='V_y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.fill_between(frames, 0, v_y, where=v_y > 0, alpha=0.3, color='red', label='Downward')
    ax4.fill_between(frames, 0, v_y, where=v_y < 0, alpha=0.3, color='blue', label='Upward')
    
    for cand in candidates:
        event_type = cand.get('event_type', 'unknown')
        color = event_colors.get(event_type, 'black')
        ax4.axvline(x=cand['frame'], color=color, linestyle='--', alpha=0.7, linewidth=1)
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('V_y (pixels/frame)')
    ax4.set_title('Y Velocity over Time (positive = downward)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
