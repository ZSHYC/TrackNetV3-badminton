"""
运动学特征计算模块
Kinematics Calculator for Badminton Trajectory

基于 TrackNetV3 (InpaintNet 修复后) 的轨迹坐标，计算速度、加速度等运动学特征。
"""

import numpy as np
from typing import Dict, Tuple, Optional


class KinematicsCalculator:
    """
    运动学特征计算器
    
    输入: InpaintNet 修复后的轨迹坐标序列
    输出: 速度、加速度、运动方向等特征
    
    注意: 输入轨迹应该是 InpaintNet 修复后的，不需要额外平滑处理
    """
    
    def __init__(self, fps: int = 30):
        """
        Args:
            fps: 视频帧率，用于将像素/帧转换为实际速度（可选）
        """
        self.fps = fps
        self.dt = 1.0 / fps
    
    def compute(self, 
                x: np.ndarray, 
                y: np.ndarray, 
                visibility: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算完整的运动学特征
        
        Args:
            x: x 坐标序列 (N,)
            y: y 坐标序列 (N,)
            visibility: 可见性序列 (N,), 1=可见, 0=不可见
        
        Returns:
            dict: 包含以下运动学特征
                - v_x: x 方向速度 (像素/帧)
                - v_y: y 方向速度 (像素/帧), 正值表示向下
                - a_x: x 方向加速度 (像素/帧²)
                - a_y: y 方向加速度 (像素/帧²)
                - speed: 速度大小
                - direction: 运动方向角度 (弧度, [-π, π])
                - curvature: 轨迹曲率
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        
        n = len(x)
        
        # 计算速度 (一阶差分)
        # 使用 np.gradient 比简单差分处理边界更好
        v_x = np.gradient(x)
        v_y = np.gradient(y)
        
        # 对于不可见的帧，速度设为 0 或 NaN
        # 这里选择保留计算值，但标记不可见帧
        invisible_mask = visibility == 0
        
        # 计算加速度 (二阶差分)
        a_x = np.gradient(v_x)
        a_y = np.gradient(v_y)
        
        # 计算速度大小
        speed = np.sqrt(v_x**2 + v_y**2)
        
        # 计算运动方向 (弧度)
        # arctan2(y, x): 向右为0, 向下为π/2, 向左为±π, 向上为-π/2
        direction = np.arctan2(v_y, v_x)
        
        # 计算曲率: κ = |v × a| / |v|³
        # 对于 2D: κ = |v_x * a_y - v_y * a_x| / (v_x² + v_y²)^(3/2)
        cross_product = np.abs(v_x * a_y - v_y * a_x)
        speed_cubed = np.power(speed, 3)
        # 避免除以零
        curvature = np.where(speed_cubed > 1e-6, 
                            cross_product / speed_cubed, 
                            0.0)
        
        return {
            'v_x': v_x,
            'v_y': v_y,
            'a_x': a_x,
            'a_y': a_y,
            'speed': speed,
            'direction': direction,
            'curvature': curvature,
            'visibility': visibility
        }
    
    def compute_at_frame(self, 
                         x: np.ndarray, 
                         y: np.ndarray, 
                         visibility: np.ndarray,
                         frame_idx: int,
                         window_size: int = 5) -> Dict[str, float]:
        """
        计算指定帧的运动学特征（使用局部窗口）
        
        Args:
            x, y, visibility: 完整轨迹
            frame_idx: 目标帧索引
            window_size: 局部窗口大小（单侧）
        
        Returns:
            dict: 该帧的运动学特征
        """
        n = len(x)
        start = max(0, frame_idx - window_size)
        end = min(n, frame_idx + window_size + 1)
        
        # 提取局部窗口
        x_local = x[start:end]
        y_local = y[start:end]
        vis_local = visibility[start:end]
        
        # 计算局部特征
        features = self.compute(x_local, y_local, vis_local)
        
        # 找到目标帧在局部窗口中的索引
        local_idx = frame_idx - start
        
        return {
            'v_x': features['v_x'][local_idx],
            'v_y': features['v_y'][local_idx],
            'a_x': features['a_x'][local_idx],
            'a_y': features['a_y'][local_idx],
            'speed': features['speed'][local_idx],
            'direction': features['direction'][local_idx],
            'curvature': features['curvature'][local_idx]
        }
    
    def get_trajectory_window(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              visibility: np.ndarray,
                              center_frame: int,
                              window_size: int = 5,
                              normalize: bool = True,
                              img_size: Tuple[int, int] = (512, 288)) -> np.ndarray:
        """
        提取以指定帧为中心的轨迹特征窗口，用于 BounceNet 输入
        
        Args:
            x, y, visibility: 完整轨迹
            center_frame: 中心帧索引
            window_size: 窗口大小（单侧），总窗口 = 2*window_size + 1
            normalize: 是否归一化坐标
            img_size: (width, height) 用于归一化
        
        Returns:
            features: (seq_len, 4) 数组，每帧包含 [x, y, v_x, v_y]
        """
        n = len(x)
        seq_len = 2 * window_size + 1
        features = np.zeros((seq_len, 4), dtype=np.float32)
        
        # 计算完整轨迹的速度
        full_kinematics = self.compute(x, y, visibility)
        v_x = full_kinematics['v_x']
        v_y = full_kinematics['v_y']
        
        for i, offset in enumerate(range(-window_size, window_size + 1)):
            frame_idx = center_frame + offset
            
            if 0 <= frame_idx < n:
                if normalize:
                    features[i, 0] = x[frame_idx] / img_size[0]  # 归一化 x
                    features[i, 1] = y[frame_idx] / img_size[1]  # 归一化 y
                    # 速度也需要归一化（除以图像对角线长度）
                    diag = np.sqrt(img_size[0]**2 + img_size[1]**2)
                    features[i, 2] = v_x[frame_idx] / diag
                    features[i, 3] = v_y[frame_idx] / diag
                else:
                    features[i, 0] = x[frame_idx]
                    features[i, 1] = y[frame_idx]
                    features[i, 2] = v_x[frame_idx]
                    features[i, 3] = v_y[frame_idx]
            else:
                # padding: 使用边界值
                edge_idx = 0 if frame_idx < 0 else n - 1
                if normalize:
                    features[i, 0] = x[edge_idx] / img_size[0]
                    features[i, 1] = y[edge_idx] / img_size[1]
                    diag = np.sqrt(img_size[0]**2 + img_size[1]**2)
                    features[i, 2] = v_x[edge_idx] / diag
                    features[i, 3] = v_y[edge_idx] / diag
                else:
                    features[i, 0] = x[edge_idx]
                    features[i, 1] = y[edge_idx]
                    features[i, 2] = v_x[edge_idx]
                    features[i, 3] = v_y[edge_idx]
        
        return features


def compute_direction_change(direction: np.ndarray) -> np.ndarray:
    """
    计算方向变化量（处理角度环绕问题）
    
    Args:
        direction: 方向角度序列 (弧度)
    
    Returns:
        direction_change: 方向变化量 (弧度)
    """
    diff = np.diff(direction)
    # 处理角度环绕: 如果差值 > π, 说明跨越了 ±π 边界
    diff = np.where(diff > np.pi, diff - 2*np.pi, diff)
    diff = np.where(diff < -np.pi, diff + 2*np.pi, diff)
    # 在开头补一个 0
    return np.concatenate([[0], diff])
