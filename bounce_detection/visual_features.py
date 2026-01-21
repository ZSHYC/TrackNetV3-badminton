"""
视觉特征提取器
Visual Feature Extractor for BounceNet

从视频帧中提取候选事件周围的局部视觉特征。
支持两种模式：
1. 裁剪模式：提取球周围的局部 patch
2. 全局模式：使用预训练 CNN 提取全局特征
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class VisualFeatureExtractor:
    """
    视觉特征提取器
    
    从视频帧中提取候选事件的局部视觉特征，用于 BounceNet 分类。
    
    特征类型：
    1. 局部 patch: 以球为中心的 crop (默认 64x64)
    2. 时序 patch: 连续多帧的 crop stack
    3. 运动历史图 (MHI): 编码最近几帧的运动
    """
    
    def __init__(self,
                 patch_size: int = 64,
                 temporal_window: int = 5,
                 use_motion_history: bool = True,
                 img_size: Tuple[int, int] = (512, 288),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            patch_size: 局部 patch 大小 (正方形)
            temporal_window: 时序窗口大小（单侧），总窗口 = 2*window + 1
            use_motion_history: 是否使用运动历史图
            img_size: 输入图像尺寸 (width, height)
            device: 计算设备
        """
        self.patch_size = patch_size
        self.temporal_window = temporal_window
        self.use_motion_history = use_motion_history
        self.img_size = img_size
        self.device = device
        
        # MHI 参数
        self.mhi_duration = temporal_window * 2 + 1  # MHI 持续时间
    
    def extract_patch(self,
                      frame: np.ndarray,
                      x: float,
                      y: float,
                      patch_size: Optional[int] = None) -> np.ndarray:
        """
        从帧中提取以 (x, y) 为中心的 patch
        
        Args:
            frame: 输入帧 (H, W, C) 或 (H, W)
            x, y: 中心坐标
            patch_size: patch 大小，默认使用初始化时的值
        
        Returns:
            patch: (patch_size, patch_size, C) 或 (patch_size, patch_size)
        """
        if patch_size is None:
            patch_size = self.patch_size
        
        h, w = frame.shape[:2]
        half = patch_size // 2
        
        # 计算裁剪区域（考虑边界）
        x, y = int(round(x)), int(round(y))
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)
        
        # 提取 patch
        if len(frame.shape) == 3:
            patch = np.zeros((patch_size, patch_size, frame.shape[2]), dtype=frame.dtype)
        else:
            patch = np.zeros((patch_size, patch_size), dtype=frame.dtype)
        
        # 计算在 patch 中的偏移
        px1 = half - (x - x1)
        py1 = half - (y - y1)
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)
        
        patch[py1:py2, px1:px2] = frame[y1:y2, x1:x2]
        
        return patch
    
    def extract_temporal_patches(self,
                                 frames: List[np.ndarray],
                                 x_coords: np.ndarray,
                                 y_coords: np.ndarray,
                                 center_frame_idx: int,
                                 visibility: Optional[np.ndarray] = None) -> np.ndarray:
        """
        提取时序 patch 序列
        
        Args:
            frames: 视频帧列表
            x_coords, y_coords: 轨迹坐标
            center_frame_idx: 中心帧索引
            visibility: 可见性数组
        
        Returns:
            patches: (seq_len, patch_size, patch_size, C) 数组
        """
        seq_len = 2 * self.temporal_window + 1
        n_frames = len(frames)
        
        # 获取第一帧的形状
        sample_frame = frames[0]
        if len(sample_frame.shape) == 3:
            channels = sample_frame.shape[2]
            patches = np.zeros((seq_len, self.patch_size, self.patch_size, channels), 
                              dtype=np.uint8)
        else:
            patches = np.zeros((seq_len, self.patch_size, self.patch_size), 
                              dtype=np.uint8)
        
        for i, offset in enumerate(range(-self.temporal_window, self.temporal_window + 1)):
            frame_idx = center_frame_idx + offset
            
            if 0 <= frame_idx < n_frames and 0 <= frame_idx < len(x_coords):
                # 检查可见性
                if visibility is not None and visibility[frame_idx] == 0:
                    # 不可见帧，使用零填充
                    continue
                
                x, y = x_coords[frame_idx], y_coords[frame_idx]
                patches[i] = self.extract_patch(frames[frame_idx], x, y)
        
        return patches
    
    def compute_motion_history_image(self,
                                     frames: List[np.ndarray],
                                     center_frame_idx: int,
                                     threshold: int = 25) -> np.ndarray:
        """
        计算运动历史图 (Motion History Image)
        
        编码最近几帧的运动信息到一张图中。
        
        Args:
            frames: 视频帧列表
            center_frame_idx: 中心帧索引
            threshold: 帧差阈值
        
        Returns:
            mhi: (H, W) 运动历史图，值在 [0, 255]
        """
        n_frames = len(frames)
        h, w = frames[0].shape[:2]
        mhi = np.zeros((h, w), dtype=np.float32)
        
        # 计算时序范围
        start = max(0, center_frame_idx - self.temporal_window)
        end = min(n_frames, center_frame_idx + self.temporal_window + 1)
        
        duration = end - start
        if duration < 2:
            return (mhi * 255).astype(np.uint8)
        
        # 转换为灰度图
        def to_gray(frame):
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        
        # 计算帧差并累积
        prev_gray = to_gray(frames[start])
        
        for i in range(start + 1, end):
            curr_gray = to_gray(frames[i])
            diff = cv2.absdiff(curr_gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
            
            # 时间权重：越近的帧权重越高
            timestamp = (i - start) / duration
            mhi = np.where(motion_mask > 0, timestamp, mhi * (1 - 1.0/duration))
            
            prev_gray = curr_gray
        
        return (mhi * 255).astype(np.uint8)
    
    def extract_features(self,
                         frames: List[np.ndarray],
                         x_coords: np.ndarray,
                         y_coords: np.ndarray,
                         center_frame_idx: int,
                         visibility: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        提取候选事件的完整视觉特征
        
        Args:
            frames: 视频帧列表
            x_coords, y_coords: 轨迹坐标
            center_frame_idx: 候选事件帧索引
            visibility: 可见性数组
        
        Returns:
            features: {
                'center_patch': 中心帧 patch (patch_size, patch_size, 3),
                'temporal_patches': 时序 patches (seq_len, patch_size, patch_size, 3),
                'mhi_patch': 运动历史图 patch (patch_size, patch_size),
                'center_coords': (x, y) 归一化坐标
            }
        """
        n_frames = len(frames)
        
        # 安全获取坐标
        if center_frame_idx < 0 or center_frame_idx >= len(x_coords):
            center_frame_idx = max(0, min(center_frame_idx, len(x_coords) - 1))
        
        x, y = x_coords[center_frame_idx], y_coords[center_frame_idx]
        
        # 1. 提取中心帧 patch
        if 0 <= center_frame_idx < n_frames:
            center_patch = self.extract_patch(frames[center_frame_idx], x, y)
        else:
            center_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # 2. 提取时序 patches
        temporal_patches = self.extract_temporal_patches(
            frames, x_coords, y_coords, center_frame_idx, visibility
        )
        
        # 3. 计算运动历史图 patch
        if self.use_motion_history and n_frames > 1:
            mhi = self.compute_motion_history_image(frames, center_frame_idx)
            mhi_patch = self.extract_patch(mhi, x, y)
        else:
            mhi_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # 4. 归一化坐标
        center_coords = (x / self.img_size[0], y / self.img_size[1])
        
        return {
            'center_patch': center_patch,
            'temporal_patches': temporal_patches,
            'mhi_patch': mhi_patch,
            'center_coords': center_coords
        }
    
    def extract_batch_features(self,
                               frames: List[np.ndarray],
                               x_coords: np.ndarray,
                               y_coords: np.ndarray,
                               candidate_frames: List[int],
                               visibility: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        批量提取多个候选事件的特征
        
        Args:
            frames: 视频帧列表
            x_coords, y_coords: 轨迹坐标
            candidate_frames: 候选事件帧索引列表
            visibility: 可见性数组
        
        Returns:
            batch_features: {
                'center_patches': (N, patch_size, patch_size, 3),
                'temporal_patches': (N, seq_len, patch_size, patch_size, 3),
                'mhi_patches': (N, patch_size, patch_size),
                'center_coords': (N, 2)
            }
        """
        n_candidates = len(candidate_frames)
        seq_len = 2 * self.temporal_window + 1
        
        center_patches = []
        temporal_patches = []
        mhi_patches = []
        center_coords = []
        
        for frame_idx in candidate_frames:
            features = self.extract_features(
                frames, x_coords, y_coords, frame_idx, visibility
            )
            center_patches.append(features['center_patch'])
            temporal_patches.append(features['temporal_patches'])
            mhi_patches.append(features['mhi_patch'])
            center_coords.append(features['center_coords'])
        
        return {
            'center_patches': np.stack(center_patches),
            'temporal_patches': np.stack(temporal_patches),
            'mhi_patches': np.stack(mhi_patches),
            'center_coords': np.array(center_coords)
        }
    
    def to_tensor(self, 
                  features: Dict[str, np.ndarray],
                  normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        将特征转换为 PyTorch tensor
        
        Args:
            features: numpy 特征字典
            normalize: 是否归一化到 [0, 1]
        
        Returns:
            tensor_features: PyTorch tensor 字典
        """
        result = {}
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.astype(np.float32))
                
                if normalize and 'patch' in key:
                    tensor = tensor / 255.0
                
                # 调整通道顺序 (H, W, C) -> (C, H, W)
                if key == 'center_patches' and len(tensor.shape) == 4:
                    tensor = tensor.permute(0, 3, 1, 2)
                elif key == 'center_patch' and len(tensor.shape) == 3:
                    tensor = tensor.permute(2, 0, 1)
                elif key == 'temporal_patches' and len(tensor.shape) == 5:
                    # (N, seq, H, W, C) -> (N, seq, C, H, W)
                    tensor = tensor.permute(0, 1, 4, 2, 3)
                
                result[key] = tensor.to(self.device)
            elif isinstance(value, tuple):
                result[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        return result


class PatchEncoder(nn.Module):
    """
    简单的 Patch 编码器
    
    将 patch 编码为固定长度的特征向量。
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 64,
                 feature_dim: int = 128):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) patch tensor
        
        Returns:
            features: (N, feature_dim)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalPatchEncoder(nn.Module):
    """
    时序 Patch 编码器
    
    处理连续多帧的 patch 序列，提取时序特征。
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 64,
                 seq_len: int = 11,
                 feature_dim: int = 128):
        super().__init__()
        
        self.seq_len = seq_len
        
        # 单帧编码器 (共享权重)
        self.patch_encoder = PatchEncoder(in_channels, patch_size, feature_dim)
        
        # 时序聚合 (使用 1D 卷积)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, seq_len, C, H, W) 时序 patch tensor
        
        Returns:
            features: (N, feature_dim)
        """
        N, T, C, H, W = x.shape
        
        # 编码每一帧
        x = x.view(N * T, C, H, W)
        x = self.patch_encoder(x)  # (N*T, feature_dim)
        x = x.view(N, T, -1)  # (N, T, feature_dim)
        
        # 时序卷积
        x = x.permute(0, 2, 1)  # (N, feature_dim, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # (N, T, feature_dim)
        
        # 注意力池化
        attn = self.attention(x)  # (N, T, 1)
        x = (x * attn).sum(dim=1)  # (N, feature_dim)
        
        return x
