"""
BounceNet 数据集
Dataset for BounceNet Training

支持从标注文件加载训练数据。
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union

from .bouncenet import EVENT_LABELS
from .kinematics import KinematicsCalculator
from .visual_features import VisualFeatureExtractor


class BounceNetDataset(Dataset):
    """
    BounceNet 训练数据集
    
    从标注 JSON 文件加载事件标注，生成训练样本。
    
    数据格式:
    - CSV: 轨迹数据 (Frame, X, Y, Visibility)
    - JSON: 标注文件，包含确认的事件列表
    """
    
    def __init__(self,
                 csv_paths: List[str],
                 label_paths: List[str],
                 video_paths: Optional[List[str]] = None,
                 mode: str = 'trajectory_only',
                 window_size: int = 5,
                 patch_size: int = 64,
                 img_size: Tuple[int, int] = (512, 288),
                 augment: bool = True,
                 include_negatives: bool = True,
                 negative_ratio: float = 1.0,
                 min_distance_from_event: int = 10):
        """
        Args:
            csv_paths: 轨迹 CSV 文件路径列表
            label_paths: 标注 JSON 文件路径列表
            video_paths: 视频文件路径列表（可选，用于视觉特征）
            mode: 'trajectory_only', 'visual_only', 'fusion'
            window_size: 特征窗口大小（单侧）
            patch_size: 视觉 patch 大小
            img_size: 图像尺寸
            augment: 是否数据增强
            include_negatives: 是否包含负样本
            negative_ratio: 负样本与正样本的比例
            min_distance_from_event: 负样本与事件的最小距离（帧）
        """
        self.mode = mode
        self.window_size = window_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.augment = augment
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        self.min_distance_from_event = min_distance_from_event
        
        self.seq_len = 2 * window_size + 1
        
        # 运动学计算器
        self.kinematics_calculator = KinematicsCalculator()
        
        # 视觉特征提取器
        if mode in ['visual_only', 'fusion']:
            self.feature_extractor = VisualFeatureExtractor(
                patch_size=patch_size,
                temporal_window=window_size,
                img_size=img_size
            )
        else:
            self.feature_extractor = None
        
        # 加载数据
        self.samples = []
        self._load_data(csv_paths, label_paths, video_paths)
    
    def _load_data(self,
                   csv_paths: List[str],
                   label_paths: List[str],
                   video_paths: Optional[List[str]]):
        """加载所有数据"""
        
        if video_paths is None:
            video_paths = [None] * len(csv_paths)
        
        for csv_path, label_path, video_path in zip(csv_paths, label_paths, video_paths):
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found: {csv_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Label not found: {label_path}")
                continue
            
            # 加载轨迹
            df = pd.read_csv(csv_path)
            df = df.sort_values(by='Frame').fillna(0)
            
            x = df['X'].values.astype(np.float32)
            y = df['Y'].values.astype(np.float32)
            visibility = df['Visibility'].values.astype(np.float32)
            n_frames = len(x)
            
            # 加载标注
            with open(label_path, 'r') as f:
                labels = json.load(f)
            
            events = labels.get('events', [])
            
            # 提取正样本（确认的事件）
            event_frames = set()
            for event in events:
                if event.get('confirmed', False):
                    frame = event['frame']
                    event_type = event.get('event_type', 'none')
                    
                    if event_type in EVENT_LABELS:
                        self.samples.append({
                            'csv_path': csv_path,
                            'video_path': video_path,
                            'frame': frame,
                            'label': EVENT_LABELS[event_type],
                            'event_type': event_type,
                            'x': x,
                            'y': y,
                            'visibility': visibility
                        })
                        event_frames.add(frame)
            
            # 生成负样本
            if self.include_negatives and len(event_frames) > 0:
                n_positives = len(event_frames)
                n_negatives = int(n_positives * self.negative_ratio)
                
                # 找出远离所有事件的帧
                valid_neg_frames = []
                for f in range(self.window_size, n_frames - self.window_size):
                    if visibility[f] > 0:  # 只选择可见的帧
                        min_dist = min(abs(f - ef) for ef in event_frames)
                        if min_dist >= self.min_distance_from_event:
                            valid_neg_frames.append(f)
                
                # 随机选择负样本
                if len(valid_neg_frames) > 0:
                    neg_frames = np.random.choice(
                        valid_neg_frames,
                        size=min(n_negatives, len(valid_neg_frames)),
                        replace=False
                    )
                    
                    for frame in neg_frames:
                        self.samples.append({
                            'csv_path': csv_path,
                            'video_path': video_path,
                            'frame': int(frame),
                            'label': EVENT_LABELS['none'],
                            'event_type': 'none',
                            'x': x,
                            'y': y,
                            'visibility': visibility
                        })
        
        print(f"Loaded {len(self.samples)} samples")
        
        # 统计类别分布
        label_counts = {}
        for s in self.samples:
            label_counts[s['event_type']] = label_counts.get(s['event_type'], 0) + 1
        print(f"Class distribution: {label_counts}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        x = sample['x']
        y = sample['y']
        visibility = sample['visibility']
        center_frame = sample['frame']
        label = sample['label']
        
        # 计算运动学特征
        kinematics = self.kinematics_calculator.compute(x, y, visibility)
        v_x = kinematics['v_x']
        v_y = kinematics['v_y']
        
        # 提取轨迹特征窗口
        traj_features = self._extract_trajectory_features(
            x, y, v_x, v_y, visibility, center_frame
        )
        
        # 数据增强
        if self.augment:
            traj_features = self._augment_trajectory(traj_features)
        
        result = {
            'traj_features': torch.from_numpy(traj_features),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # 视觉特征（如果需要）
        if self.mode in ['visual_only', 'fusion'] and sample['video_path'] is not None:
            # 延迟加载视频帧
            visual_features = self._extract_visual_features(sample, center_frame)
            result['visual_features'] = visual_features
        
        return result
    
    def _extract_trajectory_features(self,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     v_x: np.ndarray,
                                     v_y: np.ndarray,
                                     visibility: np.ndarray,
                                     center_frame: int) -> np.ndarray:
        """提取轨迹特征窗口"""
        n_frames = len(x)
        features = np.zeros((self.seq_len, 5), dtype=np.float32)
        
        # 归一化参数
        diag = np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)
        
        for i, offset in enumerate(range(-self.window_size, self.window_size + 1)):
            frame_idx = center_frame + offset
            
            if 0 <= frame_idx < n_frames:
                features[i, 0] = x[frame_idx] / self.img_size[0]
                features[i, 1] = y[frame_idx] / self.img_size[1]
                features[i, 2] = v_x[frame_idx] / diag
                features[i, 3] = v_y[frame_idx] / diag
                features[i, 4] = visibility[frame_idx]
            else:
                # 边界填充
                edge_idx = 0 if frame_idx < 0 else n_frames - 1
                features[i, 0] = x[edge_idx] / self.img_size[0]
                features[i, 1] = y[edge_idx] / self.img_size[1]
                features[i, 2] = v_x[edge_idx] / diag
                features[i, 3] = v_y[edge_idx] / diag
                features[i, 4] = visibility[edge_idx]
        
        return features
    
    def _augment_trajectory(self, features: np.ndarray) -> np.ndarray:
        """轨迹数据增强"""
        # 随机水平翻转
        if np.random.random() > 0.5:
            features[:, 0] = 1.0 - features[:, 0]  # 翻转 x
            features[:, 2] = -features[:, 2]  # 翻转 v_x
        
        # 随机添加噪声
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, features[:, :2].shape)
            features[:, :2] += noise
        
        # 随机时间偏移（轻微）
        if np.random.random() > 0.7:
            shift = np.random.choice([-1, 0, 1])
            if shift != 0:
                features = np.roll(features, shift, axis=0)
        
        return features
    
    def _extract_visual_features(self,
                                 sample: Dict,
                                 center_frame: int) -> torch.Tensor:
        """提取视觉特征（延迟加载）"""
        video_path = sample['video_path']
        x = sample['x']
        y = sample['y']
        visibility = sample['visibility']
        
        # 加载视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        start_frame = max(0, center_frame - self.window_size)
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                       center_frame + self.window_size + 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.img_size)
                frames.append(frame)
            else:
                frames.append(np.zeros((self.img_size[1], self.img_size[0], 3), 
                                       dtype=np.uint8))
        
        cap.release()
        
        # 提取特征
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_features(
                frames, x[start_frame:end_frame], y[start_frame:end_frame],
                center_frame - start_frame, visibility[start_frame:end_frame]
            )
            
            # 转换为 tensor
            patches = features['temporal_patches']  # (seq_len, H, W, C)
            patches = patches.astype(np.float32) / 255.0
            patches = np.transpose(patches, (0, 3, 1, 2))  # (seq_len, C, H, W)
            return torch.from_numpy(patches)
        
        return torch.zeros(self.seq_len, 3, self.patch_size, self.patch_size)


def create_dataloaders(csv_paths: List[str],
                       label_paths: List[str],
                       video_paths: Optional[List[str]] = None,
                       mode: str = 'trajectory_only',
                       batch_size: int = 32,
                       val_split: float = 0.2,
                       num_workers: int = 4,
                       **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        csv_paths: CSV 文件路径列表
        label_paths: 标注文件路径列表
        video_paths: 视频文件路径列表
        mode: 特征模式
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
        **kwargs: 传递给 Dataset 的其他参数
    
    Returns:
        train_loader, val_loader
    """
    # 划分训练集和验证集
    n_samples = len(csv_paths)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_csv = [csv_paths[i] for i in train_indices]
    train_label = [label_paths[i] for i in train_indices]
    train_video = [video_paths[i] for i in train_indices] if video_paths else None
    
    val_csv = [csv_paths[i] for i in val_indices]
    val_label = [label_paths[i] for i in val_indices]
    val_video = [video_paths[i] for i in val_indices] if video_paths else None
    
    # 创建数据集
    train_dataset = BounceNetDataset(
        train_csv, train_label, train_video,
        mode=mode, augment=True, **kwargs
    )
    
    val_dataset = BounceNetDataset(
        val_csv, val_label, val_video,
        mode=mode, augment=False, **kwargs
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
