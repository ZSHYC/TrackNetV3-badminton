"""
BounceNet 事件分类网络
BounceNet - Event Classification Network for Badminton

将规则检测的候选事件分类为:
- landing: 落地点
- hit: 击球点
- none: 非事件（误检）

支持多种输入特征:
- 运动学特征 (轨迹序列)
- 视觉特征 (局部 patch)
- 融合特征 (运动学 + 视觉)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .visual_features import PatchEncoder, TemporalPatchEncoder


# 事件类型标签
EVENT_LABELS = {
    'none': 0,      # 非事件
    'landing': 1,   # 落地点
    'hit': 2,       # 击球点
}

LABEL_TO_EVENT = {v: k for k, v in EVENT_LABELS.items()}


class TrajectoryEncoder(nn.Module):
    """
    轨迹编码器
    
    将轨迹特征序列编码为固定长度的特征向量。
    输入: (x, y, v_x, v_y, visibility) 序列
    """
    
    def __init__(self,
                 input_dim: int = 5,
                 hidden_dim: int = 64,
                 feature_dim: int = 64,
                 num_layers: int = 2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.bidirectional = bidirectional
        
        # 1D 卷积预处理
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 输出投影
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, seq_len, input_dim) 轨迹特征序列
        
        Returns:
            features: (N, feature_dim)
        """
        # 1D 卷积: (N, seq_len, input_dim) -> (N, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (N, seq_len, hidden_dim)
        
        # LSTM
        output, (h_n, c_n) = self.lstm(x)
        
        # 使用最后时刻的隐藏状态
        if self.bidirectional:
            # 拼接双向最后状态
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]
        
        # 投影到输出维度
        features = self.fc(h)
        
        return features


class BounceNet(nn.Module):
    """
    BounceNet 事件分类网络
    
    融合运动学特征和视觉特征，对候选事件进行分类。
    
    模式:
    1. trajectory_only: 仅使用轨迹特征
    2. visual_only: 仅使用视觉特征
    3. fusion: 融合两种特征
    """
    
    def __init__(self,
                 mode: str = 'trajectory_only',
                 # 轨迹编码器参数
                 traj_input_dim: int = 5,
                 traj_hidden_dim: int = 64,
                 traj_feature_dim: int = 64,
                 traj_seq_len: int = 11,
                 # 视觉编码器参数
                 patch_size: int = 64,
                 visual_feature_dim: int = 128,
                 use_temporal_patches: bool = True,
                 # 分类器参数
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.mode = mode
        self.num_classes = num_classes
        self.traj_seq_len = traj_seq_len
        
        # 初始化编码器
        if mode in ['trajectory_only', 'fusion']:
            self.traj_encoder = TrajectoryEncoder(
                input_dim=traj_input_dim,
                hidden_dim=traj_hidden_dim,
                feature_dim=traj_feature_dim
            )
        else:
            self.traj_encoder = None
        
        if mode in ['visual_only', 'fusion']:
            if use_temporal_patches:
                self.visual_encoder = TemporalPatchEncoder(
                    in_channels=3,
                    patch_size=patch_size,
                    seq_len=traj_seq_len,
                    feature_dim=visual_feature_dim
                )
            else:
                self.visual_encoder = PatchEncoder(
                    in_channels=3,
                    patch_size=patch_size,
                    feature_dim=visual_feature_dim
                )
            self.use_temporal_patches = use_temporal_patches
        else:
            self.visual_encoder = None
            self.use_temporal_patches = False
        
        # 计算融合特征维度
        if mode == 'trajectory_only':
            fusion_dim = traj_feature_dim
        elif mode == 'visual_only':
            fusion_dim = visual_feature_dim
        else:  # fusion
            fusion_dim = traj_feature_dim + visual_feature_dim
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # 置信度输出（可选，用于异常检测）
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                traj_features: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None,
                return_confidence: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            traj_features: (N, seq_len, traj_input_dim) 轨迹特征
            visual_features: 视觉特征
                - 如果 use_temporal_patches: (N, seq_len, C, H, W)
                - 否则: (N, C, H, W)
            return_confidence: 是否返回置信度
        
        Returns:
            logits: (N, num_classes) 分类 logits
            confidence: (N, 1) 置信度（可选）
        """
        features = []
        
        # 轨迹特征编码
        if self.traj_encoder is not None:
            if traj_features is None:
                raise ValueError("traj_features required for mode: " + self.mode)
            traj_feat = self.traj_encoder(traj_features)
            features.append(traj_feat)
        
        # 视觉特征编码
        if self.visual_encoder is not None:
            if visual_features is None:
                raise ValueError("visual_features required for mode: " + self.mode)
            visual_feat = self.visual_encoder(visual_features)
            features.append(visual_feat)
        
        # 特征融合
        if len(features) > 1:
            fused = torch.cat(features, dim=1)
        else:
            fused = features[0]
        
        # 分类
        logits = self.classifier(fused)
        
        if return_confidence:
            confidence = self.confidence_head(fused)
            return logits, confidence
        
        return logits
    
    def predict(self,
                traj_features: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None,
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        预测事件类型
        
        Args:
            traj_features: 轨迹特征
            visual_features: 视觉特征
            threshold: 置信度阈值
        
        Returns:
            predictions: {
                'labels': (N,) 预测标签,
                'probs': (N, num_classes) 类别概率,
                'confidence': (N,) 置信度
            }
        """
        self.eval()
        with torch.no_grad():
            logits, confidence = self.forward(
                traj_features, visual_features, return_confidence=True
            )
            probs = F.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
        
        return {
            'labels': labels,
            'probs': probs,
            'confidence': confidence.squeeze()
        }
    
    def predict_event_type(self, labels: torch.Tensor) -> List[str]:
        """将预测标签转换为事件类型字符串"""
        return [LABEL_TO_EVENT[l.item()] for l in labels]


class BounceNetLoss(nn.Module):
    """
    BounceNet 损失函数
    
    组合:
    1. 交叉熵损失 (分类)
    2. 焦点损失 (处理类别不平衡)
    3. 置信度损失 (可选)
    """
    
    def __init__(self,
                 class_weights: Optional[List[float]] = None,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 confidence_weight: float = 0.1):
        super().__init__()
        
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.confidence_weight = confidence_weight
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            # 默认权重：none 较低，landing/hit 较高
            self.class_weights = torch.tensor([0.5, 1.0, 1.0], dtype=torch.float32)
    
    def focal_loss(self,
                   logits: torch.Tensor,
                   targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - p_t) ** self.focal_gamma
        loss = focal_weight * ce_loss
        return loss.mean()
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            logits: (N, num_classes) 预测 logits
            targets: (N,) 目标标签
            confidence: (N, 1) 预测置信度（可选）
        
        Returns:
            losses: {
                'total': 总损失,
                'classification': 分类损失,
                'confidence': 置信度损失（如果有）
            }
        """
        # 确保 class_weights 在正确的设备上
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
        
        # 分类损失
        if self.use_focal_loss:
            cls_loss = self.focal_loss(logits, targets)
        else:
            cls_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        losses = {
            'classification': cls_loss,
            'total': cls_loss
        }
        
        # 置信度损失（可选）
        if confidence is not None and self.confidence_weight > 0:
            # 正确分类的样本应该有高置信度
            with torch.no_grad():
                pred_correct = (torch.argmax(logits, dim=1) == targets).float()
            conf_loss = F.binary_cross_entropy(confidence.squeeze(), pred_correct)
            losses['confidence'] = conf_loss
            losses['total'] = cls_loss + self.confidence_weight * conf_loss
        
        return losses


class BounceNetPredictor:
    """
    BounceNet 推理包装器
    
    提供简便的推理接口，处理特征提取和后处理。
    """
    
    def __init__(self,
                 model: BounceNet,
                 feature_extractor=None,
                 kinematics_calculator=None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: BounceNet 模型
            feature_extractor: VisualFeatureExtractor (可选)
            kinematics_calculator: KinematicsCalculator (可选)
            device: 计算设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.feature_extractor = feature_extractor
        self.kinematics_calculator = kinematics_calculator
        
        # 如果没有提供运动学计算器，创建一个默认的
        if self.kinematics_calculator is None:
            from .kinematics import KinematicsCalculator
            self.kinematics_calculator = KinematicsCalculator()
    
    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: str,
                        mode: str = 'trajectory_only',
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """从检查点加载模型"""
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 从检查点获取模型配置
        config = checkpoint.get('config', {})
        
        model = BounceNet(
            mode=config.get('mode', mode),
            traj_input_dim=config.get('traj_input_dim', 5),
            traj_hidden_dim=config.get('traj_hidden_dim', 64),
            traj_feature_dim=config.get('traj_feature_dim', 64),
            traj_seq_len=config.get('traj_seq_len', 11),
            patch_size=config.get('patch_size', 64),
            visual_feature_dim=config.get('visual_feature_dim', 128),
            num_classes=config.get('num_classes', 3)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device=device)
    
    def prepare_trajectory_features(self,
                                    x: np.ndarray,
                                    y: np.ndarray,
                                    visibility: np.ndarray,
                                    candidate_frames: List[int],
                                    window_size: int = 5,
                                    img_size: Tuple[int, int] = (512, 288)) -> torch.Tensor:
        """
        为候选事件准备轨迹特征
        
        Args:
            x, y, visibility: 完整轨迹
            candidate_frames: 候选事件帧列表
            window_size: 特征窗口大小（单侧）
            img_size: 图像尺寸，用于归一化
        
        Returns:
            features: (N, seq_len, 5) tensor
                      每个时间步包含 [x, y, v_x, v_y, visibility]
        """
        # 计算运动学特征
        kinematics = self.kinematics_calculator.compute(x, y, visibility)
        v_x = kinematics['v_x']
        v_y = kinematics['v_y']
        
        seq_len = 2 * window_size + 1
        n_candidates = len(candidate_frames)
        n_frames = len(x)
        
        features = np.zeros((n_candidates, seq_len, 5), dtype=np.float32)
        
        # 归一化参数
        diag = np.sqrt(img_size[0]**2 + img_size[1]**2)
        
        for i, center_frame in enumerate(candidate_frames):
            for j, offset in enumerate(range(-window_size, window_size + 1)):
                frame_idx = center_frame + offset
                
                if 0 <= frame_idx < n_frames:
                    features[i, j, 0] = x[frame_idx] / img_size[0]
                    features[i, j, 1] = y[frame_idx] / img_size[1]
                    features[i, j, 2] = v_x[frame_idx] / diag
                    features[i, j, 3] = v_y[frame_idx] / diag
                    features[i, j, 4] = visibility[frame_idx]
                else:
                    # 边界填充
                    edge_idx = 0 if frame_idx < 0 else n_frames - 1
                    features[i, j, 0] = x[edge_idx] / img_size[0]
                    features[i, j, 1] = y[edge_idx] / img_size[1]
                    features[i, j, 2] = v_x[edge_idx] / diag
                    features[i, j, 3] = v_y[edge_idx] / diag
                    features[i, j, 4] = visibility[edge_idx]
        
        return torch.from_numpy(features).to(self.device)
    
    def classify_candidates(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           visibility: np.ndarray,
                           candidates: List[Dict],
                           frames: Optional[List[np.ndarray]] = None,
                           threshold: float = 0.5) -> List[Dict]:
        """
        对候选事件进行分类
        
        Args:
            x, y, visibility: 轨迹数据
            candidates: 候选事件列表
            frames: 视频帧列表（用于视觉特征）
            threshold: 分类置信度阈值
        
        Returns:
            classified_candidates: 带有分类结果的候选列表
        """
        if len(candidates) == 0:
            return []
        
        candidate_frames = [c['frame'] for c in candidates]
        
        # 准备轨迹特征
        traj_features = self.prepare_trajectory_features(
            x, y, visibility, candidate_frames
        )
        
        # 准备视觉特征（如果需要）
        visual_features = None
        if self.model.mode in ['visual_only', 'fusion'] and frames is not None:
            if self.feature_extractor is not None:
                batch_features = self.feature_extractor.extract_batch_features(
                    frames, x, y, candidate_frames, visibility
                )
                if self.model.use_temporal_patches:
                    visual_features = torch.from_numpy(
                        batch_features['temporal_patches'].astype(np.float32) / 255.0
                    ).permute(0, 1, 4, 2, 3).to(self.device)
                else:
                    visual_features = torch.from_numpy(
                        batch_features['center_patches'].astype(np.float32) / 255.0
                    ).permute(0, 3, 1, 2).to(self.device)
        
        # 预测
        predictions = self.model.predict(traj_features, visual_features, threshold)
        
        # 更新候选信息
        labels = predictions['labels'].cpu().numpy()
        probs = predictions['probs'].cpu().numpy()
        confidence = predictions['confidence'].cpu().numpy()
        
        classified_candidates = []
        for i, cand in enumerate(candidates):
            new_cand = cand.copy()
            pred_label = int(labels[i])
            
            new_cand['predicted_type'] = LABEL_TO_EVENT[pred_label]
            new_cand['prediction_probs'] = {
                'none': float(probs[i, 0]),
                'landing': float(probs[i, 1]),
                'hit': float(probs[i, 2])
            }
            new_cand['prediction_confidence'] = float(confidence[i])
            
            # 如果预测为 none 且置信度高，标记为可能的误检
            if pred_label == 0 and confidence[i] > threshold:
                new_cand['is_false_positive'] = True
            else:
                new_cand['is_false_positive'] = False
            
            classified_candidates.append(new_cand)
        
        return classified_candidates


def create_bouncenet(mode: str = 'trajectory_only',
                     pretrained: bool = False,
                     ckpt_path: Optional[str] = None,
                     **kwargs) -> BounceNet:
    """
    创建 BounceNet 模型的工厂函数
    
    Args:
        mode: 'trajectory_only', 'visual_only', 'fusion'
        pretrained: 是否加载预训练权重
        ckpt_path: 检查点路径
        **kwargs: 传递给 BounceNet 的其他参数
    
    Returns:
        model: BounceNet 模型
    """
    model = BounceNet(mode=mode, **kwargs)
    
    if pretrained and ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {ckpt_path}")
    
    return model
