"""
数据管理模块
Data Manager for Labeling Tool

负责:
- 加载视频帧和轨迹数据
- 管理标注数据（加载/保存）
- 数据格式转换
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 使用统一的事件类型定义
try:
    from bounce_detection import EVENT_TYPES
except ImportError:
    # 回退定义，防止导入失败
    EVENT_TYPES = ['landing', 'hit', 'out_of_frame', 'other']


class LabelingDataManager:
    """
    标注数据管理器
    
    管理一个回合(rally)的:
    - 视频帧
    - 轨迹数据 (x, y, visibility)
    - 标注结果 (events)
    """
    
    # 使用模块级别的事件类型定义
    EVENT_TYPES = EVENT_TYPES
    
    def __init__(self, 
                 video_path: Optional[str] = None,
                 csv_path: Optional[str] = None,
                 label_path: Optional[str] = None,
                 lazy_load_video: bool = False):
        """
        初始化数据管理器
        
        Args:
            video_path: 视频文件路径
            csv_path: 轨迹 CSV 文件路径
            label_path: 已有标注文件路径 (可选)
            lazy_load_video: 是否延迟加载视频帧（节省内存，适用于长视频）
        """
        self.video_path = video_path
        self.csv_path = csv_path
        self.label_path = label_path
        self.lazy_load_video = lazy_load_video
        
        # 数据存储
        self.frames: List[np.ndarray] = []
        self._video_cap: Optional[cv2.VideoCapture] = None  # 延迟加载时使用
        self._frame_cache: Dict[int, np.ndarray] = {}  # 帧缓存
        self._cache_size: int = 100  # 最大缓存帧数
        
        self.trajectory: Dict[str, np.ndarray] = {}
        self.events: List[Dict] = []
        
        # 元数据
        self.metadata: Dict = {
            'video_path': video_path,
            'csv_path': csv_path,
            'total_frames': 0,
            'fps': 30,
            'frame_size': (0, 0)
        }
        
        # 加载数据
        if video_path and os.path.exists(video_path):
            self._load_video()
        if csv_path and os.path.exists(csv_path):
            self._load_trajectory()
        if label_path and os.path.exists(label_path):
            self._load_labels()
    
    def _load_video(self):
        """加载视频帧（或初始化延迟加载）"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
        self.metadata['frame_size'] = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.lazy_load_video:
            # 延迟加载模式：保存 VideoCapture 对象
            self._video_cap = cap
            self.metadata['total_frames'] = total_frames
            print(f"Video initialized (lazy mode): {total_frames} frames from {self.video_path}")
        else:
            # 立即加载所有帧
            self.frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 转换 BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
            
            cap.release()
            self.metadata['total_frames'] = len(self.frames)
            print(f"Loaded {len(self.frames)} frames from {self.video_path}")
        self.metadata['total_frames'] = len(self.frames)
        print(f"Loaded {len(self.frames)} frames from {self.video_path}")
    
    def _load_trajectory(self):
        """加载轨迹数据"""
        df = pd.read_csv(self.csv_path)
        df = df.sort_values(by='Frame').fillna(0)
        
        self.trajectory = {
            'frame': df['Frame'].values.astype(int),
            'x': df['X'].values.astype(float),
            'y': df['Y'].values.astype(float),
            'visibility': df['Visibility'].values.astype(int)
        }
        
        # 如果没有视频，从轨迹推断帧数
        if self.metadata['total_frames'] == 0:
            self.metadata['total_frames'] = int(df['Frame'].max()) + 1
        
        print(f"Loaded trajectory with {len(df)} frames from {self.csv_path}")
    
    def _load_labels(self):
        """加载已有标注"""
        with open(self.label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.events = data.get('events', [])
        
        # 合并元数据
        if 'metadata' in data:
            self.metadata.update(data['metadata'])
        
        print(f"Loaded {len(self.events)} events from {self.label_path}")
    
    def save_labels(self, save_path: Optional[str] = None):
        """
        保存标注结果
        
        Args:
            save_path: 保存路径，默认使用 label_path 或自动生成
        """
        if save_path is None:
            if self.label_path:
                save_path = self.label_path
            elif self.csv_path:
                # 自动生成路径: xxx_ball.csv -> xxx_ball_labels.json
                save_path = self.csv_path.replace('_ball.csv', '_labels.json')
                save_path = save_path.replace('.csv', '_labels.json')
            else:
                raise ValueError("No save path specified")
        
        # 构建保存数据
        data = {
            'metadata': {
                'video_path': self.video_path,
                'csv_path': self.csv_path,
                'total_frames': self.metadata['total_frames'],
                'fps': self.metadata['fps'],
                'labeled_at': pd.Timestamp.now().isoformat()
            },
            'events': self.events
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.label_path = save_path
        print(f"Saved {len(self.events)} events to {save_path}")
        return save_path
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        获取指定帧
        
        支持两种模式:
        - 立即加载模式: 直接从 self.frames 列表获取
        - 延迟加载模式: 按需从视频文件读取，带 LRU 缓存
        """
        if self.lazy_load_video:
            # 延迟加载模式
            return self._get_frame_lazy(frame_idx)
        else:
            # 立即加载模式
            if 0 <= frame_idx < len(self.frames):
                return self.frames[frame_idx]
            return None
    
    def _get_frame_lazy(self, frame_idx: int) -> Optional[np.ndarray]:
        """延迟加载帧（带缓存）"""
        if frame_idx < 0 or frame_idx >= self.metadata['total_frames']:
            return None
        
        # 检查缓存
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
        
        # 从视频读取
        if self._video_cap is None:
            return None
        
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_cap.read()
        
        if not ret:
            return None
        
        # 转换 BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 更新缓存（简单 LRU：超过限制时删除最早的）
        if len(self._frame_cache) >= self._cache_size:
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
        self._frame_cache[frame_idx] = frame_rgb
        
        return frame_rgb
    
    def __del__(self):
        """析构函数：释放视频资源"""
        if hasattr(self, '_video_cap') and self._video_cap is not None:
            self._video_cap.release()
    
    def get_trajectory_at_frame(self, frame_idx: int) -> Tuple[float, float, int]:
        """获取指定帧的轨迹坐标"""
        if 'frame' not in self.trajectory:
            return 0.0, 0.0, 0
        
        # 找到对应帧的索引
        frame_indices = self.trajectory['frame']
        idx = np.where(frame_indices == frame_idx)[0]
        
        if len(idx) > 0:
            i = idx[0]
            return (
                self.trajectory['x'][i],
                self.trajectory['y'][i],
                int(self.trajectory['visibility'][i])
            )
        return 0.0, 0.0, 0
    
    def get_trajectory_window(self, 
                              center_frame: int, 
                              window_size: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取指定帧周围的轨迹窗口"""
        if 'frame' not in self.trajectory:
            return np.array([]), np.array([]), np.array([])
        
        start = max(0, center_frame - window_size)
        end = min(self.metadata['total_frames'], center_frame + window_size + 1)
        
        mask = (self.trajectory['frame'] >= start) & (self.trajectory['frame'] < end)
        
        return (
            self.trajectory['x'][mask],
            self.trajectory['y'][mask],
            self.trajectory['visibility'][mask]
        )
    
    # ==================== 事件管理 ====================
    
    def add_event(self, event: Dict):
        """添加事件"""
        # 确保必要字段
        required_fields = ['frame', 'event_type']
        for field in required_fields:
            if field not in event:
                raise ValueError(f"Event must have '{field}' field")
        
        # 检查事件类型
        if event['event_type'] not in self.EVENT_TYPES:
            print(f"Warning: Unknown event type '{event['event_type']}'")
        
        # 添加默认字段
        if 'confirmed' not in event:
            event['confirmed'] = False
        if 'confidence' not in event:
            event['confidence'] = 1.0
        
        # 添加坐标（如果没有）
        if 'x' not in event or 'y' not in event:
            x, y, _ = self.get_trajectory_at_frame(event['frame'])
            event['x'] = x
            event['y'] = y
        
        self.events.append(event)
        self._sort_events()
    
    def remove_event(self, frame: int) -> bool:
        """删除指定帧的事件"""
        for i, event in enumerate(self.events):
            if event['frame'] == frame:
                self.events.pop(i)
                return True
        return False
    
    def update_event(self, frame: int, updates: Dict) -> bool:
        """更新指定帧的事件"""
        for event in self.events:
            if event['frame'] == frame:
                event.update(updates)
                return True
        return False
    
    def get_event_at_frame(self, frame: int) -> Optional[Dict]:
        """获取指定帧的事件"""
        for event in self.events:
            if event['frame'] == frame:
                return event
        return None
    
    def get_events_in_range(self, start_frame: int, end_frame: int) -> List[Dict]:
        """获取指定帧范围内的事件"""
        return [e for e in self.events if start_frame <= e['frame'] <= end_frame]
    
    def confirm_event(self, frame: int) -> bool:
        """确认事件"""
        return self.update_event(frame, {'confirmed': True})
    
    def _sort_events(self):
        """按帧排序事件"""
        self.events.sort(key=lambda e: e['frame'])
    
    def set_events_from_candidates(self, candidates: List[Dict]):
        """从 Phase 1 候选设置事件（预填充）
        
        保存的字段:
        - frame, x, y: 位置信息
        - event_type: 事件类型 (landing/hit/out_of_frame)
        - rule: 主规则名称
        - confidence: 置信度 (可能被辅助规则提升)
        - all_rules: 所有触发的规则列表 (包括主规则和辅助规则)
        - auxiliary_rules: 辅助规则列表 (用于置信度提升)
        - features: 运动学特征字典
        """
        self.events = []
        for cand in candidates:
            event = {
                'frame': cand['frame'],
                'x': cand.get('x', 0),
                'y': cand.get('y', 0),
                'event_type': cand.get('event_type', 'unknown'),
                'rule': cand.get('rule', 'manual'),
                'confidence': cand.get('confidence', 0.5),
                'confirmed': False,  # 待人工确认
                'features': cand.get('features', {}),
                # 新增字段：混合方案的规则信息
                'all_rules': cand.get('all_rules', [cand.get('rule', 'manual')]),
                'auxiliary_rules': cand.get('auxiliary_rules', [])
            }
            self.events.append(event)
        self._sort_events()
        print(f"Pre-filled {len(self.events)} events from candidates")
    
    # ==================== 统计 ====================
    
    def get_statistics(self) -> Dict:
        """获取标注统计"""
        stats = {
            'total_events': len(self.events),
            'confirmed_events': sum(1 for e in self.events if e.get('confirmed', False)),
            'unconfirmed_events': sum(1 for e in self.events if not e.get('confirmed', False)),
            'by_type': {}
        }
        
        for event_type in self.EVENT_TYPES:
            count = sum(1 for e in self.events if e.get('event_type') == event_type)
            if count > 0:
                stats['by_type'][event_type] = count
        
        return stats
    
    def __len__(self):
        return self.metadata['total_frames']


class LabelExporter:
    """
    标注数据导出器
    
    支持导出为多种格式用于训练
    """
    
    @staticmethod
    def to_training_csv(data_manager: LabelingDataManager, 
                        output_path: str,
                        only_confirmed: bool = True):
        """
        导出为训练用 CSV 格式
        
        格式: Frame, X, Y, Visibility, EventType, IsEvent
        """
        trajectory = data_manager.trajectory
        events = data_manager.events
        
        if only_confirmed:
            events = [e for e in events if e.get('confirmed', False)]
        
        # 创建事件帧集合
        event_frames = {e['frame']: e for e in events}
        
        rows = []
        for i, frame_idx in enumerate(trajectory['frame']):
            event = event_frames.get(frame_idx)
            rows.append({
                'Frame': int(frame_idx),
                'X': trajectory['x'][i],
                'Y': trajectory['y'][i],
                'Visibility': int(trajectory['visibility'][i]),
                'EventType': event['event_type'] if event else 'none',
                'IsEvent': 1 if event else 0
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported training CSV to {output_path}")
        return output_path
    
    @staticmethod
    def to_event_list(data_manager: LabelingDataManager,
                      output_path: str,
                      only_confirmed: bool = True):
        """
        导出为事件列表格式 (仅事件帧)
        
        格式: Frame, X, Y, EventType, Confidence
        """
        events = data_manager.events
        
        if only_confirmed:
            events = [e for e in events if e.get('confirmed', False)]
        
        rows = [{
            'Frame': e['frame'],
            'X': e.get('x', 0),
            'Y': e.get('y', 0),
            'EventType': e['event_type'],
            'Confidence': e.get('confidence', 1.0)
        } for e in events]
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported event list to {output_path}")
        return output_path


def find_matching_video(csv_path: str, video_dirs: List[str] = None) -> Optional[str]:
    """
    根据 CSV 路径查找对应的视频文件
    
    Args:
        csv_path: CSV 文件路径 (如 data/test/match1/csv/1_05_02_ball.csv)
        video_dirs: 可选的视频目录列表
    
    Returns:
        视频文件路径 或 None
    """
    csv_path = Path(csv_path)
    
    # 从 CSV 文件名推断视频名
    # 1_05_02_ball.csv -> 1_05_02.mp4
    video_name = csv_path.stem.replace('_ball', '') + '.mp4'
    
    # 候选目录
    candidates = []
    
    # 1. 同级 video 目录
    video_dir = csv_path.parent.parent / 'video'
    candidates.append(video_dir / video_name)
    
    # 2. 用户指定的目录
    if video_dirs:
        for vd in video_dirs:
            candidates.append(Path(vd) / video_name)
    
    # 查找存在的文件
    for cand in candidates:
        if cand.exists():
            return str(cand)
    
    return None


def get_match_csv_files(csv_path: str) -> List[str]:
    """
    获取当前match目录下的所有CSV文件列表（已排序）
    
    Args:
        csv_path: 当前CSV文件路径
    
    Returns:
        CSV文件路径列表
    """
    csv_path = Path(csv_path)
    
    # 查找CSV目录
    csv_dir = csv_path.parent
    if not csv_dir.exists():
        return [str(csv_path)]
    
    # 获取所有 *_ball.csv 文件并排序
    csv_files = sorted(csv_dir.glob('*_ball.csv'))
    return [str(f) for f in csv_files]


def get_previous_csv(csv_path: str) -> Optional[str]:
    """
    获取当前match中的上一个CSV文件
    
    Args:
        csv_path: 当前CSV文件路径
    
    Returns:
        上一个CSV文件路径，如果是第一个则返回None
    """
    all_csvs = get_match_csv_files(csv_path)
    current_path = str(Path(csv_path).resolve())
    
    try:
        current_idx = [str(Path(f).resolve()) for f in all_csvs].index(current_path)
        if current_idx > 0:
            return all_csvs[current_idx - 1]
    except (ValueError, IndexError):
        pass
    
    return None


def get_next_csv(csv_path: str) -> Optional[str]:
    """
    获取当前match中的下一个CSV文件
    
    Args:
        csv_path: 当前CSV文件路径
    
    Returns:
        下一个CSV文件路径，如果是最后一个则返回None
    """
    all_csvs = get_match_csv_files(csv_path)
    current_path = str(Path(csv_path).resolve())
    
    try:
        current_idx = [str(Path(f).resolve()) for f in all_csvs].index(current_path)
        if current_idx < len(all_csvs) - 1:
            return all_csvs[current_idx + 1]
    except (ValueError, IndexError):
        pass
    
    return None


def get_csv_index_info(csv_path: str) -> Tuple[int, int, str, str]:
    """
    获取当前CSV在match中的位置信息
    
    Args:
        csv_path: 当前CSV文件路径
    
    Returns:
        (当前索引, 总数, match名称, 视频名称)
    """
    csv_path = Path(csv_path)
    all_csvs = get_match_csv_files(str(csv_path))
    
    # 获取match名称（从路径中提取）
    match_name = "Unknown"
    for part in csv_path.parts:
        if part.startswith('match'):
            match_name = part
            break
    
    # 获取视频名称
    video_name = csv_path.stem.replace('_ball', '')
    
    # 获取当前索引
    current_path = str(csv_path.resolve())
    try:
        current_idx = [str(Path(f).resolve()) for f in all_csvs].index(current_path)
        return (current_idx + 1, len(all_csvs), match_name, video_name)
    except (ValueError, IndexError):
        return (1, 1, match_name, video_name)
