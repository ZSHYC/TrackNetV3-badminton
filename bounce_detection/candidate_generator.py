"""
候选落点生成器
Bounce Candidate Generator for Badminton

使用规则方法检测可能的落点位置，作为后续分类网络的输入。

羽毛球落点特性:
1. 落地后几乎不反弹（与网球不同）
2. 落点表现为: 下落→停止/消失
3. 落点一般在场地底部区域
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .kinematics import KinematicsCalculator, compute_direction_change


class BounceCandidateGenerator:
    """
    羽毛球落点候选生成器
    
    羽毛球场景区分:
    - 落地点 (bounce/landing): 球触地后停止/消失，通常在场地边缘底部
    - 击球点 (hit point): 球拍击球导致方向反转，通常在场地中部
    
    检测规则:
    1. 可见→不可见转换点（球落地后静止或被捡起） - 落地
    2. 速度急剧下降点 - 可能是落地
    3. y方向速度反转点 - 更可能是击球
    4. 轨迹在底部区域结束 - 落地
    """
    
    def __init__(self,
                 speed_drop_ratio: float = 0.4,
                 min_y_ratio: float = 0.35,
                 landing_y_ratio: float = 0.6,  # 落地点必须在更底部
                 direction_change_th: float = 120,
                 min_visible_before: int = 3,
                 min_speed_before: float = 5.0,
                 merge_window: int = 3,
                 detect_hit_points: bool = True):  # 是否同时检测击球点
        """
        Args:
            speed_drop_ratio: 速度下降比例阈值，当 speed[t]/speed[t-1] < ratio 时触发
            min_y_ratio: 最小 y 坐标比例 (相对于画面高度)，过滤掉太高的位置
            landing_y_ratio: 落地点最小 y 坐标比例，必须在画面底部
            direction_change_th: 方向变化阈值 (度)，用于检测急转弯
            min_visible_before: 落点前最少可见帧数
            min_speed_before: 落点前最小平均速度（过滤静止球）
            merge_window: 合并相邻候选的窗口大小
            detect_hit_points: 是否同时检测击球点（用于完整事件分析）
        """
        self.speed_drop_ratio = speed_drop_ratio
        self.min_y_ratio = min_y_ratio
        self.landing_y_ratio = landing_y_ratio
        self.direction_change_th = np.radians(direction_change_th)
        self.min_visible_before = min_visible_before
        self.min_speed_before = min_speed_before
        self.merge_window = merge_window
        self.detect_hit_points = detect_hit_points
    
    def generate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 visibility: np.ndarray,
                 kinematics: Optional[Dict[str, np.ndarray]] = None,
                 img_height: int = 288,
                 only_landing: bool = True) -> List[Dict]:
        """
        生成候选落点
        
        Args:
            x: x 坐标序列
            y: y 坐标序列
            visibility: 可见性序列
            kinematics: 预计算的运动学特征 (可选，如果未提供会自动计算)
            img_height: 图像高度，用于计算相对位置
            only_landing: 如果为 True，只返回落地点；否则也返回击球点
        
        Returns:
            candidates: 候选落点列表，每个候选包含:
                - frame: 帧索引
                - x, y: 坐标
                - event_type: 'landing' (落地) 或 'hit' (击球)
                - rule: 触发的规则名称
                - confidence: 初始置信度
                - features: 附加特征信息
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        
        n = len(x)
        if n < self.min_visible_before + 2:
            return []
        
        # 计算运动学特征
        if kinematics is None:
            calculator = KinematicsCalculator()
            kinematics = calculator.compute(x, y, visibility)
        
        v_x = kinematics['v_x']
        v_y = kinematics['v_y']
        speed = kinematics['speed']
        direction = kinematics['direction']
        
        # 计算方向变化
        direction_change = compute_direction_change(direction)
        
        min_y = img_height * self.min_y_ratio
        landing_y = img_height * self.landing_y_ratio  # 落地点需要在更底部
        candidates = []
        
        for t in range(self.min_visible_before, n - 1):
            # 检查前面是否有足够的可见帧
            visible_count = np.sum(visibility[max(0, t-self.min_visible_before):t+1])
            if visible_count < self.min_visible_before:
                continue
            
            # 检查前面的平均速度（过滤静止球）
            recent_speed = speed[max(0, t-self.min_visible_before):t+1]
            if np.mean(recent_speed) < self.min_speed_before:
                continue
            
            # ==================== 落地点检测规则 ====================
            
            # 规则1: 可见→不可见 且 在下落中 且 位置在底部 (v_y > 0 表示向下)
            # 这是最可靠的落地点信号
            if visibility[t] == 1 and visibility[t+1] == 0 and v_y[t] > 0:
                if y[t] >= landing_y:
                    candidates.append({
                        'frame': t,
                        'x': float(x[t]),
                        'y': float(y[t]),
                        'event_type': 'landing',
                        'rule': 'visibility_drop_landing',
                        'confidence': 0.85,
                        'features': {
                            'v_y': float(v_y[t]),
                            'speed': float(speed[t]),
                            'visible_before': int(visible_count)
                        }
                    })
                    continue
                elif y[t] >= min_y:
                    # 位置较高的消失点，可能是出界
                    candidates.append({
                        'frame': t,
                        'x': float(x[t]),
                        'y': float(y[t]),
                        'event_type': 'landing',
                        'rule': 'visibility_drop_out',
                        'confidence': 0.6,
                        'features': {
                            'v_y': float(v_y[t]),
                            'speed': float(speed[t])
                        }
                    })
                    continue
            
            # 规则2: 速度急剧下降 + 在底部区域
            if t > 0 and speed[t-1] > self.min_speed_before and y[t] >= landing_y:
                speed_ratio = speed[t] / (speed[t-1] + 1e-6)
                if speed_ratio < self.speed_drop_ratio and v_y[t-1] > 0:
                    # 额外条件: 下一帧速度继续很低或不可见
                    if t+1 < n and (speed[t+1] < speed[t-1] * 0.5 or visibility[t+1] == 0):
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'landing',
                            'rule': 'speed_drop_landing',
                            'confidence': 0.7,
                            'features': {
                                'speed_ratio': float(speed_ratio),
                                'speed_before': float(speed[t-1]),
                                'speed_after': float(speed[t])
                            }
                        })
                        continue
            
            # ==================== 击球点检测规则 (可选) ====================
            
            if self.detect_hit_points and not only_landing:
                # 规则3: y方向速度从正变负 (反弹/击球)
                # 羽毛球中这通常是击球点，不是落地点
                if t > 0 and y[t] >= min_y:
                    if v_y[t-1] > 3 and v_y[t] < -3:  # 明显的反转
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'hit',
                            'rule': 'y_velocity_reversal',
                            'confidence': 0.75,
                            'features': {
                                'v_y_before': float(v_y[t-1]),
                                'v_y_after': float(v_y[t])
                            }
                        })
                        continue
                
                # 规则4: 轨迹方向急剧变化 (击球)
                if np.abs(direction_change[t]) > self.direction_change_th and y[t] >= min_y:
                    if speed[t] > self.min_speed_before:
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'hit',
                            'rule': 'direction_change',
                            'confidence': 0.5,
                            'features': {
                                'direction_change': float(np.degrees(direction_change[t]))
                            }
                        })
        
        # 检查轨迹末尾（球落地后停止跟踪）
        last_visible_idx = self._find_last_visible(visibility)
        if last_visible_idx is not None and last_visible_idx > self.min_visible_before:
            if y[last_visible_idx] >= landing_y:
                # 检查结束前是否在下落
                if v_y[last_visible_idx] > 0:
                    # 避免与其他规则重复
                    if not any(c['frame'] == last_visible_idx for c in candidates):
                        candidates.append({
                            'frame': last_visible_idx,
                            'x': float(x[last_visible_idx]),
                            'y': float(y[last_visible_idx]),
                            'event_type': 'landing',
                            'rule': 'trajectory_end',
                            'confidence': 0.75,
                            'features': {
                                'v_y': float(v_y[last_visible_idx]),
                                'speed': float(speed[last_visible_idx])
                            }
                        })
        
        # 合并相邻候选（分别处理 landing 和 hit）
        if only_landing:
            candidates = [c for c in candidates if c['event_type'] == 'landing']
        
        candidates = self._merge_nearby_candidates(candidates)
        
        # 按帧排序
        candidates.sort(key=lambda c: c['frame'])
        
        return candidates
    
    def _find_last_visible(self, visibility: np.ndarray) -> Optional[int]:
        """找到最后一个可见帧的索引"""
        visible_indices = np.where(visibility == 1)[0]
        if len(visible_indices) == 0:
            return None
        return int(visible_indices[-1])
    
    def _merge_nearby_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        合并相邻的候选点
        保留置信度最高的或规则优先级最高的
        """
        if len(candidates) <= 1:
            return candidates
        
        # 规则优先级
        rule_priority = {
            'y_velocity_reversal': 4,
            'visibility_drop': 3,
            'trajectory_end': 2,
            'speed_drop': 1,
            'direction_change': 0
        }
        
        # 按帧排序
        candidates.sort(key=lambda c: c['frame'])
        
        merged = []
        i = 0
        while i < len(candidates):
            group = [candidates[i]]
            j = i + 1
            
            # 收集在 merge_window 内的所有候选
            while j < len(candidates) and candidates[j]['frame'] - candidates[i]['frame'] <= self.merge_window:
                group.append(candidates[j])
                j += 1
            
            # 选择最佳候选
            best = max(group, key=lambda c: (
                rule_priority.get(c['rule'], 0),
                c['confidence']
            ))
            merged.append(best)
            
            i = j
        
        return merged
    
    def generate_from_csv(self, 
                          csv_path: str,
                          img_height: int = 288) -> List[Dict]:
        """
        从 CSV 文件读取轨迹并生成候选
        
        Args:
            csv_path: CSV 文件路径 (格式: Frame, Visibility, X, Y)
            img_height: 图像高度
        
        Returns:
            candidates: 候选落点列表
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='Frame').fillna(0)
        
        x = df['X'].values
        y = df['Y'].values
        visibility = df['Visibility'].values
        
        return self.generate(x, y, visibility, img_height=img_height)


class RallyBounceDetector:
    """
    基于回合的落点检测器
    
    一个回合（rally）中可能有多个落点，需要区分:
    1. 正常击球点（球拍击球）
    2. 落地点（球触地）
    """
    
    def __init__(self, candidate_generator: Optional[BounceCandidateGenerator] = None):
        self.generator = candidate_generator or BounceCandidateGenerator()
    
    def detect_in_rally(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        visibility: np.ndarray,
                        img_height: int = 288) -> Dict:
        """
        检测一个回合中的所有落点
        
        Returns:
            result: {
                'candidates': 候选列表,
                'rally_info': {
                    'total_frames': 总帧数,
                    'visible_frames': 可见帧数,
                    'trajectory_segments': 轨迹段数
                }
            }
        """
        candidates = self.generator.generate(x, y, visibility, img_height=img_height)
        
        # 计算回合信息
        visibility = np.asarray(visibility)
        segments = self._count_trajectory_segments(visibility)
        
        return {
            'candidates': candidates,
            'rally_info': {
                'total_frames': len(x),
                'visible_frames': int(np.sum(visibility)),
                'trajectory_segments': segments
            }
        }
    
    def _count_trajectory_segments(self, visibility: np.ndarray) -> int:
        """计算轨迹段数（连续可见的段）"""
        if len(visibility) == 0:
            return 0
        
        segments = 0
        in_segment = False
        
        for v in visibility:
            if v == 1 and not in_segment:
                segments += 1
                in_segment = True
            elif v == 0:
                in_segment = False
        
        return segments
