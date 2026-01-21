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
    羽毛球事件检测候选生成器
    
    检测两种事件:
    - 落地点 (landing): 球落地后停止/消失
      特征: 速度急剧下降 → 接近静止或消失
    - 击球点 (hit): 球被球拍击中
      特征: 速度方向突然反转，但速度大小保持较高
    
    注意: 不依赖于画面位置，因为落点可能出现在场地任何位置
    """
    
    def __init__(self,
                 # 落地点检测参数
                 speed_drop_ratio: float = 0.3,          # 速度下降到原来的比例以下视为急剧下降
                 min_speed_before_landing: float = 8.0,  # 落地前最小速度（过滤静止球）
                 max_speed_after_landing: float = 5.0,   # 落地后最大速度（接近静止）
                 # 击球点检测参数
                 min_speed_at_hit: float = 5.0,          # 击球时最小速度
                 vy_reversal_threshold: float = 3.0,     # v_y 反转阈值
                 # 通用参数
                 min_visible_before: int = 3,            # 事件前最少可见帧数
                 merge_window: int = 3):                 # 合并相邻候选的窗口大小
        """
        Args:
            speed_drop_ratio: 速度下降比例阈值
            min_speed_before_landing: 落地前最小速度，过滤本来就静止的球
            max_speed_after_landing: 落地后最大速度，确认球确实停下来了
            min_speed_at_hit: 击球时最小速度
            vy_reversal_threshold: y方向速度反转阈值
            min_visible_before: 事件前最少可见帧数
            merge_window: 合并相邻候选的窗口大小
        """
        self.speed_drop_ratio = speed_drop_ratio
        self.min_speed_before_landing = min_speed_before_landing
        self.max_speed_after_landing = max_speed_after_landing
        self.min_speed_at_hit = min_speed_at_hit
        self.vy_reversal_threshold = vy_reversal_threshold
        self.min_visible_before = min_visible_before
        self.merge_window = merge_window
    
    def generate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 visibility: np.ndarray,
                 kinematics: Optional[Dict[str, np.ndarray]] = None,
                 img_height: int = 288) -> List[Dict]:
        """
        生成候选事件点（落地点 + 击球点）
        
        Args:
            x: x 坐标序列
            y: y 坐标序列
            visibility: 可见性序列
            kinematics: 预计算的运动学特征 (可选)
            img_height: 图像高度 (保留参数兼容性，但不再用于位置过滤)
        
        Returns:
            candidates: 候选事件列表，每个候选包含:
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
        
        candidates = []
        
        for t in range(self.min_visible_before, n - 1):
            # 检查当前帧是否可见
            if visibility[t] == 0:
                continue
            
            # 检查前面是否有足够的可见帧
            visible_count = np.sum(visibility[max(0, t-self.min_visible_before):t+1])
            if visible_count < self.min_visible_before:
                continue
            
            # ==================== 落地点检测规则 ====================
            
            # 规则1: 可见→不可见 + 之前速度较高
            # 球落地后消失（被捡起或静止不动超出检测范围）
            # 或者球飞出画面
            if visibility[t] == 1 and visibility[t+1] == 0:
                # 检查之前是否有运动（不是本来就静止的球）
                recent_speed = np.mean(speed[max(0, t-3):t+1])
                if recent_speed >= self.min_speed_before_landing:
                    # 判断是否是出画面（边缘位置）
                    # 假设图像尺寸约 512x288 或类似比例
                    is_edge = (x[t] < 20 or x[t] > img_height * 1.7 or 
                               y[t] < 20 or y[t] > img_height * 0.95)
                    
                    if is_edge:
                        # 可能是出画面
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'out_of_frame',
                            'rule': 'visibility_drop_edge',
                            'confidence': 0.50,
                            'features': {
                                'speed_before': float(recent_speed),
                                'v_y': float(v_y[t]),
                                'visible_before': int(visible_count)
                            }
                        })
                    else:
                        # 场内消失，更可能是落地
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'landing',
                            'rule': 'visibility_drop',
                            'confidence': 0.85,
                            'features': {
                                'speed_before': float(recent_speed),
                                'v_y': float(v_y[t]),
                                'visible_before': int(visible_count)
                            }
                        })
                    continue
            
            # 规则2: 速度急剧下降 + 之后保持低速或消失
            # 典型的落地特征：快速运动 → 突然减速 → 接近静止
            if t > 0 and t + 2 < n:
                speed_before = speed[t-1]
                speed_current = speed[t]
                speed_after = speed[t+1] if visibility[t+1] == 1 else 0
                
                if speed_before >= self.min_speed_before_landing:
                    speed_ratio = speed_current / (speed_before + 1e-6)
                    
                    # 速度急剧下降 且 之后保持低速
                    if (speed_ratio < self.speed_drop_ratio and 
                        speed_after <= self.max_speed_after_landing):
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'landing',
                            'rule': 'speed_drop',
                            'confidence': 0.75,
                            'features': {
                                'speed_ratio': float(speed_ratio),
                                'speed_before': float(speed_before),
                                'speed_current': float(speed_current),
                                'speed_after': float(speed_after)
                            }
                        })
                        continue
            
            # ==================== 击球点检测规则 ====================
            
            # 规则3: Y方向速度反转 + 速度保持较高
            # 击球特征：球被击中后方向反转，但速度不会降到很低
            if t > 0 and t + 1 < n and visibility[t+1] == 1:
                vy_before = v_y[t-1]
                vy_after = v_y[t]
                
                # 检测明显的 y 方向反转
                is_reversal = ((vy_before > self.vy_reversal_threshold and vy_after < -self.vy_reversal_threshold) or
                               (vy_before < -self.vy_reversal_threshold and vy_after > self.vy_reversal_threshold))
                
                if is_reversal:
                    # 击球后速度应该保持较高（与落地不同）
                    speed_after = speed[t+1] if t+1 < n else speed[t]
                    if speed_after >= self.min_speed_at_hit:
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'hit',
                            'rule': 'vy_reversal',
                            'confidence': 0.80,
                            'features': {
                                'vy_before': float(vy_before),
                                'vy_after': float(vy_after),
                                'speed_after': float(speed_after)
                            }
                        })
                        continue
            
            # 规则4: X方向速度反转 (水平击球)
            if t > 0 and t + 1 < n and visibility[t+1] == 1:
                vx_before = v_x[t-1]
                vx_after = v_x[t]
                
                # 检测明显的 x 方向反转
                vx_threshold = self.vy_reversal_threshold  # 使用相同阈值
                is_x_reversal = ((vx_before > vx_threshold and vx_after < -vx_threshold) or
                                 (vx_before < -vx_threshold and vx_after > vx_threshold))
                
                if is_x_reversal:
                    speed_after = speed[t+1] if t+1 < n else speed[t]
                    if speed_after >= self.min_speed_at_hit:
                        # 检查是否已经被 vy_reversal 检测到
                        if not any(c['frame'] == t for c in candidates):
                            candidates.append({
                                'frame': t,
                                'x': float(x[t]),
                                'y': float(y[t]),
                                'event_type': 'hit',
                                'rule': 'vx_reversal',
                                'confidence': 0.75,
                                'features': {
                                    'vx_before': float(vx_before),
                                    'vx_after': float(vx_after),
                                    'speed_after': float(speed_after)
                                }
                            })
                            continue
        
        # 检查轨迹末尾（回合结束时的落地）
        last_visible_idx = self._find_last_visible(visibility)
        if last_visible_idx is not None and last_visible_idx > self.min_visible_before:
            # 检查结束前是否有运动
            recent_speed = np.mean(speed[max(0, last_visible_idx-3):last_visible_idx+1])
            if recent_speed >= self.min_speed_before_landing * 0.5:  # 放宽条件
                # 避免与其他规则重复
                if not any(c['frame'] == last_visible_idx for c in candidates):
                    candidates.append({
                        'frame': last_visible_idx,
                        'x': float(x[last_visible_idx]),
                        'y': float(y[last_visible_idx]),
                        'event_type': 'landing',
                        'rule': 'trajectory_end',
                        'confidence': 0.70,
                        'features': {
                            'speed': float(speed[last_visible_idx]),
                            'recent_avg_speed': float(recent_speed)
                        }
                    })
        
        # 合并相邻候选
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
