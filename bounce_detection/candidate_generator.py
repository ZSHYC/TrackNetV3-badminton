"""
候选落点生成器
Bounce Candidate Generator for Badminton

使用规则方法检测可能的落点和击球点，作为后续分类网络的输入。

"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .kinematics import KinematicsCalculator, compute_direction_change


class BounceCandidateGenerator:
    """
    羽毛球事件检测候选生成器
    
    检测两种事件:
        Landing落点特征：
        ├── 速度下降 → 接近静止
        ├── 之后消失或不再运动
        └── 回合结束

        Hit 击球点特征：
        ├── 速度方向反转
        ├── 之后速度保持较高（继续飞行）
        └── 加速度突变
    
    注意: 不依赖于画面位置，因为落点可能出现在场地任何位置
    """
    
    def __init__(self,
                 # 落地点检测参数
                 speed_drop_ratio: float = 0.3,          # 速度下降到原来的比例以下视为急剧下降
                 min_speed_before_landing: float = 8.0,  # 落地前最小速度（过滤静止球）
                 max_speed_after_landing: float = 5.0,   # 落地后最大速度（接近静止）
                 # 击球点检测参数
                 min_speed_at_hit: float = 3.0,          # 击球时最小速度
                 vy_reversal_threshold: float = 1.0,     # v_y 反转阈值
                 acc_threshold: float = 3.0,             # 加速度阈值（像素/帧²）
                 # 局部极值检测参数
                 min_height_diff: float = 5.0,           # Y极值最小高度差（像素）
                 min_peak_speed: float = 10.0,           # 速度极值最小峰值
                 min_speed_diff: float = 2.0,            # 速度极值最小差值
                 # 通用参数
                 min_visible_before: int = 3,            # 事件前最少可见帧数
                 merge_window: int = 3,                  # 合并相邻候选的窗口大小
                 edge_margin: int = 20):                 # 边缘判定距离（像素）
        """
        Args:
            speed_drop_ratio: 速度下降比例阈值
            min_speed_before_landing: 落地前最小速度，过滤本来就静止的球
            max_speed_after_landing: 落地后最大速度，确认球确实停下来了
            min_speed_at_hit: 击球时最小速度
            vy_reversal_threshold: y方向速度反转阈值
            acc_threshold: 加速度阈值（像素/帧²）
            min_height_diff: Y极值最小高度差（像素）
            min_peak_speed: 速度极值最小峰值
            min_speed_diff: 速度极值最小差值
            min_visible_before: 事件前最少可见帧数
            merge_window: 合并相邻候选的窗口大小
            edge_margin: 边缘判定距离（像素）
        """
        self.speed_drop_ratio = speed_drop_ratio
        self.min_speed_before_landing = min_speed_before_landing
        self.max_speed_after_landing = max_speed_after_landing
        self.min_speed_at_hit = min_speed_at_hit
        self.vy_reversal_threshold = vy_reversal_threshold
        self.acc_threshold = acc_threshold
        self.min_height_diff = min_height_diff
        self.min_peak_speed = min_peak_speed
        self.min_speed_diff = min_speed_diff
        self.min_visible_before = min_visible_before
        self.merge_window = merge_window
        self.edge_margin = edge_margin
    
    def generate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 visibility: np.ndarray,
                 kinematics: Optional[Dict[str, np.ndarray]] = None,
                 img_height: int = 288,
                 img_width: int = 512) -> List[Dict]:
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
        a_x = kinematics['a_x']  # 加速度特征
        a_y = kinematics['a_y']
        speed = kinematics['speed']
        
        # 预计算加速度大小，用于击球检测
        acceleration = np.sqrt(a_x**2 + a_y**2)
        
        # 处理 NaN/Inf 值，避免计算异常
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)
        v_x = np.nan_to_num(v_x, nan=0.0, posinf=0.0, neginf=0.0)
        v_y = np.nan_to_num(v_y, nan=0.0, posinf=0.0, neginf=0.0)
        
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
                    is_edge = (x[t] < self.edge_margin or x[t] > img_width - self.edge_margin or 
                               y[t] < self.edge_margin or y[t] > img_height - self.edge_margin)
                    edge_distance = min(x[t], img_width - x[t], y[t], img_height - y[t])
                    
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
                                'visible_before': int(visible_count),
                                'edge_distance': float(edge_distance)
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
            # 注意：检测 t 帧前后的速度变化，击球发生在 t 帧
            if t > 0 and t + 1 < n and visibility[t-1] == 1 and visibility[t+1] == 1:
                # 使用 t 帧前后的速度来检测反转
                vy_before = v_y[t-1]  # t-1 到 t 的速度
                vy_after = v_y[t+1]   # t+1 帧的速度
                
                # 检测 t 帧处的 Y 方向反转（前后速度符号相反）
                is_reversal = ((vy_before > self.vy_reversal_threshold and vy_after < -self.vy_reversal_threshold) or
                               (vy_before < -self.vy_reversal_threshold and vy_after > self.vy_reversal_threshold))
                
                if is_reversal:
                    # 击球后速度应该保持较高（与落地不同）
                    speed_after = speed[t+1]
                    if speed_after >= self.min_speed_at_hit:
                        candidates.append({
                            'frame': t,
                            'x': float(x[t]),
                            'y': float(y[t]),
                            'event_type': 'hit',
                            'rule': 'vy_reversal',
                            'confidence': 0.85,
                            'features': {
                                'vy_before': float(vy_before),
                                'vy_after': float(vy_after),
                                'speed_after': float(speed_after),
                                'reversal_magnitude': float(abs(vy_before - vy_after))
                            }
                        })
                        continue
            
            # 规则4: X方向速度反转 (水平击球)
            if t > 0 and t + 1 < n and visibility[t-1] == 1 and visibility[t+1] == 1:
                vx_before = v_x[t-1]
                vx_after = v_x[t+1]
                
                # 检测 t 帧处的 X 方向反转（使用与Y相同的阈值）
                is_x_reversal = ((vx_before > self.vy_reversal_threshold and vx_after < -self.vy_reversal_threshold) or
                                 (vx_before < -self.vy_reversal_threshold and vx_after > self.vy_reversal_threshold))
                
                if is_x_reversal:
                    speed_after = speed[t+1]
                    if speed_after >= self.min_speed_at_hit:
                        # 检查是否已经被 vy_reversal 检测到
                        if not any(c['frame'] == t for c in candidates):
                            candidates.append({
                                'frame': t,
                                'x': float(x[t]),
                                'y': float(y[t]),
                                'event_type': 'hit',
                                'rule': 'vx_reversal',
                                'confidence': 0.80,
                                'features': {
                                    'vx_before': float(vx_before),
                                    'vx_after': float(vx_after),
                                    'speed_after': float(speed_after),
                                    'reversal_magnitude': float(abs(vx_before - vx_after))
                                }
                            })
                            continue
            
            # 规则5: 加速度突变（击球瞬间加速度会有明显突变）
            # 这是比较可靠的击球检测特征
            if t > 1 and t + 1 < n:
                if visibility[t-1] == 1 and visibility[t] == 1 and visibility[t+1] == 1:
                    acc_current = acceleration[t]
                    acc_before = acceleration[t-1]
                    acc_after = acceleration[t+1]
                    
                    # 加速度局部极大值，且足够显著
                    is_acc_peak = (acc_current > acc_before and acc_current > acc_after and 
                                   acc_current > self.acc_threshold)
                    
                    if is_acc_peak:
                        # 避免重复
                        if not any(c['frame'] == t for c in candidates):
                            candidates.append({
                                'frame': t,
                                'x': float(x[t]),
                                'y': float(y[t]),
                                'event_type': 'hit',
                                'rule': 'acceleration_peak',
                                'confidence': 0.75,
                                'features': {
                                    'acc_before': float(acc_before),
                                    'acc_current': float(acc_current),
                                    'acc_after': float(acc_after),
                                    'speed': float(speed[t]),
                                    'acc_ratio': float(acc_current / (max(acc_before, acc_after) + 1e-6))
                                }
                            })
                            continue
            
            # 规则6: Y坐标局部极大值 (球到达最低点)
            # 图像坐标系: Y向下为正，所以 Y 极大值 = 球的最低点
            # 这可能是低手击球的位置
            if t > 0 and t + 1 < n:
                if visibility[t-1] == 1 and visibility[t] == 1 and visibility[t+1] == 1:
                    is_y_maximum = (y[t] > y[t-1] and y[t] > y[t+1])
                    
                    # 确保不是噪声（需要有一定的高度差）
                    height_diff = min(y[t] - y[t-1], y[t] - y[t+1])
                    
                    # 使用可配置的高度差阈值
                    if is_y_maximum and height_diff > self.min_height_diff:
                        # 避免重复
                        if not any(c['frame'] == t for c in candidates):
                            candidates.append({
                                'frame': t,
                                'x': float(x[t]),
                                'y': float(y[t]),
                                'event_type': 'hit',
                                'rule': 'y_local_max',
                                'confidence': 0.60,
                                'features': {
                                    'y_before': float(y[t-1]),
                                    'y_current': float(y[t]),
                                    'y_after': float(y[t+1]),
                                    'height_diff': float(height_diff)
                                }
                            })
                            continue
            
            # 规则7: 速度局部极大值 (击球瞬间速度达到峰值)
            if t > 0 and t + 1 < n:
                if visibility[t-1] == 1 and visibility[t] == 1 and visibility[t+1] == 1:
                    # 检查是否是局部极大值
                    is_speed_maximum = (speed[t] > speed[t-1] and speed[t] > speed[t+1])
                    
                    # 确保速度足够高，不是噪声（使用可配置阈值）
                    speed_diff = min(speed[t] - speed[t-1], speed[t] - speed[t+1])
                    
                    if is_speed_maximum and speed[t] > self.min_peak_speed and speed_diff > self.min_speed_diff:
                        # 避免重复
                        if not any(c['frame'] == t for c in candidates):
                            candidates.append({
                                'frame': t,
                                'x': float(x[t]),
                                'y': float(y[t]),
                                'event_type': 'hit',
                                'rule': 'speed_local_max',
                                'confidence': 0.60,
                                'features': {
                                    'speed_before': float(speed[t-1]),
                                    'speed_current': float(speed[t]),
                                    'speed_after': float(speed[t+1]),
                                    'speed_diff': float(speed_diff)
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
        
        # 规则优先级 (数值越大优先级越高)
        # 优先级设计原则：
        # - 速度反转规则最可靠（击球必然导致方向改变）
        # - 加速度峰值次之（击球瞬间有明显加速度变化）
        # - 可见性变化中等（可能是落地或出界）
        # - 局部极值规则最低（可能是正常轨迹的一部分）
        rule_priority = {
            'vy_reversal': 7,           # Y速度反转 - 最可靠的击球检测
            'vx_reversal': 6,           # X速度反转
            'acceleration_peak': 5,     # 加速度峰值
            'visibility_drop': 4,       # 可见性消失 - 落地
            'speed_drop': 4,            # 速度骤降 - 落地
            'visibility_drop_edge': 3,  # 边缘消失 - 出界
            'trajectory_end': 3,        # 轨迹结束 - 落地
            'y_local_max': 2,           # Y坐标局部极大值
            'speed_local_max': 2,       # 速度局部极大值
            'direction_change': 0       # 方向变化（保留扩展用）
        }
        
        # 按帧排序
        candidates.sort(key=lambda c: c['frame'])
        
        merged = []
        i = 0
        while i < len(candidates):
            group = [candidates[i]]
            base_frame = candidates[i]['frame']
            j = i + 1
            
            # 收集在 merge_window 内的所有候选（使用滑动窗口）
            while j < len(candidates) and candidates[j]['frame'] - base_frame <= self.merge_window:
                group.append(candidates[j])
                j += 1
            
            # 选择最佳候选（优先级 > 置信度）
            best = max(group, key=lambda c: (
                rule_priority.get(c['rule'], 0),
                c['confidence']
            ))
            merged.append(best)
            
            # 跳过所有已合并的候选
            i = j
        
        return merged
    
    def generate_from_csv(self, 
                          csv_path: str,
                          img_height: int = 288,
                          img_width: int = 512) -> List[Dict]:
        """
        从 CSV 文件读取轨迹并生成候选
        
        Args:
            csv_path: CSV 文件路径 (格式: Frame, Visibility, X, Y)
            img_height: 图像高度
            img_width: 图像宽度
        
        Returns:
            candidates: 候选落点列表
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='Frame').fillna(0)
        
        x = df['X'].values
        y = df['Y'].values
        visibility = df['Visibility'].values
        
        return self.generate(x, y, visibility, img_height=img_height, img_width=img_width)


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
                        img_height: int = 288,
                        img_width: int = 512) -> Dict:
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
        candidates = self.generator.generate(x, y, visibility, 
                                            img_height=img_height, img_width=img_width)
        
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
