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
                 future_check_window: int = 5,           # 落地后检查“复活”的窗口帧数
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
            future_check_window: 落地检查"复活"的窗口帧数
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
        self.future_check_window = future_check_window
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
        
        使用混合方案：
        - 主规则（Primary）：决定事件类型，互斥
        - 辅助规则（Auxiliary）：提供额外证据，可叠加增强置信度
        
        Args:
            x: x 坐标序列
            y: y 坐标序列
            visibility: 可见性序列
            kinematics: 预计算的运动学特征 (可选)
            img_height: 图像高度
            img_width: 图像宽度
        
        Returns:
            candidates: 候选事件列表，每个候选包含:
                - frame: 帧索引
                - x, y: 坐标
                - event_type: 'landing' / 'hit' / 'out_of_frame'
                - rule: 主规则名称
                - confidence: 置信度（可被辅助规则增强）
                - all_rules: 所有触发的规则列表
                - auxiliary_rules: 辅助规则列表
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
        a_x = kinematics['a_x']
        a_y = kinematics['a_y']
        speed = kinematics['speed']
        
        # 预计算加速度大小
        acceleration = np.sqrt(a_x**2 + a_y**2)
        
        # 处理 NaN/Inf 值
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
            
            # ==================== 第一步：检测主规则（互斥）====================
            primary_candidate = None
            primary_features = {}
            
            # ---------- 击球类主规则（按优先级排序）----------
            
            # 主规则1: Y方向速度反转（最可靠的击球检测）
            if primary_candidate is None:
                result = self._check_vy_reversal(t, n, visibility, v_y, speed)
                if result:
                    primary_candidate = {
                        'event_type': 'hit',
                        'rule': 'vy_reversal',
                        'confidence': 0.85
                    }
                    primary_features = result
            
            # 主规则2: X方向速度反转（水平击球）
            if primary_candidate is None:
                result = self._check_vx_reversal(t, n, visibility, v_x, speed)
                if result:
                    primary_candidate = {
                        'event_type': 'hit',
                        'rule': 'vx_reversal',
                        'confidence': 0.80
                    }
                    primary_features = result
            
            # 主规则3: 加速度突变
            if primary_candidate is None:
                result = self._check_acceleration_peak(t, n, visibility, acceleration, speed)
                if result:
                    primary_candidate = {
                        'event_type': 'hit',
                        'rule': 'acceleration_peak',
                        'confidence': 0.75
                    }
                    primary_features = result
            
            # ---------- 落地类主规则 ----------
            
            # 主规则4: 可见→不可见（落地或出界）
            if primary_candidate is None:
                result = self._check_visibility_drop(t, n, x, y, visibility, speed, v_y, 
                                                     img_height, img_width, visible_count)
                if result:
                    primary_candidate = {
                        'event_type': result['event_type'],
                        'rule': result['rule'],
                        'confidence': result['confidence']
                    }
                    primary_features = result['features']
            
            # 主规则5: 速度急剧下降
            if primary_candidate is None:
                # 显式传入 y 和 v_y 进行空间和方向约束
                result = self._check_speed_drop(t, n, visibility, speed, y=y, v_y=v_y, img_height=img_height)
                if result:
                    primary_candidate = {
                        'event_type': 'landing',
                        'rule': 'speed_drop',
                        'confidence': 0.75
                    }
                    primary_features = result
            
            # ==================== 第二步：检测辅助规则（可叠加）====================
            auxiliary_rules = []
            auxiliary_features = {}
            confidence_boost = 0.0
            
            # 辅助规则1: Y坐标局部极大值
            aux_result = self._check_y_local_max(t, n, y, visibility)
            if aux_result:
                auxiliary_rules.append('y_local_max')
                auxiliary_features['y_local_max'] = aux_result
                # 对 hit 和 landing 均增强置信度
                # 对于 landing，这是非常强的特征（落地往往是Y坐标最大值/最低点）
                if primary_candidate:
                    if primary_candidate['event_type'] == 'landing':
                        confidence_boost += 0.10  # 落地时Y极大值是非常强的证据
                    elif primary_candidate['event_type'] == 'hit':
                        confidence_boost += 0.05  # 击球也可能在低点（挑球）

            # 辅助规则2: 速度局部极大值
            aux_result = self._check_speed_local_max(t, n, speed, visibility)
            if aux_result:
                auxiliary_rules.append('speed_local_max')
                auxiliary_features['speed_local_max'] = aux_result
                # 仅对 hit 类事件增强置信度
                if primary_candidate and primary_candidate['event_type'] == 'hit':
                    confidence_boost += 0.05
            
            # ==================== 第三步：生成候选 ====================
            if primary_candidate:
                # 应用置信度增益（上限 0.95）
                final_confidence = min(0.95, primary_candidate['confidence'] + confidence_boost)
                
                # 合并特征
                all_features = {**primary_features, **auxiliary_features}
                
                candidate = {
                    'frame': t,
                    'x': float(x[t]),
                    'y': float(y[t]),
                    'event_type': primary_candidate['event_type'],
                    'rule': primary_candidate['rule'],
                    'confidence': final_confidence,
                    'all_rules': [primary_candidate['rule']] + auxiliary_rules,
                    'auxiliary_rules': auxiliary_rules,
                    'features': all_features
                }
                candidates.append(candidate)
            
            # 特殊情况：只有辅助规则触发（没有主规则）
            elif auxiliary_rules:
                # 辅助规则单独触发时，作为低置信度候选
                main_aux_rule = auxiliary_rules[0]
                main_aux_features = auxiliary_features.get(main_aux_rule, {})
                
                candidate = {
                    'frame': t,
                    'x': float(x[t]),
                    'y': float(y[t]),
                    'event_type': 'hit',  # 辅助规则默认为 hit
                    'rule': main_aux_rule,
                    'confidence': 0.50,  # 低置信度
                    'all_rules': auxiliary_rules,
                    'auxiliary_rules': auxiliary_rules[1:] if len(auxiliary_rules) > 1 else [],
                    'features': auxiliary_features
                }
                candidates.append(candidate)
        
        # 检查轨迹末尾（回合结束时的落地）
        self._check_trajectory_end(x, y, visibility, speed, candidates)
        
        # 合并相邻候选
        candidates = self._merge_nearby_candidates(candidates)
        
        # 按帧排序
        candidates.sort(key=lambda c: c['frame'])
        
        return candidates
    
    # ==================== 主规则检测方法 ====================
    
    def _check_vy_reversal(self, t: int, n: int, visibility: np.ndarray,
                           v_y: np.ndarray, speed: np.ndarray) -> Optional[Dict]:
        """主规则: Y方向速度反转检测"""
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t-1] != 1 or visibility[t+1] != 1:
            return None
        
        vy_before = v_y[t-1]
        vy_after = v_y[t+1]
        
        # 检测反转
        is_reversal = ((vy_before > self.vy_reversal_threshold and vy_after < -self.vy_reversal_threshold) or
                       (vy_before < -self.vy_reversal_threshold and vy_after > self.vy_reversal_threshold))
        
        if is_reversal:
            speed_after = speed[t+1]
            if speed_after >= self.min_speed_at_hit:
                return {
                    'vy_before': float(vy_before),
                    'vy_after': float(vy_after),
                    'speed_after': float(speed_after),
                    'reversal_magnitude': float(abs(vy_before - vy_after))
                }
        return None
    
    def _check_vx_reversal(self, t: int, n: int, visibility: np.ndarray,
                           v_x: np.ndarray, speed: np.ndarray) -> Optional[Dict]:
        """主规则: X方向速度反转检测"""
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t-1] != 1 or visibility[t+1] != 1:
            return None
        
        vx_before = v_x[t-1]
        vx_after = v_x[t+1]
        
        is_reversal = ((vx_before > self.vy_reversal_threshold and vx_after < -self.vy_reversal_threshold) or
                       (vx_before < -self.vy_reversal_threshold and vx_after > self.vy_reversal_threshold))
        
        if is_reversal:
            speed_after = speed[t+1]
            if speed_after >= self.min_speed_at_hit:
                return {
                    'vx_before': float(vx_before),
                    'vx_after': float(vx_after),
                    'speed_after': float(speed_after),
                    'reversal_magnitude': float(abs(vx_before - vx_after))
                }
        return None
    
    def _check_acceleration_peak(self, t: int, n: int, visibility: np.ndarray,
                                  acceleration: np.ndarray, speed: np.ndarray) -> Optional[Dict]:
        """主规则: 加速度突变检测"""
        if t <= 1 or t + 1 >= n:
            return None
        if visibility[t-1] != 1 or visibility[t] != 1 or visibility[t+1] != 1:
            return None
        
        acc_before = acceleration[t-1]
        acc_current = acceleration[t]
        acc_after = acceleration[t+1]
        
        is_acc_peak = (acc_current > acc_before and 
                       acc_current > acc_after and 
                       acc_current > self.acc_threshold)
        
        if is_acc_peak:
            return {
                'acc_before': float(acc_before),
                'acc_current': float(acc_current),
                'acc_after': float(acc_after),
                'speed': float(speed[t]),
                'acc_ratio': float(acc_current / (max(acc_before, acc_after) + 1e-6))
            }
        return None
    
    def _check_visibility_drop(self, t: int, n: int, x: np.ndarray, y: np.ndarray,
                               visibility: np.ndarray, speed: np.ndarray, v_y: np.ndarray,
                               img_height: int, img_width: int, visible_count: int) -> Optional[Dict]:
        """主规则: 可见性消失检测（落地或出界）"""
        if t + 1 >= n:
            return None
        if visibility[t] != 1 or visibility[t+1] != 0:
            return None
        
        recent_speed = np.mean(speed[max(0, t-3):t+1])
        if recent_speed < self.min_speed_before_landing:
            return None
            
        # 物理约束：落地必须是向下运动 (v_y > 0)
        # 如果 v_y < 0 (向上运动) 但消失了，可能是飞出画面上方或被遮挡，不应判为落地
        if v_y[t] < -1.0: # 给一点宽容度防止噪声
            # 向上消失，更有可能是 out_of_frame (上方) 或 漏检
            # 这里保守起见，不作为 landing 返回，或者作为 out_of_frame
            return {
                'event_type': 'out_of_frame',
                'rule': 'visibility_drop_upward', # 新增规则类型标识
                'confidence': 0.60,
                'features': {
                    'speed_before': float(recent_speed),
                    'v_y': float(v_y[t]),
                    'visible_before': int(visible_count)
                }
            }
        
        # 判断是否在边缘
        is_edge = (x[t] < self.edge_margin or x[t] > img_width - self.edge_margin or
                   y[t] < self.edge_margin or y[t] > img_height - self.edge_margin)
        edge_distance = float(min(x[t], img_width - x[t], y[t], img_height - y[t]))
        
        features = {
            'speed_before': float(recent_speed),
            'v_y': float(v_y[t]),
            'visible_before': int(visible_count),
            'edge_distance': edge_distance
        }
        
        if is_edge:
            return {
                'event_type': 'out_of_frame',
                'rule': 'visibility_drop_edge',
                'confidence': 0.50,
                'features': features
            }
        
        # 新增规则：最小落地高度约束
        # 如果消失点在画面顶部 15% 区域，这绝不可能是落地，而是高空球出界/丢失
        if y[t] < img_height * 0.15:
            return {
                'event_type': 'out_of_frame',
                'rule': 'visibility_drop_high_altitude', # 新增规则：高空丢失
                'confidence': 0.60,
                'features': features
            }
        
        return {
            'event_type': 'landing',
            'rule': 'visibility_drop',
            'confidence': 0.85,
            'features': features
        }
    
    def _check_speed_drop(self, t: int, n: int, visibility: np.ndarray,
                          speed: np.ndarray,
                          y: Optional[np.ndarray] = None, 
                          v_y: Optional[np.ndarray] = None,
                          img_height: int = 288) -> Optional[Dict]:
        """主规则: 速度急剧下降检测"""
        if t <= 0 or t + 2 >= n:
            return None
        
        speed_before = speed[t-1]
        speed_current = speed[t]
        speed_after = speed[t+1] if visibility[t+1] == 1 else 0
        
        if speed_before < self.min_speed_before_landing:
            return None
            
        # 改进 1: 空间高度约束 (Apex Protection)
        # 如果 Y 坐标在画面上部 (前 30%)，这极有可能是高远球的最高点，而不是落地
        # 落地应当发生在地面（通常在画面下方）
        if y is not None:
            if y[t] < img_height * 0.30:
                return None

        # 改进 2: 垂直方向约束
        # 落地必须不处于上升状态。允许极小的负值容错，但不能明显向上。
        if v_y is not None:
            if v_y[t] < -2.0: 
                return None
        
        speed_ratio = speed_current / (speed_before + 1e-6)
        
        if speed_ratio < self.speed_drop_ratio and speed_after <= self.max_speed_after_landing:
            # 新增规则：Future Activity Check (检查是否有"复活")
            # 真正的落地后，球应该保持静止。如果在未来几帧内速度又变大了，说明这帧只是暂时的减速（如网前晃动）
            check_end = min(n, t + self.future_check_window + 1)
            if check_end > t + 1:
                future_max_speed = np.max(speed[t+1 : check_end] * visibility[t+1 : check_end])
                # 如果未来几帧有任何一帧速度超过击球阈值或落地前速度的一半，则认为它复活了
                resurrection_threshold = 10.0 # 像素/帧，约为慢速球的速度
                if future_max_speed > resurrection_threshold:
                    return None

            return {
                'speed_ratio': float(speed_ratio),
                'speed_before': float(speed_before),
                'speed_current': float(speed_current),
                'speed_after': float(speed_after)
            }
        return None
    
    # ==================== 辅助规则检测方法 ====================
    
    def _check_y_local_max(self, t: int, n: int, y: np.ndarray,
                           visibility: np.ndarray) -> Optional[Dict]:
        """辅助规则: Y坐标局部极大值（球最低点）"""
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t-1] != 1 or visibility[t] != 1 or visibility[t+1] != 1:
            return None
        
        is_y_maximum = (y[t] > y[t-1] and y[t] > y[t+1])
        
        if is_y_maximum:
            height_diff = min(y[t] - y[t-1], y[t] - y[t+1])
            if height_diff > self.min_height_diff:
                return {
                    'y_before': float(y[t-1]),
                    'y_current': float(y[t]),
                    'y_after': float(y[t+1]),
                    'height_diff': float(height_diff)
                }
        return None
    
    def _check_speed_local_max(self, t: int, n: int, speed: np.ndarray,
                               visibility: np.ndarray) -> Optional[Dict]:
        """辅助规则: 速度局部极大值"""
        if t <= 0 or t + 1 >= n:
            return None
        if visibility[t-1] != 1 or visibility[t] != 1 or visibility[t+1] != 1:
            return None
        
        is_speed_maximum = (speed[t] > speed[t-1] and speed[t] > speed[t+1])
        
        if is_speed_maximum:
            speed_diff = min(speed[t] - speed[t-1], speed[t] - speed[t+1])
            if speed[t] > self.min_peak_speed and speed_diff > self.min_speed_diff:
                return {
                    'speed_before': float(speed[t-1]),
                    'speed_current': float(speed[t]),
                    'speed_after': float(speed[t+1]),
                    'speed_diff': float(speed_diff)
                }
        return None
    
    def _check_trajectory_end(self, x: np.ndarray, y: np.ndarray,
                              visibility: np.ndarray, speed: np.ndarray,
                              candidates: List[Dict]) -> None:
        """检查轨迹末尾（回合结束时的落地）"""
        last_visible_idx = self._find_last_visible(visibility)
        
        if last_visible_idx is None or last_visible_idx <= self.min_visible_before:
            return
        
        # 检查是否已有候选
        if any(c['frame'] == last_visible_idx for c in candidates):
            return
        
        recent_speed = np.mean(speed[max(0, last_visible_idx-3):last_visible_idx+1])
        if recent_speed >= self.min_speed_before_landing * 0.5:
            candidates.append({
                'frame': last_visible_idx,
                'x': float(x[last_visible_idx]),
                'y': float(y[last_visible_idx]),
                'event_type': 'landing',
                'rule': 'trajectory_end',
                'confidence': 0.70,
                'all_rules': ['trajectory_end'],
                'auxiliary_rules': [],
                'features': {
                    'speed': float(speed[last_visible_idx]),
                    'recent_avg_speed': float(recent_speed)
                }
            })
    
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
