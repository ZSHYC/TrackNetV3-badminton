"""
可视化组件
Visualizer for Labeling Tool

提供:
- 视频帧显示
- 轨迹绘制
- 事件标记
- 交互式 matplotlib 界面
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, RadioButtons
from typing import Dict, List, Optional, Tuple, Callable


# 事件类型的颜色和标记
EVENT_COLORS = {
    'landing': '#FF4444',      # 红色
    'hit': '#4444FF',          # 蓝色
    'out_of_frame': '#888888', # 灰色
    'other': '#44FF44',        # 绿色
    'none': '#CCCCCC'          # 浅灰色
}

EVENT_MARKERS = {
    'landing': '*',
    'hit': 'o',
    'out_of_frame': 'x',
    'other': 's'
}


class FrameVisualizer:
    """
    单帧可视化器
    
    在视频帧上绘制:
    - 球的位置
    - 轨迹线
    - 事件标记
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.frame_image = None
        self.trajectory_line = None
        self.ball_marker = None
        self.event_markers = []
        
    def draw_frame(self, 
                   frame: Optional[np.ndarray],
                   frame_idx: int,
                   ball_pos: Tuple[float, float, int],
                   trajectory: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   events: List[Dict],
                   current_event: Optional[Dict] = None):
        """
        绘制帧
        
        Args:
            frame: 视频帧图像 (RGB)
            frame_idx: 帧索引
            ball_pos: (x, y, visibility) 当前帧球的位置
            trajectory: (x_array, y_array, vis_array) 轨迹窗口
            events: 当前窗口内的事件列表
            current_event: 当前帧的事件（如果有）
        """
        self.ax.clear()
        
        # 1. 绘制视频帧或空白背景
        if frame is not None:
            self.ax.imshow(frame)
        else:
            # 没有视频帧时使用空白背景
            self.ax.set_xlim(0, 512)
            self.ax.set_ylim(288, 0)  # Y轴翻转
            self.ax.set_facecolor('#1a1a2e')
        
        # 2. 绘制轨迹
        x_traj, y_traj, vis_traj = trajectory
        if len(x_traj) > 0:
            # 可见点
            visible = vis_traj == 1
            self.ax.plot(x_traj[visible], y_traj[visible], 
                        'g-', alpha=0.5, linewidth=1.5, label='Trajectory')
            self.ax.scatter(x_traj[visible], y_traj[visible], 
                           c='lime', s=15, alpha=0.6, zorder=3)
        
        # 3. 绘制事件标记
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, EVENT_COLORS['other'])
            marker = EVENT_MARKERS.get(event_type, 's')
            
            # 已确认的事件用实心，未确认的用空心
            is_confirmed = event.get('confirmed', False)
            facecolor = color if is_confirmed else 'none'
            
            self.ax.scatter(event['x'], event['y'], 
                           c=facecolor, edgecolors=color,
                           s=200, marker=marker, linewidths=2, zorder=5)
            
            # 标注帧号
            self.ax.annotate(f"F{event['frame']}", 
                            (event['x'], event['y']),
                            textcoords="offset points",
                            xytext=(10, 10),
                            fontsize=8,
                            color=color)
        
        # 4. 绘制当前帧的球
        x, y, vis = ball_pos
        if vis == 1:
            if current_event:
                # 当前帧有事件
                event_type = current_event.get('event_type', 'other')
                color = EVENT_COLORS.get(event_type, 'yellow')
                self.ax.scatter(x, y, c='yellow', edgecolors=color, 
                               s=300, marker='o', linewidths=3, zorder=10)
            else:
                # 普通帧
                self.ax.scatter(x, y, c='yellow', edgecolors='orange', 
                               s=200, marker='o', linewidths=2, zorder=10)
        
        # 5. 标题和信息
        title = f"Frame {frame_idx}"
        if current_event:
            event_type = current_event.get('event_type', 'unknown')
            confirmed = '✓' if current_event.get('confirmed', False) else '?'
            title += f" | Event: {event_type} [{confirmed}]"
        
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # 确保 Y 轴翻转（图像坐标系）
        if self.ax.get_ylim()[0] < self.ax.get_ylim()[1]:
            self.ax.invert_yaxis()


class TrajectoryPlot:
    """
    轨迹时序图
    
    显示 X/Y 坐标和速度随时间的变化
    """
    
    def __init__(self, ax_y: plt.Axes, ax_speed: plt.Axes):
        self.ax_y = ax_y
        self.ax_speed = ax_speed
    
    def draw(self,
             frames: np.ndarray,
             x: np.ndarray,
             y: np.ndarray,
             visibility: np.ndarray,
             speed: np.ndarray,
             current_frame: int,
             events: List[Dict]):
        """
        绘制轨迹时序图
        """
        self.ax_y.clear()
        self.ax_speed.clear()
        
        visible = visibility == 1
        
        # 1. Y 坐标图
        self.ax_y.plot(frames[visible], y[visible], 'b-', linewidth=1, label='Y')
        self.ax_y.axvline(x=current_frame, color='red', linestyle='--', 
                         linewidth=2, label='Current')
        
        # 绘制事件
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            self.ax_y.axvline(x=event['frame'], color=color, 
                             linestyle=':', alpha=0.7, linewidth=1.5)
            self.ax_y.scatter(event['frame'], event['y'], 
                             c=color, s=80, marker='*', zorder=5)
        
        self.ax_y.set_ylabel('Y')
        self.ax_y.set_title('Y Coordinate', fontsize=10)
        self.ax_y.invert_yaxis()
        self.ax_y.grid(True, alpha=0.3)
        
        # 2. 速度图
        self.ax_speed.plot(frames, speed, 'g-', linewidth=1, label='Speed')
        self.ax_speed.fill_between(frames, 0, speed, alpha=0.3, color='green')
        self.ax_speed.axvline(x=current_frame, color='red', linestyle='--', 
                             linewidth=2, label='Current')
        
        # 绘制事件
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            self.ax_speed.axvline(x=event['frame'], color=color, 
                                 linestyle=':', alpha=0.7, linewidth=1.5)
        
        self.ax_speed.set_xlabel('Frame')
        self.ax_speed.set_ylabel('Speed')
        self.ax_speed.set_title('Speed', fontsize=10)
        self.ax_speed.grid(True, alpha=0.3)


class InfoPanel:
    """
    信息面板
    
    显示:
    - 当前状态
    - 统计信息
    - 快捷键提示
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.ax.axis('off')
    
    def draw(self, 
             current_frame: int,
             total_frames: int,
             current_event: Optional[Dict],
             statistics: Dict,
             mode: str = 'navigate'):
        """
        绘制信息面板
        """
        self.ax.clear()
        self.ax.axis('off')
        
        lines = []
        
        # 基本信息
        lines.append(f"━━━ Frame Info ━━━")
        lines.append(f"Frame: {current_frame} / {total_frames - 1}")
        lines.append(f"Progress: {100 * current_frame / max(1, total_frames-1):.1f}%")
        lines.append("")
        
        # 当前事件
        lines.append(f"━━━ Current Event ━━━")
        if current_event:
            lines.append(f"Type: {current_event.get('event_type', 'N/A')}")
            lines.append(f"Confirmed: {'Yes' if current_event.get('confirmed') else 'No'}")
            lines.append(f"Rule: {current_event.get('rule', 'N/A')}")
            lines.append(f"Conf: {current_event.get('confidence', 0):.2f}")
        else:
            lines.append("No event at this frame")
        lines.append("")
        
        # 统计
        lines.append(f"━━━ Statistics ━━━")
        lines.append(f"Total events: {statistics.get('total_events', 0)}")
        lines.append(f"Confirmed: {statistics.get('confirmed_events', 0)}")
        lines.append(f"Unconfirmed: {statistics.get('unconfirmed_events', 0)}")
        
        by_type = statistics.get('by_type', {})
        for event_type, count in by_type.items():
            lines.append(f"  {event_type}: {count}")
        lines.append("")
        
        # 快捷键
        lines.append(f"━━━ Shortcuts ━━━")
        lines.append("← → : Prev/Next frame")
        lines.append("↑ ↓ : Prev/Next event")
        lines.append("Y   : Confirm event")
        lines.append("D   : Delete event")
        lines.append("L   : Set as Landing")
        lines.append("H   : Set as Hit")
        lines.append("O   : Set as Out-of-frame")
        lines.append("A   : Add event here")
        lines.append("S   : Save labels")
        lines.append("Q   : Quit")
        
        text = '\n'.join(lines)
        self.ax.text(0.05, 0.95, text, 
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


class EventListPanel:
    """
    事件列表面板
    
    显示所有事件的列表视图
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
    
    def draw(self, events: List[Dict], current_event_idx: int = -1):
        """
        绘制事件列表
        """
        self.ax.clear()
        self.ax.axis('off')
        
        lines = ["━━━ Event List ━━━", ""]
        
        if not events:
            lines.append("No events detected")
        else:
            for i, event in enumerate(events):
                prefix = "▶ " if i == current_event_idx else "  "
                confirmed = "✓" if event.get('confirmed', False) else "○"
                event_type = event.get('event_type', '?')[:3].upper()
                frame = event.get('frame', 0)
                
                # 颜色标记（用符号代替）
                type_symbol = {'landing': '●', 'hit': '◆', 'out_of_frame': '○'}.get(
                    event.get('event_type', ''), '?')
                
                line = f"{prefix}[{confirmed}] F{frame:4d} {type_symbol} {event_type}"
                lines.append(line)
        
        text = '\n'.join(lines)
        self.ax.text(0.05, 0.95, text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
