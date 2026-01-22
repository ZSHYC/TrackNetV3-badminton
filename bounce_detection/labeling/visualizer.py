"""
可视化组件
Visualizer for Labeling Tool

提供:
- 视频帧显示
- 轨迹绘制
- 事件标记
- 交互式 matplotlib 界面

优化版本 - 美观清晰的现代化界面
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.widgets import Button, RadioButtons
from typing import Dict, List, Optional, Tuple, Callable

# 设置负号显示
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 配色方案 ====================
# 现代化配色方案 - 高对比度 + 柔和色调

# 深色主题背景
THEME = {
    'bg_dark': '#1a1a2e',      # 深蓝黑色背景
    'bg_panel': '#16213e',     # 面板背景
    'bg_highlight': '#0f3460', # 高亮背景
    'text_primary': '#ffffff', # 主要文字
    'text_secondary': '#a0a0a0', # 次要文字
    'text_accent': '#e94560',  # 强调文字
    'border': '#3a3a5c',       # 边框
    'grid': '#2a2a4a',         # 网格线
}

# 事件类型的颜色和标记 - 明亮清晰
EVENT_COLORS = {
    'landing': '#FF6B6B',      # 珊瑚红 - 落地
    'hit': '#4ECDC4',          # 青绿色 - 击打
    'out_of_frame': '#95A5A6', # 灰色 - 出画
    'other': '#F39C12',        # 橙黄色 - 其他
    'none': '#BDC3C7'          # 浅灰色
}

# 事件类型的深色版本（用于边框）
EVENT_COLORS_DARK = {
    'landing': '#C0392B',      # 深红
    'hit': '#16A085',          # 深青
    'out_of_frame': '#7F8C8D', # 深灰
    'other': '#D68910',        # 深橙
    'none': '#95A5A6'
}

EVENT_MARKERS = {
    'landing': '*',   # 星形 - 醒目
    'hit': 'D',       # 菱形
    'out_of_frame': 'X', # X形
    'other': 's'      # 方形
}

# Event type display names
EVENT_NAMES = {
    'landing': 'Land',
    'hit': 'Hit',
    'out_of_frame': 'OOF',
    'other': 'Other'
}

# Event type symbols (ASCII-safe)
EVENT_SYMBOLS = {
    'landing': '*',
    'hit': '+',
    'out_of_frame': 'x',
    'other': 'o'
}


class FrameVisualizer:
    """
    单帧可视化器
    
    在视频帧上绘制:
    - 球的位置（带动态效果）
    - 轨迹线（渐变透明度）
    - 事件标记（清晰图标）
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
            # 添加半透明叠加层使标记更清晰
            overlay_alpha = 0.1
        else:
            # 没有视频帧时使用美观的深色背景
            self.ax.set_xlim(0, 512)
            self.ax.set_ylim(288, 0)  # Y轴翻转
            self.ax.set_facecolor(THEME['bg_dark'])
            # 添加网格
            self.ax.grid(True, alpha=0.1, color=THEME['grid'], linestyle='--')
            overlay_alpha = 0
        
        # 2. 绘制轨迹（带渐变效果）
        x_traj, y_traj, vis_traj = trajectory
        if len(x_traj) > 0:
            visible = vis_traj == 1
            x_vis, y_vis = x_traj[visible], y_traj[visible]
            
            if len(x_vis) > 1:
                # 轨迹线 - 使用渐变颜色
                for i in range(len(x_vis) - 1):
                    alpha = 0.3 + 0.5 * (i / len(x_vis))  # 从淡到浓
                    self.ax.plot([x_vis[i], x_vis[i+1]], [y_vis[i], y_vis[i+1]], 
                                color='#00FF88', alpha=alpha, linewidth=2)
                
                # 轨迹点 - 从小到大
                sizes = np.linspace(10, 40, len(x_vis))
                alphas = np.linspace(0.3, 0.8, len(x_vis))
                for i, (x, y, s, a) in enumerate(zip(x_vis, y_vis, sizes, alphas)):
                    self.ax.scatter(x, y, c='#00FF88', s=s, alpha=a, zorder=3, edgecolors='none')
        
        # 3. 绘制事件标记
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, EVENT_COLORS['other'])
            dark_color = EVENT_COLORS_DARK.get(event_type, color)
            marker = EVENT_MARKERS.get(event_type, 's')
            
            # 已确认的事件用实心，未确认的用空心
            is_confirmed = event.get('confirmed', False)
            facecolor = color if is_confirmed else 'none'
            edge_width = 2 if is_confirmed else 3
            
            # 绘制事件标记
            self.ax.scatter(event['x'], event['y'], 
                           c=facecolor, edgecolors=dark_color,
                           s=250, marker=marker, linewidths=edge_width, zorder=5)
            
            # 添加光晕效果
            if is_confirmed:
                self.ax.scatter(event['x'], event['y'], 
                               c=color, s=400, alpha=0.2, marker='o', zorder=4)
            
            # Label with frame number
            label_text = f"F{event['frame']}"
            self.ax.annotate(label_text, 
                            (event['x'], event['y']),
                            textcoords="offset points",
                            xytext=(12, -12),
                            fontsize=9,
                            fontweight='bold',
                            color=color,
                            bbox=dict(boxstyle='round,pad=0.2', 
                                     facecolor=THEME['bg_dark'], 
                                     edgecolor=color, alpha=0.8))
        
        # 4. 绘制当前帧的球（突出显示）
        x, y, vis = ball_pos
        if vis == 1:
            if current_event:
                # 当前帧有事件 - 使用事件颜色
                event_type = current_event.get('event_type', 'other')
                color = EVENT_COLORS.get(event_type, '#FFFF00')
                
                # 多层光晕效果
                self.ax.scatter(x, y, c=color, s=600, alpha=0.15, marker='o', zorder=8)
                self.ax.scatter(x, y, c=color, s=400, alpha=0.25, marker='o', zorder=9)
                self.ax.scatter(x, y, c='#FFFF00', edgecolors=color, 
                               s=200, marker='o', linewidths=3, zorder=10)
            else:
                # 普通帧 - 黄色球
                self.ax.scatter(x, y, c='#FFFF00', s=400, alpha=0.2, marker='o', zorder=8)
                self.ax.scatter(x, y, c='#FFFF00', edgecolors='#FF8C00', 
                               s=150, marker='o', linewidths=2, zorder=10)
        
        # 5. Title bar
        title_parts = [f"Frame {frame_idx}"]
        if current_event:
            event_type = current_event.get('event_type', 'unknown')
            type_name = EVENT_NAMES.get(event_type, event_type)
            symbol = EVENT_SYMBOLS.get(event_type, '?')
            confirmed = '[OK]' if current_event.get('confirmed', False) else '[?]'
            title_parts.append(f"{symbol} {type_name} {confirmed}")
        
        self.ax.set_title(' | '.join(title_parts), 
                         fontsize=13, fontweight='bold',
                         color=THEME['text_primary'],
                         pad=10)
        
        # 移除坐标轴刻度，保持简洁
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # 添加边框
        for spine in self.ax.spines.values():
            spine.set_color(THEME['border'])
            spine.set_linewidth(2)
        
        # 确保 Y 轴翻转（图像坐标系）
        if self.ax.get_ylim()[0] < self.ax.get_ylim()[1]:
            self.ax.invert_yaxis()


class TrajectoryPlot:
    """
    轨迹时序图
    
    显示 X/Y 坐标和速度随时间的变化
    使用现代化深色主题
    """
    
    def __init__(self, ax_y: plt.Axes, ax_speed: plt.Axes):
        self.ax_y = ax_y
        self.ax_speed = ax_speed
        self._setup_style()
    
    def _setup_style(self):
        """设置图表样式"""
        for ax in [self.ax_y, self.ax_speed]:
            ax.set_facecolor(THEME['bg_panel'])
            for spine in ax.spines.values():
                spine.set_color(THEME['border'])
    
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
        self._setup_style()
        
        visible = visibility == 1
        
        # ========== Y 坐标图 ==========
        # 背景区域高亮当前帧附近
        if len(frames) > 0:
            frame_range = frames[-1] - frames[0]
            if frame_range > 0:
                highlight_width = max(10, frame_range * 0.02)
                self.ax_y.axvspan(current_frame - highlight_width, 
                                 current_frame + highlight_width,
                                 facecolor=THEME['bg_highlight'], alpha=0.3)
        
        # Y 坐标线
        if np.any(visible):
            self.ax_y.plot(frames[visible], y[visible], 
                          color='#5DADE2', linewidth=2, label='Y坐标', alpha=0.9)
            self.ax_y.fill_between(frames[visible], y[visible], 
                                  alpha=0.1, color='#5DADE2')
        
        # 当前帧标记
        self.ax_y.axvline(x=current_frame, color='#E74C3C', linestyle='-', 
                         linewidth=2.5, label='当前帧', zorder=10)
        
        # 绘制事件标记
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            symbol = EVENT_SYMBOLS.get(event_type, '?')
            
            # 垂直线
            self.ax_y.axvline(x=event['frame'], color=color, 
                             linestyle='--', alpha=0.6, linewidth=1.5)
            
            # 事件点
            self.ax_y.scatter(event['frame'], event['y'], 
                             c=color, s=120, marker='*', zorder=5,
                             edgecolors='white', linewidths=0.5)
        
        self.ax_y.set_ylabel('Y', fontsize=10, color=THEME['text_primary'])
        self.ax_y.set_title('Y Coordinate', fontsize=11, fontweight='bold',
                           color=THEME['text_primary'], pad=8)
        self.ax_y.invert_yaxis()
        self.ax_y.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_y.tick_params(colors=THEME['text_secondary'], labelsize=8)
        
        # ========== 速度图 ==========
        # 背景区域高亮
        if len(frames) > 0:
            frame_range = frames[-1] - frames[0]
            if frame_range > 0:
                highlight_width = max(10, frame_range * 0.02)
                self.ax_speed.axvspan(current_frame - highlight_width, 
                                     current_frame + highlight_width,
                                     facecolor=THEME['bg_highlight'], alpha=0.3)
        
        # 速度线和填充
        self.ax_speed.plot(frames, speed, color='#2ECC71', linewidth=2, 
                          label='速度', alpha=0.9)
        self.ax_speed.fill_between(frames, 0, speed, alpha=0.2, color='#2ECC71')
        
        # 当前帧标记
        self.ax_speed.axvline(x=current_frame, color='#E74C3C', linestyle='-', 
                             linewidth=2.5, label='当前帧', zorder=10)
        
        # 绘制事件标记
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            self.ax_speed.axvline(x=event['frame'], color=color, 
                                 linestyle='--', alpha=0.6, linewidth=1.5)
        
        self.ax_speed.set_xlabel('Frame', fontsize=10, color=THEME['text_primary'])
        self.ax_speed.set_ylabel('Speed', fontsize=10, color=THEME['text_primary'])
        self.ax_speed.set_title('Speed', fontsize=11, fontweight='bold',
                               color=THEME['text_primary'], pad=8)
        self.ax_speed.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_speed.tick_params(colors=THEME['text_secondary'], labelsize=8)


class InfoPanel:
    """
    Info Panel
    
    Shows:
    - Current status
    - Statistics
    - Shortcut hints
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
        Draw info panel
        """
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_facecolor(THEME['bg_panel'])
        
        lines = []
        
        # ===== Frame Info =====
        progress = 100 * current_frame / max(1, total_frames - 1)
        progress_bar = self._make_progress_bar(progress, width=15)
        
        lines.append("+----------------------+")
        lines.append("|    POSITION          |")
        lines.append("+----------------------+")
        lines.append(f"| Frame: {current_frame:5d}/{total_frames-1:<5d} |")
        lines.append(f"| {progress_bar} {progress:5.1f}% |")
        lines.append("+----------------------+")
        lines.append("")
        
        # ===== Event Info =====
        lines.append("+----------------------+")
        lines.append("|    CURRENT EVENT     |")
        lines.append("+----------------------+")
        if current_event:
            event_type = current_event.get('event_type', 'N/A')
            type_name = EVENT_NAMES.get(event_type, event_type)
            symbol = EVENT_SYMBOLS.get(event_type, '?')
            confirmed = '[OK]' if current_event.get('confirmed') else '[?]'
            rule = current_event.get('rule', 'N/A')[:12]
            conf = current_event.get('confidence', 0)
            
            lines.append(f"| Type: {symbol} {type_name:<11} |")
            lines.append(f"| Status: {confirmed:<12} |")
            lines.append(f"| Rule: {rule:<14} |")
            lines.append(f"| Conf: {conf:.2f}            |")
            
            # 显示辅助规则（如果有）
            aux_rules = current_event.get('auxiliary_rules', [])
            if aux_rules:
                lines.append(f"| Boost: +{len(aux_rules)*0.05:.2f} ({len(aux_rules)} aux) |")
            else:
                lines.append("|                      |")
        else:
            lines.append("|   (No event here)    |")
            lines.append("|                      |")
            lines.append("|                      |")
            lines.append("|                      |")
            lines.append("|                      |")
        lines.append("+----------------------+")
        lines.append("")
        
        # ===== Statistics =====
        lines.append("+----------------------+")
        lines.append("|    STATISTICS        |")
        lines.append("+----------------------+")
        total = statistics.get('total_events', 0)
        confirmed = statistics.get('confirmed_events', 0)
        unconfirmed = statistics.get('unconfirmed_events', 0)
        lines.append(f"| Total: {total:<14} |")
        lines.append(f"| Confirmed: {confirmed:<10} |")
        lines.append(f"| Pending: {unconfirmed:<12} |")
        
        by_type = statistics.get('by_type', {})
        for etype, count in list(by_type.items())[:3]:
            symbol = EVENT_SYMBOLS.get(etype, '?')
            name = EVENT_NAMES.get(etype, etype)[:6]
            lines.append(f"|  {symbol} {name}: {count:<10} |")
        lines.append("+----------------------+")
        
        text = '\n'.join(lines)
        self.ax.text(0.02, 0.98, text, 
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='monospace',
                    verticalalignment='top',
                    color=THEME['text_primary'],
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=THEME['bg_panel'], 
                             edgecolor=THEME['border'],
                             alpha=0.95))
    
    def _make_progress_bar(self, percent: float, width: int = 10) -> str:
        """Create text progress bar"""
        filled = int(width * percent / 100)
        empty = width - filled
        return '#' * filled + '-' * empty


class EventListPanel:
    """
    Event List Panel
    
    Shows all events in a list view
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
    
    def draw(self, events: List[Dict], current_event_idx: int = -1):
        """
        Draw event list
        """
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_facecolor(THEME['bg_panel'])
        
        lines = []
        
        # Title
        lines.append("+----------------------+")
        lines.append("|    EVENT LIST        |")
        lines.append("+----------------------+")
        
        if not events:
            lines.append("|  (No events found)   |")
            lines.append("|                      |")
            lines.append("|  Tip: Use Phase1     |")
            lines.append("|  or add manually     |")
        else:
            # Show event list (max 15, centered on current)
            max_display = 15
            start_idx = max(0, current_event_idx - max_display // 2)
            end_idx = min(len(events), start_idx + max_display)
            
            if start_idx > 0:
                lines.append(f"|  ... ({start_idx} more above)  |")
            
            for i in range(start_idx, end_idx):
                event = events[i]
                
                # Current event highlight
                if i == current_event_idx:
                    prefix = ">"
                    suffix = "<"
                else:
                    prefix = " "
                    suffix = " "
                
                # Confirmed status
                confirmed = "Y" if event.get('confirmed', False) else "?"
                
                # Event type
                event_type = event.get('event_type', 'other')
                symbol = EVENT_SYMBOLS.get(event_type, '?')
                type_abbr = EVENT_NAMES.get(event_type, event_type)[:4]
                
                # Frame number
                frame = event.get('frame', 0)
                
                # Format line
                line = f"|{prefix}[{confirmed}] F{frame:04d} {symbol}{type_abbr}{suffix}|"
                lines.append(line)
            
            if end_idx < len(events):
                remaining = len(events) - end_idx
                lines.append(f"|  ... ({remaining} more below)  |")
        
        # Legend
        lines.append("+----------------------+")
        lines.append("|     -- Legend --     |")
        lines.append(f"| {EVENT_SYMBOLS['landing']}Land {EVENT_SYMBOLS['hit']}Hit {EVENT_SYMBOLS['out_of_frame']}OOF    |")
        lines.append("| Y=Confirmed ?=Pending|")
        lines.append("+----------------------+")
        
        text = '\n'.join(lines)
        self.ax.text(0.02, 0.98, text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='monospace',
                    verticalalignment='top',
                    color=THEME['text_primary'],
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=THEME['bg_panel'], 
                             edgecolor=THEME['border'],
                             alpha=0.95))


class ShortcutPanel:
    """
    Shortcut Panel
    
    Shows all available keyboard shortcuts
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
    
    def draw(self, highlight_key: Optional[str] = None):
        """
        Draw shortcut panel
        """
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_facecolor(THEME['bg_panel'])
        
        shortcuts = [
            ("SHORTCUTS", "", True),  # Title
            ("", "", False),  # Empty line
            ("Navigate", "", True),  # Category
            ("<- / A", "Prev frame", False),
            ("-> / D", "Next frame", False),
            ("Up / W", "Prev event", False),
            ("Down", "Next event", False),
            ("Home", "First frame", False),
            ("End", "Last frame", False),
            ("PgUp/Dn", "+/-30 frm", False),
            ("", "", False),
            ("Edit", "", True),
            ("Y", "Confirm", False),
            ("Delete", "Delete", False),
            ("L", "Set Land", False),
            ("H", "Set Hit", False),
            ("O", "Set OOF", False),
            ("N", "Add new", False),
            ("", "", False),
            ("System", "", True),
            ("Ctrl+S", "Save", False),
            ("Q / Esc", "Quit", False),
        ]
        
        lines = []
        lines.append("+----------------------+")
        
        for key, desc, is_header in shortcuts:
            if is_header and key:
                lines.append(f"|    {key:<18} |")
            elif is_header:
                lines.append("+----------------------+")
            elif key == "":
                lines.append("|                      |")
            else:
                # Highlight current key
                if highlight_key and key.lower().startswith(highlight_key.lower()):
                    lines.append(f"| >{key:<6} {desc:<12} |")
                else:
                    lines.append(f"|  {key:<6} {desc:<12} |")
        
        lines.append("+----------------------+")
        
        text = '\n'.join(lines)
        self.ax.text(0.02, 0.98, text,
                    transform=self.ax.transAxes,
                    fontsize=8,
                    fontfamily='monospace',
                    verticalalignment='top',
                    color=THEME['text_primary'],
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=THEME['bg_panel'], 
                             edgecolor=THEME['border'],
                             alpha=0.95))
