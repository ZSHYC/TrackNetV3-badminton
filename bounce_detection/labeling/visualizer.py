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

# 设置负号显示与全局字体
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10


# ==================== 配色方案 ====================
# 现代化配色方案 - 高对比度 + 柔和色调

# 深色主题背景
THEME = {
    'bg_dark': '#141824',      # 深蓝黑背景
    'bg_panel': '#1B2233',     # 面板背景
    'bg_highlight': '#22304A', # 高亮背景
    'text_primary': '#F2F4F8', # 主要文字
    'text_secondary': '#A8B0C0', # 次要文字
    'text_accent': '#FF6B6B',  # 强调文字
    'border': '#2D374D',       # 边框
    'grid': '#273047',         # 网格线
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
    - 轨迹线（渐变效果）
    - 事件标记（简洁清晰）
    """
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        
    def draw_frame(self, 
                   frame: Optional[np.ndarray],
                   frame_idx: int,
                   ball_pos: Tuple[float, float, int],
                   trajectory: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   events: List[Dict],
                   current_event: Optional[Dict] = None):
        """
        绘制帧
        """
        self.ax.clear()
        
        # 1. 绘制视频帧或空白背景
        if frame is not None:
            self.ax.imshow(frame)
        else:
            self.ax.set_xlim(0, 512)
            self.ax.set_ylim(288, 0)
            self.ax.set_facecolor(THEME['bg_dark'])
            self.ax.grid(True, alpha=0.1, color=THEME['grid'], linestyle='-', linewidth=0.5)
        
        # 2. 绘制轨迹（渐变效果，从淡到浓）
        x_traj, y_traj, vis_traj = trajectory
        if len(x_traj) > 0:
            visible = vis_traj == 1
            x_vis, y_vis = x_traj[visible], y_traj[visible]
            
            if len(x_vis) > 1:
                n_points = len(x_vis)
                # 渐变轨迹点 - 从小到大，从淡到浓
                sizes = np.linspace(8, 35, n_points)
                alphas = np.linspace(0.2, 0.7, n_points)
                
                for i in range(n_points):
                    self.ax.scatter(x_vis[i], y_vis[i], 
                                   c='#00FF88', s=sizes[i], alpha=alphas[i], 
                                   zorder=3, edgecolors='none')
                
                # 轨迹连线（细线，半透明）
                self.ax.plot(x_vis, y_vis, color='#00FF88', alpha=0.4, 
                            linewidth=1.5, linestyle='-', zorder=2)
        
        # 3. 绘制事件标记（只显示当前帧附近的，不显示标签避免遮挡）
        # 只显示当前帧的事件，其他事件只用小标记
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, EVENT_COLORS['other'])
            dark_color = EVENT_COLORS_DARK.get(event_type, color)
            marker = EVENT_MARKERS.get(event_type, 's')
            
            is_confirmed = event.get('confirmed', False)
            is_current = (event['frame'] == frame_idx)
            
            if is_current:
                # 当前帧的事件 - 大且醒目
                facecolor = color if is_confirmed else 'none'
                self.ax.scatter(event['x'], event['y'], 
                               c=facecolor, edgecolors=dark_color,
                               s=100, marker=marker, linewidths=2.5, zorder=6)
                if is_confirmed:
                    self.ax.scatter(event['x'], event['y'], 
                                   c=color, s=150, alpha=0.2, marker='o', zorder=5)
            else:
                # 其他事件 - 小标记，无标签
                facecolor = color if is_confirmed else 'none'
                self.ax.scatter(event['x'], event['y'], 
                               c=facecolor, edgecolors=dark_color,
                               s=35, marker=marker, linewidths=1.2, zorder=4, alpha=0.7)
        
        # 4. 绘制当前帧的球
        x, y, vis = ball_pos
        if vis == 1:
            if current_event:
                event_type = current_event.get('event_type', 'other')
                color = EVENT_COLORS.get(event_type, '#FFFF00')
                self.ax.scatter(x, y, c=color, s=140, alpha=0.2, marker='o', zorder=8)
                self.ax.scatter(x, y, c='#FFFF00', edgecolors=color, 
                               s=55, marker='o', linewidths=2.5, zorder=10)
            else:
                self.ax.scatter(x, y, c='#FFFF00', s=90, alpha=0.2, marker='o', zorder=8)
                self.ax.scatter(x, y, c='#FFFF00', edgecolors='#FF8C00', 
                               s=38, marker='o', linewidths=1.8, zorder=10)
        
        # 5. 标题栏
        title_parts = [f"Frame {frame_idx}"]
        if vis == 1:
            title_parts.append(f"({x:.0f}, {y:.0f})")
        if current_event:
            event_type = current_event.get('event_type', 'unknown')
            type_name = EVENT_NAMES.get(event_type, event_type)
            confirmed = '✓' if current_event.get('confirmed', False) else '?'
            title_parts.append(f"{type_name} [{confirmed}]")
        
        self.ax.set_title(' | '.join(title_parts), 
                         fontsize=11, fontweight='bold',
                         color=THEME['text_primary'], pad=6)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        for spine in self.ax.spines.values():
            spine.set_color(THEME['border'])
            spine.set_linewidth(1.5)
        
        if self.ax.get_ylim()[0] < self.ax.get_ylim()[1]:
            self.ax.invert_yaxis()


class TrajectoryPlot:
    """
    轨迹时序图（4个子图）
    
    显示 X坐标、Y坐标、速度、加速度 随时间的变化
    使用现代化深色主题，支持事件标记和当前帧高亮
    """
    
    def __init__(self, ax_x: plt.Axes, ax_y: plt.Axes, 
                 ax_speed: plt.Axes, ax_acc: plt.Axes):
        self.ax_x = ax_x
        self.ax_y = ax_y
        self.ax_speed = ax_speed
        self.ax_acc = ax_acc
        self._setup_style()
    
    def _setup_style(self):
        """设置图表样式"""
        for ax in [self.ax_x, self.ax_y, self.ax_speed, self.ax_acc]:
            ax.set_facecolor(THEME['bg_panel'])
            for spine in ax.spines.values():
                spine.set_color(THEME['border'])
    
    def _draw_current_frame_highlight(self, ax: plt.Axes, current_frame: int, 
                                      frames: np.ndarray):
        """绘制当前帧高亮区域"""
        if len(frames) > 0:
            frame_range = frames[-1] - frames[0]
            if frame_range > 0:
                highlight_width = max(5, frame_range * 0.015)
                ax.axvspan(current_frame - highlight_width, 
                          current_frame + highlight_width,
                          facecolor=THEME['bg_highlight'], alpha=0.4)
    
    def _draw_event_markers(self, ax: plt.Axes, events: List[Dict], 
                            data_y: Optional[np.ndarray] = None,
                            frames: Optional[np.ndarray] = None):
        """绘制事件标记（垂直线 + 可选的数据点）"""
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            
            # 垂直虚线
            ax.axvline(x=event['frame'], color=color, 
                      linestyle='--', alpha=0.35, linewidth=1.0)
            
            # 如果提供了数据，在数据线上标记点
            if data_y is not None and frames is not None:
                idx = np.where(frames == event['frame'])[0]
                if len(idx) > 0:
                    ax.scatter(event['frame'], data_y[idx[0]], 
                              c=color, s=55, marker='*', zorder=5,
                              edgecolors='white', linewidths=0.4)
    
    def draw(self,
             frames: np.ndarray,
             x: np.ndarray,
             y: np.ndarray,
             visibility: np.ndarray,
             speed: np.ndarray,
             acceleration: np.ndarray,
             current_frame: int,
             events: List[Dict]):
        """
        绘制4个轨迹时序图
        """
        # 清理并重置样式
        for ax in [self.ax_x, self.ax_y, self.ax_speed, self.ax_acc]:
            ax.clear()
        self._setup_style()
        
        visible = visibility == 1
        
        # ========== X 坐标图 ==========
        self._draw_current_frame_highlight(self.ax_x, current_frame, frames)
        
        if np.any(visible):
            self.ax_x.plot(frames[visible], x[visible], 
                          color='#E74C3C', linewidth=1.8, alpha=0.9)
            self.ax_x.fill_between(frames[visible], x[visible], 
                                  alpha=0.1, color='#E74C3C')
        
        # 当前帧竖线
        self.ax_x.axvline(x=current_frame, color='#FFFFFF', linestyle='-', 
                         linewidth=2, alpha=0.8, zorder=10)
        
        self._draw_event_markers(self.ax_x, events, x, frames)
        
        self.ax_x.set_ylabel('X', fontsize=9, color=THEME['text_primary'])
        self.ax_x.set_title('X Coordinate', fontsize=10, fontweight='bold',
                           color=THEME['text_primary'], pad=5)
        self.ax_x.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_x.tick_params(colors=THEME['text_secondary'], labelsize=7)
        self.ax_x.set_xticklabels([])  # 隐藏X轴标签（与下面的图共享）
        
        # ========== Y 坐标图 ==========
        self._draw_current_frame_highlight(self.ax_y, current_frame, frames)
        
        if np.any(visible):
            self.ax_y.plot(frames[visible], y[visible], 
                          color='#5DADE2', linewidth=1.8, alpha=0.9)
            self.ax_y.fill_between(frames[visible], y[visible], 
                                  alpha=0.1, color='#5DADE2')
        
        self.ax_y.axvline(x=current_frame, color='#FFFFFF', linestyle='-', 
                         linewidth=2, alpha=0.8, zorder=10)
        
        # 绘制事件标记（Y图上显示Y位置）
        for event in events:
            event_type = event.get('event_type', 'other')
            color = EVENT_COLORS.get(event_type, 'gray')
            self.ax_y.axvline(x=event['frame'], color=color, 
                             linestyle='--', alpha=0.5, linewidth=1.2)
            self.ax_y.scatter(event['frame'], event['y'], 
                             c=color, s=55, marker='*', zorder=5,
                             edgecolors='white', linewidths=0.4)
        
        self.ax_y.set_ylabel('Y', fontsize=9, color=THEME['text_primary'])
        self.ax_y.set_title('Y Coordinate', fontsize=10, fontweight='bold',
                           color=THEME['text_primary'], pad=5)
        self.ax_y.invert_yaxis()  # Y轴翻转（图像坐标系）
        self.ax_y.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_y.tick_params(colors=THEME['text_secondary'], labelsize=7)
        self.ax_y.set_xticklabels([])
        
        # ========== 速度图 ==========
        self._draw_current_frame_highlight(self.ax_speed, current_frame, frames)
        
        # 速度线和填充
        self.ax_speed.plot(frames, speed, color='#2ECC71', linewidth=1.8, alpha=0.9)
        self.ax_speed.fill_between(frames, 0, speed, alpha=0.15, color='#2ECC71')
        
        self.ax_speed.axvline(x=current_frame, color='#FFFFFF', linestyle='-', 
                             linewidth=2, alpha=0.8, zorder=10)
        
        self._draw_event_markers(self.ax_speed, events, speed, frames)
        
        self.ax_speed.set_xlabel('Frame', fontsize=9, color=THEME['text_primary'])
        self.ax_speed.set_ylabel('Speed', fontsize=9, color=THEME['text_primary'])
        self.ax_speed.set_title('Speed (px/frame)', fontsize=10, fontweight='bold',
                               color=THEME['text_primary'], pad=5)
        self.ax_speed.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_speed.tick_params(colors=THEME['text_secondary'], labelsize=7)
        self.ax_speed.set_ylim(bottom=0)  # 速度从0开始
        
        # ========== 加速度图 ==========
        self._draw_current_frame_highlight(self.ax_acc, current_frame, frames)
        
        # 处理加速度数据
        acc_safe = np.nan_to_num(acceleration, nan=0, posinf=0, neginf=0)
        
        # 使用百分位数确定合理的显示范围（去除极端值）
        if len(acc_safe) > 0 and np.any(acc_safe != 0):
            p5, p95 = np.percentile(acc_safe, [5, 95])
            # 确保范围有意义
            acc_range = max(abs(p5), abs(p95), 1.0)
            # 稍微扩展范围
            y_limit = acc_range * 1.3
        else:
            y_limit = 10.0
        
        # 绘制加速度曲线
        self.ax_acc.plot(frames, acc_safe, color='#F39C12', linewidth=1.8, alpha=0.9)
        self.ax_acc.fill_between(frames, 0, acc_safe, 
                                where=(acc_safe >= 0), alpha=0.15, color='#F39C12')
        self.ax_acc.fill_between(frames, 0, acc_safe, 
                                where=(acc_safe < 0), alpha=0.15, color='#9B59B6')
        
        # 零线
        self.ax_acc.axhline(y=0, color=THEME['text_secondary'], linestyle='-', 
                           linewidth=0.8, alpha=0.5)
        
        self.ax_acc.axvline(x=current_frame, color='#FFFFFF', linestyle='-', 
                           linewidth=2, alpha=0.8, zorder=10)
        
        self._draw_event_markers(self.ax_acc, events, acc_safe, frames)
        
        self.ax_acc.set_xlabel('Frame', fontsize=9, color=THEME['text_primary'])
        self.ax_acc.set_ylabel('Acc', fontsize=9, color=THEME['text_primary'])
        self.ax_acc.set_title('Acceleration (px/frame²)', fontsize=10, fontweight='bold',
                             color=THEME['text_primary'], pad=5)
        self.ax_acc.grid(True, alpha=0.2, color=THEME['grid'], linestyle='--')
        self.ax_acc.tick_params(colors=THEME['text_secondary'], labelsize=7)
        # 动态设置Y轴范围
        self.ax_acc.set_ylim(-y_limit, y_limit)


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
             match_info: Dict = None,
             mode: str = 'navigate'):
        """
        Draw info panel
        
        Args:
            match_info: {'match_name': str, 'video_name': str, 'video_idx': int, 'video_total': int}
        """
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_facecolor(THEME['bg_panel'])

        # Card background
        self.ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                         boxstyle='round,pad=0.015,rounding_size=0.02',
                                         transform=self.ax.transAxes,
                                         facecolor=THEME['bg_panel'],
                                         edgecolor=THEME['border'],
                                         linewidth=1.2, alpha=0.95, zorder=0))
        # Header bar
        self.ax.add_patch(Rectangle((0.01, 0.93), 0.98, 0.05,
                                    transform=self.ax.transAxes,
                                    facecolor=THEME['bg_highlight'],
                                    edgecolor='none', alpha=0.9, zorder=1))
        self.ax.text(0.03, 0.955, "INFO",
                     transform=self.ax.transAxes,
                     fontsize=11, fontweight='bold',
                     color=THEME['text_primary'], va='center')
        
        lines = []
        
        # ===== Match/Video Info =====
        if match_info:
            lines.append("MATCH/VIDEO")
            lines.append(f"Match: {match_info.get('match_name', 'N/A')}")
            lines.append(f"Video: {match_info.get('video_name', 'N/A')}")
            video_idx = match_info.get('video_idx', 1)
            video_total = match_info.get('video_total', 1)
            lines.append(f"[{video_idx}/{video_total}]")
            lines.append("")
        
        # ===== Frame Info =====
        progress = 100 * current_frame / max(1, total_frames - 1)
        progress_bar = self._make_progress_bar(progress, width=15)
        
        lines.append("POSITION")
        lines.append(f"Frame: {current_frame}/{total_frames-1}")
        lines.append(f"{progress_bar} {progress:5.1f}%")
        lines.append("")
        
        # ===== Event Info =====
        lines.append("CURRENT EVENT")
        if current_event:
            event_type = current_event.get('event_type', 'N/A')
            type_name = EVENT_NAMES.get(event_type, event_type)
            symbol = EVENT_SYMBOLS.get(event_type, '?')
            confirmed = '[OK]' if current_event.get('confirmed') else '[?]'
            rule = current_event.get('rule', 'N/A')[:12]
            conf = current_event.get('confidence', 0)
            
            lines.append(f"Type: {symbol} {type_name}")
            lines.append(f"Status: {confirmed}")
            lines.append(f"Rule: {rule}")
            lines.append(f"Conf: {conf:.2f}")
            
            # 显示辅助规则（如果有）
            aux_rules = current_event.get('auxiliary_rules', [])
            if aux_rules:
                aux_count = len(aux_rules)
                aux_text = ",".join(aux_rules[:2])
                if aux_count > 2:
                    aux_text += f"+{aux_count-2}"
                lines.append(f"Aux: {aux_count} {aux_text}")
            else:
                lines.append("Aux: -")
        else:
            lines.append("(No event here)")
        lines.append("")
        
        # ===== Statistics =====
        lines.append("STATISTICS")
        total = statistics.get('total_events', 0)
        confirmed = statistics.get('confirmed_events', 0)
        unconfirmed = statistics.get('unconfirmed_events', 0)
        lines.append(f"Total: {total}")
        lines.append(f"Confirmed: {confirmed}")
        lines.append(f"Pending: {unconfirmed}")
        
        by_type = statistics.get('by_type', {})
        for etype, count in list(by_type.items())[:3]:
            symbol = EVENT_SYMBOLS.get(etype, '?')
            name = EVENT_NAMES.get(etype, etype)[:6]
            lines.append(f"{symbol} {name}: {count}")
        
        text = '\n'.join(lines)
        self.ax.text(0.03, 0.90, text, 
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='DejaVu Sans Mono',
                    verticalalignment='top',
                    color=THEME['text_primary'])
    
    def _make_progress_bar(self, percent: float, width: int = 10) -> str:
        """Create text progress bar"""
        filled = int(width * percent / 100)
        empty = width - filled
        return '█' * filled + '░' * empty


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

        # Card background
        self.ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                         boxstyle='round,pad=0.015,rounding_size=0.02',
                                         transform=self.ax.transAxes,
                                         facecolor=THEME['bg_panel'],
                                         edgecolor=THEME['border'],
                                         linewidth=1.2, alpha=0.95, zorder=0))
        # Header bar
        self.ax.add_patch(Rectangle((0.01, 0.93), 0.98, 0.05,
                                    transform=self.ax.transAxes,
                                    facecolor=THEME['bg_highlight'],
                                    edgecolor='none', alpha=0.9, zorder=1))
        self.ax.text(0.03, 0.955, "EVENTS",
                     transform=self.ax.transAxes,
                     fontsize=11, fontweight='bold',
                     color=THEME['text_primary'], va='center')
        
        lines = []
        
        if not events:
            lines.append("(No events found)")
            lines.append("Tip: Use Phase1 or add manually")
        else:
            # Show event list (max 15, centered on current)
            max_display = 15
            start_idx = max(0, current_event_idx - max_display // 2)
            end_idx = min(len(events), start_idx + max_display)
            
            if start_idx > 0:
                lines.append(f"... ({start_idx} more above)")
            
            for i in range(start_idx, end_idx):
                event = events[i]
                
                # Current event highlight
                if i == current_event_idx:
                    prefix = "▶"
                    suffix = "◀"
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
                line = f"{prefix}[{confirmed}] F{frame:04d} {symbol}{type_abbr} {suffix}"
                lines.append(line)
            
            if end_idx < len(events):
                remaining = len(events) - end_idx
                lines.append(f"... ({remaining} more below)")
        
        # Legend
        lines.append("")
        lines.append("Legend")
        lines.append(f"{EVENT_SYMBOLS['landing']}Land {EVENT_SYMBOLS['hit']}Hit {EVENT_SYMBOLS['out_of_frame']}OOF")
        lines.append("Y=Confirmed  ?=Pending")
        
        text = '\n'.join(lines)
        self.ax.text(0.03, 0.90, text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='DejaVu Sans Mono',
                    verticalalignment='top',
                    color=THEME['text_primary'])


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

        # Card background
        self.ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                         boxstyle='round,pad=0.015,rounding_size=0.02',
                                         transform=self.ax.transAxes,
                                         facecolor=THEME['bg_panel'],
                                         edgecolor=THEME['border'],
                                         linewidth=1.2, alpha=0.95, zorder=0))
        # Header bar
        self.ax.add_patch(Rectangle((0.01, 0.93), 0.98, 0.05,
                                    transform=self.ax.transAxes,
                                    facecolor=THEME['bg_highlight'],
                                    edgecolor='none', alpha=0.9, zorder=1))
        self.ax.text(0.03, 0.955, "SHORTCUTS",
                     transform=self.ax.transAxes,
                     fontsize=11, fontweight='bold',
                     color=THEME['text_primary'], va='center')
        
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
            ("Video", "", True),
            ("Ctrl+<-", "Prev video", False),
            ("Ctrl+->", "Next video", False),
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
        
        for key, desc, is_header in shortcuts:
            if is_header and key:
                lines.append(f"{key}")
            elif is_header:
                lines.append("")
            elif key == "":
                lines.append("")
            else:
                # Highlight current key
                if highlight_key and key.lower().startswith(highlight_key.lower()):
                    lines.append(f"▶ {key:<6} {desc}")
                else:
                    lines.append(f"  {key:<6} {desc}")
        
        text = '\n'.join(lines)
        self.ax.text(0.03, 0.90, text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    fontfamily='DejaVu Sans Mono',
                    verticalalignment='top',
                    color=THEME['text_primary'])
