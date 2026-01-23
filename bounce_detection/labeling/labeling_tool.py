"""
Semi-automatic Labeling Tool for Bounce Detection

Interactive UI supports:
- Video frame browsing
- Phase 1 candidate pre-fill
- Manual confirm/edit/delete/add events
- Save labels

Optimized version - Clean UI + Complete keyboard shortcuts
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Dict, List, Optional, Callable

# Fix minus sign display
matplotlib.rcParams['axes.unicode_minus'] = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .data_manager import LabelingDataManager, find_matching_video
from .visualizer import FrameVisualizer, TrajectoryPlot, InfoPanel, EventListPanel, ShortcutPanel, THEME


class LabelingTool:
    """
    Semi-automatic Labeling Tool
    
    Usage:
    1. Init: tool = LabelingTool(csv_path, video_path)
    2. 预填充: tool.prefill_from_detector()
    3. 启动: tool.run()
    4. 使用键盘快捷键进行标注
    5. 按 S 保存, 按 Q 退出
    """
    
    def __init__(self,
                 csv_path: str,
                 video_path: Optional[str] = None,
                 label_path: Optional[str] = None,
                 auto_detect: bool = True,
                 lazy_load_video: bool = False):
        """
        初始化标注工具
        
        Args:
            csv_path: 轨迹 CSV 文件路径
            video_path: 视频文件路径 (可选，自动查找)
            label_path: 已有标注文件路径 (可选)
            auto_detect: 是否自动运行 Phase 1 检测
            lazy_load_video: 是否延迟加载视频帧（节省内存，适用于长视频）
        """
        # 自动查找视频
        if video_path is None:
            video_path = find_matching_video(csv_path)
            if video_path:
                print(f"Found matching video: {video_path}")
        
        # 初始化数据管理器
        self.data_manager = LabelingDataManager(
            video_path=video_path,
            csv_path=csv_path,
            label_path=label_path,
            lazy_load_video=lazy_load_video
        )
        
        # 状态
        self.current_frame = 0
        self.current_event_idx = -1
        self.modified = False
        self.running = False
        
        # 计算运动学特征
        self._compute_kinematics()
        
        # 自动检测
        if auto_detect and not label_path:
            self.prefill_from_detector()
        
        # UI 组件 (延迟初始化)
        self.fig = None
        self.axes = {}
        self.visualizers = {}
    
    def _compute_kinematics(self):
        """计算运动学特征"""
        traj = self.data_manager.trajectory
        if 'x' not in traj or len(traj['x']) == 0:
            self.kinematics = {'speed': np.array([])}
            return
        
        from bounce_detection.kinematics import KinematicsCalculator
        calc = KinematicsCalculator()
        self.kinematics = calc.compute(
            traj['x'], traj['y'], traj['visibility']
        )
    
    def prefill_from_detector(self):
        """使用 Phase 1 检测器预填充候选事件"""
        from bounce_detection.detector import BounceDetector
        
        detector = BounceDetector()
        traj = self.data_manager.trajectory
        
        if 'x' not in traj or len(traj['x']) == 0:
            print("No trajectory data to detect")
            return
        
        candidates = detector.detect(
            traj['x'], traj['y'], traj['visibility']
        )
        
        self.data_manager.set_events_from_candidates(candidates)
        print(f"Pre-filled {len(candidates)} events from Phase 1 detector")
    
    def _setup_ui(self):
        """设置 UI 布局 - 美观的现代化界面"""
        # 设置深色主题
        plt.style.use('dark_background')
        
        # 创建图形
        self.fig = plt.figure(figsize=(22, 13), facecolor=THEME['bg_dark'])
        self.fig.canvas.manager.set_window_title('🏸 羽毛球落点检测标注工具')
        self.fig.suptitle('🏸 Bounce Labeling Studio',
                  fontsize=16, fontweight='bold',
                  color=THEME['text_primary'], y=0.985)
        
        # 优化布局: 3行布局
        # Row 0: [视频帧 (大)] [信息面板] [事件列表] [快捷键]
        # Row 1: [X坐标图    ] [Y坐标图 ]
        # Row 2: [速度图     ] [加速度图] [按钮区域]
        
        gs = self.fig.add_gridspec(3, 4, height_ratios=[4.2, 0.55, 0.55], 
                       width_ratios=[4.6, 1.05, 0.95, 0.9],
                       hspace=0.22, wspace=0.12,
                       left=0.03, right=0.97, top=0.95, bottom=0.05)
        
        # 创建子图
        self.axes['frame'] = self.fig.add_subplot(gs[0, 0])
        self.axes['info'] = self.fig.add_subplot(gs[0, 1])
        self.axes['events'] = self.fig.add_subplot(gs[0, 2])
        self.axes['shortcuts'] = self.fig.add_subplot(gs[0, 3])
        
        # 新增：4个时序图
        self.axes['x_plot'] = self.fig.add_subplot(gs[1, 0:2])
        self.axes['y_plot'] = self.fig.add_subplot(gs[1, 2:4])
        self.axes['speed_plot'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['acc_plot'] = self.fig.add_subplot(gs[2, 2])
        self.axes['buttons'] = self.fig.add_subplot(gs[2, 3])
        
        # 设置所有面板的背景色
        for name, ax in self.axes.items():
            ax.set_facecolor(THEME['bg_panel'])
        
        # 初始化可视化组件
        self.visualizers['frame'] = FrameVisualizer(self.axes['frame'])
        self.visualizers['trajectory'] = TrajectoryPlot(
            self.axes['x_plot'], self.axes['y_plot'], 
            self.axes['speed_plot'], self.axes['acc_plot']
        )
        self.visualizers['info'] = InfoPanel(self.axes['info'])
        self.visualizers['events'] = EventListPanel(self.axes['events'])
        self.visualizers['shortcuts'] = ShortcutPanel(self.axes['shortcuts'])
        
        # 绘制快捷键面板（静态）
        self.visualizers['shortcuts'].draw()
        
        # 设置按钮区域
        self._setup_buttons()
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Add status bar
        self._status_text = self.fig.text(
            0.5, 0.01, 'Ready | Press Q to quit, S to save',
            ha='center', va='bottom',
            fontsize=10, color=THEME['text_secondary'],
            fontfamily='monospace'
        )
        
        # Initial draw
        self._update_display()
    
    def _setup_buttons(self):
        """Setup buttons with modern style - 紧凑布局"""
        ax = self.axes['buttons']
        ax.axis('off')
        ax.set_facecolor(THEME['bg_panel'])
        
        # 紧凑的按钮配置: (x, y, w, h, label, callback, color)
        # 4行3列布局
        button_specs = [
            # Row 1 - Navigation
            (0.02, 0.76, 0.30, 0.22, '< Prev', self._prev_frame, '#3498DB'),
            (0.35, 0.76, 0.30, 0.22, 'Next >', self._next_frame, '#3498DB'),
            (0.68, 0.76, 0.30, 0.22, 'PrevEvt', self._prev_event, '#9B59B6'),
            
            # Row 2
            (0.02, 0.52, 0.30, 0.22, 'NextEvt', self._next_event, '#9B59B6'),
            (0.35, 0.52, 0.30, 0.22, 'Confirm', lambda e: self._confirm_event(), '#27AE60'),
            (0.68, 0.52, 0.30, 0.22, 'Delete', lambda e: self._delete_event(), '#E74C3C'),
            
            # Row 3 - Event types
            (0.02, 0.28, 0.30, 0.22, '* Land', lambda e: self._set_event_type('landing'), '#FF6B6B'),
            (0.35, 0.28, 0.30, 0.22, '+ Hit', lambda e: self._set_event_type('hit'), '#4ECDC4'),
            (0.68, 0.28, 0.30, 0.22, 'Add New', lambda e: self._add_event(), '#1ABC9C'),
            
            # Row 4 - System
            (0.02, 0.04, 0.47, 0.22, 'Save (S)', lambda e: self._save(), '#F39C12'),
            (0.51, 0.04, 0.47, 0.22, 'Quit (Q)', lambda e: self._quit(), '#95A5A6'),
        ]
        
        self.buttons = []
        for x, y, w, h, label, callback, color in button_specs:
            btn_ax = self.fig.add_axes([
                ax.get_position().x0 + x * ax.get_position().width,
                ax.get_position().y0 + y * ax.get_position().height,
                w * ax.get_position().width * 0.95,
                h * ax.get_position().height * 0.9
            ])
            btn_ax.set_facecolor(color)
            for spine in btn_ax.spines.values():
                spine.set_edgecolor('#000000')
                spine.set_linewidth(0.8)
            
            btn = Button(btn_ax, label, color=color, hovercolor=self._lighten_color(color))
            btn.label.set_fontsize(9)
            btn.label.set_fontweight('bold')
            btn.label.set_color('white')
            btn.label.set_fontfamily('DejaVu Sans')
            btn.on_clicked(callback)
            self.buttons.append(btn)
    
    def _lighten_color(self, hex_color: str, factor: float = 0.3) -> str:
        """Lighten a color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _update_status(self, message: str):
        """Update status bar"""
        if hasattr(self, '_status_text'):
            modified_mark = ' [Modified]' if self.modified else ''
            self._status_text.set_text(f'{message}{modified_mark}')
            self.fig.canvas.draw_idle()
    
    def _update_display(self):
        """更新显示"""
        if self.fig is None:
            return
        
        dm = self.data_manager
        traj = dm.trajectory
        
        # 获取当前帧数据
        frame_img = dm.get_frame(self.current_frame)
        ball_pos = dm.get_trajectory_at_frame(self.current_frame)
        # 减小轨迹窗口大小，避免画面杂乱（前后各0帧）
        traj_window = dm.get_trajectory_window(self.current_frame, window_size=0)
         
        # 获取当前事件
        current_event = dm.get_event_at_frame(self.current_frame)
        
        # 视频帧上只显示当前帧的事件（避免过去和未来的事件遮挡视线）
        frame_events = [current_event] if current_event else []
        
        # 时序图显示所有事件（用于查看整体分布）
        all_events = dm.events
        
        # 更新当前事件索引
        if current_event:
            for i, e in enumerate(dm.events):
                if e['frame'] == self.current_frame:
                    self.current_event_idx = i
                    break
        
        # 1. 更新帧可视化（只显示当前事件）
        self.visualizers['frame'].draw_frame(
            frame_img, self.current_frame, ball_pos,
            traj_window, frame_events, current_event
        )
        
        # 2. 更新轨迹图（显示所有事件的标记线）
        if 'frame' in traj and len(traj['frame']) > 0:
            # 计算总加速度 = sqrt(a_x^2 + a_y^2)
            a_x = self.kinematics.get('a_x', np.zeros(len(traj['frame'])))
            a_y = self.kinematics.get('a_y', np.zeros(len(traj['frame'])))
            acceleration = np.sqrt(a_x**2 + a_y**2)
            
            self.visualizers['trajectory'].draw(
                traj['frame'], traj['x'], traj['y'], traj['visibility'],
                self.kinematics.get('speed', np.zeros(len(traj['frame']))),
                acceleration,
                self.current_frame, all_events
            )
        
        # 3. 更新信息面板
        self.visualizers['info'].draw(
            self.current_frame,
            dm.metadata['total_frames'],
            current_event,
            dm.get_statistics()
        )
        
        # 4. 更新事件列表
        self.visualizers['events'].draw(dm.events, self.current_event_idx)
        
        # 刷新
        self.fig.canvas.draw_idle()
    
    # ==================== 导航 ====================
    
    def _prev_frame(self, event=None):
        """上一帧"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_display()
    
    def _next_frame(self, event=None):
        """下一帧"""
        max_frame = self.data_manager.metadata['total_frames'] - 1
        if self.current_frame < max_frame:
            self.current_frame += 1
            self._update_display()
    
    def _prev_event(self, event=None):
        """跳转到上一个事件"""
        events = self.data_manager.events
        if not events:
            return
        
        # 找到当前帧之前的最近事件
        for i in range(len(events) - 1, -1, -1):
            if events[i]['frame'] < self.current_frame:
                self.current_frame = events[i]['frame']
                self.current_event_idx = i
                self._update_display()
                return
        
        # 没有更早的事件，跳到最后一个
        self.current_frame = events[-1]['frame']
        self.current_event_idx = len(events) - 1
        self._update_display()
    
    def _next_event(self, event=None):
        """跳转到下一个事件"""
        events = self.data_manager.events
        if not events:
            return
        
        # 找到当前帧之后的最近事件
        for i, e in enumerate(events):
            if e['frame'] > self.current_frame:
                self.current_frame = e['frame']
                self.current_event_idx = i
                self._update_display()
                return
        
        # 没有更晚的事件，跳到第一个
        self.current_frame = events[0]['frame']
        self.current_event_idx = 0
        self._update_display()
    
    def _jump_to_frame(self, frame: int):
        """跳转到指定帧"""
        frame = max(0, min(frame, self.data_manager.metadata['total_frames'] - 1))
        self.current_frame = frame
        self._update_display()
    
    # ==================== Event Editing ====================
    
    def _confirm_event(self):
        """Confirm current event"""
        if self.data_manager.confirm_event(self.current_frame):
            self.modified = True
            print(f"[OK] Confirmed event at frame {self.current_frame}")
            self._update_status(f'Confirmed: F{self.current_frame}')
            self._update_display()
        else:
            self._update_status('No event to confirm')
    
    def _delete_event(self):
        """Delete current event"""
        if self.data_manager.remove_event(self.current_frame):
            self.modified = True
            print(f"[X] Deleted event at frame {self.current_frame}")
            self._update_status(f'Deleted: F{self.current_frame}')
            self._update_display()
        else:
            self._update_status('No event to delete')
    
    def _add_event(self, event_type: str = 'landing'):
        """Add event at current frame"""
        x, y, vis = self.data_manager.get_trajectory_at_frame(self.current_frame)
        
        event = {
            'frame': self.current_frame,
            'x': x,
            'y': y,
            'event_type': event_type,
            'rule': 'manual',
            'confidence': 1.0,
            'confirmed': True  # Manual events auto-confirmed
        }
        
        self.data_manager.add_event(event)
        self.modified = True
        print(f"[+] Added {event_type} at frame {self.current_frame}")
        self._update_status(f'Added: F{self.current_frame}')
        self._update_display()
    
    def _set_event_type(self, event_type: str):
        """Set event type"""
        if self.data_manager.update_event(self.current_frame, 
                                          {'event_type': event_type, 'confirmed': True}):
            self.modified = True
            print(f"[*] Set event at frame {self.current_frame} to {event_type}")
        else:
            # No event at current frame, add one
            self._add_event(event_type)
        self._update_display()
    
    # ==================== Save/Quit ====================
    
    def _save(self):
        """Save labels"""
        save_path = self.data_manager.save_labels()
        self.modified = False
        print(f"[Save] Labels saved to: {save_path}")
        self._update_status(f'Saved: {save_path}')
    
    def _quit(self):
        """Quit"""
        if self.modified:
            # Prompt save
            print("\n[!] You have unsaved changes!")
            response = input("Save before quit? (y=save/n=no/c=cancel): ").strip().lower()
            if response == 'y':
                self._save()
            elif response == 'c':
                print("Cancelled")
                return  # Cancel quit
            else:
                print("Discarded changes")
        
        self.running = False
        plt.close(self.fig)
        print("\n[Bye] Labeling tool closed")
    
    # ==================== Keyboard Events ====================
    
    def _on_key_press(self, event):
        """
        Keyboard event handler
        
        Full shortcut mapping:
        - Navigate: <- -> Up Down A D W (D for next frame, not delete)
        - Quick jump: Home End PageUp PageDown
        - Edit: Y(confirm) Delete(delete) L(land) H(hit) O(oof) N(add)
        - System: Ctrl+S(save) S(save) Q Escape(quit)
        """
        if event.key is None:
            return
        
        key = event.key.lower()
        
        # ===== Navigation =====
        # Previous frame: Left arrow or A
        if key in ['left', 'a']:
            self._prev_frame()
            self._update_status(f'Prev: F{self.current_frame}')
            return
        
        # Next frame: Right arrow or D
        if key in ['right', 'd']:
            self._next_frame()
            self._update_status(f'Next: F{self.current_frame}')
            return
        
        # Previous event: Up arrow or W
        if key in ['up', 'w']:
            self._prev_event()
            self._update_status(f'Prev event: F{self.current_frame}')
            return
        
        # Next event: Down arrow (S is for save)
        if key == 'down':
            self._next_event()
            self._update_status(f'Next event: F{self.current_frame}')
            return
        
        # ===== Quick Jump =====
        if key == 'home':
            self._jump_to_frame(0)
            self._update_status('Jump to first')
            return
        
        if key == 'end':
            self._jump_to_frame(self.data_manager.metadata['total_frames'] - 1)
            self._update_status('Jump to last')
            return
        
        if key == 'pageup':
            self._jump_to_frame(self.current_frame - 30)
            self._update_status(f'-30: F{self.current_frame}')
            return
        
        if key == 'pagedown':
            self._jump_to_frame(self.current_frame + 30)
            self._update_status(f'+30: F{self.current_frame}')
            return
        
        # ===== Event Editing =====
        # Confirm: Y
        if key == 'y':
            self._confirm_event()
            return
        
        # Delete: Delete or Backspace
        if key in ['delete', 'backspace']:
            self._delete_event()
            return
        
        # Set event type
        if key == 'l':
            self._set_event_type('landing')
            self._update_status('Set: Landing')
            return
        
        if key == 'h':
            self._set_event_type('hit')
            self._update_status('Set: Hit')
            return
        
        if key == 'o':
            self._set_event_type('out_of_frame')
            self._update_status('Set: Out-of-frame')
            return
        
        # Add new event: N
        if key == 'n':
            self._add_event('landing')  # Default to landing
            return
        
        # ===== System =====
        # Save: Ctrl+S or S
        if key == 's' or key == 'ctrl+s':
            self._save()
            return
        
        # Quit: Q or Escape
        if key in ['q', 'escape']:
            self._quit()
            return
    
    # ==================== Run ====================
    
    def run(self):
        """Start labeling tool"""
        # Print startup info
        print("\n" + "="*65)
        print("  Badminton Bounce Detection Labeling Tool v2.0")
        print("="*65)
        print(f"  CSV: {self.data_manager.csv_path}")
        print(f"  Video: {self.data_manager.video_path or 'Not loaded'}")
        print(f"  Total frames: {self.data_manager.metadata['total_frames']}")
        print(f"  Pre-filled events: {len(self.data_manager.events)}")
        print("="*65)
        print("\n  Keyboard Shortcuts:")
        print("  +-----------------------------------------------------------+")
        print("  | Navigate                                                  |")
        print("  |   <- / A    Prev frame     -> / D    Next frame           |")
        print("  |   Up / W    Prev event     Down      Next event           |")
        print("  |   Home      First frame    End       Last frame           |")
        print("  |   PageUp    -30 frames     PageDown  +30 frames           |")
        print("  +-----------------------------------------------------------+")
        print("  | Edit                                                      |")
        print("  |   Y         Confirm        Delete    Delete event         |")
        print("  |   L         Set Landing    H         Set Hit              |")
        print("  |   O         Set OOF        N         Add new event        |")
        print("  +-----------------------------------------------------------+")
        print("  | System                                                    |")
        print("  |   S / Ctrl+S  Save         Q / Esc   Quit                 |")
        print("  +-----------------------------------------------------------+")
        print("="*65 + "\n")
        
        self._setup_ui()
        self.running = True
        
        # If there are pre-filled events, jump to first one
        if self.data_manager.events:
            self.current_frame = self.data_manager.events[0]['frame']
            self.current_event_idx = 0
            self._update_display()
        
        plt.show()
        
        return self.data_manager.events


def main():
    """CLI entry"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bounce Detection Labeling Tool')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to trajectory CSV file')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (optional, auto-detected)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to existing labels file (optional)')
    parser.add_argument('--no-detect', action='store_true',
                       help='Disable automatic Phase 1 detection')
    parser.add_argument('--lazy-load', action='store_true',
                       help='Enable lazy video loading (save memory for long videos)')
    
    args = parser.parse_args()
    
    tool = LabelingTool(
        csv_path=args.csv,
        video_path=args.video,
        label_path=args.labels,
        auto_detect=not args.no_detect,
        lazy_load_video=args.lazy_load
    )
    
    tool.run()


if __name__ == '__main__':
    main()
