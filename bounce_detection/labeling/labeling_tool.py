"""
半自动标注工具主类
Semi-automatic Labeling Tool for Bounce Detection

交互式界面，支持:
- 视频帧浏览
- Phase 1 候选预填充
- 人工确认/修正/删除/添加事件
- 保存标注结果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Dict, List, Optional, Callable

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .data_manager import LabelingDataManager, find_matching_video
from .visualizer import FrameVisualizer, TrajectoryPlot, InfoPanel, EventListPanel


class LabelingTool:
    """
    半自动标注工具
    
    使用方法:
    1. 初始化: tool = LabelingTool(csv_path, video_path)
    2. 预填充: tool.prefill_from_detector()
    3. 启动: tool.run()
    4. 使用键盘快捷键进行标注
    5. 按 S 保存, 按 Q 退出
    """
    
    def __init__(self,
                 csv_path: str,
                 video_path: Optional[str] = None,
                 label_path: Optional[str] = None,
                 auto_detect: bool = True):
        """
        初始化标注工具
        
        Args:
            csv_path: 轨迹 CSV 文件路径
            video_path: 视频文件路径 (可选，自动查找)
            label_path: 已有标注文件路径 (可选)
            auto_detect: 是否自动运行 Phase 1 检测
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
            label_path=label_path
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
        """设置 UI 布局"""
        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Bounce Detection Labeling Tool')
        
        # 布局: 2行3列
        # [视频帧 (大)      ] [信息面板] [事件列表]
        # [Y坐标图] [速度图 ] [按钮区域]
        
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1], 
                                   width_ratios=[2, 1, 1],
                                   hspace=0.3, wspace=0.3)
        
        # 创建子图
        self.axes['frame'] = self.fig.add_subplot(gs[0, 0])
        self.axes['info'] = self.fig.add_subplot(gs[0, 1])
        self.axes['events'] = self.fig.add_subplot(gs[0, 2])
        self.axes['y_plot'] = self.fig.add_subplot(gs[1, 0])
        self.axes['speed_plot'] = self.fig.add_subplot(gs[1, 1])
        self.axes['buttons'] = self.fig.add_subplot(gs[1, 2])
        
        # 初始化可视化组件
        self.visualizers['frame'] = FrameVisualizer(self.axes['frame'])
        self.visualizers['trajectory'] = TrajectoryPlot(
            self.axes['y_plot'], self.axes['speed_plot']
        )
        self.visualizers['info'] = InfoPanel(self.axes['info'])
        self.visualizers['events'] = EventListPanel(self.axes['events'])
        
        # 设置按钮区域
        self._setup_buttons()
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 初始绘制
        self._update_display()
    
    def _setup_buttons(self):
        """设置按钮"""
        ax = self.axes['buttons']
        ax.axis('off')
        
        # 按钮位置
        button_specs = [
            (0.1, 0.75, 0.35, 0.15, 'Prev Frame (←)', self._prev_frame),
            (0.55, 0.75, 0.35, 0.15, 'Next Frame (→)', self._next_frame),
            (0.1, 0.55, 0.35, 0.15, 'Prev Event (↑)', self._prev_event),
            (0.55, 0.55, 0.35, 0.15, 'Next Event (↓)', self._next_event),
            (0.1, 0.35, 0.35, 0.15, 'Confirm (Y)', lambda e: self._confirm_event()),
            (0.55, 0.35, 0.35, 0.15, 'Delete (D)', lambda e: self._delete_event()),
            (0.1, 0.15, 0.35, 0.15, 'Save (S)', lambda e: self._save()),
            (0.55, 0.15, 0.35, 0.15, 'Quit (Q)', lambda e: self._quit()),
        ]
        
        self.buttons = []
        for x, y, w, h, label, callback in button_specs:
            btn_ax = self.fig.add_axes([
                ax.get_position().x0 + x * ax.get_position().width,
                ax.get_position().y0 + y * ax.get_position().height,
                w * ax.get_position().width * 0.9,
                h * ax.get_position().height * 0.8
            ])
            btn = Button(btn_ax, label)
            btn.on_clicked(callback)
            self.buttons.append(btn)
    
    def _update_display(self):
        """更新显示"""
        if self.fig is None:
            return
        
        dm = self.data_manager
        traj = dm.trajectory
        
        # 获取当前帧数据
        frame_img = dm.get_frame(self.current_frame)
        ball_pos = dm.get_trajectory_at_frame(self.current_frame)
        traj_window = dm.get_trajectory_window(self.current_frame, window_size=50)
        
        # 获取窗口内的事件
        events = dm.get_events_in_range(
            max(0, self.current_frame - 50),
            min(dm.metadata['total_frames'], self.current_frame + 50)
        )
        current_event = dm.get_event_at_frame(self.current_frame)
        
        # 更新当前事件索引
        if current_event:
            for i, e in enumerate(dm.events):
                if e['frame'] == self.current_frame:
                    self.current_event_idx = i
                    break
        
        # 1. 更新帧可视化
        self.visualizers['frame'].draw_frame(
            frame_img, self.current_frame, ball_pos,
            traj_window, events, current_event
        )
        
        # 2. 更新轨迹图
        if 'frame' in traj and len(traj['frame']) > 0:
            self.visualizers['trajectory'].draw(
                traj['frame'], traj['x'], traj['y'], traj['visibility'],
                self.kinematics.get('speed', np.zeros(len(traj['frame']))),
                self.current_frame, events
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
    
    # ==================== 事件编辑 ====================
    
    def _confirm_event(self):
        """确认当前事件"""
        if self.data_manager.confirm_event(self.current_frame):
            self.modified = True
            print(f"Confirmed event at frame {self.current_frame}")
            self._update_display()
    
    def _delete_event(self):
        """删除当前事件"""
        if self.data_manager.remove_event(self.current_frame):
            self.modified = True
            print(f"Deleted event at frame {self.current_frame}")
            self._update_display()
    
    def _add_event(self, event_type: str = 'landing'):
        """在当前帧添加事件"""
        x, y, vis = self.data_manager.get_trajectory_at_frame(self.current_frame)
        
        event = {
            'frame': self.current_frame,
            'x': x,
            'y': y,
            'event_type': event_type,
            'rule': 'manual',
            'confidence': 1.0,
            'confirmed': True  # 手动添加的自动确认
        }
        
        self.data_manager.add_event(event)
        self.modified = True
        print(f"Added {event_type} event at frame {self.current_frame}")
        self._update_display()
    
    def _set_event_type(self, event_type: str):
        """设置当前事件的类型"""
        if self.data_manager.update_event(self.current_frame, 
                                          {'event_type': event_type, 'confirmed': True}):
            self.modified = True
            print(f"Set event type to {event_type} at frame {self.current_frame}")
        else:
            # 当前帧没有事件，添加一个
            self._add_event(event_type)
        self._update_display()
    
    # ==================== 保存/退出 ====================
    
    def _save(self):
        """保存标注"""
        save_path = self.data_manager.save_labels()
        self.modified = False
        print(f"Labels saved to {save_path}")
    
    def _quit(self):
        """退出"""
        if self.modified:
            # 提示保存
            print("\n⚠️  You have unsaved changes!")
            response = input("Save before quitting? (y/n/c): ").strip().lower()
            if response == 'y':
                self._save()
            elif response == 'c':
                return  # 取消退出
        
        self.running = False
        plt.close(self.fig)
    
    # ==================== 键盘事件 ====================
    
    def _on_key_press(self, event):
        """键盘事件处理"""
        key = event.key.lower() if event.key else ''
        
        # 导航
        if key in ['left', 'a']:
            self._prev_frame()
        elif key in ['right', 'd']:
            self._next_frame()
        elif key in ['up', 'w']:
            self._prev_event()
        elif key in ['down', 's'] and not event.key == 's':
            self._next_event()
        
        # 快速跳转
        elif key == 'home':
            self._jump_to_frame(0)
        elif key == 'end':
            self._jump_to_frame(self.data_manager.metadata['total_frames'] - 1)
        elif key == 'pageup':
            self._jump_to_frame(self.current_frame - 30)
        elif key == 'pagedown':
            self._jump_to_frame(self.current_frame + 30)
        
        # 事件编辑
        elif key == 'y':
            self._confirm_event()
        elif key == 'delete' or (key == 'd' and not event.key == 'right'):
            self._delete_event()
        elif key == 'l':
            self._set_event_type('landing')
        elif key == 'h':
            self._set_event_type('hit')
        elif key == 'o':
            self._set_event_type('out_of_frame')
        elif key == 'n':
            self._add_event('other')
        
        # 保存/退出
        elif key == 's':
            self._save()
        elif key == 'q' or key == 'escape':
            self._quit()
    
    # ==================== 运行 ====================
    
    def run(self):
        """启动标注工具"""
        print("\n" + "="*60)
        print("  Bounce Detection Labeling Tool")
        print("="*60)
        print(f"CSV: {self.data_manager.csv_path}")
        print(f"Video: {self.data_manager.video_path or 'Not loaded'}")
        print(f"Total frames: {self.data_manager.metadata['total_frames']}")
        print(f"Pre-filled events: {len(self.data_manager.events)}")
        print("="*60)
        print("\nKeyboard shortcuts:")
        print("  ← → : Previous/Next frame")
        print("  ↑ ↓ : Previous/Next event")
        print("  Y   : Confirm event")
        print("  D   : Delete event")
        print("  L/H/O : Set as Landing/Hit/Out-of-frame")
        print("  N   : Add new event")
        print("  S   : Save labels")
        print("  Q   : Quit")
        print("="*60 + "\n")
        
        self._setup_ui()
        self.running = True
        
        # 如果有预填充的事件，跳转到第一个
        if self.data_manager.events:
            self.current_frame = self.data_manager.events[0]['frame']
            self.current_event_idx = 0
            self._update_display()
        
        plt.show()
        
        return self.data_manager.events


def main():
    """命令行入口"""
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
    
    args = parser.parse_args()
    
    tool = LabelingTool(
        csv_path=args.csv,
        video_path=args.video,
        label_path=args.labels,
        auto_detect=not args.no_detect
    )
    
    tool.run()


if __name__ == '__main__':
    main()
