"""
标注工具测试脚本
Test Script for Labeling Tool

测试各组件功能，不启动 GUI
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bounce_detection.labeling.data_manager import LabelingDataManager, LabelExporter, find_matching_video
from bounce_detection.detector import BounceDetector


def test_data_manager():
    """测试数据管理器"""
    print("\n" + "="*60)
    print("Testing LabelingDataManager")
    print("="*60)
    
    csv_path = 'data/test/match1/csv/1_05_02_ball.csv'
    
    # 创建数据管理器
    dm = LabelingDataManager(csv_path=csv_path)
    
    print(f"✓ Loaded CSV: {csv_path}")
    print(f"  Total frames: {dm.metadata['total_frames']}")
    print(f"  Trajectory points: {len(dm.trajectory.get('x', []))}")
    
    # 测试轨迹访问
    x, y, vis = dm.get_trajectory_at_frame(100)
    print(f"✓ Frame 100 position: ({x:.1f}, {y:.1f}), visible={vis}")
    
    # 测试窗口获取
    x_win, y_win, vis_win = dm.get_trajectory_window(100, window_size=10)
    print(f"✓ Window around frame 100: {len(x_win)} points")
    
    return dm


def test_phase1_prefill(dm):
    """测试 Phase 1 预填充"""
    print("\n" + "="*60)
    print("Testing Phase 1 Prefill")
    print("="*60)
    
    detector = BounceDetector()
    candidates = detector.detect(
        dm.trajectory['x'], 
        dm.trajectory['y'], 
        dm.trajectory['visibility']
    )
    
    print(f"✓ Detected {len(candidates)} candidates")
    
    dm.set_events_from_candidates(candidates)
    print(f"✓ Pre-filled {len(dm.events)} events")
    
    # 按类型统计
    stats = dm.get_statistics()
    print(f"  By type: {stats['by_type']}")
    
    return dm


def test_event_operations(dm):
    """测试事件操作"""
    print("\n" + "="*60)
    print("Testing Event Operations")
    print("="*60)
    
    # 测试确认事件
    if dm.events:
        first_frame = dm.events[0]['frame']
        dm.confirm_event(first_frame)
        print(f"✓ Confirmed event at frame {first_frame}")
    
    # 测试添加事件
    dm.add_event({
        'frame': 250,
        'event_type': 'landing',
        'rule': 'manual_test'
    })
    print(f"✓ Added manual event at frame 250")
    
    # 测试更新事件
    dm.update_event(250, {'event_type': 'hit'})
    print(f"✓ Updated event at frame 250 to 'hit'")
    
    # 测试获取事件
    event = dm.get_event_at_frame(250)
    print(f"✓ Got event at frame 250: type={event['event_type']}")
    
    # 测试删除事件
    dm.remove_event(250)
    print(f"✓ Removed event at frame 250")
    
    # 最终统计
    stats = dm.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total: {stats['total_events']}")
    print(f"  Confirmed: {stats['confirmed_events']}")
    print(f"  By type: {stats['by_type']}")
    
    return dm


def test_save_load(dm):
    """测试保存和加载"""
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)
    
    # 保存
    save_path = 'test_labels_temp.json'
    dm.save_labels(save_path)
    print(f"✓ Saved labels to {save_path}")
    
    # 加载
    dm2 = LabelingDataManager(
        csv_path='data/test/match1/csv/1_05_02_ball.csv',
        label_path=save_path
    )
    print(f"✓ Loaded labels: {len(dm2.events)} events")
    
    # 清理
    os.remove(save_path)
    print(f"✓ Cleaned up temp file")
    
    return dm


def test_video_matching():
    """测试视频匹配"""
    print("\n" + "="*60)
    print("Testing Video Matching")
    print("="*60)
    
    csv_path = 'data/test/match1/csv/1_05_02_ball.csv'
    video_path = find_matching_video(csv_path)
    
    if video_path:
        print(f"✓ Found video: {video_path}")
    else:
        print(f"○ No video found (this is OK if video doesn't exist)")
    
    return video_path


def test_export():
    """测试导出功能"""
    print("\n" + "="*60)
    print("Testing Export")
    print("="*60)
    
    # 创建带确认事件的数据管理器
    dm = LabelingDataManager(csv_path='data/test/match1/csv/1_05_02_ball.csv')
    
    # 添加一些确认的事件
    dm.add_event({'frame': 30, 'event_type': 'hit', 'confirmed': True})
    dm.add_event({'frame': 72, 'event_type': 'hit', 'confirmed': True})
    dm.add_event({'frame': 516, 'event_type': 'landing', 'confirmed': True})
    
    # 导出事件列表
    export_path = 'test_export_temp.csv'
    LabelExporter.to_event_list(dm, export_path, only_confirmed=True)
    print(f"✓ Exported to {export_path}")
    
    # 读取验证
    import pandas as pd
    df = pd.read_csv(export_path)
    print(f"  Exported {len(df)} events")
    print(df)
    
    # 清理
    os.remove(export_path)
    print(f"✓ Cleaned up temp file")


def main():
    print("\n" + "="*60)
    print("  Labeling Tool Test Suite")
    print("="*60)
    
    # 运行测试
    dm = test_data_manager()
    dm = test_phase1_prefill(dm)
    dm = test_event_operations(dm)
    dm = test_save_load(dm)
    test_video_matching()
    test_export()
    
    print("\n" + "="*60)
    print("  All tests passed! ✓")
    print("="*60)
    print("\nTo launch the GUI labeling tool:")
    print("  python labeling_launcher.py --csv data/test/match1/csv/1_05_02_ball.csv")
    print("")


if __name__ == '__main__':
    main()
