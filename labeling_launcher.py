#!/usr/bin/env python
"""
启动标注工具
Launch Labeling Tool

用法:
    python labeling_launcher.py --csv data/test/match1/csv/1_05_02_ball.csv
    python labeling_launcher.py --csv data/test/match1/csv/1_05_02_ball.csv --video data/test/match1/video/1_05_02.mp4
    python labeling_launcher.py --match_dir data/test/match1
"""

import os
import sys
import argparse
from glob import glob
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bounce_detection.labeling import LabelingTool, LabelingDataManager
from bounce_detection.labeling.data_manager import find_matching_video, LabelExporter


def label_single_file(csv_path: str, video_path: str = None, labels_path: str = None, lazy_load: bool = False):
    """标注单个文件"""
    tool = LabelingTool(
        csv_path=csv_path,
        video_path=video_path,
        label_path=labels_path,
        auto_detect=True,
        lazy_load_video=lazy_load
    )
    return tool.run()


def label_match_directory(match_dir: str, output_dir: str = None, review_mode: bool = False):
    """
    批量标注整个 match 目录
    
    Args:
        match_dir: match 目录路径 (如 data/test/match1)
        output_dir: 标注输出目录
        review_mode: 是否进入查看/修改已标注文件模式
    """
    # 查找 CSV 文件
    csv_dir = os.path.join(match_dir, 'csv')
    if not os.path.exists(csv_dir):
        csv_dir = os.path.join(match_dir, 'corrected_csv')
    
    if not os.path.exists(csv_dir):
        print(f"Error: Cannot find csv directory in {match_dir}")
        return
    
    csv_files = sorted(glob(os.path.join(csv_dir, '*_ball.csv')))
    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(match_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查已标注文件
    labeled_files = set()
    for f in glob(os.path.join(output_dir, '*_labels.json')):
        base = os.path.basename(f).replace('_labels.json', '_ball.csv')
        labeled_files.add(base)
    
    # 过滤未标注文件
    pending_files = [f for f in csv_files if os.path.basename(f) not in labeled_files]
    
    print(f"\nLabeled: {len(labeled_files)}")
    print(f"Pending: {len(pending_files)}")
    
    # 根据模式选择要处理的文件
    if review_mode:
        # 查看/修改模式：处理已标注的文件
        files_to_process = [f for f in csv_files if os.path.basename(f) in labeled_files]
        if not files_to_process:
            print("No labeled files found to review!")
            return
        print(f"\n[Review Mode] Will review {len(files_to_process)} labeled files")
    else:
        # 标注模式：处理未标注的文件
        files_to_process = pending_files
        if not files_to_process:
            print("All files have been labeled!")
            print("\nTip: Use --review flag to view/modify labeled files")
            print("Example: python labeling_launcher.py --match_dir data/test/match1 --review")
            return
    
    # 逐个处理
    for i, csv_path in enumerate(files_to_process):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(files_to_process)}] {os.path.basename(csv_path)}")
        print('='*60)
        
        # 查找对应视频
        video_path = find_matching_video(csv_path)
        
        # 计算标注文件路径
        label_name = os.path.basename(csv_path).replace('_ball.csv', '_labels.json')
        label_path = os.path.join(output_dir, label_name)
        
        # 启动标注工具
        tool = LabelingTool(
            csv_path=csv_path,
            video_path=video_path,
            label_path=label_path if review_mode else None,  # 查看模式加载已有标注
            auto_detect=not review_mode  # 查看模式不自动检测
        )
        
        events = tool.run()
        
        # 自动保存
        if events:
            tool.data_manager.save_labels(label_path)
        
        # 询问是否继续
        if i < len(files_to_process) - 1:
            response = input("\nContinue to next file? (y/n): ").strip().lower()
            if response != 'y':
                break


def export_all_labels(labels_dir: str, output_path: str, format: str = 'csv'):
    """
    导出所有标注为训练数据
    
    Args:
        labels_dir: 标注文件目录
        output_path: 输出文件路径
        format: 输出格式 ('csv' 或 'events')
    """
    import json
    import pandas as pd
    
    label_files = glob(os.path.join(labels_dir, '*_labels.json'))
    print(f"Found {len(label_files)} label files")
    
    all_events = []
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rally_name = os.path.basename(label_file).replace('_labels.json', '')
        
        for event in data.get('events', []):
            if event.get('confirmed', False):
                all_events.append({
                    'rally': rally_name,
                    'frame': event['frame'],
                    'x': event.get('x', 0),
                    'y': event.get('y', 0),
                    'event_type': event.get('event_type', 'unknown'),
                    'confidence': event.get('confidence', 1.0)
                })
    
    df = pd.DataFrame(all_events)
    df.to_csv(output_path, index=False)
    
    print(f"\nExported {len(all_events)} confirmed events to {output_path}")
    
    # 统计
    print("\nStatistics:")
    print(df['event_type'].value_counts())


def show_statistics(labels_dir: str):
    """显示标注统计"""
    import json
    
    label_files = glob(os.path.join(labels_dir, '*_labels.json'))
    
    total_events = 0
    confirmed_events = 0
    by_type = {}
    by_rally = {}
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rally_name = os.path.basename(label_file).replace('_labels.json', '')
        events = data.get('events', [])
        
        rally_stats = {'total': 0, 'confirmed': 0}
        
        for event in events:
            total_events += 1
            rally_stats['total'] += 1
            
            if event.get('confirmed', False):
                confirmed_events += 1
                rally_stats['confirmed'] += 1
                
                event_type = event.get('event_type', 'unknown')
                by_type[event_type] = by_type.get(event_type, 0) + 1
        
        by_rally[rally_name] = rally_stats
    
    print("\n" + "="*60)
    print("Labeling Statistics")
    print("="*60)
    print(f"Total label files: {len(label_files)}")
    print(f"Total events: {total_events}")
    print(f"Confirmed events: {confirmed_events}")
    print(f"Confirmation rate: {100*confirmed_events/max(1,total_events):.1f}%")
    
    print("\nBy event type:")
    for event_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count}")
    
    print("\nBy rally:")
    for rally, stats in sorted(by_rally.items()):
        print(f"  {rally}: {stats['confirmed']}/{stats['total']} confirmed")


def main():
    parser = argparse.ArgumentParser(
        description='Bounce Detection Labeling Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label a single file
  python labeling_launcher.py --csv data/test/match1/csv/1_05_02_ball.csv

  # Label with video
  python labeling_launcher.py --csv data/test/match1/csv/1_05_02_ball.csv --video data/test/match1/video/1_05_02.mp4

  # Batch label a match directory
  python labeling_launcher.py --match_dir data/test/match1

  # Review/modify labeled files
  python labeling_launcher.py --match_dir data/test/match1 --review

  # Export all labels to CSV
  python labeling_launcher.py --export data/test/match1/labels --output training_events.csv

  # Show statistics
  python labeling_launcher.py --stats data/test/match1/labels
        """
    )
    
    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--csv', type=str, help='Path to single CSV file')
    mode_group.add_argument('--match_dir', type=str, help='Path to match directory for batch labeling')
    mode_group.add_argument('--export', type=str, help='Path to labels directory for export')
    mode_group.add_argument('--stats', type=str, help='Path to labels directory for statistics')
    
    # 可选参数
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--labels', type=str, default=None, help='Path to existing labels file')
    parser.add_argument('--output', type=str, default=None, help='Output path for export')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for batch labeling')
    parser.add_argument('--review', action='store_true',
                       help='Review/modify already labeled files (use with --match_dir)')
    parser.add_argument('--lazy-load', action='store_true',
                       help='Enable lazy video loading (save memory for long videos)')
    
    args = parser.parse_args()
    
    if args.csv:
        # 单文件标注
        label_single_file(args.csv, args.video, args.labels, args.lazy_load)
    
    elif args.match_dir:
        # 批量标注
        label_match_directory(args.match_dir, args.output_dir, args.review)
    
    elif args.export:
        # 导出
        output_path = args.output or 'training_events.csv'
        export_all_labels(args.export, output_path)
    
    elif args.stats:
        # 统计
        show_statistics(args.stats)


if __name__ == '__main__':
    main()
