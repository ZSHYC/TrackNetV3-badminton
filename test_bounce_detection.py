"""
测试落点检测模块 (Phase 1)
Test Bounce Detection Module

用法:
    # 测试单个 CSV 文件
    python test_bounce_detection.py --csv data/test/match1/csv/1_05_02_ball.csv
    
    # 测试整个 match 目录
    python test_bounce_detection.py --match_dir data/test/match1
    
    # 可视化结果
    python test_bounce_detection.py --csv data/test/match1/csv/1_05_02_ball.csv --visualize
"""

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob

from bounce_detection import KinematicsCalculator, BounceCandidateGenerator
from bounce_detection.detector import BounceDetector, visualize_candidates


def test_single_csv(csv_path: str, visualize: bool = False, save_dir: str = None):
    """测试单个 CSV 文件"""
    print(f"\n{'='*60}")
    print(f"Testing: {csv_path}")
    print('='*60)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='Frame').fillna(0)
    
    x = df['X'].values
    y = df['Y'].values
    visibility = df['Visibility'].values
    
    print(f"Total frames: {len(x)}")
    print(f"Visible frames: {int(np.sum(visibility))} ({100*np.mean(visibility):.1f}%)")
    
    # 创建检测器
    detector = BounceDetector(
        speed_drop_ratio=0.3,
        min_speed_before_landing=8.0,
        max_speed_after_landing=5.0,
        min_speed_at_hit=5.0,
        vy_reversal_threshold=3.0,
        min_visible_before=3
    )
    
    # 检测事件
    candidates = detector.detect(x, y, visibility, img_height=288)
    
    # 分类统计
    landing_candidates = [c for c in candidates if c.get('event_type') == 'landing']
    hit_candidates = [c for c in candidates if c.get('event_type') == 'hit']
    out_candidates = [c for c in candidates if c.get('event_type') == 'out_of_frame']
    
    print(f"\n=== Detected {len(landing_candidates)} LANDING candidate(s) ===")
    for i, cand in enumerate(landing_candidates):
        print(f"  [{i+1}] Frame {cand['frame']:4d} | "
              f"Position ({cand['x']:.0f}, {cand['y']:.0f}) | "
              f"Rule: {cand['rule']:20s} | "
              f"Confidence: {cand['confidence']:.2f}")
        if 'features' in cand:
            feat_str = ', '.join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in cand['features'].items()])
            print(f"       Features: {feat_str}")
    
    print(f"\n=== Detected {len(hit_candidates)} HIT candidate(s) ===")
    for i, cand in enumerate(hit_candidates):
        print(f"  [{i+1}] Frame {cand['frame']:4d} | "
              f"Position ({cand['x']:.0f}, {cand['y']:.0f}) | "
              f"Rule: {cand['rule']:20s} | "
              f"Confidence: {cand['confidence']:.2f}")
        if 'features' in cand:
            feat_str = ', '.join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in cand['features'].items()])
            print(f"       Features: {feat_str}")
    
    if out_candidates:
        print(f"\n=== Detected {len(out_candidates)} OUT_OF_FRAME candidate(s) ===")
        for i, cand in enumerate(out_candidates):
            print(f"  [{i+1}] Frame {cand['frame']:4d} | "
                  f"Position ({cand['x']:.0f}, {cand['y']:.0f}) | "
                  f"Rule: {cand['rule']:20s} | "
                  f"Confidence: {cand['confidence']:.2f}")
    
    # 轨迹分析
    analysis = detector.analyze_trajectory(x, y, visibility)
    print(f"\nTrajectory Analysis:")
    print(f"  Y range: {analysis['y_range'][0]:.0f} - {analysis['y_range'][1]:.0f}")
    print(f"  X range: {analysis['x_range'][0]:.0f} - {analysis['x_range'][1]:.0f}")
    print(f"  Avg speed: {analysis['avg_speed']:.2f} pixels/frame")
    print(f"  Max speed: {analysis['max_speed']:.2f} pixels/frame")
    
    # 可视化
    if visualize:
        rally_name = os.path.basename(csv_path).replace('_ball.csv', '')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{rally_name}_bounce_analysis.png")
        else:
            save_path = None
        
        visualize_candidates(
            x, y, visibility, candidates,
            save_path=save_path,
            title=f"Bounce Analysis: {rally_name}"
        )
    
    return candidates


def test_match_directory(match_dir: str, visualize: bool = False, save_dir: str = None):
    """测试整个 match 目录"""
    # 如果目录名已经是 csv 目录
    if match_dir.endswith('csv') or match_dir.endswith('corrected_csv'):
        csv_dir = match_dir
    else:
        csv_dir = os.path.join(match_dir, 'csv')
        if not os.path.exists(csv_dir):
            # 尝试 corrected_csv 目录
            csv_dir = os.path.join(match_dir, 'corrected_csv')
    
    if not os.path.exists(csv_dir):
        print(f"Error: Cannot find csv directory in {match_dir}")
        return
    
    csv_files = sorted(glob(os.path.join(csv_dir, '*_ball.csv')))
    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    
    all_results = []
    for csv_path in csv_files:
        try:
            candidates = test_single_csv(csv_path, visualize=visualize, save_dir=save_dir)
            all_results.append({
                'file': os.path.basename(csv_path),
                'num_candidates': len(candidates),
                'candidates': candidates
            })
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    total_candidates = sum(r['num_candidates'] for r in all_results)
    print(f"Total rallies: {len(all_results)}")
    print(f"Total candidates: {total_candidates}")
    print(f"Avg candidates per rally: {total_candidates/len(all_results):.2f}")
    
    # 按规则统计
    rule_counts = {}
    for r in all_results:
        for c in r['candidates']:
            rule = c['rule']
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    print(f"\nCandidates by rule:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")
    
    return all_results


def demo_kinematics():
    """演示运动学计算"""
    print("\n" + "="*60)
    print("Kinematics Calculator Demo")
    print("="*60)
    
    # 创建一个模拟的抛物线轨迹
    t = np.linspace(0, 2, 60)  # 2秒，30fps
    x = 100 + 200 * t  # 水平匀速运动
    y = 50 + 100 * t - 50 * t**2  # 抛物线 (向上抛出，然后下落)
    y = np.maximum(y, 50)  # 模拟触地反弹
    visibility = np.ones(len(t))
    
    calc = KinematicsCalculator(fps=30)
    kinematics = calc.compute(x, y, visibility)
    
    print(f"Simulated trajectory: {len(t)} frames")
    print(f"X range: {x.min():.0f} - {x.max():.0f}")
    print(f"Y range: {y.min():.0f} - {y.max():.0f}")
    print(f"V_x range: {kinematics['v_x'].min():.2f} - {kinematics['v_x'].max():.2f}")
    print(f"V_y range: {kinematics['v_y'].min():.2f} - {kinematics['v_y'].max():.2f}")
    print(f"Speed range: {kinematics['speed'].min():.2f} - {kinematics['speed'].max():.2f}")


def main():
    parser = argparse.ArgumentParser(description='Test Bounce Detection Module')
    parser.add_argument('--csv', type=str, help='Path to a single CSV file')
    parser.add_argument('--match_dir', type=str, help='Path to a match directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--save_dir', type=str, default='bounce_analysis', 
                        help='Directory to save visualizations')
    parser.add_argument('--demo', action='store_true', help='Run kinematics demo')
    args = parser.parse_args()
    
    if args.demo:
        demo_kinematics()
        return
    
    if args.csv:
        test_single_csv(args.csv, visualize=args.visualize, save_dir=args.save_dir)
    elif args.match_dir:
        test_match_directory(args.match_dir, visualize=args.visualize, save_dir=args.save_dir)
    else:
        # 默认测试
        print("No input specified. Running demo on sample data...")
        
        # 尝试找一个测试文件
        test_files = [
            'data/test/match1/csv/1_05_02_ball.csv',
            'data/train/match1/csv/1_01_00_ball.csv',
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                test_single_csv(test_file, visualize=args.visualize, save_dir=args.save_dir)
                break
        else:
            print("No test files found. Please specify --csv or --match_dir")
            demo_kinematics()


if __name__ == '__main__':
    main()
