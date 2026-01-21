"""
BounceNet 测试脚本
Test Script for BounceNet

验证 BounceNet 和 VisualFeatureExtractor 的功能。
"""

import os
import sys
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bounce_detection import (
    BounceNet, 
    BounceNetPredictor,
    VisualFeatureExtractor,
    create_bouncenet,
    BounceDetector,
    EVENT_LABELS,
    LABEL_TO_EVENT
)


def test_visual_feature_extractor():
    """测试视觉特征提取器"""
    print("\n" + "="*60)
    print("Testing VisualFeatureExtractor")
    print("="*60)
    
    extractor = VisualFeatureExtractor(
        patch_size=64,
        temporal_window=5,
        img_size=(512, 288)
    )
    
    # 创建模拟数据
    n_frames = 50
    frames = [np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8) 
              for _ in range(n_frames)]
    x = np.random.uniform(100, 400, n_frames).astype(np.float32)
    y = np.random.uniform(50, 230, n_frames).astype(np.float32)
    visibility = np.ones(n_frames)
    
    # 测试单帧特征提取
    center_frame = 25
    features = extractor.extract_features(
        frames, x, y, center_frame, visibility
    )
    
    print(f"Center patch shape: {features['center_patch'].shape}")
    print(f"Temporal patches shape: {features['temporal_patches'].shape}")
    print(f"MHI patch shape: {features['mhi_patch'].shape}")
    print(f"Center coords: {features['center_coords']}")
    
    # 测试批量提取
    candidate_frames = [10, 20, 30, 40]
    batch_features = extractor.extract_batch_features(
        frames, x, y, candidate_frames, visibility
    )
    
    print(f"\nBatch center patches shape: {batch_features['center_patches'].shape}")
    print(f"Batch temporal patches shape: {batch_features['temporal_patches'].shape}")
    
    print("\n✓ VisualFeatureExtractor test passed!")
    return True


def test_bouncenet_trajectory_only():
    """测试仅使用轨迹特征的 BounceNet"""
    print("\n" + "="*60)
    print("Testing BounceNet (trajectory_only mode)")
    print("="*60)
    
    # 创建模型
    model = create_bouncenet(
        mode='trajectory_only',
        traj_input_dim=5,
        traj_hidden_dim=64,
        traj_feature_dim=64,
        traj_seq_len=11,
        num_classes=3
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 11
    traj_features = torch.randn(batch_size, seq_len, 5)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(traj_features)
        predictions = model.predict(traj_features)
    
    print(f"Input shape: {traj_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted labels: {predictions['labels'].tolist()}")
    print(f"Predicted types: {model.predict_event_type(predictions['labels'])}")
    print(f"Confidence scores: {predictions['confidence'].tolist()}")
    
    print("\n✓ BounceNet (trajectory_only) test passed!")
    return True


def test_bouncenet_fusion():
    """测试融合模式的 BounceNet"""
    print("\n" + "="*60)
    print("Testing BounceNet (fusion mode)")
    print("="*60)
    
    # 创建模型
    model = create_bouncenet(
        mode='fusion',
        traj_input_dim=5,
        traj_hidden_dim=64,
        traj_feature_dim=64,
        traj_seq_len=11,
        patch_size=64,
        visual_feature_dim=128,
        use_temporal_patches=True,
        num_classes=3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 11
    patch_size = 64
    
    traj_features = torch.randn(batch_size, seq_len, 5)
    visual_features = torch.randn(batch_size, seq_len, 3, patch_size, patch_size)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits, confidence = model(
            traj_features, visual_features, return_confidence=True
        )
    
    print(f"Trajectory features shape: {traj_features.shape}")
    print(f"Visual features shape: {visual_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    print("\n✓ BounceNet (fusion) test passed!")
    return True


def test_bouncenet_predictor():
    """测试 BounceNetPredictor"""
    print("\n" + "="*60)
    print("Testing BounceNetPredictor")
    print("="*60)
    
    # 创建模型和预测器
    model = create_bouncenet(mode='trajectory_only')
    predictor = BounceNetPredictor(model, device='cpu')
    
    # 创建模拟轨迹数据
    n_frames = 100
    x = np.random.uniform(100, 400, n_frames).astype(np.float32)
    y = np.cumsum(np.random.uniform(-5, 5, n_frames)).astype(np.float32) + 150
    visibility = np.ones(n_frames)
    
    # 模拟候选事件
    candidates = [
        {'frame': 20, 'event_type': 'hit', 'x': x[20], 'y': y[20], 'confidence': 0.8},
        {'frame': 50, 'event_type': 'landing', 'x': x[50], 'y': y[50], 'confidence': 0.7},
        {'frame': 80, 'event_type': 'hit', 'x': x[80], 'y': y[80], 'confidence': 0.75},
    ]
    
    # 分类候选
    classified = predictor.classify_candidates(
        x, y, visibility, candidates, frames=None
    )
    
    print(f"Number of candidates: {len(candidates)}")
    print(f"Classified candidates:")
    for cand in classified:
        print(f"  Frame {cand['frame']}: "
              f"original={cand['event_type']}, "
              f"predicted={cand.get('predicted_type', 'N/A')}, "
              f"confidence={cand.get('prediction_confidence', 0):.3f}")
    
    print("\n✓ BounceNetPredictor test passed!")
    return True


def test_integration_with_detector():
    """测试与 BounceDetector 的集成"""
    print("\n" + "="*60)
    print("Testing Integration with BounceDetector")
    print("="*60)
    
    # 不带 BounceNet 的检测器 (Phase 1 only)
    detector = BounceDetector()
    print("✓ BounceDetector created (Phase 1 mode)")
    
    # 测试检测
    n_frames = 100
    x = np.cumsum(np.random.uniform(-2, 5, n_frames)).astype(np.float32) + 200
    y = np.cumsum(np.random.uniform(-3, 3, n_frames)).astype(np.float32) + 150
    
    # 添加一些事件特征
    # 模拟击球 (y 方向反转)
    y[30:35] = y[29] - np.arange(1, 6) * 5
    y[35:40] = y[34] + np.arange(1, 6) * 5
    
    visibility = np.ones(n_frames)
    visibility[60:65] = 0  # 模拟消失
    
    events = detector.detect(x, y, visibility)
    
    print(f"Detected {len(events)} events:")
    for event in events[:5]:  # 只显示前5个
        print(f"  Frame {event['frame']}: {event['event_type']} "
              f"(rule: {event.get('rule', 'N/A')}, conf: {event.get('confidence', 0):.2f})")
    
    print("\n✓ Integration test passed!")
    return True


def main():
    print("="*60)
    print("  BounceNet Test Suite")
    print("="*60)
    
    tests = [
        ("Visual Feature Extractor", test_visual_feature_extractor),
        ("BounceNet (trajectory_only)", test_bouncenet_trajectory_only),
        ("BounceNet (fusion)", test_bouncenet_fusion),
        ("BounceNet Predictor", test_bouncenet_predictor),
        ("Integration with Detector", test_integration_with_detector),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {name}")
        if not success:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
