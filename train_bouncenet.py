"""
BounceNet 训练脚本
Training Script for BounceNet

使用标注数据训练事件分类网络。
"""

import os
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from bounce_detection.bouncenet import (
    BounceNet, BounceNetLoss, EVENT_LABELS, LABEL_TO_EVENT, create_bouncenet
)
from bounce_detection.dataset import BounceNetDataset, create_dataloaders


class BounceNetTrainer:
    """BounceNet 训练器"""
    
    def __init__(self,
                 model: BounceNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: BounceNetLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = 'ckpts/bouncenet'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            traj_features = batch['traj_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 视觉特征（如果有）
            visual_features = None
            if 'visual_features' in batch:
                visual_features = batch['visual_features'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits, confidence = self.model(
                traj_features, visual_features, return_confidence=True
            )
            
            # 计算损失
            losses = self.criterion(logits, labels, confidence)
            loss = losses['total']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 每类统计
        class_correct = {i: 0 for i in range(self.model.num_classes)}
        class_total = {i: 0 for i in range(self.model.num_classes)}
        
        for batch in self.val_loader:
            traj_features = batch['traj_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            visual_features = None
            if 'visual_features' in batch:
                visual_features = batch['visual_features'].to(self.device)
            
            logits, confidence = self.model(
                traj_features, visual_features, return_confidence=True
            )
            
            losses = self.criterion(logits, labels, confidence)
            loss = losses['total']
            
            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # 每类统计
            for i in range(self.model.num_classes):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += ((pred == labels) & mask).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # 计算每类准确率
        class_acc = {}
        for i in range(self.model.num_classes):
            if class_total[i] > 0:
                class_acc[LABEL_TO_EVENT[i]] = class_correct[i] / class_total[i]
            else:
                class_acc[LABEL_TO_EVENT[i]] = 0.0
        
        return avg_loss, accuracy, class_acc
    
    def train(self, num_epochs: int, early_stopping: int = 10):
        """完整训练流程"""
        no_improve = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, class_acc = self.validate()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Class Acc: {class_acc}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best.pt', epoch, val_acc)
                print(f"  -> New best model saved!")
                no_improve = 0
            else:
                no_improve += 1
            
            # 早停
            if no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存最终模型
        self.save_checkpoint('last.pt', epoch, val_acc)
        
        # 保存训练历史
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'mode': self.model.mode,
                'num_classes': self.model.num_classes,
                'traj_seq_len': self.model.traj_seq_len
            }
        }
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)


def find_label_files(label_dir: str) -> Tuple[List[str], List[str]]:
    """查找标注文件和对应的 CSV 文件"""
    csv_paths = []
    label_paths = []
    
    # 查找所有 JSON 标注文件
    for label_path in glob.glob(os.path.join(label_dir, '**', '*.json'), recursive=True):
        # 尝试找到对应的 CSV 文件
        label_name = os.path.basename(label_path)
        csv_name = label_name.replace('_labels.json', '.csv')
        
        # 在常见位置查找 CSV
        possible_csv_paths = [
            label_path.replace('_labels.json', '.csv'),
            label_path.replace('labels', 'csv').replace('_labels.json', '.csv'),
            os.path.join(os.path.dirname(label_path), '..', 'csv', csv_name)
        ]
        
        for csv_path in possible_csv_paths:
            if os.path.exists(csv_path):
                csv_paths.append(csv_path)
                label_paths.append(label_path)
                break
    
    return csv_paths, label_paths


def main():
    parser = argparse.ArgumentParser(description='Train BounceNet')
    
    # 数据参数
    parser.add_argument('--label_dir', type=str, required=True,
                        help='标注文件目录')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='CSV 文件目录（可选）')
    
    # 模型参数
    parser.add_argument('--mode', type=str, default='trajectory_only',
                        choices=['trajectory_only', 'visual_only', 'fusion'],
                        help='特征模式')
    parser.add_argument('--window_size', type=int, default=5,
                        help='特征窗口大小（单侧）')
    parser.add_argument('--traj_hidden_dim', type=int, default=64,
                        help='轨迹编码器隐藏维度')
    parser.add_argument('--traj_feature_dim', type=int, default=64,
                        help='轨迹特征维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='早停轮数')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='ckpts/bouncenet',
                        help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 查找数据文件
    print("Looking for label files...")
    csv_paths, label_paths = find_label_files(args.label_dir)
    
    if len(csv_paths) == 0:
        print("No labeled data found!")
        return
    
    print(f"Found {len(csv_paths)} labeled files")
    
    # 创建数据加载器
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        csv_paths=csv_paths,
        label_paths=label_paths,
        mode=args.mode,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        window_size=args.window_size
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # 创建模型
    print("Creating model...")
    model = create_bouncenet(
        mode=args.mode,
        traj_seq_len=2 * args.window_size + 1,
        traj_hidden_dim=args.traj_hidden_dim,
        traj_feature_dim=args.traj_feature_dim
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 创建损失函数
    criterion = BounceNetLoss(use_focal_loss=True)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建训练器
    trainer = BounceNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 开始训练
    print("Starting training...")
    print("=" * 60)
    trainer.train(num_epochs=args.epochs, early_stopping=args.early_stopping)
    
    print("=" * 60)
    print(f"Training complete! Best val acc: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
