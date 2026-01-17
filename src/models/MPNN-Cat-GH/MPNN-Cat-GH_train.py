"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time

from models import SoluteSolventGraphModel
from utils import GraphDataLoader, calculate_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    skipped_batches = 0
    total_samples = 0
    skipped_samples = 0
    
    for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="训练中"):
        # 移动到设备
        solute_data = solute_data.to(device)
        solvent_data = solvent_data.to(device)
        labels = labels.to(device)
        temperatures = temperatures.to(device)
        
        batch_size = labels.size(0)
        total_samples += batch_size
        
        # 前向传播
        optimizer.zero_grad()
        predictions, _ = model(solute_data, solvent_data, temperatures)
        predictions = predictions.squeeze()
        
        # 检查预测值中是否有 NaN 或 Inf
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            skipped_batches += 1
            skipped_samples += batch_size
            continue
        
        # 计算损失
        loss = criterion(predictions, labels)
        
        # 检查损失是否为 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_batches += 1
            skipped_samples += batch_size
            continue
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否有NaN/Inf（优化：只在检测到时才处理）
        # 先快速检查是否有NaN/Inf，避免遍历所有参数
        has_nan_grad = False
        # 只检查前几个参数，如果发现NaN/Inf再处理所有参数
        param_list = list(model.parameters())
        if len(param_list) > 0 and param_list[0].grad is not None:
            # 快速检查：只检查第一个参数的梯度
            first_grad = param_list[0].grad
            if torch.isnan(first_grad).any() or torch.isinf(first_grad).any():
                has_nan_grad = True
                # 如果发现NaN/Inf，处理所有参数
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if has_nan_grad:
            print(f"警告: 检测到 NaN 或 Inf 梯度，已替换为有限值")
        
        # 增强梯度裁剪（更严格的裁剪）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # 记录（只有成功处理的批次才记录）
        total_loss += loss.item()
        pred_np = predictions.detach().cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # 过滤掉 NaN 和 Inf 值
        valid_mask = np.isfinite(pred_np) & np.isfinite(label_np)
        if valid_mask.sum() > 0:
            all_preds.extend(pred_np[valid_mask])
            all_labels.extend(label_np[valid_mask])
        
        num_batches += 1
    
    # 计算平均损失（只基于成功处理的批次）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 添加统计信息
    if skipped_batches > 0:
        skip_ratio = skipped_batches / (num_batches + skipped_batches) * 100
        sample_skip_ratio = skipped_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"  警告: 跳过了 {skipped_batches} 个批次 ({skip_ratio:.1f}%), {skipped_samples} 个样本 ({sample_skip_ratio:.1f}%)")
    
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    skipped_batches = 0
    total_samples = 0
    skipped_samples = 0
    
    with torch.no_grad():
        for solute_data, solvent_data, labels, temperatures in tqdm(dataloader, desc="验证中"):
            # 移动到设备
            solute_data = solute_data.to(device)
            solvent_data = solvent_data.to(device)
            labels = labels.to(device)
            temperatures = temperatures.to(device)
            
            batch_size = labels.size(0)
            total_samples += batch_size
            
            # 前向传播
            predictions, _ = model(solute_data, solvent_data, temperatures)
            predictions = predictions.squeeze()
            
            # 检查预测值中是否有 NaN 或 Inf
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                skipped_batches += 1
                skipped_samples += batch_size
                continue
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 检查损失是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                skipped_samples += batch_size
                continue
            
            # 记录（只有成功处理的批次才记录）
            total_loss += loss.item()
            pred_np = predictions.cpu().numpy()
            label_np = labels.cpu().numpy()
            
            # 过滤掉 NaN 和 Inf 值
            valid_mask = np.isfinite(pred_np) & np.isfinite(label_np)
            if valid_mask.sum() > 0:
                all_preds.extend(pred_np[valid_mask])
                all_labels.extend(label_np[valid_mask])
            
            num_batches += 1
    
    # 计算平均损失（只基于成功处理的批次）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 添加统计信息
    if skipped_batches > 0:
        skip_ratio = skipped_batches / (num_batches + skipped_batches) * 100
        sample_skip_ratio = skipped_samples / total_samples * 100 if total_samples > 0 else 0
        print(f"  警告: 跳过了 {skipped_batches} 个批次 ({skip_ratio:.1f}%), {skipped_samples} 个样本 ({sample_skip_ratio:.1f}%)")
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='训练溶质-溶剂交互图学习模型')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['training']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    data_loader = GraphDataLoader(
        data_path=config['data']['data_path'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['data']['random_seed']
    )
    
    # 创建模型
    print("创建模型...")
    model = SoluteSolventGraphModel(
        input_dim=128,  # 根据实际数据调整
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm'],
        output_dim=1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        # 恢复scheduler状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"从第 {start_epoch} 个epoch恢复训练，最佳验证损失: {best_val_loss:.4f}")
    
    # 训练循环
    print("开始训练...")
    log_file = os.path.join(
        config['logging']['log_dir'],
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Epoch\tTrain Loss\tTrain MAE\tTrain R2\tVal Loss\tVal MAE\tVal R2\n")
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        current_epoch = epoch + 1
        print(f"\n{'='*60}")
        print(f"第 {current_epoch}/{config['training']['num_epochs']} 轮训练")
        print(f"{'='*60}")
        
        # 训练
        train_dataloader = data_loader.get_dataloader('train')
        train_loss, train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        
        # 验证
        val_dataloader = data_loader.get_dataloader('val')
        val_loss, val_metrics = validate_epoch(
            model, val_dataloader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印结果
        print(f"\n[第 {current_epoch} 轮] 训练结果:")
        print(f"  损失: {train_loss:.4f}, MAE: {train_metrics['MAE']:.4f}, R²: {train_metrics['R2']:.4f}")
        print(f"[第 {current_epoch} 轮] 验证结果:")
        print(f"  损失: {val_loss:.4f}, MAE: {val_metrics['MAE']:.4f}, R²: {val_metrics['R2']:.4f}")
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_metrics['MAE']:.4f}\t{train_metrics['R2']:.4f}\t"
                   f"{val_loss:.4f}\t{val_metrics['MAE']:.4f}\t{val_metrics['R2']:.4f}\n")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                config['logging']['save_dir'], 'best_model.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config
            }, checkpoint_path)
            print(f"保存最佳模型到 {checkpoint_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"验证损失在 {config['training']['early_stopping_patience']} 个epoch内未改善，提前停止训练")
            break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config['logging']['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
    
    # 记录训练结束时间并计算总时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    # 格式化时间输出
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print("="*60)


if __name__ == '__main__':
    main()

