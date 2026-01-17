"""
GEATCat训练
基于 GHGEAT 架构，采用 GNNCat 的输出方式
"""
import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GEATCat_architecture import GEATCat, count_parameters
from utilities_v2.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from utilities_v2.Train_eval_T import train, eval, MAE, R2
from utilities_v2.save_info import save_train_traj

# External utilities
from tqdm import tqdm
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr


def train_GNNGH_T(df, model_name, hyperparameters, resume=False):
    # 模型保存路径：D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models\{model_name}
    base_path = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models'
    path = os.path.join(base_path, model_name)
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report_path = os.path.join(path, f'Report_training_{model_name}.txt')
    report = open(report_path, 'w', encoding='utf-8')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-' * 50)
    
    # Build molecule from SMILES
    mol_column_solvent = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute = 'Molecule_Solute'
    df[mol_column_solute] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    train_index = df.index.tolist()
    
    target = 'log-gamma'
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']
    lr = hyperparameters['lr']
    n_epochs = hyperparameters['n_epochs']
    batch_size = hyperparameters['batch_size']
    attention_weight = hyperparameters.get('attention_weight', 1.0)  # 默认完全使用注意力
    
    start = time.time()
    
    # Data loaders - 优化数据加载速度
    # Windows系统对多进程支持有限，使用单进程模式避免共享内存错误
    import platform
    if platform.system() == 'Windows':
        num_workers = hyperparameters.get('num_workers', 0)  # Windows使用单进程
        persistent_workers = False  # Windows不支持persistent_workers
    else:
        num_workers = hyperparameters.get('num_workers', 8)  # Linux/Mac可以使用多进程
        persistent_workers = hyperparameters.get('persistent_workers', True)
    
    pin_memory = hyperparameters.get('pin_memory', True)  # 启用内存固定
    prefetch_factor = hyperparameters.get('prefetch_factor', 4)  # 增加预取批次数量
    
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory,
                                          persistent_workers=persistent_workers,
                                          prefetch_factor=prefetch_factor)
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3  # ap, bp, topopsa
    model = GEATCat(v_in, e_in, u_in, hidden_dim, attention_weight=attention_weight)
    
    # 确保使用GPU
    if not torch.cuda.is_available():
        print_report('⚠️ 警告: CUDA不可用，将使用CPU训练（速度会很慢）')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print_report(f'✓ 使用GPU训练: {torch.cuda.get_device_name(0)}')
    
    model = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    task_type = 'regression'
    
    # 学习率预热和调度器
    warmup_epochs = hyperparameters.get('warmup_epochs', 5)  # 默认5个epoch预热
    base_scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7)
    
    # 创建带warmup的调度器
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup阶段：线性增加学习率从0到lr
            return float(epoch + 1) / float(warmup_epochs)
        else:
            return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    use_warmup = warmup_epochs > 0
    if use_warmup:
        print_report(f'✓ 学习率预热: {warmup_epochs} 个epoch (从0线性增加到 {lr:.6f})')

    # Mixed precision training with GradScaler
    scaler = None
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print_report('✓ 已启用混合精度训练 (AMP)')
        
        # GPU预热 - 减少第一个epoch的训练时间
        print_report('正在进行GPU预热...')
        model.train()
        # 获取一个小的batch进行预热
        warmup_batch = next(iter(train_loader))
        if len(warmup_batch) == 3:
            warmup_solvent, warmup_solute, warmup_T = warmup_batch
            warmup_T = warmup_T.to(device, non_blocking=True)
        else:
            warmup_solvent, warmup_solute = warmup_batch
            warmup_T = None
        warmup_solvent = warmup_solvent.to(device, non_blocking=True)
        warmup_solute = warmup_solute.to(device, non_blocking=True)
        
        # 执行几次前向和反向传播进行预热
        for _ in range(3):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if warmup_T is not None:
                    _ = model(warmup_solvent, warmup_solute, warmup_T)
                else:
                    _ = model(warmup_solvent, warmup_solute)
            # 不需要真实的loss，只是预热
        torch.cuda.synchronize()  # 等待GPU完成
        print_report('✓ GPU预热完成')
        
        pbar = range(n_epochs)
    else:
        pbar = tqdm(range(n_epochs))

    # To save trajectory
    mae_train = []
    r2_train = []
    train_loss = []
    best_MAE = np.inf

    # 为了兼容checkpoint保存/加载，创建一个统一的scheduler引用
    # 在warmup阶段使用warmup_scheduler，之后使用base_scheduler
    current_scheduler = warmup_scheduler if use_warmup else base_scheduler

    # Check if we are resuming training
    if resume:
        # Load checkpoint
        checkpoint_path = os.path.join(path, f'{model_name}.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 根据start_epoch决定加载哪个scheduler
            start_epoch = checkpoint['epoch'] + 1
            if use_warmup and start_epoch < warmup_epochs:
                warmup_scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
                current_scheduler = warmup_scheduler
            else:
                # 尝试加载base_scheduler，如果checkpoint中有的话
                if 'scheduler_state_dict' in checkpoint:
                    base_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                current_scheduler = base_scheduler
            best_MAE = checkpoint['best_MAE']
            mae_train = checkpoint.get('mae_train', [])
            r2_train = checkpoint.get('r2_train', [])
            train_loss = checkpoint.get('train_loss', [])
            print_report(f"Resuming training from epoch {start_epoch}")
        else:
            print_report("No checkpoint found, starting training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        print(f'\n开始训练 Epoch {epoch+1}/{n_epochs}...', flush=True)
        stats = OrderedDict()
        
        # 学习率预热
        if use_warmup and epoch < warmup_epochs:
            warmup_scheduler.step()
            current_scheduler = warmup_scheduler
            current_lr = optimizer.param_groups[0]['lr']
            if epoch == 0 or (epoch + 1) % max(1, warmup_epochs // 5) == 0:
                print_report(f'Warmup Epoch {epoch+1}/{warmup_epochs}: lr = {current_lr:.8f}')
        
        # Train - 传递 scaler 以启用混合精度训练
        stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler=scaler))
        
        # 第一个epoch跳过评估以加速（可选）
        skip_first_eval = hyperparameters.get('skip_first_eval', False) and epoch == start_epoch
        
        if not skip_first_eval:
            # Evaluation - MAE
            stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
            # Evaluation - R²
            r2_stats = eval(model, device, train_loader, R2, OrderedDict(), 'Train', task_type)
            r2_value = r2_stats.get('R2_Train', 0.0)
            stats.update(r2_stats)
        else:
            # 第一个epoch使用训练loss估算MAE
            r2_value = 0.0
            stats['MAE_Train'] = np.sqrt(stats['Train_loss'])  # 简单估算
            print_report('  第一个epoch跳过完整评估以加速训练')
        
        # Scheduler - warmup后切换到ReduceLROnPlateau
        if use_warmup and epoch >= warmup_epochs - 1:
            if epoch == warmup_epochs - 1:
                print_report(f'Warmup完成，切换到ReduceLROnPlateau调度器')
                current_scheduler = base_scheduler
            if not skip_first_eval:
                base_scheduler.step(stats['MAE_Train'])
                current_scheduler = base_scheduler
        elif not use_warmup:
            if not skip_first_eval:
                base_scheduler.step(stats['MAE_Train'])
                current_scheduler = base_scheduler
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        r2_train.append(r2_value)
        # Save best model
        if mae_train[-1] < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE = mae_train[-1]
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch information
        epoch_info = f'Epoch {epoch+1}/{n_epochs} - MAE: {mae_train[-1]:.6f}, R²: {r2_value:.6f}, Best MAE: {best_MAE:.6f}, Time: {epoch_time:.2f}s'
        print_report(epoch_info)
        print(epoch_info, flush=True)

        # Save checkpoint
        checkpoint_path = os.path.join(path, f'{model_name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': current_scheduler.state_dict() if hasattr(current_scheduler, 'state_dict') else {},
            'best_MAE': best_MAE,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'train_loss': train_loss
        }, checkpoint_path)

    print_report('-' * 30)
    best_epoch = mae_train.index(min(mae_train)) + 1
    print_report('Best Epoch     : ' + str(best_epoch))
    print_report('Training MAE   : ' + str(mae_train[best_epoch - 1]))
    if len(r2_train) >= best_epoch:
        print_report('Training R²    : ' + str(r2_train[best_epoch - 1]))
    print_report('Training Loss  : ' + str(train_loss[best_epoch - 1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train'] = mae_train
    df_model_training['R2_Train'] = r2_train
    save_train_traj(path, df_model_training, valid=False)

    # Save best model
    best_model_path = os.path.join(path, f'{model_name}_best.pth')
    torch.save(best_model, best_model_path)

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()


hyperparameters_dict = {'hidden_dim': 38,
                        'lr': 8.00e-04,
                        'n_epochs': 434,
                        'batch_size': 104,
                        'attention_weight': 0.8,
                        'warmup_epochs': 5,  # 学习率预热epoch数
                        'skip_first_eval': False,  # 是否跳过第一个epoch的评估以加速
                        'num_workers': 0,  # Windows系统使用单进程（避免共享内存错误）
                        'pin_memory': True,  # 启用内存固定，加速GPU传输
                        'persistent_workers': False,  # Windows不支持persistent_workers
                        'prefetch_factor': 4  # 增加预取批次数量
                        }

# 训练数据路径
train_data_path = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\data\processed\new_dataset\train_dataset\v2\molecule\molecule_train.csv'

if __name__ == '__main__':
    # 加载训练数据
    df = pd.read_csv(train_data_path)
    
    # 训练模型
    train_GNNGH_T(df, 'GEATCat', hyperparameters_dict)

