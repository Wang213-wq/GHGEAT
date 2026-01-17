"""
训练脚本
用于训练张量补全模型
"""
import numpy as np
import pandas as pd
from data_loader import DataLoader, load_example_data
from tensor_completion import TensorCompletionModel, calculate_metrics
from config import MODEL_CONFIG, DATA_CONFIG
import os
import pickle
from sklearn.model_selection import KFold
import json


def cross_validation(data_loader: DataLoader, n_folds: int = 10, 
                     ranks: tuple = (5, 5, 2), max_iter: int = 100):
    """
    执行10折交叉验证
    
    参数:
        data_loader: 数据加载器
        n_folds: 折数
        ranks: Tucker分解的秩
        max_iter: 最大迭代次数
        
    返回:
        所有折的评估结果
    """
    tensor, mask = data_loader.get_tensor()
    all_results = []
    
    print(f"\n开始 {n_folds} 折交叉验证...")
    print(f"数据张量形状: {tensor.shape}")
    print(f"总数据点数: {np.sum(mask)}")
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"折 {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # 创建系统级分割
        train_mask, test_mask = data_loader.create_system_wise_split(
            n_folds=n_folds, fold=fold, random_state=DATA_CONFIG['random_state']
        )
        
        train_size = np.sum(train_mask)
        test_size = np.sum(test_mask)
        print(f"训练集大小: {train_size}")
        print(f"测试集大小: {test_size}")
        
        # 创建训练张量（缺失值用NaN）
        train_tensor = tensor.copy()
        train_tensor[~train_mask] = np.nan
        
        # 训练模型
        print("\n训练模型...")
        import time
        start_time = time.time()
        model = TensorCompletionModel(ranks=ranks)
        # 使用MAE为监控指标，迭代次数按要求为 max_iter
        from config import MODEL_CONFIG
        patience = MODEL_CONFIG.get('early_stopping_patience', 10)
        model.fit(train_tensor, train_mask, max_iter=max_iter, verbose=True, early_stopping_patience=patience)
        total_time = time.time() - start_time
        print(f"\n本折训练总用时: {total_time:.2f} 秒")
        
        # 预测（使用完整重构张量，在测试掩码位置评估）
        print("\n进行预测...")
        reconstructed = model.predict(tensor, test_mask)
        
        # 评估：仅在测试掩码位置取预测
        y_true = tensor[test_mask]
        y_pred = reconstructed[test_mask]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"\n折 {fold + 1} 评估结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        all_results.append({
            'fold': fold + 1,
            'metrics': metrics,
            'train_size': train_size,
            'test_size': test_size
        })
    
    # 计算平均指标
    print(f"\n{'='*60}")
    print("交叉验证总结")
    print(f"{'='*60}")
    
    avg_metrics = {}
    for key in all_results[0]['metrics'].keys():
        avg_metrics[key] = np.mean([r['metrics'][key] for r in all_results])
        std_metrics = np.std([r['metrics'][key] for r in all_results])
        print(f"{key}: {avg_metrics[key]:.6f} ± {std_metrics:.6f}")
    
    return all_results, avg_metrics


def train_full_model(data_loader: DataLoader, ranks: tuple = (5, 5, 2), 
                     max_iter: int = 100, save_path: str = None,
                     use_full_data: bool = False,
                     train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                     resume_from: str = None,
                     checkpoint_dir: str = None,
                     checkpoint_every: int = None):
    """
    使用全部数据训练模型
    
    参数:
        data_loader: 数据加载器
        ranks: Tucker分解的秩
        max_iter: 最大迭代次数
        save_path: 模型保存路径
        
    返回:
        训练好的模型
    """
    tensor, mask = data_loader.get_tensor()
    
    if use_full_data:
        print(f"\n使用全部数据训练模型...")
        print(f"数据张量形状: {tensor.shape}")
        print(f"数据点数量: {np.sum(mask)}")
        
        # 训练集使用全部已知点，不单独划分验证/测试
        train_mask = mask
        val_mask = None
        test_mask = None
    else:
        print(f"\n按比例划分 训练/验证/测试 (系统级)...")
        print(f"比例: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
        train_mask, val_mask, test_mask = data_loader.create_system_wise_split_with_validation(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
        print(f"训练集点数: {int(np.sum(train_mask))}")
        print(f"验证集点数: {int(np.sum(val_mask))}")
        print(f"测试集点数: {int(np.sum(test_mask))}")
    
    # 训练张量（未知点置 NaN，仅在 train_mask 的位置贡献损失）
    train_tensor = tensor.copy()
    if not use_full_data:
        # 在训练时屏蔽掉非训练集的观测（防止泄漏）
        train_tensor[~train_mask] = np.nan
    
    # 训练模型（基于训练集早停）
    import time
    start_time = time.time()
    model = TensorCompletionModel(ranks=ranks)
    from config import MODEL_CONFIG
    patience = MODEL_CONFIG.get('early_stopping_patience', 10)
    conv_window = MODEL_CONFIG.get('convergence_window', 5)
    conv_rel_tol = MODEL_CONFIG.get('convergence_rel_tol', 1e-4)
    min_epochs = MODEL_CONFIG.get('min_epochs', 10)
    reg_lambda = MODEL_CONFIG.get('ridge_lambda', 1e-3)
    resume_state = None
    if resume_from and os.path.exists(resume_from):
        print(f"\n从断点加载: {resume_from}")
        try:
            ckpt = TensorCompletionModel.load_checkpoint(resume_from)
            resume_state = ckpt.get('model_state', None)
        except Exception as e:
            print(f"断点加载失败: {e}")
    
    # 默认启用断点保存：如果未指定，则每10轮保存一次
    if checkpoint_dir is None:
        checkpoint_dir = 'models/checkpoints'
    if checkpoint_every is None:
        checkpoint_every = 10  # 默认每10轮保存一次
    
    if checkpoint_every > 0:
        print(f"\n断点保存设置: 目录={checkpoint_dir}, 每{checkpoint_every}轮保存一次")
    
    model.fit(
        train_tensor, train_mask,
        max_iter=max_iter, verbose=True,
        early_stopping_patience=patience,
        convergence_window=conv_window,
        convergence_rel_tol=conv_rel_tol,
        min_epochs=min_epochs,
        resume_state=resume_state,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        reg_lambda=reg_lambda
    )
    total_time = time.time() - start_time
    print(f"\n训练总用时: {total_time:.2f} 秒")

    # 若存在验证/测试划分，训练后分别评估
    if not use_full_data:
        from tensor_completion import calculate_metrics
        reconstructed = model.tucker.reconstruct()
        # 使用重构值在对应掩码位置做评估，避免从原始张量拷贝造成泄漏
        if val_mask is not None:
            val_metrics = calculate_metrics(tensor[val_mask], reconstructed[val_mask])
            print("\n验证集评估指标:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")
        if test_mask is not None:
            test_metrics = calculate_metrics(tensor[test_mask], reconstructed[test_mask])
            print("\n测试集评估指标:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.6f}")
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'data_loader': data_loader,
                'ranks': ranks
            }, f)
        print(f"\n模型已保存到: {save_path}")
    
    return model


def rank_selection(data_loader: DataLoader, rank_candidates: list, 
                   n_folds: int = 5, max_iter: int = 50):
    """
    选择最优的秩组合
    
    参数:
        data_loader: 数据加载器
        rank_candidates: 候选秩组合列表，例如 [(3,3,2), (5,5,2), (7,7,3)]
        n_folds: 用于秩选择的折数（可以使用较少的折数以加快速度）
        max_iter: 最大迭代次数
        
    返回:
        最优秩组合和对应的指标
    """
    print(f"\n开始秩选择...")
    print(f"候选秩组合: {rank_candidates}")
    
    best_ranks = None
    best_wmse = float('inf')
    results = []
    
    for ranks in rank_candidates:
        print(f"\n{'='*60}")
        print(f"测试秩组合: {ranks}")
        print(f"{'='*60}")
        
        try:
            all_results, avg_metrics = cross_validation(
                data_loader, n_folds=n_folds, ranks=ranks, max_iter=max_iter
            )
            
            wmse = avg_metrics['wMSE']
            results.append({
                'ranks': ranks,
                'metrics': avg_metrics
            })
            
            print(f"\n秩组合 {ranks} 的 wMSE: {wmse:.6f}")
            
            if wmse < best_wmse:
                best_wmse = wmse
                best_ranks = ranks
                print(f"✓ 新的最佳秩组合!")
        
        except Exception as e:
            print(f"秩组合 {ranks} 训练失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("秩选择总结")
    print(f"{'='*60}")
    print(f"最佳秩组合: {best_ranks}")
    print(f"最佳 wMSE: {best_wmse:.6f}")
    
    return best_ranks, results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练张量补全模型')
    parser.add_argument('--data_path', type=str, default='data/raw/Brouwer_2021.csv',
                       help='数据文件路径（CSV格式）')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cv', 'rank_selection'],
                       help='运行模式: train=训练完整模型, cv=交叉验证, rank_selection=秩选择')
    parser.add_argument('--ranks', type=int, nargs=3, default=[5, 5, 2],
                       help='Tucker分解的秩 (r1 r2 r3)')
    parser.add_argument('--max_iter', type=int, default=0,
                       help='最大迭代次数（设为 0 表示不限定，直到早停触发）')
    parser.add_argument('--use_full_data', action='store_true',
                       help='是否使用全部数据进行训练（不划分验证/测试）')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='系统级训练集比例（与 val_ratio, test_ratio 之和为 1）')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='系统级验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='系统级测试集比例')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='交叉验证折数')
    parser.add_argument('--save_path', type=str, default='models/tcm_model.pkl',
                       help='模型保存路径')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints',
                       help='断点保存目录（默认: models/checkpoints）')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                       help='每多少个epoch保存一次断点（默认: 10，设为0表示不保存）')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从指定断点文件恢复训练（.ckpt）')
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    if args.data_path and os.path.exists(args.data_path):
        # 从文件加载数据
        df = pd.read_csv(args.data_path)
        # 假设CSV包含列: solute, solvent, temperature, ln_gamma_inf
    else:
        # 使用示例数据
        print("使用示例数据...")
        df = load_example_data()
    
    # 创建数据加载器
    loader = DataLoader()
    loader.load_from_dataframe(df)
    loader.create_temperature_bins(bin_width=1.0)  # 根据论文，使用1K bins
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 训练完整模型
        model = train_full_model(
            loader, 
            ranks=tuple(args.ranks),
            max_iter=args.max_iter,
            save_path=args.save_path,
            use_full_data=args.use_full_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            resume_from=args.resume_from,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every
        )
    
    elif args.mode == 'cv':
        # 交叉验证
        all_results, avg_metrics = cross_validation(
            loader,
            n_folds=args.n_folds,
            ranks=tuple(args.ranks),
            max_iter=args.max_iter
        )
        
        # 保存结果
        results_path = 'results/cv_results.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'all_results': all_results,
                'avg_metrics': avg_metrics,
                'ranks': args.ranks
            }, f, indent=2)
        print(f"\n交叉验证结果已保存到: {results_path}")
    
    elif args.mode == 'rank_selection':
        # 秩选择
        rank_candidates = [
            (3, 3, 2), (4, 4, 2), (5, 5, 2), (6, 6, 2),
            (5, 5, 3), (7, 7, 2), (7, 7, 3)
        ]
        best_ranks, results = rank_selection(
            loader,
            rank_candidates=rank_candidates,
            n_folds=5,  # 使用较少的折数以加快速度
            max_iter=50
        )
        
        # 保存结果
        results_path = 'results/rank_selection_results.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'best_ranks': best_ranks,
                'all_results': results
            }, f, indent=2)
        print(f"\n秩选择结果已保存到: {results_path}")


if __name__ == "__main__":
    main()


