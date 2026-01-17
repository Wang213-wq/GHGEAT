"""
使用标准GNNExplainer进行特征重要性分析
针对原子特征和化学键特征

GNNExplainer通过学习特征掩码来识别重要的特征子集，适用于双图模型（溶剂和溶质）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import warnings
# 抑制RDKit的弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')
from pathlib import Path
import sys
import time
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'scr' / 'models'))
sys.path.append(str(project_root / 'scr' / 'models' / 'utilities'))
sys.path.append(str(project_root / 'scr' / 'model'))

try:
    from scr.models.utilities_v2.mol2graph import n_atom_features, n_bond_features, get_dataloader_pairs_T, sys2graph
    # 尝试导入带input_projection的版本（用于full_search模型）
    try:
        from scr.model.GH_pyGEAT_architecture_0615_v0 import GHGEAT as GHGEAT
        print("使用 GH_pyGEAT_architecture_0615_v0 (带input_projection)")
    except ImportError:
        from scr.model.GH_pyGEAT_wo_architecture_0615_v0 import GHGEAT
        print("使用 GH_pyGEAT_wo_architecture_0615_v0")
except ImportError:
    # 尝试其他路径
    from scr.models.utilities_v2.mol2graph import n_atom_features, n_bond_features, get_dataloader_pairs_T, sys2graph
    from scr.model.GHGEAT_wo_architecture import GHGEAT_wo

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 原子特征维度映射表
ATOM_FEATURE_MAPPING = {
    'Atomic_Symbol': (0, 16),      # 维度 0-15: 原子类型 (C, N, O, Cl, S, F, Br, I, Si, Sn, Pb, Ge, H, P, Hg, Te)
    'Ring_Presence': (16, 1),      # 维度 16: 是否在环中
    'Aromaticity': (17, 1),        # 维度 17: 是否芳香
    'Hybridization': (18, 4),      # 维度 18-21: 杂化方式 (S, SP, SP2, SP3)
    'Neighbor_Count': (22, 5),     # 维度 22-26: 键数量 (0,1,2,3,4)
    'Formal_Charge': (27, 3),      # 维度 27-29: 形式电荷 (0,1,-1)
    'Hydrogen_Count': (30, 4),     # 维度 30-33: 连接氢原子数 (0,1,2,3)
    'Chirality': (34, 3)          # 维度 34-36: 手性 (Unspecified, CW, CCW)
}

# 化学键特征维度映射表
BOND_FEATURE_MAPPING = {
    'Bond_Type': (0, 4),           # 维度 0-3: 键类型 (Single, Double, Triple, Aromatic)
    'Conjugated': (4, 1),          # 维度 4: 是否共轭
    'Ring_Part': (5, 1),           # 维度 5: 是否为环的一部分
    'Stereochemistry': (6, 3)      # 维度 6-8: 立体化学 (None, Z-type, E-type)
}


class GNNExplainer:
    """
    标准的GNNExplainer实现
    通过学习特征掩码来识别重要的特征子集
    
    原理：
    -----
    GNNExplainer通过优化以下目标函数来学习掩码：
    
    max_{M} log P_Φ(Y = y_G | G_c) - λ ||M||_1
    
    其中：
    - M: 特征掩码（mask）
    - G_c: 掩码后的图
    - λ: 正则化系数（控制掩码稀疏性）
    - ||M||_1: L1正则化项（鼓励稀疏掩码）
    
    优化过程：
    1. 初始化掩码（全1或随机）
    2. 使用梯度下降优化掩码
    3. 通过sigmoid函数将掩码限制在[0,1]范围内
    4. 迭代优化直到收敛
    """
    
    def __init__(self, model, epochs=100, lr=0.01, lambda_reg=0.01, 
                 temperature=1.0, device='cuda'):
        """
        初始化GNNExplainer
        
        参数:
        -----
        model: 要解释的模型
        epochs: 优化迭代次数
        lr: 学习率
        lambda_reg: L1正则化系数
        temperature: 温度参数（用于sigmoid的锐化）
        device: 计算设备
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.device = device
    
    def explain(self, batch_solvent, batch_solute, T, target=None):
        """
        解释单个样本，学习特征掩码
        
        参数:
        -----
        batch_solvent: 溶剂图数据
        batch_solute: 溶质图数据
        T: 温度
        target: 目标值（如果为None，使用模型预测值）
        
        返回:
        -----
        atom_mask_solvent: 溶剂原子特征掩码
        atom_mask_solute: 溶质原子特征掩码
        bond_mask_solvent: 溶剂化学键特征掩码
        bond_mask_solute: 溶质化学键特征掩码
        """
        # 保存原始模式
        was_training = self.model.training
        
        # 初始化掩码（使用可学习的参数）
        atom_mask_solvent = torch.nn.Parameter(
            torch.ones(batch_solvent.x.shape[1], device=self.device)
        )
        atom_mask_solute = torch.nn.Parameter(
            torch.ones(batch_solute.x.shape[1], device=self.device)
        )
        
        bond_mask_solvent = None
        bond_mask_solute = None
        if batch_solvent.edge_attr is not None:
            bond_mask_solvent = torch.nn.Parameter(
                torch.ones(batch_solvent.edge_attr.shape[1], device=self.device)
            )
        if batch_solute.edge_attr is not None:
            bond_mask_solute = torch.nn.Parameter(
                torch.ones(batch_solute.edge_attr.shape[1], device=self.device)
            )
        
        # 创建优化器
        params = [atom_mask_solvent, atom_mask_solute]
        if bond_mask_solvent is not None:
            params.append(bond_mask_solvent)
        if bond_mask_solute is not None:
            params.append(bond_mask_solute)
        
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        # 获取目标值（如果未提供，使用模型预测）
        # 在eval模式下获取目标值（不需要梯度）
        if target is None:
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch_solvent, batch_solute, T)
                target = output.detach()
        
        # 优化掩码（需要切换到训练模式以支持反向传播）
        self.model.train()
        try:
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                
                # 应用掩码（通过sigmoid确保值在[0,1]范围内）
                atom_mask_solvent_sigmoid = torch.sigmoid(atom_mask_solvent * self.temperature)
                atom_mask_solute_sigmoid = torch.sigmoid(atom_mask_solute * self.temperature)
                
                # 创建掩码后的特征
                masked_solvent_x = batch_solvent.x * atom_mask_solvent_sigmoid.unsqueeze(0)
                masked_solute_x = batch_solute.x * atom_mask_solute_sigmoid.unsqueeze(0)
                
                # 创建掩码后的图数据
                masked_solvent = Data(
                    x=masked_solvent_x,
                    edge_index=batch_solvent.edge_index,
                    edge_attr=batch_solvent.edge_attr,
                    batch=batch_solvent.batch,
                    ap=batch_solvent.ap,
                    bp=batch_solvent.bp,
                    topopsa=batch_solvent.topopsa,
                    inter_hb=batch_solvent.inter_hb,
                    hb=batch_solvent.hb,
                    y=batch_solvent.y
                ).to(self.device)
                
                masked_solute = Data(
                    x=masked_solute_x,
                    edge_index=batch_solute.edge_index,
                    edge_attr=batch_solute.edge_attr,
                    batch=batch_solute.batch,
                    ap=batch_solute.ap,
                    bp=batch_solute.bp,
                    topopsa=batch_solute.topopsa,
                    inter_hb=batch_solute.inter_hb,
                    hb=batch_solute.hb,
                    y=batch_solute.y
                ).to(self.device)
                
                # 如果有化学键掩码，也应用
                if bond_mask_solvent is not None and masked_solvent.edge_attr is not None:
                    bond_mask_solvent_sigmoid = torch.sigmoid(bond_mask_solvent * self.temperature)
                    masked_solvent.edge_attr = masked_solvent.edge_attr * bond_mask_solvent_sigmoid.unsqueeze(0)
                
                if bond_mask_solute is not None and masked_solute.edge_attr is not None:
                    bond_mask_solute_sigmoid = torch.sigmoid(bond_mask_solute * self.temperature)
                    masked_solute.edge_attr = masked_solute.edge_attr * bond_mask_solute_sigmoid.unsqueeze(0)
                
                # 前向传播
                output = self.model(masked_solvent, masked_solute, T)
                
                # 计算损失：最大化预测概率（最小化负对数似然）
                # 使用MSE损失（假设回归任务）
                loss_pred = F.mse_loss(output, target)
                
                # L1正则化项（鼓励稀疏掩码）
                loss_reg = self.lambda_reg * (
                    torch.norm(atom_mask_solvent_sigmoid, p=1) +
                    torch.norm(atom_mask_solute_sigmoid, p=1)
                )
                
                if bond_mask_solvent is not None:
                    loss_reg += self.lambda_reg * torch.norm(bond_mask_solvent_sigmoid, p=1)
                if bond_mask_solute is not None:
                    loss_reg += self.lambda_reg * torch.norm(bond_mask_solute_sigmoid, p=1)
                
                # 总损失
                loss = loss_pred + loss_reg
                
                # 反向传播
                loss.backward()
                optimizer.step()
        finally:
            # 恢复原始模式
            self.model.train(was_training)
        
        # 返回最终的掩码（应用sigmoid）
        with torch.no_grad():
            atom_mask_solvent_final = torch.sigmoid(atom_mask_solvent * self.temperature)
            atom_mask_solute_final = torch.sigmoid(atom_mask_solute * self.temperature)
            
            bond_mask_solvent_final = None
            bond_mask_solute_final = None
            if bond_mask_solvent is not None:
                bond_mask_solvent_final = torch.sigmoid(bond_mask_solvent * self.temperature)
            if bond_mask_solute is not None:
                bond_mask_solute_final = torch.sigmoid(bond_mask_solute * self.temperature)
        
        return (atom_mask_solvent_final, atom_mask_solute_final, 
                bond_mask_solvent_final, bond_mask_solute_final)


def explain_atom_features_gnnexplainer(model, dataloader, device, n_samples=50, 
                                        epochs=100, lr=0.01, lambda_reg=0.01):
    """
    使用标准的GNNExplainer方法分析原子特征重要性
    
    参数:
    -----
    model: 要解释的模型
    dataloader: 数据加载器
    device: 计算设备
    n_samples: 分析的样本数
    epochs: GNNExplainer优化迭代次数
    lr: 学习率
    lambda_reg: L1正则化系数
    """
    model.eval()
    
    # 创建GNNExplainer实例
    explainer = GNNExplainer(model, epochs=epochs, lr=lr, 
                             lambda_reg=lambda_reg, device=device)
    
    # 存储每个特征的重要性分数
    feature_importance = {feat_name: [] for feat_name in ATOM_FEATURE_MAPPING.keys()}
    
    sample_count = 0
    processing_times = []
    with tqdm(total=n_samples, desc="GNNExplainer分析原子特征") as pbar:
        for batch_solvent, batch_solute, T in dataloader:
            if sample_count >= n_samples:
                break
            
            # 记录开始时间
            start_time = time.time()
                
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            T = T.to(device)
            
            try:
                # 使用GNNExplainer学习掩码
                atom_mask_solvent, atom_mask_solute, _, _ = explainer.explain(
                    batch_solvent, batch_solute, T
                )
                
                # 合并溶剂和溶质的掩码（平均）
                if atom_mask_solvent is not None and atom_mask_solute is not None:
                    # 对每个图分别处理
                    for graph_idx in range(batch_solvent.num_graphs):
                        # 获取该图的节点索引
                        if hasattr(batch_solvent, 'batch'):
                            graph_mask = batch_solvent.batch == graph_idx
                            solvent_mask = atom_mask_solvent  # 掩码是特征级别的，对所有节点相同
                        else:
                            solvent_mask = atom_mask_solvent
                        
                        if hasattr(batch_solute, 'batch'):
                            graph_mask = batch_solute.batch == graph_idx
                            solute_mask = atom_mask_solute
                        else:
                            solute_mask = atom_mask_solute
                        
                        # 合并溶剂和溶质的掩码（平均）
                        combined_mask = (solvent_mask + solute_mask) / 2.0
                        
                        # 分析每个原子特征的重要性
                        for feat_name, (start_idx, feat_dim) in ATOM_FEATURE_MAPPING.items():
                            # 掩码值越大，特征越重要
                            feat_importance = combined_mask[start_idx:start_idx+feat_dim].mean().item()
                            feature_importance[feat_name].append(feat_importance)
                        
                        # 记录处理时间
                        elapsed_time = time.time() - start_time
                        processing_times.append(elapsed_time)
                        pbar.set_postfix({'时间': f'{elapsed_time:.2f}s', '平均': f'{np.mean(processing_times):.2f}s'})
                        
                        sample_count += 1
                        pbar.update(1)
                        
                        if sample_count >= n_samples:
                            break
                            
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n处理样本 {sample_count} 时出错 (耗时 {elapsed_time:.2f}s): {e}")
                continue
    
    # 计算平均重要性（掩码值）
    avg_importance_raw = {feat: np.mean(scores) if scores else 0.0 
                          for feat, scores in feature_importance.items()}
    
    # 归一化重要性：转换为相对重要性（百分比）
    importance_values = np.array(list(avg_importance_raw.values()))
    if importance_values.max() > importance_values.min():
        # Min-Max归一化
        min_val = importance_values.min()
        max_val = importance_values.max()
        normalized_importance = {feat: (score - min_val) / (max_val - min_val) 
                                for feat, score in avg_importance_raw.items()}
        
        # 转换为百分比（总和归一化）
        total_importance = sum(normalized_importance.values())
        if total_importance > 0:
            percentage_importance = {feat: (score / total_importance) * 100 
                                     for feat, score in normalized_importance.items()}
        else:
            percentage_importance = {feat: 0.0 for feat in normalized_importance.keys()}
    else:
        # 所有特征重要性相同
        n_features = len(avg_importance_raw)
        percentage_importance = {feat: 100.0 / n_features if n_features > 0 else 0.0 
                                 for feat in avg_importance_raw.keys()}
        normalized_importance = {feat: 1.0 / n_features if n_features > 0 else 0.0 
                                 for feat in avg_importance_raw.keys()}
    
    # 输出统计信息
    print(f"\n原子特征重要性统计 (GNNExplainer):")
    print(f"  掩码值范围: [{importance_values.min():.6f}, {importance_values.max():.6f}]")
    print(f"  平均掩码值: {importance_values.mean():.6f}")
    print(f"  标准差: {importance_values.std():.6f}")
    
    # 输出时间统计
    if processing_times:
        print(f"\n原子特征分析时间统计:")
        print(f"  总样本数: {len(processing_times)}")
        print(f"  总耗时: {sum(processing_times):.2f}s")
        print(f"  平均耗时: {np.mean(processing_times):.2f}s/样本")
        print(f"  最短耗时: {np.min(processing_times):.2f}s")
        print(f"  最长耗时: {np.max(processing_times):.2f}s")
    
    # 返回归一化后的重要性（包含原始值、归一化值和百分比）
    return {
        'raw': avg_importance_raw,
        'normalized': normalized_importance,
        'percentage': percentage_importance
    }


def explain_atom_features(model, dataloader, device, n_samples=50, 
                          epochs=100, lr=0.01, lambda_reg=0.01):
    """
    使用标准GNNExplainer分析原子特征重要性
    
    参数:
    -----
    model: 要解释的模型
    dataloader: 数据加载器
    device: 计算设备
    n_samples: 分析的样本数
    epochs: GNNExplainer优化迭代次数
    lr: 学习率
    lambda_reg: L1正则化系数
    """
    return explain_atom_features_gnnexplainer(model, dataloader, device, n_samples, 
                                               epochs, lr, lambda_reg)


def explain_bond_features_gnnexplainer(model, dataloader, device, n_samples=50,
                                       epochs=100, lr=0.01, lambda_reg=0.01):
    """
    使用标准的GNNExplainer方法分析化学键特征重要性
    
    参数:
    -----
    model: 要解释的模型
    dataloader: 数据加载器
    device: 计算设备
    n_samples: 分析的样本数
    epochs: GNNExplainer优化迭代次数
    lr: 学习率
    lambda_reg: L1正则化系数
    """
    model.eval()
    
    # 创建GNNExplainer实例
    explainer = GNNExplainer(model, epochs=epochs, lr=lr, 
                             lambda_reg=lambda_reg, device=device)
    
    # 存储每个特征的重要性分数
    feature_importance = {feat_name: [] for feat_name in BOND_FEATURE_MAPPING.keys()}
    
    sample_count = 0
    processing_times = []
    with tqdm(total=n_samples, desc="GNNExplainer分析化学键特征") as pbar:
        for batch_solvent, batch_solute, T in dataloader:
            if sample_count >= n_samples:
                break
            
            # 记录开始时间
            start_time = time.time()
                
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            T = T.to(device)
            
            try:
                # 使用GNNExplainer学习掩码
                _, _, bond_mask_solvent, bond_mask_solute = explainer.explain(
                    batch_solvent, batch_solute, T
                )
                
                # 合并溶剂和溶质的掩码（平均）
                if bond_mask_solvent is not None and bond_mask_solute is not None:
                    # 对每个图分别处理
                    for graph_idx in range(batch_solvent.num_graphs):
                        # 合并溶剂和溶质的掩码（平均）
                        combined_mask = (bond_mask_solvent + bond_mask_solute) / 2.0
                        
                        # 分析每个化学键特征的重要性
                        for feat_name, (start_idx, feat_dim) in BOND_FEATURE_MAPPING.items():
                            # 掩码值越大，特征越重要
                            feat_importance = combined_mask[start_idx:start_idx+feat_dim].mean().item()
                            feature_importance[feat_name].append(feat_importance)
                        
                        # 记录处理时间
                        elapsed_time = time.time() - start_time
                        processing_times.append(elapsed_time)
                        pbar.set_postfix({'时间': f'{elapsed_time:.2f}s', '平均': f'{np.mean(processing_times):.2f}s'})
                        
                        sample_count += 1
                        pbar.update(1)
                        
                        if sample_count >= n_samples:
                            break
                            
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n处理样本 {sample_count} 时出错 (耗时 {elapsed_time:.2f}s): {e}")
                continue
    
    # 计算平均重要性（掩码值）
    avg_importance_raw = {feat: np.mean(scores) if scores else 0.0 
                          for feat, scores in feature_importance.items()}
    
    # 归一化重要性：转换为相对重要性（百分比）
    importance_values = np.array(list(avg_importance_raw.values()))
    if importance_values.max() > importance_values.min():
        # Min-Max归一化
        min_val = importance_values.min()
        max_val = importance_values.max()
        normalized_importance = {feat: (score - min_val) / (max_val - min_val) 
                                for feat, score in avg_importance_raw.items()}
        
        # 转换为百分比（总和归一化）
        total_importance = sum(normalized_importance.values())
        if total_importance > 0:
            percentage_importance = {feat: (score / total_importance) * 100 
                                     for feat, score in normalized_importance.items()}
        else:
            percentage_importance = {feat: 0.0 for feat in normalized_importance.keys()}
    else:
        # 所有特征重要性相同
        n_features = len(avg_importance_raw)
        percentage_importance = {feat: 100.0 / n_features if n_features > 0 else 0.0 
                                 for feat in avg_importance_raw.keys()}
        normalized_importance = {feat: 1.0 / n_features if n_features > 0 else 0.0 
                                 for feat in avg_importance_raw.keys()}
    
    # 输出统计信息
    print(f"\n化学键特征重要性统计 (GNNExplainer):")
    print(f"  掩码值范围: [{importance_values.min():.6f}, {importance_values.max():.6f}]")
    print(f"  平均掩码值: {importance_values.mean():.6f}")
    print(f"  标准差: {importance_values.std():.6f}")
    
    # 输出时间统计
    if processing_times:
        print(f"\n化学键特征分析时间统计:")
        print(f"  总样本数: {len(processing_times)}")
        print(f"  总耗时: {sum(processing_times):.2f}s")
        print(f"  平均耗时: {np.mean(processing_times):.2f}s/样本")
        print(f"  最短耗时: {np.min(processing_times):.2f}s")
        print(f"  最长耗时: {np.max(processing_times):.2f}s")
    
    # 返回归一化后的重要性（包含原始值、归一化值和百分比）
    return {
        'raw': avg_importance_raw,
        'normalized': normalized_importance,
        'percentage': percentage_importance
    }


def explain_bond_features(model, dataloader, device, n_samples=50,
                          epochs=100, lr=0.01, lambda_reg=0.01):
    """
    使用标准GNNExplainer分析化学键特征重要性
    
    参数:
    -----
    model: 要解释的模型
    dataloader: 数据加载器
    device: 计算设备
    n_samples: 分析的样本数
    epochs: GNNExplainer优化迭代次数
    lr: 学习率
    lambda_reg: L1正则化系数
    """
    return explain_bond_features_gnnexplainer(model, dataloader, device, n_samples,
                                             epochs, lr, lambda_reg)


def stratified_sampling(df, target_col='log-gamma', n_samples=500, n_strata=10):
    """
    分层采样：根据目标变量的数值范围进行采样
    确保高值和低值样本都被包含
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    target_col : str
        目标变量列名
    n_samples : int
        总采样数
    n_strata : int
        分层数量
    
    Returns:
    --------
    pd.Index
        采样后的索引
    """
    if target_col not in df.columns:
        print(f"警告: 目标变量 '{target_col}' 不存在，使用随机采样")
        return df.index[:n_samples]
    
    # 移除缺失值
    df_valid = df.dropna(subset=[target_col])
    
    if len(df_valid) < n_samples:
        print(f"警告: 有效样本数 ({len(df_valid)}) 少于所需采样数 ({n_samples})，使用所有有效样本")
        return df_valid.index
    
    # 根据目标变量值进行分层
    df_valid = df_valid.copy()
    df_valid['stratum'] = pd.qcut(df_valid[target_col], q=n_strata, labels=False, duplicates='drop')
    
    # 计算每层应采样的数量
    samples_per_stratum = n_samples // n_strata
    remainder = n_samples % n_strata
    
    sampled_indices = []
    for stratum in range(n_strata):
        stratum_data = df_valid[df_valid['stratum'] == stratum]
        if len(stratum_data) > 0:
            # 计算该层的采样数
            n_stratum_samples = samples_per_stratum + (1 if stratum < remainder else 0)
            n_stratum_samples = min(n_stratum_samples, len(stratum_data))
            
            # 从该层随机采样
            stratum_indices = stratum_data.sample(n=n_stratum_samples, random_state=42).index
            sampled_indices.extend(stratum_indices)
    
    # 如果采样数不足，从剩余样本中补充
    if len(sampled_indices) < n_samples:
        remaining = df_valid[~df_valid.index.isin(sampled_indices)]
        n_needed = n_samples - len(sampled_indices)
        if len(remaining) > 0:
            additional = remaining.sample(n=min(n_needed, len(remaining)), random_state=42).index
            sampled_indices.extend(additional)
    
    return pd.Index(sampled_indices[:n_samples])


def compute_molecular_fingerprints(df, mol_col_solvent='Molecule_Solvent', mol_col_solute='Molecule_Solute', 
                                   radius=2, n_bits=2048):
    """
    计算分子指纹（Morgan指纹）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框，包含分子对象
    mol_col_solvent : str
        溶剂分子列名
    mol_col_solute : str
        溶质分子列名
    radius : int
        Morgan指纹半径
    n_bits : int
        指纹位数
    
    Returns:
    --------
    np.ndarray
        组合的分子指纹矩阵
    """
    fingerprints = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算分子指纹"):
        try:
            mol_solvent = row[mol_col_solvent]
            mol_solute = row[mol_col_solute]
            
            if mol_solvent is None or mol_solute is None:
                fingerprints.append(np.zeros(n_bits * 2))
                continue
            
            # 计算Morgan指纹
            # 注意：GetMorganFingerprintAsBitVect已被弃用，但功能正常
            # 已通过warnings.filterwarnings抑制弃用警告
            fp_solvent = GetMorganFingerprintAsBitVect(mol_solvent, radius, nBits=n_bits)
            fp_solute = GetMorganFingerprintAsBitVect(mol_solute, radius, nBits=n_bits)
            
            # 转换为numpy数组并组合
            fp_solvent_arr = np.array(fp_solvent)
            fp_solute_arr = np.array(fp_solute)
            combined_fp = np.concatenate([fp_solvent_arr, fp_solute_arr])
            
            fingerprints.append(combined_fp)
        except Exception as e:
            print(f"计算指纹时出错 (索引 {idx}): {e}")
            fingerprints.append(np.zeros(n_bits * 2))
    
    return np.array(fingerprints)


def cluster_sampling(df, mol_col_solvent='Molecule_Solvent', mol_col_solute='Molecule_Solute', 
                     n_samples=500, n_clusters=20, random_state=42):
    """
    聚类采样：对分子指纹进行聚类，从每个聚类中抽取代表性样本
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    mol_col_solvent : str
        溶剂分子列名
    mol_col_solute : str
        溶质分子列名
    n_samples : int
        总采样数
    n_clusters : int
        聚类数量
    random_state : int
        随机种子
    
    Returns:
    --------
    pd.Index
        采样后的索引
    """
    print(f"计算分子指纹...")
    fingerprints = compute_molecular_fingerprints(df, mol_col_solvent, mol_col_solute)
    
    if len(fingerprints) < n_samples:
        print(f"警告: 样本数 ({len(fingerprints)}) 少于所需采样数 ({n_samples})，使用所有样本")
        return df.index[:len(fingerprints)]
    
    # 标准化指纹
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints)
    
    # K-means聚类
    print(f"进行K-means聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=min(n_clusters, len(df)), random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(fingerprints_scaled)
    
    # 从每个聚类中采样
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    sampled_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = df.index[cluster_mask]
        
        if len(cluster_indices) > 0:
            # 计算该聚类的采样数
            n_cluster_samples = samples_per_cluster + (1 if cluster_id < remainder else 0)
            n_cluster_samples = min(n_cluster_samples, len(cluster_indices))
            
            # 从该聚类中采样（选择距离聚类中心最近的样本作为代表性样本）
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_fps = fingerprints_scaled[cluster_mask]
            distances = np.linalg.norm(cluster_fps - cluster_center, axis=1)
            
            # 选择距离中心最近的n_cluster_samples个样本
            closest_indices = np.argsort(distances)[:n_cluster_samples]
            sampled_cluster_indices = cluster_indices[closest_indices]
            sampled_indices.extend(sampled_cluster_indices)
    
    # 如果采样数不足，从剩余样本中补充
    if len(sampled_indices) < n_samples:
        remaining = df.index[~df.index.isin(sampled_indices)]
        n_needed = n_samples - len(sampled_indices)
        if len(remaining) > 0:
            additional = remaining[np.random.choice(len(remaining), min(n_needed, len(remaining)), replace=False)]
            sampled_indices.extend(additional)
    
    return pd.Index(sampled_indices[:n_samples])


def visualize_feature_importance(atom_importance, bond_importance, save_path='feature_importance_gnnexplainer.png'):
    """
    可视化特征重要性
    使用归一化的相对重要性（百分比）进行可视化，更易于理解
    """
    # 提取百分比重要性（如果输入是字典格式）
    if isinstance(atom_importance, dict) and 'percentage' in atom_importance:
        atom_percentage = atom_importance['percentage']
    else:
        # 兼容旧格式
        atom_percentage = atom_importance
    
    if isinstance(bond_importance, dict) and 'percentage' in bond_importance:
        bond_percentage = bond_importance['percentage']
    else:
        bond_percentage = bond_importance
    
    # 使用1x2布局，只显示相对重要性百分比
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原子特征重要性（百分比）
    sorted_atom_features = sorted(atom_percentage.items(), key=lambda x: x[1], reverse=True)
    atom_features = [feat for feat, _ in sorted_atom_features]
    atom_scores_pct = [score for _, score in sorted_atom_features]
    
    bars1 = axes[0].barh(atom_features, atom_scores_pct, color='steelblue')
    axes[0].set_xlabel('Relative Importance (%)', fontsize=12)
    axes[0].set_title('Atom Feature Importance Ranking (Relative Importance %)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    if atom_scores_pct:
        axes[0].set_xlim(0, max(atom_scores_pct) * 1.15)
        # 在条形图上添加百分比标签
        for i, (bar, score) in enumerate(zip(bars1, atom_scores_pct)):
            width = bar.get_width()
            axes[0].text(width + max(atom_scores_pct) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.1f}%', ha='left', va='center', fontsize=10)
    
    # 化学键特征重要性（百分比）
    sorted_bond_features = sorted(bond_percentage.items(), key=lambda x: x[1], reverse=True)
    bond_features = [feat for feat, _ in sorted_bond_features]
    bond_scores_pct = [score for _, score in sorted_bond_features]
    
    bars2 = axes[1].barh(bond_features, bond_scores_pct, color='coral')
    axes[1].set_xlabel('Relative Importance (%)', fontsize=12)
    axes[1].set_title('Bond Feature Importance Ranking (Relative Importance %)', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    if bond_scores_pct:
        axes[1].set_xlim(0, max(bond_scores_pct) * 1.15)
        # 在条形图上添加百分比标签
        for i, (bar, score) in enumerate(zip(bars2, bond_scores_pct)):
            width = bar.get_width()
            axes[1].text(width + max(bond_scores_pct) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性可视化已保存到: {save_path}")
    plt.show()


def main():
    """
    主函数：执行特征重要性分析
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3  # ap, bp, topopsa
    hidden_dim = 38
    
    print(f"原子特征维度: {v_in}")
    print(f"化学键特征维度: {e_in}")
    
    # 加载模型
    # 检查模型是否需要attention_weight参数
    try:
        # 尝试使用attention_weight参数（用于GH_pyGEAT_architecture_0615_v0）
        model = GHGEAT(v_in, e_in, u_in, hidden_dim, attention_weight=0.8).to(device)
    except TypeError:
        # 如果不支持attention_weight，使用默认参数
        model = GHGEAT(v_in, e_in, u_in, hidden_dim).to(device)
    
    # 模型路径（需要根据实际情况修改）
    model_path = 'D:\\化工预测\\论文复现结果\\GH-GAT - 副本 (2)\\GHGEAT_full_search_lr8.00e-04_hd38_bs104_ep434_pretrained_attn0.8\\GHGEAT.pth'
    
    # 确保model_path是Path对象
    if isinstance(model_path, str):
        model_path = Path(model_path)
    
    try:
        # 加载权重文件
        checkpoint = torch.load(str(model_path), map_location=device)
        
        # 第一步：检查并初始化所有input_projection层
        input_projection_keys = {k: v for k, v in checkpoint.items() if 'input_projection' in k}
        
        if input_projection_keys:
            print(f"检测到 {len(input_projection_keys)} 个input_projection权重，正在初始化...")
            
            # 按graphnet分组处理
            graphnet_proj_info = {}
            for key, value in input_projection_keys.items():
                parts = key.split('.')
                if len(parts) >= 4:
                    graphnet_name = parts[0]  # graphnet1 or graphnet2
                    param_type = parts[-1]  # weight or bias
                    
                    if graphnet_name not in graphnet_proj_info:
                        graphnet_proj_info[graphnet_name] = {}
                    
                    if param_type == 'weight':
                        graphnet_proj_info[graphnet_name]['input_dim'] = value.shape[1]
                        graphnet_proj_info[graphnet_name]['output_dim'] = value.shape[0]
                        graphnet_proj_info[graphnet_name]['weight'] = value
                    elif param_type == 'bias':
                        graphnet_proj_info[graphnet_name]['bias'] = value
            
            # 初始化所有input_projection层
            for graphnet_name, proj_info in graphnet_proj_info.items():
                try:
                    node_model = getattr(getattr(model, graphnet_name), 'node_model')
                    if 'input_dim' in proj_info and 'output_dim' in proj_info:
                        input_dim = proj_info['input_dim']
                        output_dim = proj_info['output_dim']
                        
                        # 初始化input_projection
                        node_model._input_proj_dim = input_dim
                        node_model.input_projection = nn.Linear(input_dim, output_dim).to(device)
                        
                        # 加载权重
                        if 'weight' in proj_info:
                            node_model.input_projection.weight.data = proj_info['weight'].clone()
                        if 'bias' in proj_info:
                            node_model.input_projection.bias.data = proj_info['bias'].clone()
                        
                        print(f"  已初始化 {graphnet_name}.node_model.input_projection "
                              f"({input_dim} -> {output_dim})")
                except Exception as e:
                    print(f"  警告: 初始化 {graphnet_name}.input_projection 时出错: {e}")
        
        # 第二步：加载所有权重（现在input_projection已经初始化，可以正常加载）
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"警告: 以下键缺失: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"警告: 以下键被忽略: {unexpected_keys[:5]}...")
        
        model.eval()
        print(f"模型已加载: {model_path}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据（需要根据实际情况修改）
    data_path = project_root / 'data' / 'raw' / 'mini_dataset.csv'
    if not data_path.exists():
        alternative_data_paths = [
            project_root / 'data' / 'raw' / 'dataset.csv',
            project_root / 'scr' / 'isothermal' / 'T_dataset' / 'train' / '25_train.csv',
        ]
        # 用户指定的数据路径
        user_data_path = 'data\\processed\\new_dataset\\train_dataset\\v2\\molecule\\molecule_test.csv'
        # 检查用户指定的路径是否存在
        if Path(user_data_path).exists():
            data_path = Path(user_data_path)
        else:
            # 尝试相对路径
            user_data_path_rel = project_root / user_data_path
            if user_data_path_rel.exists():
                data_path = user_data_path_rel
            else:
                # 尝试其他路径
                for alt_path in alternative_data_paths:
                    if alt_path.exists():
                        data_path = alt_path
                        break
        
        if not Path(data_path).exists() if isinstance(data_path, str) else not data_path.exists():
            print(f"警告: 未找到数据文件")
            print("请修改data_path变量指向正确的数据文件")
            return
    
    # 确保data_path是Path对象
    if isinstance(data_path, str):
        data_path = Path(data_path)
    
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 检查必要的列
    required_cols = ['Solvent_SMILES', 'Solute_SMILES']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 数据文件缺少必要的列: {required_cols}")
        return
    
    df['Molecule_Solvent'] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)
    df['Molecule_Solute'] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    # 构建图数据
    target = 'log-gamma' if 'log-gamma' in df.columns else df.columns[-1]
    print(f"正在构建图数据，目标变量: {target}...")
    df['g_solv'], df['g_solu'] = sys2graph(df, 'Molecule_Solvent', 'Molecule_Solute', target)
    
    # 使用分层采样和聚类采样策略
    n_samples = 500
    print(f"\n采用智能采样策略，目标采样数: {n_samples}")
    
    # 策略1: 分层采样（根据log-gamma值）
    print("\n策略1: 分层采样（根据目标变量范围）...")
    stratified_indices = stratified_sampling(df, target_col=target, n_samples=n_samples//2, n_strata=10)
    print(f"分层采样获得 {len(stratified_indices)} 个样本")
    
    # 策略2: 聚类采样（根据分子指纹）
    print("\n策略2: 聚类采样（根据分子指纹）...")
    cluster_indices = cluster_sampling(df, mol_col_solvent='Molecule_Solvent', 
                                      mol_col_solute='Molecule_Solute', 
                                      n_samples=n_samples//2, n_clusters=20)
    print(f"聚类采样获得 {len(cluster_indices)} 个样本")
    
    # 合并两种采样策略的结果（去重）
    combined_indices = pd.Index(list(set(stratified_indices) | set(cluster_indices)))
    
    # 如果合并后样本数不足，补充随机采样
    if len(combined_indices) < n_samples:
        remaining = df.index[~df.index.isin(combined_indices)]
        n_needed = n_samples - len(combined_indices)
        if len(remaining) > 0:
            additional = remaining[np.random.choice(len(remaining), min(n_needed, len(remaining)), replace=False)]
            combined_indices = combined_indices.union(pd.Index(additional))
    
    # 限制最终样本数
    final_indices = combined_indices[:n_samples]
    print(f"\n最终采样数: {len(final_indices)} (分层: {len(stratified_indices)}, 聚类: {len(cluster_indices)})")
    
    # 创建数据加载器
    test_loader = get_dataloader_pairs_T(df, final_indices.tolist(), 'g_solv', 'g_solu', 
                                        batch_size=1, shuffle=False, drop_last=False)
    
    # 使用标准GNNExplainer方法
    print(f"\n使用分析方法: GNNExplainer")
    print("  - 方法: 标准GNNExplainer（学习特征掩码）")
    print("  - 优点: 更精确，能够识别重要的特征子集")
    print("  - 注意: 计算时间较长（每个样本需要多次迭代优化）")
    
    # 分析原子特征重要性
    print(f"\n开始分析原子特征重要性 (使用 {len(final_indices)} 个样本)...")
    atom_importance = explain_atom_features(model, test_loader, device, 
                                            n_samples=len(final_indices))
    
    # 分析化学键特征重要性
    print(f"\n开始分析化学键特征重要性 (使用 {len(final_indices)} 个样本)...")
    bond_importance = explain_bond_features(model, test_loader, device, 
                                           n_samples=len(final_indices))
    
    # 提取重要性数据（处理字典格式）
    if isinstance(atom_importance, dict) and 'percentage' in atom_importance:
        atom_percentage = atom_importance['percentage']
        atom_raw = atom_importance['raw']
    else:
        atom_percentage = atom_importance
        atom_raw = atom_importance
    
    if isinstance(bond_importance, dict) and 'percentage' in bond_importance:
        bond_percentage = bond_importance['percentage']
        bond_raw = bond_importance['raw']
    else:
        bond_percentage = bond_importance
        bond_raw = bond_importance
    
    # 打印结果（使用百分比，更直观）
    print("\n" + "="*80)
    print("原子特征重要性排序 (基于GNNExplainer - 相对重要性%)")
    print("="*80)
    print(f"{'排名':<6} {'特征名称':<25} {'相对重要性(%)':<15} {'原始掩码值':<15}")
    print("-" * 80)
    for i, (feature, score_pct) in enumerate(sorted(atom_percentage.items(), key=lambda x: x[1], reverse=True), 1):
        score_raw = atom_raw[feature] if isinstance(atom_raw, dict) else score_pct
        print(f"{i:2d}.   {feature:25s} {score_pct:>13.2f}%    {score_raw:>13.6f}")
    
    print("\n" + "="*80)
    print("化学键特征重要性排序 (基于GNNExplainer - 相对重要性%)")
    print("="*80)
    print(f"{'排名':<6} {'特征名称':<25} {'相对重要性(%)':<15} {'原始掩码值':<15}")
    print("-" * 80)
    for i, (feature, score_pct) in enumerate(sorted(bond_percentage.items(), key=lambda x: x[1], reverse=True), 1):
        score_raw = bond_raw[feature] if isinstance(bond_raw, dict) else score_pct
        print(f"{i:2d}.   {feature:25s} {score_pct:>13.2f}%    {score_raw:>13.6f}")
    
    # 创建结果保存文件夹
    output_dir = project_root / 'Feature_importance_analysis'
    output_dir.mkdir(exist_ok=True)
    print(f"\n结果将保存到: {output_dir}")
    
    # 可视化
    print("\n正在生成可视化图表...")
    save_path = output_dir / 'feature_importance_visualization.png'
    visualize_feature_importance(atom_importance, bond_importance, str(save_path))
    
    # 保存结果到CSV（包含原始值、归一化值和百分比）
    atom_features_list = list(atom_percentage.keys())
    bond_features_list = list(bond_percentage.keys())
    
    results_df = pd.DataFrame({
        'Feature_Type': ['Atom'] * len(atom_features_list) + ['Bond'] * len(bond_features_list),
        'Feature_Name': atom_features_list + bond_features_list,
        'Relative_Importance_Percent': (list(atom_percentage.values()) + list(bond_percentage.values())),
        'Raw_Mask_Value': ([atom_raw[f] if isinstance(atom_raw, dict) else atom_percentage[f] 
                                for f in atom_features_list] + 
                               [bond_raw[f] if isinstance(bond_raw, dict) else bond_percentage[f] 
                                for f in bond_features_list])
    })
    results_df = results_df.sort_values('Relative_Importance_Percent', ascending=False)
    results_path = output_dir / 'feature_importance_results.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"CSV结果已保存到: {results_path}")
    
    # 分别保存原子特征和化学键特征的结果
    atom_df = pd.DataFrame({
        'Feature_Name': atom_features_list,
        'Relative_Importance_Percent': [atom_percentage[f] for f in atom_features_list],
        'Raw_Mask_Value': [atom_raw[f] if isinstance(atom_raw, dict) else atom_percentage[f] 
                               for f in atom_features_list]
    }).sort_values('Relative_Importance_Percent', ascending=False)
    atom_path = output_dir / 'atom_feature_importance.csv'
    atom_df.to_csv(atom_path, index=False, encoding='utf-8-sig')
    print(f"原子特征结果已保存到: {atom_path}")
    
    bond_df = pd.DataFrame({
        'Feature_Name': bond_features_list,
        'Relative_Importance_Percent': [bond_percentage[f] for f in bond_features_list],
        'Raw_Mask_Value': [bond_raw[f] if isinstance(bond_raw, dict) else bond_percentage[f] 
                               for f in bond_features_list]
    }).sort_values('Relative_Importance_Percent', ascending=False)
    bond_path = output_dir / 'bond_feature_importance.csv'
    bond_df.to_csv(bond_path, index=False, encoding='utf-8-sig')
    print(f"化学键特征结果已保存到: {bond_path}")
    
    # 保存文本格式的汇总报告
    report_path = output_dir / 'feature_importance_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("特征重要性分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"分析方法: 标准GNNExplainer（学习特征掩码）\n")
        f.write(f"分析样本数: {len(final_indices)}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("原子特征重要性排序（相对重要性%）\n")
        f.write("="*80 + "\n")
        f.write(f"{'排名':<6} {'特征名称':<25} {'相对重要性(%)':<15} {'原始掩码值':<15}\n")
        f.write("-" * 80 + "\n")
        for i, (feature, score_pct) in enumerate(sorted(atom_percentage.items(), key=lambda x: x[1], reverse=True), 1):
            score_raw = atom_raw[feature] if isinstance(atom_raw, dict) else score_pct
            f.write(f"{i:2d}.   {feature:25s} {score_pct:>13.2f}%    {score_raw:>13.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("化学键特征重要性排序（相对重要性%）\n")
        f.write("="*80 + "\n")
        f.write(f"{'排名':<6} {'特征名称':<25} {'相对重要性(%)':<15} {'原始掩码值':<15}\n")
        f.write("-" * 80 + "\n")
        for i, (feature, score_pct) in enumerate(sorted(bond_percentage.items(), key=lambda x: x[1], reverse=True), 1):
            score_raw = bond_raw[feature] if isinstance(bond_raw, dict) else score_pct
            f.write(f"{i:2d}.   {feature:25s} {score_pct:>13.2f}%    {score_raw:>13.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("特征说明\n")
        f.write("="*80 + "\n")
        f.write("原子特征:\n")
        f.write("  - Atomic_Symbol: 原子类型 (C, N, O, Cl, S, F, Br, I, Si, Sn, Pb, Ge, H, P, Hg, Te)\n")
        f.write("  - Ring_Presence: 是否在环中 (Boolean)\n")
        f.write("  - Aromaticity: 是否芳香 (Boolean)\n")
        f.write("  - Hybridization: 杂化方式 (S, SP, SP2, SP3)\n")
        f.write("  - Neighbor_Count: 键数量 (0,1,2,3,4)\n")
        f.write("  - Formal_Charge: 形式电荷 (0,1,-1)\n")
        f.write("  - Hydrogen_Count: 连接氢原子数 (0,1,2,3)\n")
        f.write("  - Chirality: 手性 (Unspecified, CW, CCW)\n\n")
        
        f.write("化学键特征:\n")
        f.write("  - Bond_Type: 键类型 (Single, Double, Triple, Aromatic)\n")
        f.write("  - Conjugated: 是否共轭 (Boolean)\n")
        f.write("  - Ring_Part: 是否为环的一部分 (Boolean)\n")
        f.write("  - Stereochemistry: 立体化学 (None, Z-type, E-type)\n")
    
    print(f"文本报告已保存到: {report_path}")
    print(f"\n所有结果已保存到文件夹: {output_dir}")


if __name__ == "__main__":
    main()

