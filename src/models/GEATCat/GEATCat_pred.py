"""
GEATCat预测
基于 GHGEAT 架构，采用 GNNCat 的输出方式
"""
import pandas as pd
from rdkit import Chem
from utilities_v2.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GEATCat_architecture import GEATCat
import torch
import os
import numpy as np


def pred_GEATCat(df, model_name, hyperparameters, model_path=None, temperature=None):
    """
    使用 GEATCat 模型进行预测
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 Solute_SMILES, Solvent_SMILES 的数据框
    model_name : str
        模型名称（用于保存预测列名）
    hyperparameters : dict
        超参数字典，包含 hidden_dim, attention_weight
    model_path : str, optional
        模型权重文件路径，如果为None则使用默认路径
    temperature : float, optional
        如果数据中没有温度列，使用此温度（摄氏度）
    """
    target = 'log-gamma'
    
    # 处理温度列：确保存在 'T' 列
    if 'T' not in df.columns:
        if 'T_K' in df.columns:
            # 如果存在 T_K 列，直接使用（开尔文温度）
            df['T'] = df['T_K'].copy()
        elif temperature is not None:
            # 如果指定了温度（摄氏度），转换为开尔文
            df['T'] = temperature + 273.15
        else:
            raise ValueError("数据中没有 'T' 或 'T_K' 列，且未指定 temperature 参数")
    else:
        # 如果T列存在，检查是否需要从摄氏度转换为开尔文
        # 如果T的值看起来是摄氏度（< 100），转换为开尔文
        if df['T'].max() < 100:
            df['T'] = df['T'] + 273.15
    
    # Build molecule from SMILES
    mol_column_solvent = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute = 'Molecule_Solute'
    df[mol_column_solute] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']
    attention_weight = hyperparameters.get('attention_weight', 1.0)
    
    # Dataloader
    indices = df.index.tolist()
    try:
        predict_loader = get_dataloader_pairs_T(df, 
                                              indices, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size=32, 
                                              shuffle=False, 
                                              drop_last=False,
                                              num_workers=0)  # 设置为0以避免Windows上的多进程问题
    except TypeError:
        # 如果不支持 num_workers 参数，使用旧的方式调用
        predict_loader = get_dataloader_pairs_T(df, 
                                              indices, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size=32, 
                                              shuffle=False, 
                                              drop_last=False)
    
    ######################
    # --- Prediction --- #
    ######################
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3  # ap, bp, topopsa
    model = GEATCat(v_in, e_in, u_in, hidden_dim, attention_weight=attention_weight)
    device = torch.device(available_device)
    model = model.to(device)
    
    # 加载模型权重
    if model_path is None:
        # 默认路径：与训练脚本中的保存路径一致
        base_path = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models'
        model_path = os.path.join(base_path, model_name, model_name + '_best.pth')
    
    if not os.path.exists(model_path):
        # 尝试其他可能的路径
        base_path = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models'
        alt_path = os.path.join(base_path, model_name, model_name + '.pth')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path} 或 {alt_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=available_device), strict=False)
    
    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            batch_T = batch_T.to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                    y_pred = y_pred.numpy().reshape(-1,)
                else:
                    y_pred = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df


if __name__ == '__main__':
    hyperparameters_dict = {'hidden_dim': 38,
                            'lr': 8.00e-04,
                            'batch_size': 104,
                            'attention_weight': 0.8
                            }

    print('=' * 80)
    print('GEATCat 模型预测 - 测试集')
    print('=' * 80)
    
    # 测试集路径
    test_data_path = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\data\processed\IDAC_2026 dataset.csv'
    
    print(f'\n加载测试集: {test_data_path}')
    df = pd.read_csv(test_data_path)
    print(f'测试集样本数: {len(df)}')
    
    # 进行预测
    print('\n开始预测...')
    df_pred = pred_GEATCat(df, model_name='GEATCat', hyperparameters=hyperparameters_dict)
    
    # 计算统计指标
    if 'log-gamma' in df_pred.columns:
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # 计算整体MAE和R²
        y_true = df_pred['log-gamma'].values
        y_pred = df_pred['GEATCat'].values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算绝对误差
        df_pred['abs_error'] = np.abs(y_true - y_pred)
        
        # 按二元系统分组计算平均绝对误差
        binary_systems = df_pred.groupby(['Solute_SMILES', 'Solvent_SMILES']).agg({
            'abs_error': 'mean'
        }).reset_index()
        binary_systems.columns = ['Solute_SMILES', 'Solvent_SMILES', 'mean_abs_error']
        
        # 计算满足条件的二元系统占比
        total_binary_systems = len(binary_systems)
        pct_lt_01 = (binary_systems['mean_abs_error'] < 0.1).sum() / total_binary_systems * 100
        pct_lt_02 = (binary_systems['mean_abs_error'] < 0.2).sum() / total_binary_systems * 100
        pct_lt_03 = (binary_systems['mean_abs_error'] < 0.3).sum() / total_binary_systems * 100
        
        # 保存统计结果到txt文件
        output_dir = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models\GEATCat'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        txt_output_path = os.path.join(output_dir, 'molecule_test_pred_statistics.txt')
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('GEATCat 模型预测结果统计\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'测试集样本数: {len(df_pred)}\n')
            f.write(f'二元系统数量: {total_binary_systems}\n\n')
            f.write('-' * 80 + '\n')
            f.write('预测性能指标:\n')
            f.write('-' * 80 + '\n')
            f.write(f'平均绝对误差 (MAE): {mae:.6f}\n')
            f.write(f'决定系数 (R²): {r2:.6f}\n\n')
            f.write('二元系统绝对误差分布:\n')
            f.write(f'  绝对误差 < 0.1 的二元系统占比: {pct_lt_01:.2f}% ({int(pct_lt_01/100*total_binary_systems)}/{total_binary_systems})\n')
            f.write(f'  绝对误差 < 0.2 的二元系统占比: {pct_lt_02:.2f}% ({int(pct_lt_02/100*total_binary_systems)}/{total_binary_systems})\n')
            f.write(f'  绝对误差 < 0.3 的二元系统占比: {pct_lt_03:.2f}% ({int(pct_lt_03/100*total_binary_systems)}/{total_binary_systems})\n')
            f.write('=' * 80 + '\n')
        
        print(f'\n预测完成！')
        print(f'平均绝对误差 (MAE): {mae:.6f}')
        print(f'决定系数 (R²): {r2:.6f}')
        print(f'绝对误差 < 0.1 的二元系统占比: {pct_lt_01:.2f}%')
        print(f'绝对误差 < 0.2 的二元系统占比: {pct_lt_02:.2f}%')
        print(f'绝对误差 < 0.3 的二元系统占比: {pct_lt_03:.2f}%')
        print(f'\n统计结果已保存到: {txt_output_path}')
    else:
        print('\n警告: 数据中没有 log-gamma 列，无法计算统计指标')
        output_dir = r'D:\化工预测\论文复现结果\GH-GAT - 副本 (2)\scr\models\GEATCat'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print('Done!')

