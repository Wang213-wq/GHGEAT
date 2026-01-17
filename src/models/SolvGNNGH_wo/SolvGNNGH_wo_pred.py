"""
SolvGNNGH without MTL预测
"""
import pandas as pd
from rdkit import Chem
from utilities_v2.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features
from SolvGNNGH_wo_architecture import SolvGNNGH_wo
import torch
import os
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE

def pred_SolvGNNGH_wo(df, model_name, hyperparameters, temperature=None):
    
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
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']    
    
    # Dataloader
    indices = df.index.tolist()
    # 检查 get_dataloader_pairs_T 是否支持 num_workers 参数
    try:
        predict_loader = get_dataloader_pairs_T(df, 
                                              indices, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size=23, 
                                              shuffle=False, 
                                              drop_last=False,
                                              num_workers=0)  # 设置为0以避免Windows上的多进程问题
    except TypeError:
        # 如果不支持 num_workers 参数，使用旧的方式调用
        predict_loader = get_dataloader_pairs_T(df, 
                                              indices, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size=23, 
                                              shuffle=False, 
                                              drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model    = SolvGNNGH_wo(n_atom_features(), hidden_dim)
    device   = torch.device(available_device)
    model    = model.to(device)
    model.load_state_dict(torch.load("D:\\化工预测\\论文复现结果\\GH-GAT - 副本 (2)\\scr\\models\\SolvGNNGH_epochs_250\\SolvGNNGH_epochs_250.pth",
                                     map_location=torch.device(available_device)),strict = False)
    
    
    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                    y_pred  = y_pred.numpy().reshape(-1,)
                else:
                    y_pred  = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df


if __name__ == '__main__':
    epochs = [250]


    hyperparameters_dict = {'hidden_dim'  : 193,
                            'lr'          : 0.00011559310094158379,
                            'batch_size'  : 16
                            }


    for e in epochs:
        print('-'*50)
        print('Epochs: ', e)



        ###################################
        # --- Predict Brouwer dataset --- #
        ###################################

        # Models trained on the complete train/validation set
        # print('Predicting with SolvGNNCat')
        # df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\brouwer_edge_test.csv')
        # df_pred = pred_SolvGNNCat(df, model_name='SolvGNNCat',
        #                        hyperparameters=hyperparameters_dict)
        # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\SolvGNNCat\\brouwer_edge_pred.csv')
        #
        # df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\brouwer_extrapolation_test.csv')
        # df_pred = pred_SolvGNNCat(df, model_name='SolvGNNCat',
        #                          hyperparameters=hyperparameters_dict)
        # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\SolvGNNCat\\brouwer_edge_pred.csv')
        # print('Done!')

    Ts = [20,25,30,35,40,45,50,60,70,75,80,90,100,120]
    for T in Ts:
        print('-' * 50)
        print('Temperature: ', T)

        # Models trained on the complete train/validation set

        print('Predicting with SolvGNNGH_wo')
        # df = pd.read_csv(T_path+'\\'+str(T)+'_train.csv')
        # df_pred = pred_GNNprevious(df, model_name='GNNprevious_'+str(T),
        #                   hyperparameters=hyperparameters_dict[T])
        # df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GNN\\models\\isothermal\\predictions\\GNN_previous'
        #                +str(T)+'_train_pred.csv')

        csv_path = os.path.join(
            'D:\\化工预测\\论文复现结果\\GH-GAT - 副本 (2)\\scr\\isothermal\\T_dataset\\test',str(T) + '_test.csv')
        df = pd.read_csv(csv_path)
        df_pred = pred_SolvGNNGH_wo(df, model_name='SolvGNNGH_wo_' + str(T),
                                   hyperparameters=hyperparameters_dict,
                                   temperature=T)
        df_pred.to_csv('D:\\化工预测\\论文复现结果\\GH-GAT - 副本 (2)\\scr\\isothermal\\model\\predictions\\SolvGNNGH_wo'
                           + '\\' + str(T) + '_test_pred.csv')
        print('Done!')

        


