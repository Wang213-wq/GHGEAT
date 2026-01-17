"""
GH-GNN without MTL预测
"""
# train_pred_loss:0.0233
# test_pred_loss:0.0498
import pandas as pd
from rdkit import Chem
from utilities_v2.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GH_pyGEAT_wo_architecture_0615_v0 import GHGEAT_wo
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as MAE


def pred_GHGEAT_wo(df, model_name, hyperparameters):
    # path = "F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\feature_ablation\\topopsa"
    # path = path + '\\' + model_name

    target = 'log-gamma'

    # Build molecule from SMILES
    mol_column_solvent = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute = 'Molecule_Solute'
    df[mol_column_solute] = df['Solute_SMILES'].apply(Chem.MolFromSmiles)

    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)

    # Hyperparameters
    hidden_dim = hyperparameters['hidden_dim']

    # Dataloader
    indices = df.index.tolist()
    
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
    model = GHGEAT_wo(v_in, e_in, u_in, hidden_dim)
    model.load_state_dict(
        torch.load('D:\\化工预测\\论文复现结果\\GH-GAT\\scr\\models\\ReLU\\1119_GHGEAT_wo\\GHGEAT_wo.pth', map_location=torch.device(available_device)))
    device = torch.device(available_device)
    model = model.to(device)

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
                    y_pred = y_pred.numpy().reshape(-1, )
                else:
                    y_pred = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1, )
                y_pred_final = np.concatenate((y_pred_final, y_pred))

    df[model_name] = y_pred_final

    return df


hyperparameters_dict = {'hidden_dim': 38,
                        'lr': 0.0012947540158123575,
                        'batch_size': 60,
                        'epochs': 387
                        }

print('-' * 50)
print('Epochs: ', hyperparameters_dict['epochs'])
###################################
# --- Predict Brouwer dataset --- #
###################################

# Models trained on the complete train/validation set
print('Predicting with GHGEAT_wo')
df = pd.read_csv('data\\processed\\new_dataset\\train_dataset\\v2\\molecule\\molecule_test.csv')
# df_pred = pred_GHGEAT_wo(df, model_name='GH_pyGEAT_wo_0615',
#                          hyperparameters=hyperparameters_dict)
# df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\GH_pyGEAT_wo_0615\\brouwer_edge_pred.csv')
#
# df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\brouwer_extrapolation_test.csv')
df_pred = pred_GHGEAT_wo(df, model_name='GH_pyGEAT_wo_0615',
                         hyperparameters=hyperparameters_dict)
df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\pred\\GH_pyGEAT_wo_0615\\brouwer_edge_pred.csv')
print('Done!')
print("Done")


    

