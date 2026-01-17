"""
SolvGNNGH without MTL训练
"""

# Scientific computing
import numpy as np
import pandas as pd

# RDKiT
from rdkit import Chem

# Sklearn
from sklearn.model_selection import KFold

# Internal utilities
from SolvGNNGH_wo_architecture import SolvGNNGH_wo, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features
from utilities.Train_eval_T import train, eval, MAE
from utilities.save_info import save_train_traj

# External utilities
from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.cuda.amp import GradScaler

    
def train_SolvGNNGH(df, model_name, hyperparameters):
    
    path = "F:\\化工预测\\论文复现结果\\GH-GEAT\\scr\\models"
    path = path + '\\' + model_name
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open(path+'/Report_training_' + model_name + '.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    train_index = df.index.tolist()
    
    target = 'log-gamma'
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    
    start       = time.time()
    
    # Data loaders
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    
    # Model
    model    = SolvGNNGH_wo(n_atom_features(), hidden_dim)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = range(n_epochs)
        scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        scaler=None

    # To save trajectory
    mae_train = []
    train_loss = []
    best_MAE = np.inf

    for epoch in pbar:
        print("epoch:{}".format(epoch))
        stats = OrderedDict()
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        # pbar.set_postfix(stats) # include stats in the progress bar
        # Save best model
        if mae_train[-1] < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE = mae_train[-1]

    print_report('-' * 30)
    best_epoch = mae_train.index(min(mae_train)) + 1
    print_report('Best Epoch     : ' + str(best_epoch))
    print_report('Training MAE   : ' + str(mae_train[best_epoch - 1]))
    print_report('Training Loss   : ' + str(train_loss[best_epoch - 1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train'] = mae_train
    save_train_traj(path, df_model_training, valid=False)

    # Save best model

    torch.save(best_model, path + '/' + model_name + '.pth')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()

hyperparameters_dict = {'hidden_dim'  : 193,
                        'lr'          : 0.00011559310094158379,
                        'n_epochs'    : 250,
                        'batch_size'  : 16
                        }
    
df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\new_dataset\\train_dataset\\v2\\molecule_train.csv')
train_SolvGNNGH(df, 'SolvGNNGH_epochs_'+str(250), hyperparameters_dict)