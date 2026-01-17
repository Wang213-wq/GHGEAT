"""
GH-GNN without MTL训练
"""
import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GHGNN_wo_architecture import GHGNN_wo, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
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
#from torch.cuda.amp import GradScaler

    
def train_GNNGH_T(df, model_name, hyperparameters,resume = False):
    
    path = os.getcwd()
    path = path + '/' + model_name
    
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
    #targets = ['K1','K2']
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(df[target].to_numpy())
    
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
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GHGNN_wo(v_in, e_in, u_in, hidden_dim)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = range(n_epochs)
        #scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        #scaler = None

    # To save trajectory
    mae_train = []
    train_loss = []
    best_MAE = np.inf

    # Check if we are resuming training
    if resume:
        # Load checkpoint
        if os.path.exists(path + '/' + model_name + '.pth'):
            checkpoint = torch.load(path + '/' + model_name + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_MAE = checkpoint['best_MAE']
            mae_train = checkpoint['mae_train']
            train_loss = checkpoint['train_loss']
            print_report(f"Resuming training from epoch {start_epoch}")
        else:
            print_report("No checkpoint found, starting training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, n_epochs):
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

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_MAE': best_MAE,
            'mae_train': mae_train,
            'train_loss': train_loss
        }, path + '/' + model_name + '.pth')

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
    torch.save(best_model, path + '/' + model_name + '_best.pth')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()

hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : 300,
                        'batch_size'  : 32
                        }
    
df = pd.read_csv('F:\\化工预测\\论文复现结果\\GH-GEAT\\data\\processed\\train_Cls_com.csv')
train_GNNGH_T(df, 'GHGNN_wo_0615', hyperparameters_dict,resume = True)