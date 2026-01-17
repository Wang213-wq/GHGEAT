"""
GNN-Gibbs-Helmholtz温度训练
"""
import numpy as np
# Scientific computing
import pandas as pd

# RDKiT
from rdkit import Chem

# Internal utilities
from GHGNN_architecture import GHGNN, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from utilities.Train_eval_T import train, eval, MAE, R2
from utilities.save_info import save_train_traj

# External utilities
from tqdm import tqdm
#from sklearn.preprocessing import MinMaxScaler
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.cuda.amp import autocast, GradScaler

    
def train_GNNGH_T(df, model_name, hyperparameters,resume=False):
    
    path = 'scr\\models\\GHGNN'
    
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
    
    # targets = ['K1', 'K2']
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(df[targets].to_numpy())
    scaler = None
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target, y_scaler=scaler)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    early_stopping_patience = hyperparameters.get('early_stopping_patience', 20)
    
    start       = time.time()
    
    # Data loaders
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GHGNN(v_in, e_in, u_in, hidden_dim)
    model.load_state_dict(torch.load('GHGNN_MTL_epochs250\GHGNN_MTL_epochs250.pth',map_location=torch.device(available_device)),strict=False)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device.index or 0)
    else:
        device_name = 'CPU'
    print_report(f'Using device: {device_name}')
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = tqdm(range(n_epochs))
    else:
        pbar = tqdm(range(n_epochs))

    
    # To save trajectory
    mae_train = []
    r2_train = []  # 添加 R² 列表
    train_loss = []
    best_MAE = np.inf
    best_model = None
    epochs_no_improve = 0

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
            r2_train = checkpoint.get('r2_train', [])  # 加载 R²，如果不存在则使用空列表
            train_loss = checkpoint['train_loss']
            print_report(f"Resuming training from epoch {start_epoch}")
        else:
            print_report("No checkpoint found, starting training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        total_graphs = 0
        y_true_batches = []
        y_pred_batches = []
        
        for batch_data in train_loader:
            if len(batch_data) == 3:
                batch_solvent, batch_solute, T = batch_data
                T = T.to(device)
                has_temperature = True
            elif len(batch_data) == 2:
                batch_solvent, batch_solute = batch_data
                has_temperature = False
            else:
                raise ValueError(f"Unexpected batch size {len(batch_data)}")
            
            batch_solvent = batch_solvent.to(device)
            batch_solute = batch_solute.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == 'cuda')):
                if has_temperature:
                    pred = model(batch_solvent, batch_solute, T)
                else:
                    pred = model(batch_solvent, batch_solute)
                prediction = pred.to(torch.float32)
                real = batch_solvent.y.to(torch.float32).reshape(prediction.shape)
                loss = F.mse_loss(prediction, real)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            num_graphs = batch_solvent.num_graphs
            loss_sum += loss.item() * num_graphs
            total_graphs += num_graphs
            y_true_batches.append(real.detach().cpu())
            y_pred_batches.append(prediction.detach().cpu())
        
        epoch_loss = loss_sum / total_graphs
        y_true = torch.cat(y_true_batches, dim=0).numpy()
        y_pred = torch.cat(y_pred_batches, dim=0).numpy()
        pred_dict = {"y_true": y_true, "y_pred": y_pred}
        epoch_mae = MAE(pred_dict)
        epoch_r2 = R2(pred_dict)
        
        stats = OrderedDict()
        stats['Train_loss'] = epoch_loss
        stats['MAE_Train'] = epoch_mae
        stats['R2_Train'] = epoch_r2
        
        scheduler.step(stats['MAE_Train'])
        train_loss.append(epoch_loss)
        mae_train.append(epoch_mae)
        r2_train.append(epoch_r2)  # 保存 R²
        
        # 每轮训练结束后输出 MAE 和 R²
        print_report(f'Epoch {epoch+1}/{n_epochs} - MAE: {epoch_mae:.6f}, R^2: {epoch_r2:.6f}')
        epoch_time = time.time() - epoch_start
        print_report(f'Epoch {epoch+1} duration: {epoch_time:.2f}s')
        
        if mae_train[-1] < best_MAE:
            best_model = copy.deepcopy(model.state_dict())
            best_MAE = mae_train[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print_report(f'Early stopping triggered after {epoch+1} epochs (patience={early_stopping_patience})')
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_MAE': best_MAE,
            'mae_train': mae_train,
            'r2_train': r2_train,  # 保存 R² 列表
            'train_loss': train_loss
        }, path + '/' + model_name + '.pth')

    print_report('-' * 30)
    best_epoch = mae_train.index(min(mae_train)) + 1
    print_report('Best Epoch     : ' + str(best_epoch))
    print_report('Training MAE   : ' + str(mae_train[best_epoch - 1]))
    print_report('Training R^2   : ' + str(r2_train[best_epoch - 1]))  # 添加最佳 R^2 输出
    print_report('Training Loss  : ' + str(train_loss[best_epoch - 1]))

    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train'] = mae_train
    df_model_training['R2_Train'] = r2_train  # 添加 R² 列
    save_train_traj(path, df_model_training, valid=False)

    # Save best model
    if best_model is not None:
        torch.save(best_model, path + '/' + model_name + '_best.pth')
    else:
        torch.save(model.state_dict(), path + '/' + model_name + '_best.pth')

    end = time.time()

    print_report('\nTraining time (min): ' + str((end - start) / 60))
    report.close()

hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'n_epochs'    : 300,
                        'batch_size'  : 64,
                        'early_stopping_patience' : 10
                        }
    
df = pd.read_csv('data\\processed\\new_dataset\\train_dataset\\v2\\molecule_train.csv')
train_GNNGH_T(df, 'GHGNN', hyperparameters_dict,resume=True)