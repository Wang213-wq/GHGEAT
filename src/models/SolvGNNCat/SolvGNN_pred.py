"""
SolvGNN等温模型预测
"""
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs, sys2graph, n_atom_features
from SolvGNN_architecture import SolvGNN
import torch
import os
import numpy as np
from tqdm import tqdm

def pred_SolvGNN(df, model_name, hyperparameters):
    path = 'F:\\化工预测\\论文复现结果\\GH-GAT\\scr\\isothermal'
    
    target = 'log-gamma'
    
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
    predict_loader = get_dataloader_pairs(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=df.shape[0], 
                                          shuffle=False, 
                                          drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################

    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    in_dim   = n_atom_features()
    model    = SolvGNN(in_dim, hidden_dim)
    device   = torch.device(available_device)
    model.load_state_dict(torch.load(path + '\\' + model_name + '\\' + model_name + '.pth',
                                     map_location=torch.device(available_device)))
    model    = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute in predict_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred = model(batch_solvent.cuda(), batch_solute.cuda()).cpu().numpy().reshape(-1,)
                else:
                    y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,)
            
    df[model_name] = y_pred
    
    return df
'''
温度:100
test_pred_loss:0.14510829746723175
train_pred_loss:0.002798536792397499
温度:120
test_pred_loss:0.08450791984796524
train_pred_loss:0.008761688135564327
温度:20
test_pred_loss:0.20520472526550293
train_pred_loss:0.018001560121774673
温度:25
test_pred_loss:0.10716667771339417
train_pred_loss:0.019675206393003464
温度:30
test_pred_loss:0.1391124576330185
train_pred_loss:0.020232975482940674
温度:40
test_pred_loss:0.05749998614192009
train_pred_loss:0.01243546325713396
温度:45
test_pred_loss:0.1620439738035202
train_pred_loss:0.009432061575353146
温度:50
test_pred_loss:0.1472209095954895
train_pred_loss:0.002722333185374737
温度:60
test_pred_loss:0.11408151686191559
train_pred_loss:0.010588163509964943
温度:70
test_pred_loss:0.3388596475124359
train_pred_loss:0.0031761759892106056
温度:75
test_pred_loss:0.1116972491145134
train_pred_loss:0.0020446686539798975
温度:80
test_pred_loss:0.1925126165151596
train_pred_loss:0.002237112959846854
温度:90
test_pred_loss:0.11384765058755875
train_pred_loss:0.009305636398494244
'''
def pred_SolvGNN_kfolds(df, model_name, hyperparameters, k,Temp):
    method_name = "SolvGNN_" + str(Temp)
    path = os.path.join("F:\\化工预测\\论文复现结果\\GH-GNN\\models\\isothermal\\SolvGNN\\SolvGNN_kfolds", method_name)
    
    target = 'log-gamma'
    
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
    predict_loader = get_dataloader_pairs(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=df.shape[0], 
                                          shuffle=False, 
                                          drop_last=False)

    
    ######################
    # --- Prediction --- #
    ######################

    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    in_dim   = n_atom_features()
    model    = SolvGNN(in_dim, hidden_dim)
    device   = torch.device(available_device)
    
    
    Y_pred = np.zeros((df.shape[0], k))
    
    for i in tqdm(range(k)):
        kfold_name = 'Kfold_' + str(i+1)

        model.load_state_dict(torch.load(path + '//' + kfold_name + '//' + kfold_name + '.pth',
                                         map_location=torch.device(available_device)))
        model    = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_solvent, batch_solute in predict_loader:
                if torch.cuda.is_available():
                    y_pred = model(batch_solvent.cuda(), batch_solute.cuda()).cpu().numpy().reshape(-1,)
                else:
                    y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,) 
        Y_pred[:,i] = y_pred
    
    y_pred         = np.mean(Y_pred, axis=1)
    y_pred_std     = np.std(Y_pred, axis=1)  
       
    df[model_name] = y_pred
    df[model_name+'_std'] = y_pred_std
    
    return df
'''
温度:100
test_pred_loss:0.14510826766490936
train_pred_loss:0.002798536792397499
温度:120
test_pred_loss:0.08450792729854584
train_pred_loss:0.008761689066886902
温度:20
test_pred_loss:0.20520472526550293
train_pred_loss:0.018001563847064972
温度:25
test_pred_loss:0.10716667771339417
train_pred_loss:0.019675204530358315
温度:30
test_pred_loss:0.1391124576330185
train_pred_loss:0.02023298293352127
温度:40
test_pred_loss:0.05749998241662979
train_pred_loss:0.012435461394488811
温度:45
test_pred_loss:0.16204394400119781
train_pred_loss:0.009432061575353146
温度:50
test_pred_loss:0.1472208946943283
train_pred_loss:0.002722333185374737
温度:60
test_pred_loss:0.1140814945101738
train_pred_loss:0.010588163509964943
温度:70
test_pred_loss:0.3388596475124359
train_pred_loss:0.003176175756379962
温度:75
test_pred_loss:0.1116972491145134
train_pred_loss:0.0020446686539798975
温度:80
test_pred_loss:0.1925126314163208
train_pred_loss:0.0022371127270162106
温度:90
test_pred_loss:0.11384764313697815
train_pred_loss:0.009305639192461967
'''
Ts = [20,25,30,40,50,60,70,80,100]


hyperparameters_dict ={
    20:{
        'hidden_dim'  : 242,
        'lr'          : 0.00036936165073207783,
        'n_epochs'    : 156,
        'batch_size'  : 5
        },
    25:{
        'hidden_dim'  : 226,
        'lr'          : 0.0004452246905217191,
        'n_epochs'    : 178,
        'batch_size'  : 12
        },
    30:{
        'hidden_dim'  : 236,
        'lr'          : 0.0001064539542772283,
        'n_epochs'    : 287,
        'batch_size'  : 4
        },
    40:{
        'hidden_dim'  : 186,
        'lr'          : 0.00024982359554047667,
        'n_epochs'    : 151,
        'batch_size'  : 4
        },
    45:{
        'hidden_dim'  : 204,
        'lr'          : 0.00039844252135930744,
        'n_epochs'    : 242,
        'batch_size'  : 8
        },
    50:{
        'hidden_dim'  : 197,
        'lr'          : 0.0005870732537897345,
        'n_epochs'    : 212,
        'batch_size'  : 9
        },
    60:{
        'hidden_dim'  : 182,
        'lr'          : 0.00034166273626446927,
        'n_epochs'    : 299,
        'batch_size'  : 7
        },
    70:{
        'hidden_dim'  : 252,
        'lr'          : 0.0005625406199256793,
        'n_epochs'    : 256,
        'batch_size'  : 8
        },
    75:{
        'hidden_dim'  : 232,
        'lr'          : 0.0005910334037010466,
        'n_epochs'    : 222,
        'batch_size'  : 7
        },
    80:{
        'hidden_dim'  : 162,
        'lr'          : 0.0002417916646587825,
        'n_epochs'    : 254,
        'batch_size'  : 4
        },
    90:{
        'hidden_dim'  : 183,
        'lr'          : 0.0002753542230286996,
        'n_epochs'    : 210,
        'batch_size'  : 10
        },
    100:{
        'hidden_dim'  : 177,
        'lr'          : 0.000637356514233681,
        'n_epochs'    : 260,
        'batch_size'  : 4
        },
    120:{
        'hidden_dim'  : 170,
        'lr'          : 0.00019324021058485724,
        'n_epochs'    : 262,
        'batch_size'  : 6
        }
        }


for T in Ts:
    print('-'*50)
    print('Temperature: ', T)
    
    # Models trained on the complete train/validation set
    
    # Given that the data is not open-source, the paths to the data are here 
    # just representative, the same is true for the destination paths of the 
    # predictions
    
    print('Predicting with SolvGNN')
    csv_path = os.path.join('F:\\化工预测\\论文复现结果\\GH-GAT\\scr\\isothermal'+'\\'+str(T)+'\\'+str(T)+'_test.csv')
    # df_pred = pred_SolvGNN(df, model_name='SolvGNN_'+str(T),
    #                   hyperparameters=hyperparameters_dict[T])
    # pred_path = os.path.join('F:\\化工预测\\论文复现结果\\GH-GNN\\models\\isothermal\\predictions\\SolvGNN\\SolvGNN_Train\\' , 'SolvGNN_' + str(T))
    # if not os.path.exists(pred_path):
    #     os.makedirs(pred_path)
    # df_pred.to_csv(pred_path+'\\'+str(T)+'_train_pred.csv')

    df = pd.read_csv(csv_path)
    df_pred = pred_SolvGNN(df, model_name='SolvGNN_' + str(T),
                           hyperparameters=hyperparameters_dict[T])
    df_pred.to_csv('F:\\化工预测\\论文复现结果\\GH-GAT\\scr\\isothermal\\predictions\\SolvGNN'
                   + '\\' + str(T) + '_test_pred.csv')
    print('Done!')
    
    # # Models trained using kfold CV
    # print('Predicting with SolvGNN_Kfolds')
    # T_path = os.path.join("F:\\化工预测\\论文复现结果\\GH-GNN\\data\\processed\\Train_GNN", str(T))
    # df = pd.read_csv(T_path + '\\' + str(T) + '_train.csv')
    # df_pred = pred_SolvGNN(df, model_name='SolvGNN_' + str(T),
    #                        hyperparameters=hyperparameters_dict[T])
    # pred_path = os.path.join(
    #     'F:\\化工预测\\论文复现结果\\GH-GNN\\models\\isothermal\\predictions\\SolvGNN\\SolvGNN_kfolds\\',
    #     'SolvGNN_' + str(T))
    # if not os.path.exists(pred_path):
    #     os.makedirs(pred_path)
    # df_pred.to_csv(pred_path + '\\' + str(T) + '_train_pred.csv')
    #
    # T_path = os.path.join("F:\\化工预测\\论文复现结果\\GH-GNN\\data\\processed\\Train_GNN", str(T))
    # df = pd.read_csv(T_path + '\\' + str(T) + '_test.csv')
    # df_pred = pred_SolvGNN(df, model_name='SolvGNN_' + str(T),
    #                        hyperparameters=hyperparameters_dict[T])
    # df_pred.to_csv(pred_path + '\\' + str(T) + '_test_pred.csv')
    # print('Done!')
    



