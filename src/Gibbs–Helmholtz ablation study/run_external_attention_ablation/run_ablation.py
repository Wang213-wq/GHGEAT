"""
External Attention消融实验运行脚本
运行GHGEAT_Baseline、GHGEAT_GAT和GHGEAT_GCN三个实验
"""
import sys
import os
from pathlib import Path
import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入训练函数和工具
from scr.models.GHGEAT.GHGEAT_train import train_GNNGH_T
from scr.models.utilities_v2.mol2graph import sys2graph, n_atom_features, n_bond_features
from rdkit import Chem

# 导入消融实验架构
sys.path.insert(0, str(Path(__file__).parent / "GHGEAT_GAT"))
from GHGEAT_GAT_architecture import GHGEAT_GAT

sys.path.insert(0, str(Path(__file__).parent / "GHGEAT_GCN"))
from GHGEAT_GCN_architecture import GHGEAT_GCN

# 导入基线架构（完整GHGEAT）
# 注意：训练代码实际导入的是GH_pyGEAT_architecture_0615_v0
try:
    from scr.models.GH_pyGEAT_architecture_0615_v0 import GHGEAT
except ImportError:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "GHGEAT"))
        from GHGEAT_architecture import GHGEAT
    except ImportError:
        print("警告: 无法导入GHGEAT基线模型，请检查导入路径")
        GHGEAT = None


def run_external_attention_ablation():
    """
    运行External Attention消融实验
    包括：
    1. GHGEAT_Baseline: 完整模型 (External Attention)
    2. GHGEAT_GAT: 使用GAT替换External Attention
    3. GHGEAT_GCN: 使用GCN替换External Attention
    """
    # 数据路径
    train_data_path = PROJECT_ROOT / "dataset" / "public dataset" / "molecule_train.csv"
    val_data_path = PROJECT_ROOT / "dataset" / "public dataset" / "molecule_valid.csv"
    
    # 读取数据
    print("="*60)
    print("加载训练数据...")
    df_train = pd.read_csv(train_data_path)
    print(f"训练集大小: {len(df_train)}")
    
    if os.path.exists(val_data_path):
        print("加载验证数据...")
        df_val = pd.read_csv(val_data_path)
        print(f"验证集大小: {len(df_val)}")
    else:
        print("未找到验证集，将使用训练集进行评估")
        df_val = None
    
    # 构建分子图
    print("="*60)
    print("构建分子图...")
    mol_column_solvent = 'Molecule_Solvent'
    df_train[mol_column_solvent] = df_train['Solvent_SMILES'].apply(Chem.MolFromSmiles)
    mol_column_solute = 'Molecule_Solute'
    df_train[mol_column_solute] = df_train['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    target = 'log-gamma'
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df_train[graphs_solv], df_train[graphs_solu] = sys2graph(
        df_train, mol_column_solvent, mol_column_solute, target
    )
    
    if df_val is not None:
        df_val[mol_column_solvent] = df_val['Solvent_SMILES'].apply(Chem.MolFromSmiles)
        df_val[mol_column_solute] = df_val['Solute_SMILES'].apply(Chem.MolFromSmiles)
        df_val[graphs_solv], df_val[graphs_solu] = sys2graph(
            df_val, mol_column_solvent, mol_column_solute, target
        )
    
    # 超参数配置
    hyperparameters = {
        'hidden_dim': 38,
        'lr': 0.0012947540158123575,
        'n_epochs': 500,
        'batch_size': 64,
        'attention_weight': 1.0,
        'early_stopping_patience': 34,
        'early_stopping_min_delta': 0.0001,
    }
    
    # 保存路径配置
    ablation_base_dir = Path(__file__).parent
    save_base_dir = PROJECT_ROOT / "ablation_results" / "rigorous_ablation" / "run_external_attention_ablation"
    
    # 实验配置
    experiments = [
        {
            'name': 'GHGEAT_Baseline',
            'model_class': GHGEAT,
            'description': '完整GHGEAT模型 (External Attention)'
        },
        {
            'name': 'GHGEAT_GAT',
            'model_class': GHGEAT_GAT,
            'description': '使用GAT替换External Attention'
        },
        {
            'name': 'GHGEAT_GCN',
            'model_class': GHGEAT_GCN,
            'description': '使用GCN替换External Attention'
        }
    ]
    
    # 运行实验
    print("="*60)
    print("开始运行External Attention消融实验")
    print("="*60)
    
    for exp in experiments:
        print("\n" + "="*60)
        print(f"实验: {exp['name']}")
        print(f"描述: {exp['description']}")
        print("="*60)
        
        # 设置保存路径
        exp_save_dir = save_base_dir / exp['name'] / "Training_files"
        exp_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 修改训练函数以使用自定义模型类
        # 由于train_GNNGH_T内部硬编码了GHGEAT，我们需要临时替换
        import scr.models.GHGEAT.GHGEAT_train as train_module
        # 尝试从GH_pyGEAT_architecture_0615_v0导入并替换
        try:
            import scr.models.GH_pyGEAT_architecture_0615_v0 as arch_module
            original_model_class = arch_module.GHGEAT
            arch_module.GHGEAT = exp['model_class']
        except (ImportError, AttributeError):
            # 如果无法找到，尝试从train_module替换
            if hasattr(train_module, 'GHGEAT'):
                original_model_class = train_module.GHGEAT
                train_module.GHGEAT = exp['model_class']
            else:
                print(f"警告: 无法找到GHGEAT导入路径，跳过模型替换")
                original_model_class = None
        
        try:
            # 设置自定义保存路径
            hyperparameters['custom_save_path'] = str(save_base_dir / exp['name'])
            hyperparameters['training_files_save_dir'] = str(exp_save_dir)
            
            # 运行训练
            train_GNNGH_T(
                df=df_train,
                model_name=exp['name'],
                hyperparameters=hyperparameters,
                val_df=df_val
            )
            
            print(f"\n✓ {exp['name']} 训练完成")
            print(f"结果保存在: {exp_save_dir}")
            
        except Exception as e:
            print(f"\n✗ {exp['name']} 训练失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始模型类
            if original_model_class is not None:
                try:
                    arch_module.GHGEAT = original_model_class
                except:
                    try:
                        train_module.GHGEAT = original_model_class
                    except:
                        pass
    
    print("\n" + "="*60)
    print("所有External Attention消融实验完成")
    print("="*60)


if __name__ == '__main__':
    run_external_attention_ablation()
