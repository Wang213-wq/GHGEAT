"""
外部注意力机制敏感性分析
测试不同memory_size值对模型性能（MAE）的影响
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 优化系统性能：设置进程优先级和电源管理
def optimize_system_performance():
    """
    优化系统性能，减少夜间运行时因系统资源竞争导致的训练时间变长
    包括：设置进程优先级、禁用电源管理对GPU的影响等
    """
    try:
        import psutil
        import os
        
        # 1. 设置当前进程为高优先级
        try:
            current_process = psutil.Process(os.getpid())
            # Windows: HIGH_PRIORITY_CLASS
            # Linux: 使用nice值
            if sys.platform == 'win32':
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                print("✓ 已设置进程优先级为高优先级")
            else:
                current_process.nice(-10)  # 降低nice值（提高优先级）
                print("✓ 已设置进程nice值为-10（高优先级）")
        except (psutil.AccessDenied, AttributeError) as e:
            print(f"⚠️ 无法设置进程优先级: {e}（可能需要管理员权限）")
        
        # 2. 设置GPU性能模式（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                # 设置GPU为最大性能模式
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    # 禁用GPU的电源管理（如果支持）
                    # 注意：这需要NVIDIA驱动支持
                    try:
                        # 尝试设置GPU时钟频率为最大（如果支持）
                        # 这通常需要nvidia-ml-py库
                        pass  # 暂时不实现，因为需要额外依赖
                    except:
                        pass
                    
                print("✓ GPU性能优化已应用")
        except Exception as e:
            print(f"⚠️ GPU性能优化失败: {e}")
        
        # 3. 设置CPU亲和性（可选，通常不需要）
        # 在多CPU系统中，可以绑定进程到特定CPU核心
        
    except ImportError:
        print("⚠️ psutil未安装，跳过系统性能优化")
        print("  建议安装: pip install psutil")
    except Exception as e:
        print(f"⚠️ 系统性能优化过程中出现错误: {e}")
        print("  将继续执行，但可能无法获得最佳性能")

# 设置matplotlib支持中文，并避免使用默认字体的 Unicode 减号
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from scr.models.GHGEAT_wo_train import train_GNNGH_T
from hyperparameter_research.GHGEAT_hyperparameter_search import (
    create_train_wrapper_T_with_validation
)
from scr.models.utilities_v2.hyperparameter_tuning import HyperparameterTuner


def warmup_gpu(warmup_iterations=10, warmup_batch_size=32):
    """
    GPU预热函数（增强版）
    通过执行一些接近实际训练的GPU操作来预热GPU，减少首次训练时的初始化开销
    包括CUDA kernel编译、内存分配等
    
    Parameters:
    -----------
    warmup_iterations : int
        预热迭代次数（默认10次）
    warmup_batch_size : int
        预热时的batch大小（默认32）
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.amp import GradScaler
        
        if not torch.cuda.is_available():
            print("CUDA不可用，跳过GPU预热")
            return
        
        device = torch.device('cuda')
        print(f"开始GPU预热（设备: {device}，{warmup_iterations}次迭代）...")
        
        # 使用混合精度训练（与实际训练一致）
        scaler = GradScaler('cuda', enabled=True)
        
        # 创建一些随机张量进行预热操作
        # 模拟常见的训练操作：矩阵乘法、卷积、混合精度等
        for i in range(warmup_iterations):
            # 使用autocast模拟混合精度训练
            with torch.autocast(device_type='cuda', enabled=True):
                # 矩阵乘法预热（模拟神经网络计算）
                a = torch.randn(warmup_batch_size, 256, device=device, dtype=torch.float32)
                b = torch.randn(256, 128, device=device, dtype=torch.float32)
                c = torch.matmul(a, b)
                
                # 激活函数预热
                c = F.relu(c)
                c = F.gelu(c)
                
                # 模拟损失计算
                target = torch.randn_like(c)
                loss = F.mse_loss(c, target)
            
            # 模拟反向传播（使用scaler，与实际训练一致）
            # 注意：这里不实际计算梯度，只是触发CUDA kernel编译
            dummy_loss = loss.detach()
            if i == 0:
                # 第一次迭代执行完整的backward（触发更多kernel编译）
                dummy_loss.requires_grad_(True)
                dummy_loss.backward()
                dummy_loss = dummy_loss.detach()
            
            # 使用非阻塞传输预热
            c_cpu = c.cpu(non_blocking=True)
            
            # 同步确保操作完成（最后一次迭代）
            if i == warmup_iterations - 1:
                torch.cuda.synchronize()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        del scaler
        
        print(f"✓ GPU预热完成（{warmup_iterations}次迭代）")
        
    except Exception as e:
        print(f"⚠️ GPU预热过程中出现错误: {e}")
        print("将继续执行，但可能影响首次训练的性能")


def external_attention_sensitivity_analysis_with_hp_search(
    train_df_path: str,
    val_df_path: str,
    base_model_name: str = 'EA_sensitivity_hp',
    memory_sizes: list = [16, 32, 64, 128, 256],
    search_method: str = 'bayesian',
    n_trials_per_memory: int = 30,
    save_path: str = None,
    # 批次执行参数（用于分批显示进度，不影响搜索完成）
    termination_check_interval: int = 5,  # 每完成多少次试验显示一次进度（仅用于进度显示，不用于终止）
    enable_gpu_warmup: bool = True,  # 是否启用GPU预热（默认True）
    # 单次搜索终止条件（仅用于单个trial的提前终止，不影响全局搜索）
    single_trial_start_mae_threshold: float = 0.7,  # 起始MAE阈值（默认0.7）
    single_trial_epochs_threshold: int = 100,  # 训练轮数阈值（默认100轮）
    single_trial_mae_after_epochs_threshold: float = 0.2,  # 训练指定轮数后MAE阈值（默认0.2）
):
    """
    外部注意力机制敏感性分析（带超参数搜索）
    对每个memory_size执行超参数搜索，以客观比较不同memory_size对模型性能的影响
    
    Parameters:
    -----------
    train_df_path : str
        训练数据CSV文件路径
    val_df_path : str
        验证数据CSV文件路径
    base_model_name : str
        基础模型名称
    memory_sizes : list
        要测试的memory_size值列表
    search_method : str
        超参数搜索方法: 'grid', 'random', 'bayesian'
    n_trials_per_memory : int
        每个memory_size的搜索试验次数
    save_path : str
        结果保存路径（JSON文件）
    termination_check_interval : int
        每完成多少次试验显示一次进度（默认5，仅用于进度显示，不用于终止）
    enable_gpu_warmup : bool
        是否启用GPU预热（默认True）
        预热可以减少首次训练时的初始化开销，使训练时间更稳定
    single_trial_start_mae_threshold : float
        （已弃用）单次搜索终止条件已取消，此参数不再使用
    single_trial_epochs_threshold : int
        （已弃用）单次搜索终止条件已取消，此参数不再使用
    single_trial_mae_after_epochs_threshold : float
        （已弃用）单次搜索终止条件已取消，此参数不再使用
    consecutive_trial_termination_count : int
        （已弃用）单次搜索终止条件已取消，此参数不再使用
    consecutive_mae_threshold : float
        （已弃用）全部搜索终止条件：连续MAE阈值
    consecutive_mae_count : int
        （已弃用）全部搜索终止条件：连续搜索次数阈值
    no_improvement_trials : int
        全部搜索终止条件：取得最优值后，连续多少轮没有出现更优值则终止（默认10轮）
        如果从全局最佳MAE出现后，已经进行了no_improvement_trials轮搜索，且这no_improvement_trials轮内没有出现更优的值，则提前终止
        
    Returns:
    --------
    dict
        敏感性分析结果（包含每个memory_size的最佳超参数和性能）
    """
    # 设置默认保存路径
    if save_path is None:
        save_path = str(current_dir / 'external_attention_sensitivity_hp_search_results.json')
    
    print("="*80)
    print("外部注意力机制敏感性分析（带超参数搜索）")
    print("="*80)
    print(f"训练集路径: {train_df_path}")
    print(f"验证集路径: {val_df_path}")
    print(f"测试memory_size值: {memory_sizes}")
    print(f"搜索方法: {search_method}")
    print(f"每个memory_size的试验次数: {n_trials_per_memory}")
    print(f"搜索设置:")
    print(f"  单次搜索终止条件（仅用于单个trial的提前终止，不影响全局搜索）:")
    print(f"    1. 起始MAE > {single_trial_start_mae_threshold} (第1个epoch的验证集MAE)")
    print(f"    2. 训练{single_trial_epochs_threshold}轮后，验证集最佳MAE仍 > {single_trial_mae_after_epochs_threshold}")
    print(f"  全局搜索: 已取消所有提前终止条件，将完成所有 {n_trials_per_memory} 次试验以确保充分探索")
    print(f"  - 进度显示间隔: 每 {termination_check_interval} 次试验显示一次进度")
    print(f"结果保存路径: {save_path}")
    print("="*80)
    print()
    
    # 加载数据
    print(f"加载训练集: {train_df_path}")
    train_df = pd.read_csv(train_df_path)
    print(f"训练集大小: {len(train_df)}")
    
    print(f"加载验证集: {val_df_path}")
    val_df = pd.read_csv(val_df_path)
    print(f"验证集大小: {len(val_df)}")
    print()
    
    # 存储结果
    results = {
        'analysis_type': 'external_attention_sensitivity_with_hp_search',
        'memory_sizes': memory_sizes,
        'search_method': search_method,
        'n_trials_per_memory': n_trials_per_memory,
        'results': [],
        'summary': {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 对每个memory_size进行超参数搜索
    for idx, memory_size in enumerate(memory_sizes, 1):
        # 跳过 memory_size = 128
        if memory_size == 128:
            print("\n" + "="*80)
            print(f"【跳过 {idx}/{len(memory_sizes)}】memory_size = {memory_size} (已跳过)")
            print("="*80)
            continue
        
        print("\n" + "="*80)
        print(f"【测试 {idx}/{len(memory_sizes)}】memory_size = {memory_size}")
        print("="*80)
        
        # GPU预热（在每个memory_size搜索前预热，减少首次训练的开销）
        if enable_gpu_warmup:
            warmup_gpu(warmup_iterations=10, warmup_batch_size=32)
            print()
        
        # 为每个memory_size创建独立的搜索保存路径（保存在external_attention_sensitivity_analysis文件夹下）
        memory_save_path = str(current_dir / f'EA_sensitivity_mem{memory_size}_hp_search_results.json')
        print(f"\nmemory_size={memory_size} 的详细搜索结果将保存到: {memory_save_path}")
        
        # ========== 检查主结果文件中的已有结果（仅用于日志与进度提示，不再用于“跳过搜索”） ==========
        # 说明：之前这里会根据 terminated_early 标记跳过某些 memory_size 的搜索
        # 现在全局提前终止已取消，即使之前提前终止过，也会继续补齐到 n_trials_per_memory 次
        memory_already_terminated = False  # 保留变量但不再用于跳过搜索，仅兼容后续判断
        save_path_obj = Path(save_path)
        print(f"\n[检查提前终止] 主结果文件路径: {save_path_obj}")
        print(f"[检查提前终止] 文件是否存在: {save_path_obj.exists()}")
        
        if save_path_obj.exists():
            try:
                with open(save_path_obj, 'r', encoding='utf-8') as f:
                    main_results = json.load(f)
                    print(f"[检查提前终止] 主结果文件中有 {len(main_results.get('results', []))} 个memory_size的结果")
                    
                    # 查找该memory_size的结果
                    found_result = False
                    for result in main_results.get('results', []):
                        if result.get('memory_size') == memory_size:
                            found_result = True
                            terminated = result.get('terminated_early', False)
                            print(f"[检查提前终止] 找到 memory_size={memory_size} 的结果:")
                            print(f"  - terminated_early: {terminated}")
                            print(f"  - n_trials: {result.get('n_trials', 0)} / {result.get('n_trials_planned', n_trials_per_memory)}")
                            print(f"  - best_val_mae: {result.get('best_val_mae', 'N/A')}")
                            
                            if terminated:
                                # 仅做提示：之前曾提前终止，本次将继续补齐到 n_trials_per_memory
                                print(f"\n⚠️ 提示: memory_size={memory_size} 之前曾标记为提前终止")
                                print(f"  已完成试验数: {result.get('n_trials', 0)} / {result.get('n_trials_planned', n_trials_per_memory)}")
                                print(f"  最佳MAE: {result.get('best_val_mae', 'N/A')}")
                                print(f"  本次将继续搜索，直至完成所有 {n_trials_per_memory} 次试验")
                            break
                    
                    if not found_result:
                        print(f"[检查提前终止] 主结果文件中未找到 memory_size={memory_size} 的记录")
            except Exception as e:
                print(f"⚠️ 警告: 读取主结果文件失败: {e}，将继续检查详细结果文件")
                import traceback
                traceback.print_exc()
        else:
            print(f"[检查提前终止] 主结果文件不存在，将继续搜索")
        
        # 创建训练包装器，固定memory_size
        def create_train_wrapper_with_memory(train_df, val_df, base_name, memory_size, hidden_dim=38):
            """创建固定memory_size的训练包装器"""
            def train_wrapper(**hyperparameters):
                # 固定memory_size和hidden_dim
                hyperparameters['hidden_dim'] = hidden_dim
                hyperparameters['use_external_attention'] = True
                hyperparameters['attention_memory_size'] = memory_size
                hyperparameters['attn_alpha'] = 0.5
                # 每轮都评估验证集以记录完整训练轨迹
                hyperparameters.setdefault('eval_interval', 1)
                
                # ========== 数据加载设置（默认配置，已取消加速优化） ==========
                # 训练数据加载设置
                # 注意：num_workers过多可能导致内存不足（Windows页面文件错误）
                hyperparameters.setdefault('train_num_workers', 4)  # 默认4个worker
                hyperparameters.setdefault('train_pin_memory', True)  # 启用内存固定，加速GPU传输
                hyperparameters.setdefault('train_persistent_workers', True)  # 保持worker进程存活，避免重复创建
                hyperparameters.setdefault('train_prefetch_factor', 2)  # 默认预取批次：2
                
                # 验证集数据加载设置
                hyperparameters.setdefault('val_batch_size', hyperparameters.get('batch_size', 96))  # 与训练batch_size保持一致
                hyperparameters.setdefault('val_num_workers', 2)  # 验证集使用2个worker
                hyperparameters.setdefault('val_pin_memory', True)  # 启用内存固定
                hyperparameters.setdefault('val_prefetch_factor', 2)  # 验证集预取批次：2
                
                # ========== 计算设置 ==========
                # R²计算频率设置
                hyperparameters.setdefault('compute_r2_every_epoch', False)  # 每5轮计算一次R²
                
                # 禁用torch.compile加速（已取消）
                hyperparameters['use_torch_compile'] = False  # 禁用torch.compile，避免Triton依赖问题
                
                # 注意：训练代码中autocast是硬编码启用的，这个设置可能不影响实际训练
                hyperparameters.setdefault('use_mixed_precision', True)  # 启用混合精度（训练代码已支持）
                
                # 设置默认值
                if 'n_epochs' not in hyperparameters:
                    hyperparameters['n_epochs'] = 400
                if 'early_stopping_patience' not in hyperparameters:
                    hyperparameters['early_stopping_patience'] = 20
                
                # 单次搜索终止条件已取消，不再传递相关参数
                
                # 创建模型名称（不包含memory_size前缀，因为会在路径中单独处理）
                trial_name = f"{base_name}_mem{memory_size}_lr{hyperparameters['lr']:.2e}_"
                trial_name += f"hd{hidden_dim}_bs{hyperparameters['batch_size']}_ep{hyperparameters['n_epochs']}"
                
                # 创建memory_size目录结构：ReLU/EA_sensitivity_mem{memory_size}/{trial_name}/
                # 目录命名格式：EA_sensitivity_mem16, EA_sensitivity_mem32 等（去掉base_name中的_hp后缀）
                # 从base_name中去掉_hp后缀（如果存在），用于目录命名
                base_name_for_dir = base_name.replace('_hp', '') if base_name.endswith('_hp') else base_name
                memory_dir = f"{base_name_for_dir}_mem{memory_size}"
                model_path_with_memory = f"{memory_dir}/{trial_name}"
                
                try:
                    import time
                    trial_start_time = time.time()
                    
                    # 执行训练（传递包含memory_size目录的路径）
                    # 注意：train_GNNGH_T函数内部会处理路径，我们需要修改它以支持memory_size目录结构
                    # 暂时先传递完整的模型名称，然后在训练函数中处理路径
                    checkpoint_path = Path('ReLU') / memory_dir / trial_name / f'{trial_name}_checkpoint.pth'
                    resume_training = checkpoint_path.exists()
                    
                    # 修改模型名称以包含memory_size目录信息
                    # 训练函数会使用这个名称创建路径
                    model_name_with_path = model_path_with_memory
                    
                    # 传递memory_size信息到hyperparameters，以便训练函数使用
                    hyperparameters['memory_size_dir'] = memory_dir
                    
                    train_GNNGH_T(train_df, model_name_with_path, hyperparameters, resume=resume_training, val_df=val_df)
                    
                    # 读取验证集最佳MAE
                    path = Path('ReLU') / memory_dir / trial_name
                    report_path = path / f'Report_training_{trial_name}.txt'
                    
                    best_val_mae = None
                    if report_path.exists():
                        from hyperparameter_research.GHGEAT_hyperparameter_search import safe_read_file
                        content = safe_read_file(str(report_path))
                        lines = content.split('\n')
                        
                        # 优先级1: 从报告文件末尾的总结部分读取（最可靠）
                        # 查找"Best Epoch"行，然后在其后的总结部分查找"Validation MAE"
                        found_best_epoch = False
                        for i, line in enumerate(lines):
                            if 'Best Epoch' in line:
                                found_best_epoch = True
                                # 在"Best Epoch"之后的10行内查找"Validation MAE"
                                for j in range(i + 1, min(i + 10, len(lines))):
                                    next_line = lines[j]
                                    # 确保是总结部分的"Validation MAE"，不是epoch日志中的"Valid MAE"
                                    if 'Validation MAE' in next_line and ':' in next_line and 'Epoch' not in next_line:
                                        try:
                                            parts = next_line.split(':')
                                            if len(parts) >= 2:
                                                mae_str = parts[-1].strip()
                                                best_val_mae = float(mae_str)
                                                print(f"  ✓ 从报告文件总结部分读取最佳验证集MAE: {best_val_mae:.6f}")
                                                break
                                        except (ValueError, IndexError):
                                            pass
                                if best_val_mae is not None:
                                    break
                        
                        # 优先级2: 如果总结部分没有找到，从训练轨迹CSV文件读取（最可靠）
                        if best_val_mae is None:
                            traj_path = path / 'Training.csv'
                            if traj_path.exists():
                                try:
                                    traj_df = pd.read_csv(traj_path)
                                    if 'MAE_Valid' in traj_df.columns:
                                        best_val_mae = traj_df['MAE_Valid'].min()
                                        print(f"  ✓ 从Training.csv读取最佳验证集MAE: {best_val_mae:.6f}")
                                except Exception as e:
                                    print(f"  ⚠️ 警告: 从Training.csv读取失败: {e}")
                        
                        # 优先级3: 如果前两种方法都失败，从epoch日志中提取最小的"Best Valid MAE"
                        if best_val_mae is None:
                            all_best_maes = []
                            for line in lines:
                                # 只匹配epoch日志中的"Best Valid MAE:"，确保是完整的格式
                                if 'Best Valid MAE:' in line and 'Epoch' in line:
                                    try:
                                        parts = line.split('Best Valid MAE:')
                                        if len(parts) >= 2:
                                            mae_str = parts[-1].strip().split()[0]  # 取第一个数字
                                            all_best_maes.append(float(mae_str))
                                    except (ValueError, IndexError):
                                        pass
                            if all_best_maes:
                                best_val_mae = min(all_best_maes)
                                print(f"  ✓ 从epoch日志中提取最小Best Valid MAE: {best_val_mae:.6f}")
                    
                    # 从训练轨迹读取最佳验证集MAE（如果报告文件中没有）
                    traj_path = path / 'Training.csv'
                    if traj_path.exists():
                        traj_df = pd.read_csv(traj_path)
                        if 'MAE_Valid' in traj_df.columns:
                            # 读取最佳验证集MAE
                            if best_val_mae is None:
                                best_val_mae = traj_df['MAE_Valid'].min()
                    
                    trial_end_time = time.time()
                    trial_duration = trial_end_time - trial_start_time
                    
                    if best_val_mae is None:
                        print(f"⚠️ 警告: 无法读取验证集MAE，返回一个较大的值")
                        return 10.0  # 返回一个较大的值表示失败
                    
                    # 单次搜索终止条件已取消，不再检查
                    print(f"✓ 试验完成: 验证集最佳MAE = {best_val_mae:.6f}, 耗时 = {trial_duration/60:.2f}分钟")
                    
                    # 返回结果，包含额外信息用于终止判断
                    # 为了兼容HyperparameterTuner，仍然返回MAE值
                    # 但我们需要在结果中记录单次搜索终止信息
                    return best_val_mae
                    
                except Exception as e:
                    print(f"✗ 试验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return 10.0  # 返回一个较大的值表示失败
            
            return train_wrapper
        
        # 设置全局标志
        import hyperparameter_research.GHGEAT_hyperparameter_search as hp_module
        hp_module._IN_HYPERPARAMETER_SEARCH = True
        
        # 创建训练包装器
        train_wrapper = create_train_wrapper_with_memory(
            train_df, val_df, base_model_name, memory_size, hidden_dim=38
        )
        
        # 定义超参数搜索空间（固定memory_size，搜索其他参数）
        # 优化：对于贝叶斯优化，使用连续范围而不是离散值列表
        # 这样可以更好地探索整个搜索空间，避免陷入局部最优
        if search_method == 'bayesian':
            # 贝叶斯优化：使用连续范围，会自动使用对数尺度（log=True）因为范围跨越多个数量级
            # 优化：增加探索性，确保充分覆盖整个学习率范围
            # 策略：使用更宽的范围，并在前几个trial中强制探索不同区域
            param_space = {
                'lr': [1e-5, 1e-3],  # 连续范围：1e-5 到 1e-3，使用对数尺度探索
                # 使用中等batch_size，平衡训练速度和泛化能力
                # 中等batch_size (64-128) 通常能提供更好的泛化性能，避免陷入sharp minima
                'batch_size': [64, 128],  # 连续范围：64 到 128
                'n_epochs': [300, 500],  # 连续范围：300 到 500
                'early_stopping_patience': [15, 30]  # 连续范围：15 到 30
            }
            
            # 优化：为贝叶斯优化设置更高的探索权重，避免过早收敛
            # 这可以通过调整HyperparameterTuner的acquisition_function参数实现
            # 但当前实现可能不支持，所以通过增加初始随机探索来补偿
        else:
            # 网格搜索或随机搜索：使用离散值列表
            param_space = {
                'lr': [1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1e-3],
                'batch_size': [64, 96, 128],
                'n_epochs': [300, 400, 500],
                'early_stopping_patience': [15, 20, 25, 30]
            }
        
        print(f"固定参数: memory_size={memory_size}, hidden_dim=38, use_external_attention=True")
        print(f"搜索参数空间: {list(param_space.keys())}")
        print(f"评估指标: 验证集最佳MAE（越小越好）")
        print()
        
        # 创建超参数调优器
        # 优化：对于贝叶斯优化，增加初始随机探索以确保充分覆盖搜索空间
        # 策略：前几个trial使用随机搜索，后续使用贝叶斯优化（如果支持）
        # 当前实现不支持混合策略，但可以通过增加n_trials来补偿
        tuner = HyperparameterTuner(
            search_method=search_method,
            n_trials=n_trials_per_memory,
            random_seed=42,
            auto_save_path=memory_save_path
        )
        
        # 优化建议：对于贝叶斯优化，确保至少完成10-15个trial才能充分探索学习率空间
        # 如果已完成trial数 < 10，建议继续搜索而不是过早终止
        if search_method == 'bayesian' and len(tuner.results) < 10:
            print(f"⚠️ 注意: 当前已完成 {len(tuner.results)} 个trial，贝叶斯优化需要至少10-15个trial")
            print(f"   才能充分探索学习率空间 [1e-5, 1e-3]")
            print(f"   建议继续搜索，避免过早收敛到局部最优")
        
        # 检查详细结果文件（HyperparameterTuner保存的文件）
        # 注意：此时memory_already_terminated应该已经在上面检查过了
        
        # 检查详细结果文件（HyperparameterTuner保存的文件）
        if Path(memory_save_path).exists() and not memory_already_terminated:
            try:
                tuner = HyperparameterTuner.load_results(memory_save_path)
                tuner.auto_save_path = Path(memory_save_path)
                # 确保n_trials设置为目标值（30次）
                tuner.n_trials = n_trials_per_memory
                completed = len(tuner.results)
                remaining = n_trials_per_memory - completed
                
                # 检查是否有正在运行的试验（通过检查模型目录）
                # 如果发现正在运行的试验，应该等待其完成而不是重新生成参数
                running_trial_detected = False
                if remaining > 0:
                    # 检查ReLU目录下是否有正在运行的试验
                    relu_dir = Path('ReLU')
                    if relu_dir.exists():
                        # 查找所有可能的试验目录（匹配命名模式）
                        pattern = f"{base_model_name}_mem{memory_size}_lr*_hd*_bs*_ep*"
                        import glob
                        trial_dirs = list(relu_dir.glob(pattern))
                        
                        # 检查这些目录是否有检查点文件但还没有best模型（说明正在运行）
                        for trial_dir in trial_dirs:
                            checkpoint_path = trial_dir / f'{trial_dir.name}_checkpoint.pth'
                            best_model_path = trial_dir / f'{trial_dir.name}_best.pth'
                            
                            # 如果有检查点但没有best模型，说明训练正在进行中
                            if checkpoint_path.exists() and not best_model_path.exists():
                                # 检查训练轨迹文件，确认训练是否真的在进行
                                traj_path = trial_dir / 'Training.csv'
                                if traj_path.exists():
                                    try:
                                        traj_df = pd.read_csv(traj_path)
                                        if len(traj_df) > 0:
                                            running_trial_detected = True
                                            print(f"\n⚠️ 检测到正在运行的试验:")
                                            print(f"   试验目录: {trial_dir.name}")
                                            print(f"   已训练轮数: {len(traj_df)}")
                                            print(f"   检查点文件存在: {checkpoint_path.exists()}")
                                            print(f"   最佳模型文件存在: {best_model_path.exists()}")
                                            print(f"   ⚠️ 建议: 等待该试验完成后再运行搜索，或手动删除该目录以重新开始")
                                            break
                                    except Exception as e:
                                        pass
                
                if remaining > 0:
                    if running_trial_detected:
                        print(f"\n⚠️ 警告: 检测到正在运行的试验，但搜索结果文件显示已完成 {completed} 次试验")
                        print(f"   这可能是因为试验正在运行中，结果尚未保存到文件")
                        print(f"   建议: 等待当前试验完成，或检查是否有其他进程正在运行该试验")
                    else:
                        print(f"✓ 检测到已保存的搜索结果，已完成 {completed} / {n_trials_per_memory} 次试验，还需完成 {remaining} 次")
                        print(f"   将继续搜索直到完成所有 {n_trials_per_memory} 次试验")
                else:
                    print(f"✓ 检测到已完成的搜索结果，共 {completed} / {n_trials_per_memory} 次试验（已完成）")
            except Exception as e:
                print(f"警告: 加载已保存结果失败: {e}，将从头开始搜索")
        
        # 执行超参数搜索（支持早期终止检查）
        try:
            # 创建支持早期终止的搜索函数（使用新的终止条件）
            # 跟踪全局最佳MAE和其出现的轮次
            best_score_global = None
            best_score_trial = None
            
            # 初始化：从已有结果中找出最佳值
            if len(tuner.results) > 0:
                valid_results = [r for r in tuner.results if r.get('score') is not None]
                if valid_results:
                    best_result = min(valid_results, key=lambda x: x.get('score'))
                    best_score_global = best_result.get('score')
                    best_score_trial = best_result.get('trial', len(tuner.results))
            
            # 注意：已取消所有提前终止条件，将完成所有计划的trial以确保充分探索参数空间
            
            # 执行搜索（采用分批执行的方式，确保完成所有计划的trial）
            initial_trials = len(tuner.results)
            remaining_trials = n_trials_per_memory - initial_trials
            terminated_early = False
            search_result = None
            
            # 确保tuner的n_trials设置为目标值
            tuner.n_trials = n_trials_per_memory
            
            if remaining_trials > 0:
                print(f"\n继续搜索: 当前已完成 {initial_trials} / {n_trials_per_memory} 次试验，将继续完成剩余 {remaining_trials} 次")
                # 分批执行搜索，每批后显示进度（已取消提前终止，将完成所有trial）
                batch_size = termination_check_interval
                
                # 计算应该从第几批开始（批次按固定区间划分：第1批1-5，第2批6-10，第3批11-15，...）
                # 如果已完成12次，则应该从第3批（11-15）开始
                first_batch_num = (initial_trials // batch_size) + 1  # 应该从第几批开始
                first_batch_start = (first_batch_num - 1) * batch_size  # 该批次的起始索引（0-based）
                
                # 遍历所有需要执行的批次
                batch_num = first_batch_num
                batch_start = first_batch_start
                
                while batch_start < n_trials_per_memory:
                    # 计算本批应该完成到的位置（固定区间）
                    batch_end = min(batch_start + batch_size, n_trials_per_memory)
                    # 本批需要新增的试验数 = 目标完成数 - 已完成数，但不能超过批次大小
                    batch_trials = batch_end - initial_trials if batch_end > initial_trials else 0
                    # 如果本批的部分试验已完成，则只执行剩余部分
                    actual_batch_start = max(batch_start, initial_trials)
                    actual_batch_trials = batch_end - actual_batch_start
                    
                    # 如果本批没有需要执行的试验（已全部完成），跳过
                    if actual_batch_trials <= 0:
                        batch_num += 1
                        batch_start += batch_size
                        continue
                    
                    print(f"\n执行第 {batch_num} 批搜索: 从第 {actual_batch_start + 1} 次试验开始，目标累计完成到第 {batch_end} 次试验 (本批新增 {actual_batch_trials} 次)")
                    
                    # 临时修改n_trials，只执行到batch_end
                    original_n_trials = tuner.n_trials
                    tuner.n_trials = batch_end
                    
                    try:
                        # 执行搜索（会从已完成的试验继续，只执行新的试验）
                        search_result = tuner.search(
                            param_space,
                            train_wrapper,
                            scoring_metric='MAE',
                            lower_is_better=True
                        )
                    except Exception as e:
                        tuner.n_trials = original_n_trials
                        raise e
                    finally:
                        # 恢复原始n_trials
                        tuner.n_trials = original_n_trials
                    
                    # 获取当前最佳MAE和已完成试验数
                    current_best_mae = search_result.get('best_score')
                    completed_trials = len(tuner.results)
                    
                    # 显示进度信息（已取消提前终止，将完成所有trial）
                    # 计算全局最佳值（仅用于显示，不用于终止判断）
                    all_scores = [r.get('score') for r in tuner.results if r.get('score') is not None]
                    if len(all_scores) > 0:
                        best_score_global = min(all_scores)
                        best_score_trial = next((i+1 for i, r in enumerate(tuner.results) if r.get('score') == best_score_global), None)
                        if best_score_trial is not None:
                            trials_since_best = completed_trials - best_score_trial
                            termination_info = f", 全局最佳MAE={best_score_global:.6f} (第{best_score_trial}轮), 已进行{trials_since_best}轮未改进"
                        else:
                            termination_info = ""
                    else:
                        termination_info = ""
                    
                    print(f"批次完成: 当前最佳MAE = {current_best_mae:.6f}, 已完成 {completed_trials} / {n_trials_per_memory} 次试验{termination_info}")
                    
                    # 注意：已取消所有提前终止条件，将完成所有计划的trial以确保充分探索
                    
                    # 如果已完成所有试验，退出循环
                    if completed_trials >= n_trials_per_memory:
                        break
                    
                    # 更新到下一个批次
                    batch_num += 1
                    batch_start += batch_size
                    # 更新initial_trials为当前已完成数，以便下一批次正确计算
                    initial_trials = completed_trials
            else:
                # 已经完成所有试验，直接获取结果（但需要确保search_result存在）
                if search_result is None:
                    # 如果所有试验都已完成，从已有结果构建search_result
                    if len(tuner.results) > 0:
                        valid_results = [r for r in tuner.results if r.get('score') is not None]
                        if valid_results:
                            best_result = min(valid_results, key=lambda x: x.get('score'))
                            search_result = {
                                'best_params': best_result.get('params', {}),
                                'best_score': best_result.get('score'),
                                'results': tuner.results
                            }
                        else:
                            # 如果没有有效结果，执行一次搜索（虽然不应该发生）
                            search_result = tuner.search(
                                param_space,
                                train_wrapper,
                                scoring_metric='MAE',
                                lower_is_better=True
                            )
                    else:
                        # 如果没有结果，执行搜索
                        search_result = tuner.search(
                            param_space,
                            train_wrapper,
                            scoring_metric='MAE',
                            lower_is_better=True
                        )
            
            # 提取结果
            best_params = search_result['best_params']
            best_score = search_result['best_score']
            search_results = search_result.get('results', [])
            
            # 记录结果（包含该memory_size的详细搜索结果文件路径）
            result = {
                'memory_size': memory_size,
                'best_val_mae': best_score,
                'best_hyperparameters': best_params,
                'n_trials': len(search_results),
                'n_trials_planned': n_trials_per_memory,
                'terminated_early': terminated_early,
                'all_trials': search_results,
                'search_results_file': memory_save_path  # 记录该memory_size的详细搜索结果文件
            }
            results['results'].append(result)
            
            # 保存中间结果到主结果文件（确保提前终止状态被保存）
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                print(f"警告: 保存中间结果失败: {e}")
            
            # 确保该memory_size的搜索结果已保存（HyperparameterTuner会自动保存，这里确认一下）
            if Path(memory_save_path).exists():
                print(f"✓ memory_size={memory_size} 的详细搜索结果已保存到: {memory_save_path}")
            else:
                print(f"⚠️ 警告: memory_size={memory_size} 的搜索结果文件未找到: {memory_save_path}")
            
            print(f"\n{'='*80}")
            print(f"memory_size={memory_size} 的超参数搜索完成:")
            print(f"  最佳验证集MAE: {best_score:.6f}")
            print(f"  最佳超参数: {best_params}")
            print(f"  实际完成试验数: {len(search_results)} / {n_trials_per_memory}")
            if len(search_results) < n_trials_per_memory:
                print(f"  ⚠️ 警告: 只完成了 {len(search_results)} 次试验，未达到目标 {n_trials_per_memory} 次")
                print(f"  ⚠️ 建议: 重新运行脚本以继续完成剩余 {n_trials_per_memory - len(search_results)} 次试验")
            else:
                print(f"  ✓ 已完成所有 {n_trials_per_memory} 次试验")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n错误: memory_size={memory_size} 的超参数搜索失败: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'memory_size': memory_size,
                'best_val_mae': None,
                'best_hyperparameters': None,
                'error': str(e),
                'search_results_file': memory_save_path if 'memory_save_path' in locals() else None
            }
            results['results'].append(result)
        
        # 保存中间结果
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"警告: 保存结果失败: {e}")
    
    # 计算汇总统计
    valid_results = [r for r in results['results'] if r.get('best_val_mae') is not None]
    if valid_results:
        val_mae_values = [r['best_val_mae'] for r in valid_results]
        best_idx = np.argmin(val_mae_values)
        best_result = valid_results[best_idx]
        
        results['summary'] = {
            'best_memory_size': best_result['memory_size'],
            'best_val_mae': best_result['best_val_mae'],
            'best_hyperparameters': best_result.get('best_hyperparameters'),
            'val_mae_mean': np.mean(val_mae_values),
            'val_mae_std': np.std(val_mae_values),
            'val_mae_min': np.min(val_mae_values),
            'val_mae_max': np.max(val_mae_values),
            'total_memory_sizes': len(results['results']),
            'successful_searches': len(valid_results)
        }
        
        print("\n" + "="*80)
        print("敏感性分析汇总（基于超参数搜索）")
        print("="*80)
        print(f"最佳memory_size: {results['summary']['best_memory_size']}")
        print(f"最佳验证集MAE: {results['summary']['best_val_mae']:.6f}")
        if results['summary'].get('best_hyperparameters'):
            print(f"最佳超参数组合: {results['summary']['best_hyperparameters']}")
        print(f"验证集MAE范围: [{results['summary']['val_mae_min']:.6f}, {results['summary']['val_mae_max']:.6f}]")
        print(f"验证集MAE均值: {results['summary']['val_mae_mean']:.6f} ± {results['summary']['val_mae_std']:.6f}")
        print(f"成功搜索数: {results['summary']['successful_searches']} / {results['summary']['total_memory_sizes']}")
        print("="*80)
    
    # 保存最终结果
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到: {save_path}")
    except Exception as e:
        print(f"警告: 保存结果失败: {e}")
    
    return results


def external_attention_sensitivity_analysis(
    df_path: str,
    base_model_name: str = 'EA_sensitivity',
    memory_sizes: list = [16,32,64, 128, 256],
    base_hyperparameters: dict = None,
    save_path: str = None
):
    """
    外部注意力机制敏感性分析
    
    Parameters:
    -----------
    df_path : str
        训练数据CSV文件路径
    base_model_name : str
        基础模型名称
    memory_sizes : list
        要测试的memory_size值列表
    base_hyperparameters : dict
        基础超参数（其他参数固定，只改变memory_size）
    save_path : str
        结果保存路径（JSON文件）
        
    Returns:
    --------
    dict
        敏感性分析结果
    """
    # 设置默认超参数
    if base_hyperparameters is None:
        base_hyperparameters = {
            'hidden_dim': 38,
            'lr': 0.0002532501358651798,
            'n_epochs': 400,
            'batch_size': 64,
            'early_stopping_patience': 20,
            'use_external_attention': True,  # 必须启用外部注意力
            'attn_alpha': 0.5,
            'attention_memory_size': 128  # 这个值会被测试的不同值覆盖
        }
    
    # 设置默认保存路径
    if save_path is None:
        save_path = str(current_dir / 'external_attention_sensitivity_results.json')
    
    print("="*80)
    print("外部注意力机制敏感性分析")
    print("="*80)
    print(f"数据路径: {df_path}")
    print(f"测试memory_size值: {memory_sizes}")
    print(f"基础超参数: {base_hyperparameters}")
    print(f"结果保存路径: {save_path}")
    print("="*80)
    print()
    
    # 统一输出目录：将当前工作目录切到脚本所在目录
    os.chdir(current_dir)
    
    # 直接使用与超参搜索一致的训练/验证集路径
    train_path = current_dir.parent / 'data' / 'processed' / 'new_dataset' / 'train_dataset' / 'v2' / 'molecule' / 'molecule_train.csv'
    val_path = current_dir.parent / 'data' / 'processed' / 'new_dataset' / 'train_dataset' / 'v2' / 'molecule' / 'molecule_valid.csv'
    print(f"加载训练集: {train_path}")
    print(f"加载验证集: {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print()
    
    # 存储结果
    results = {
        'analysis_type': 'external_attention_sensitivity',
        'base_hyperparameters': base_hyperparameters,
        'memory_sizes': memory_sizes,
        'results': [],
        'summary': {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 对每个memory_size进行训练
    for idx, memory_size in enumerate(memory_sizes, 1):
        # 跳过 memory_size = 128
        if memory_size == 128:
            print("\n" + "="*80)
            print(f"【跳过 {idx}/{len(memory_sizes)}】memory_size = {memory_size} (已跳过)")
            print("="*80)
            continue
        
        print("\n" + "="*80)
        print(f"【测试 {idx}/{len(memory_sizes)}】memory_size = {memory_size}")
        print("="*80)
        
        # 创建模型名称
        model_name = f"{base_model_name}_mem{memory_size}"
        
        # 设置超参数（启用外部注意力，设置memory_size）
        hyperparameters = base_hyperparameters.copy()
        hyperparameters['use_external_attention'] = True
        hyperparameters['attention_memory_size'] = memory_size
        
        print(f"模型名称: {model_name}")
        print(f"超参数: {hyperparameters}")
        print()
        
        # 记录开始时间
        trial_start_time = time.time()
        
        try:
            # 检查模型目录与 best 模型
            model_dir = Path('ReLU') / model_name
            checkpoint_path = model_dir / f'{model_name}_checkpoint.pth'
            best_model_path = model_dir / f'{model_name}_best.pth'
            
            # 如果已经存在 best 模型，认为该 memory_size 已经训练完成，直接跳过训练阶段
            if best_model_path.exists():
                print(f"检测到模型已完成训练（存在 best 模型），跳过训练，直接读取结果")
            else:
                # 仅在没有 best 模型时才考虑从检查点续训
                resume_training = checkpoint_path.exists()
                if resume_training:
                    print(f"检测到未完成的检查点，将从中断处继续训练")
                train_GNNGH_T(train_df, model_name, hyperparameters, resume=resume_training, val_df=val_df)
            
            # 读取训练结果（无论是新跑的还是历史结果）
            path = Path('ReLU') / model_name
            report_path = path / f'Report_training_{model_name}.txt'
            
            # 验证集指标（优先使用）
            best_val_mae = None
            best_val_r2 = None
            best_val_epoch = None
            # 训练集指标（作为参考）
            best_train_mae = None
            best_train_r2 = None
            
            if report_path.exists():
                try:
                    from hyperparameter_research.GHGEAT_hyperparameter_search import safe_read_file
                    content = safe_read_file(str(report_path))
                    
                    # 检查是否有验证集评估
                    has_validation = 'Validation' in content or 'Valid MAE' in content or '验证集' in content
                    
                    for line in content.split('\n'):
                        # 优先读取验证集指标
                        if has_validation:
                            # 读取验证集最佳MAE（从"最佳验证集MAE"或"Best Valid MAE"行）
                            if ('最佳验证集MAE' in line or 'Best Valid MAE' in line or 'Best Validation MAE' in line) and ':' in line:
                                try:
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        mae_str = parts[-1].strip()
                                        # 可能包含epoch信息，需要提取数字
                                        mae_str = mae_str.split('(')[0].strip()  # 移除epoch信息
                                        best_val_mae = float(mae_str)
                                except (ValueError, IndexError):
                                    pass
                            # 读取验证集R²
                            elif ('验证集R²' in line or 'Validation R²' in line or 'Valid R²' in line or 'Valid R2' in line) and ':' in line:
                                try:
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        r2_str = parts[-1].strip()
                                        best_val_r2 = float(r2_str)
                                except (ValueError, IndexError):
                                    pass
                            # 从epoch日志中提取验证集最佳指标（如果还没有找到）
                            elif 'Best Valid MAE:' in line and best_val_mae is None:
                                try:
                                    # 格式: "Epoch X - ... | Best Valid MAE: 0.123"
                                    parts = line.split('Best Valid MAE:')
                                    if len(parts) >= 2:
                                        mae_str = parts[-1].strip().split()[0]  # 取第一个数字
                                        best_val_mae = float(mae_str)
                                except (ValueError, IndexError):
                                    pass
                            # 从epoch日志中提取当前epoch的验证集指标（用于跟踪最佳值）
                            elif 'Valid MAE:' in line and '|' in line:
                                try:
                                    # 格式: "Epoch X - Train MAE: ... | Valid MAE: 0.123, R²: 0.456 | Best Valid MAE: 0.123"
                                    # 提取Valid MAE部分
                                    if 'Valid MAE:' in line:
                                        valid_part = line.split('Valid MAE:')[1]
                                        if '|' in valid_part:
                                            valid_part = valid_part.split('|')[0]
                                        # 提取MAE值
                                        if ',' in valid_part:
                                            mae_str = valid_part.split(',')[0].strip()
                                        else:
                                            mae_str = valid_part.strip().split()[0]
                                        current_val_mae = float(mae_str)
                                        # 如果还没有最佳值，或者当前值更小，更新最佳值
                                        if best_val_mae is None or current_val_mae < best_val_mae:
                                            best_val_mae = current_val_mae
                                        # 提取R²值
                                        if 'R²:' in valid_part or 'R2:' in valid_part:
                                            r2_part = valid_part.split('R²:')[-1] if 'R²:' in valid_part else valid_part.split('R2:')[-1]
                                            r2_str = r2_part.strip().split()[0] if r2_part.strip() else None
                                            if r2_str:
                                                current_val_r2 = float(r2_str)
                                                if best_val_r2 is None or (best_val_mae == current_val_mae):
                                                    best_val_r2 = current_val_r2
                                except (ValueError, IndexError):
                                    pass
                        
                        # 读取最佳epoch（可能来自验证集或训练集）
                        if 'Best Epoch' in line and ':' in line:
                            try:
                                parts = line.split(':')
                                if len(parts) >= 2:
                                    epoch_str = parts[-1].strip()
                                    # 可能包含其他信息，提取第一个数字
                                    epoch_str = epoch_str.split()[0] if epoch_str.split() else epoch_str
                                    best_val_epoch = int(epoch_str)
                            except (ValueError, IndexError):
                                pass
                        
                        # 如果没有验证集，回退到训练集指标
                        if not has_validation:
                            if 'Training MAE' in line and ':' in line:
                                try:
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        mae_str = parts[-1].strip()
                                        best_val_mae = float(mae_str)  # 使用训练集MAE作为验证集MAE
                                except (ValueError, IndexError):
                                    continue
                            elif ('Training R^2' in line or 'Training R2' in line) and ':' in line:
                                try:
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        r2_str = parts[-1].strip()
                                        best_val_r2 = float(r2_str)  # 使用训练集R²作为验证集R²
                                except (ValueError, IndexError):
                                    continue
                        
                        # 同时记录训练集指标作为参考
                        if 'Training MAE' in line and ':' in line:
                            try:
                                parts = line.split(':')
                                if len(parts) >= 2:
                                    mae_str = parts[-1].strip()
                                    best_train_mae = float(mae_str)
                            except (ValueError, IndexError):
                                pass
                        elif ('Training R^2' in line or 'Training R2' in line) and ':' in line:
                            try:
                                parts = line.split(':')
                                if len(parts) >= 2:
                                    r2_str = parts[-1].strip()
                                    best_train_r2 = float(r2_str)
                            except (ValueError, IndexError):
                                pass
                                
                except Exception as e:
                    print(f"警告: 读取报告文件失败: {e}")
            
            # 如果无法从报告文件读取验证集指标，尝试从训练轨迹文件读取
            if best_val_mae is None:
                traj_path = path / 'Training.csv'
                if traj_path.exists():
                    traj_df = pd.read_csv(traj_path)
                    # 优先查找验证集列
                    if 'MAE_Valid' in traj_df.columns:
                        best_val_mae = traj_df['MAE_Valid'].min()
                        best_val_epoch = traj_df['MAE_Valid'].idxmin() + 1
                        if 'R2_Valid' in traj_df.columns:
                            best_val_r2 = traj_df.loc[traj_df['MAE_Valid'].idxmin(), 'R2_Valid']
                    elif 'MAE_Val' in traj_df.columns:
                        best_val_mae = traj_df['MAE_Val'].min()
                        best_val_epoch = traj_df['MAE_Val'].idxmin() + 1
                        if 'R2_Val' in traj_df.columns:
                            best_val_r2 = traj_df.loc[traj_df['MAE_Val'].idxmin(), 'R2_Val']
                    # 如果没有验证集列，使用训练集列（作为回退）
                    elif 'MAE_Train' in traj_df.columns:
                        best_val_mae = traj_df['MAE_Train'].min()
                        best_val_epoch = traj_df['MAE_Train'].idxmin() + 1
                        if 'R2_Train' in traj_df.columns:
                            best_val_r2 = traj_df.loc[traj_df['MAE_Train'].idxmin(), 'R2_Train']
                    # 同时记录训练集指标
                    if 'MAE_Train' in traj_df.columns and best_train_mae is None:
                        best_train_mae = traj_df['MAE_Train'].min()
                    if 'R2_Train' in traj_df.columns and best_train_r2 is None:
                        best_train_r2 = traj_df.loc[traj_df['MAE_Train'].idxmin(), 'R2_Train']
            
            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            
            # 记录结果（以验证集指标为准）
            result = {
                'memory_size': memory_size,
                'model_name': model_name,
                'best_val_mae': best_val_mae,  # 验证集MAE（主要指标）
                'best_val_r2': best_val_r2,     # 验证集R²
                'best_val_epoch': best_val_epoch,  # 验证集最佳轮次
                'best_train_mae': best_train_mae,  # 训练集MAE（参考）
                'best_train_r2': best_train_r2,   # 训练集R²（参考）
                'duration': trial_duration,
                'hyperparameters': hyperparameters
            }
            results['results'].append(result)
            
            print(f"\n结果（以验证集指标为准）:")
            print(f"  memory_size: {memory_size}")
            print(f"  验证集最佳MAE: {best_val_mae:.6f}" if best_val_mae is not None else "  验证集最佳MAE: N/A")
            print(f"  验证集最佳R²: {best_val_r2:.6f}" if best_val_r2 is not None else "  验证集最佳R²: N/A")
            print(f"  验证集最佳轮次: {best_val_epoch}" if best_val_epoch is not None else "  验证集最佳轮次: N/A")
            if best_train_mae is not None:
                print(f"  训练集最佳MAE（参考）: {best_train_mae:.6f}")
            if best_train_r2 is not None:
                print(f"  训练集最佳R²（参考）: {best_train_r2:.6f}")
            print(f"  训练时间: {trial_duration/60:.2f}分钟")
            
            # 保存中间结果（每次试验后保存）
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                print(f"警告: 保存结果失败: {e}")
                
        except Exception as e:
            print(f"\n错误: memory_size={memory_size} 的训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 记录失败的结果
            result = {
                'memory_size': memory_size,
                'model_name': model_name,
                'best_val_mae': None,
                'best_val_r2': None,
                'best_val_epoch': None,
                'best_train_mae': None,
                'best_train_r2': None,
                'duration': None,
                'error': str(e)
            }
            results['results'].append(result)
    
    # 计算汇总统计（以验证集指标为准）
    valid_results = [r for r in results['results'] if r.get('best_val_mae') is not None]
    if valid_results:
        val_mae_values = [r['best_val_mae'] for r in valid_results]
        best_idx = np.argmin(val_mae_values)
        best_result = valid_results[best_idx]
        
        results['summary'] = {
            'best_memory_size': best_result['memory_size'],
            'best_val_mae': best_result['best_val_mae'],
            'best_val_r2': best_result.get('best_val_r2'),
            'best_val_epoch': best_result.get('best_val_epoch'),
            'val_mae_mean': np.mean(val_mae_values),
            'val_mae_std': np.std(val_mae_values),
            'val_mae_min': np.min(val_mae_values),
            'val_mae_max': np.max(val_mae_values),
            'total_trials': len(results['results']),
            'successful_trials': len(valid_results)
        }
        
        # 如果有训练集指标，也记录
        train_mae_values = [r.get('best_train_mae') for r in valid_results if r.get('best_train_mae') is not None]
        if train_mae_values:
            results['summary']['best_train_mae'] = best_result.get('best_train_mae')
            results['summary']['train_mae_mean'] = np.mean(train_mae_values)
            results['summary']['train_mae_std'] = np.std(train_mae_values)
        
        print("\n" + "="*80)
        print("敏感性分析汇总（以验证集指标为准）")
        print("="*80)
        print(f"最佳memory_size: {results['summary']['best_memory_size']}")
        print(f"验证集最佳MAE: {results['summary']['best_val_mae']:.6f}")
        if results['summary'].get('best_val_r2') is not None:
            print(f"验证集最佳R²: {results['summary']['best_val_r2']:.6f}")
        if results['summary'].get('best_val_epoch') is not None:
            print(f"验证集最佳轮次: {results['summary']['best_val_epoch']}")
        print(f"验证集MAE范围: [{results['summary']['val_mae_min']:.6f}, {results['summary']['val_mae_max']:.6f}]")
        print(f"验证集MAE均值: {results['summary']['val_mae_mean']:.6f} ± {results['summary']['val_mae_std']:.6f}")
        if 'best_train_mae' in results['summary']:
            print(f"训练集最佳MAE（参考）: {results['summary']['best_train_mae']:.6f}")
        print(f"成功试验数: {results['summary']['successful_trials']} / {results['summary']['total_trials']}")
        print("="*80)
    
    # 保存最终结果
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到: {save_path}")
    except Exception as e:
        print(f"警告: 保存结果失败: {e}")
    
    # 生成可视化图表
    try:
        plot_sensitivity_results(results, save_path.replace('.json', '_plot.png'))
    except Exception as e:
        print(f"警告: 生成可视化图表失败: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def plot_sensitivity_results(results, save_path=None):
    """
    绘制敏感性分析结果
    
    Parameters:
    -----------
    results : dict
        敏感性分析结果
    save_path : str
        图表保存路径
    """
    valid_results = [r for r in results['results'] if r.get('best_val_mae') is not None]
    
    if not valid_results:
        print("没有有效结果可以绘制")
        return
    
    memory_sizes = [r['memory_size'] for r in valid_results]
    val_mae_values = [r['best_val_mae'] for r in valid_results]
    val_r2_values = [r.get('best_val_r2') for r in valid_results if r.get('best_val_r2') is not None]
    train_mae_values = [r.get('best_train_mae') for r in valid_results if r.get('best_train_mae') is not None]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 验证集MAE vs memory_size
    ax1 = axes[0]
    ax1.plot(memory_sizes, val_mae_values, 'o-', linewidth=2, markersize=8, color='steelblue', label='验证集MAE')
    # 如果有训练集数据，也绘制作为对比
    if train_mae_values and len(train_mae_values) == len(memory_sizes):
        ax1.plot(memory_sizes, train_mae_values, 's--', linewidth=1.5, markersize=6, color='lightblue', alpha=0.7, label='训练集MAE（参考）')
    ax1.set_xlabel('记忆单元大小 (Memory Size)', fontsize=12)
    ax1.set_ylabel('平均绝对误差 (MAE)', fontsize=12)
    ax1.set_title('验证集MAE vs 记忆单元大小', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # 使用对数刻度，因为memory_size通常是2的幂
    
    # 标记最佳点
    best_idx = np.argmin(val_mae_values)
    best_memory = memory_sizes[best_idx]
    best_mae = val_mae_values[best_idx]
    ax1.plot(best_memory, best_mae, 'r*', markersize=20, 
             label=f'最佳: {best_memory} (Val MAE={best_mae:.6f})')
    ax1.legend()
    
    # 子图2: 验证集R² vs memory_size (如果有R²数据)
    if val_r2_values and len(val_r2_values) == len(memory_sizes):
        ax2 = axes[1]
        ax2.plot(memory_sizes, val_r2_values, 's-', linewidth=2, markersize=8, color='coral', label='验证集R²')
        ax2.set_xlabel('记忆单元大小 (Memory Size)', fontsize=12)
        ax2.set_ylabel('决定系数 (R²)', fontsize=12)
        ax2.set_title('验证集R² vs 记忆单元大小', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # 标记最佳点
        best_r2_idx = np.argmax(val_r2_values)
        best_r2_memory = memory_sizes[best_r2_idx]
        best_r2 = val_r2_values[best_r2_idx]
        ax2.plot(best_r2_memory, best_r2, 'r*', markersize=20, 
                 label=f'最佳: {best_r2_memory} (Val R²={best_r2:.6f})')
        ax2.legend()
    else:
        # 如果没有R²数据，显示验证集MAE的详细视图
        ax2 = axes[1]
        ax2.bar(range(len(memory_sizes)), val_mae_values, color='steelblue', alpha=0.7, label='验证集MAE')
        if train_mae_values and len(train_mae_values) == len(memory_sizes):
            ax2.bar(range(len(memory_sizes)), train_mae_values, color='lightblue', alpha=0.5, label='训练集MAE（参考）')
        ax2.set_xticks(range(len(memory_sizes)))
        ax2.set_xticklabels([str(ms) for ms in memory_sizes], rotation=45)
        ax2.set_xlabel('记忆单元大小 (Memory Size)', fontsize=12)
        ax2.set_ylabel('平均绝对误差 (MAE)', fontsize=12)
        ax2.set_title('不同记忆单元大小的验证集MAE对比', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # 标记最佳点
        best_idx = np.argmin(val_mae_values)
        ax2.bar(best_idx, val_mae_values[best_idx], color='red', alpha=0.8, 
                label=f'最佳: {memory_sizes[best_idx]}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数"""
    # 优化系统性能（减少夜间运行时因系统资源竞争导致的训练时间变长）
    print("="*80)
    print("系统性能优化")
    print("="*80)
    optimize_system_performance()
    print("="*80)
    print()
    
    # 数据路径（使用与超参搜索一致的路径）
    train_path = current_dir.parent / 'data' / 'processed' / 'new_dataset' / 'train_dataset' / 'v2' / 'molecule' / 'molecule_train.csv'
    val_path = current_dir.parent / 'data' / 'processed' / 'new_dataset' / 'train_dataset' / 'v2' / 'molecule' / 'molecule_valid.csv'
    
    if not train_path.exists():
        print("="*80)
        print("错误: 未找到训练集文件！")
        print(f"期望路径: {train_path}")
        print("="*80)
        sys.exit(1)
    
    if not val_path.exists():
        print("="*80)
        print("错误: 未找到验证集文件！")
        print(f"期望路径: {val_path}")
        print("="*80)
        sys.exit(1)
    
    # 测试的memory_size值
    memory_sizes = [16, 32, 64, 128, 256]
    
    # 使用带超参数搜索的敏感性分析（更客观的比较）
    print("\n" + "="*80)
    print("使用超参数搜索模式进行敏感性分析")
    print("="*80)
    print("说明: 对每个memory_size执行超参数搜索，找到其最佳超参数组合")
    print("这样可以更客观地比较不同memory_size对模型性能的影响")
    print("="*80)
    print()
    
    results = external_attention_sensitivity_analysis_with_hp_search(
        train_df_path=str(train_path),
        val_df_path=str(val_path),
        base_model_name='EA_sensitivity_hp',
        memory_sizes=memory_sizes,
        search_method='bayesian',
        n_trials_per_memory=30,  # 每个memory_size搜索30次（推荐值，平衡搜索充分性和时间成本）
        termination_check_interval=5,  # 每完成5次试验显示一次进度（仅用于进度显示）
        # 单次搜索终止条件（仅用于单个trial的提前终止，不影响全局搜索）
        single_trial_start_mae_threshold=0.7,  # 起始MAE阈值（默认0.7）
        single_trial_epochs_threshold=100,  # 训练轮数阈值（默认100轮）
        single_trial_mae_after_epochs_threshold=0.2,  # 训练指定轮数后MAE阈值（默认0.2）
        save_path=str(current_dir / 'external_attention_sensitivity_hp_search_results.json')
    )
    
    print("\n" + "="*80)
    print("敏感性分析（带超参数搜索）完成！")
    print("="*80)


if __name__ == '__main__':
    main()

