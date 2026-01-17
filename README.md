# GHGEAT: Graph Neural Network for Activity Coefficient Prediction

## Overview

**GHGEAT** (Graph Neural Network - Gibbs-Helmholtz with External Attention and Temperature dependency) is a deep learning model designed to predict activity coefficients (log-γ) for binary solvent-solute systems across different temperatures. The model combines graph neural networks with molecular descriptors and temperature-dependent physics to achieve accurate predictions.

## Related Work

This work is built upon the foundation established by the **Gibbs-Helmholtz Graph Neural Network (GH-GNN)** model proposed in the following paper:

**Sanchez Medina, E. I., Linke, S., Stoll, M., & Sundmacher, K.** (2023). Gibbs-Helmholtz graph neural network: capturing the temperature dependency of activity coefficients at infinite dilution. *Digital Discovery*, 2, 781-798. DOI: [10.1039/d2dd00142j](https://doi.org/10.1039/d2dd00142j)

The original GH-GNN model combines the simplicity of a Gibbs-Helmholtz-derived expression with graph neural networks that incorporate explicit molecular and intermolecular descriptors for capturing dispersion and hydrogen bonding effects. Our GHGEAT model extends this work by introducing an external attention mechanism to enhance feature extraction and improve prediction accuracy.

**Key differences and improvements in GHGEAT:**
- **External Attention Layer**: Incorporates external memory matrices (Mk, Mv) for enhanced feature learning
- **Configurable Attention Weight**: Allows ablation studies to evaluate the contribution of attention mechanisms
- **Enhanced Architecture**: Improved graph neural network architecture with better numerical stability

## Key Features

- **Graph-based molecular representation**: Processes molecular structures using graph neural networks
- **External attention mechanism**: Enhanced attention mechanism for better feature extraction
- **Temperature-dependent prediction**: Incorporates temperature as a key input variable
- **Molecular descriptors integration**: Uses MOSCED-based descriptors (ap, bp, topopsa, intra_hb)
- **Robust training pipeline**: Supports checkpointing, early stopping, and various learning rate schedulers

## Model Architecture

### Core Components

1. **Dual Graph Networks**: Separate graph neural networks for solvent and solute molecules
   - Two-layer MetaLayer architecture with GraphNorm normalization
   - Global mean pooling for molecular-level representations

2. **External Attention Layer**: 
   - External memory matrices (Mk, Mv) for enhanced feature learning
   - Configurable attention weight (0.0-1.0) for ablation studies

3. **Molecular Descriptors**:
   - **ap**: Atomic polarizability
   - **bp**: Bond polarizability  
   - **topopsa**: Topological polar surface area
   - **intra_hb**: Intramolecular hydrogen bonding

4. **Binary System Graph**: 
   - Constructs a system-level graph connecting solvent and solute
   - MPNN convolution for system-level feature extraction

5. **Temperature-Dependent Output**:
   - Predicts parameters A and B
   - Final output: `log-γ = A + B/T` (T in Kelvin)

### Architecture Diagram

```
Input: Solvent Graph, Solute Graph, Temperature
  ↓
[GraphNet1] → [GraphNorm] → [GraphNet2] → [GraphNorm] → [Global Pooling]
  ↓                                                          ↓
Solvent Features                                        Solute Features
  ↓                                                          ↓
[Concatenate with Molecular Descriptors (ap, bp, topopsa)]
  ↓
[Binary System Graph Construction]
  ↓
[MPNN Convolution]
  ↓
[MLP for A] + [MLP for B]
  ↓
Output: log-γ = A + B/T
```

## Installation

### Requirements

```bash
torch>=1.9.0
torch-geometric>=2.0.0
rdkit>=2020.09.1
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
```

### Setup

1. Clone the repository or ensure the project structure is in place:
```
project_root/
├── scr/
│   └── models/
│       ├── GH_pyGEAT_architecture_0615_v0.py
│       ├── GHGEAT_train.py
│       ├── GHGEAT_pred.py
│       └── utilities_v2/
└── data/
```

2. Install dependencies:
```bash
pip install torch torch-geometric rdkit scikit-learn pandas numpy
```

## Data Format

### Required CSV Columns

The training/prediction data should be a CSV file with the following columns:

- **`Solvent_SMILES`**: SMILES string of the solvent molecule
- **`Solute_SMILES`**: SMILES string of the solute molecule  
- **`T`**: Temperature in Celsius (°C)
- **`log-gamma`**: Target activity coefficient (log-γ) value

### Example Data Format

```csv
Solvent_SMILES,Solute_SMILES,T,log-gamma
CCO,CC(=O)O,25,0.123
CCO,CC(=O)O,50,0.145
CC(C)O,CC(=O)O,25,0.156
```

## Usage

### Training

#### Basic Training

```python
import pandas as pd
from scr.models.GHGEAT_train import train_GNNGH_T

# Load training data
df = pd.read_csv('path/to/train_data.csv')

# Define hyperparameters
hyperparameters = {
    'hidden_dim': 38,
    'lr': 0.0008,
    'n_epochs': 434,
    'batch_size': 104,
    'early_stopping_patience': 34,
    'attention_weight': 0.8  # External attention weight (0.0-1.0)
}

# Train model
train_GNNGH_T(
    df=df,
    model_name='GHGEAT_model',
    hyperparameters=hyperparameters
)
```

#### Training with Validation Set

```python
# Load training and validation data
df_train = pd.read_csv('path/to/train_data.csv')
df_val = pd.read_csv('path/to/val_data.csv')

# Train with validation set
train_GNNGH_T(
    df=df_train,
    model_name='GHGEAT_model',
    hyperparameters=hyperparameters,
    val_df=df_val
)
```

#### Resume Training from Checkpoint

```python
train_GNNGH_T(
    df=df_train,
    model_name='GHGEAT_model',
    hyperparameters=hyperparameters,
    resume_checkpoint='path/to/checkpoint.pth'
)
```

#### Training with Brouwer Data Mixing

```python
train_GNNGH_T(
    df=df_train,
    model_name='GHGEAT_model',
    hyperparameters=hyperparameters,
    mix_brouwer_ratio=0.15  # Mix 15% of Brouwer_2021 data
)
```

### Prediction

```python
import pandas as pd
from scr.models.GHGEAT_pred import pred_GNNGH_T

# Load data for prediction
df = pd.read_csv('path/to/prediction_data.csv')

# Define hyperparameters (must match training configuration)
hyperparameters = {
    'hidden_dim': 38,
    'attention_weight': 0.8
}

# Make predictions
df_pred = pred_GNNGH_T(
    df=df,
    model_name='GHGEAT',
    hyperparameters=hyperparameters
)

# Predictions are added as a new column 'GHGEAT'
print(df_pred[['log-gamma', 'GHGEAT']].head())
```

## Hyperparameters

### Key Hyperparameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `hidden_dim` | Hidden dimension size | 32-64 | 38 |
| `lr` | Learning rate | 1e-5 to 1e-3 | 0.0008 |
| `batch_size` | Batch size | 32-128 | 104 |
| `n_epochs` | Number of training epochs | 200-500 | 434 |
| `attention_weight` | External attention weight | 0.0-1.0 | 1.0 |
| `early_stopping_patience` | Early stopping patience | 10-50 | 34 |

### Learning Rate Schedulers

The model supports multiple learning rate scheduling strategies:

1. **ReduceLROnPlateau** (default): Reduces LR when validation loss plateaus
2. **CosineAnnealingWarmRestarts**: Cosine annealing with warm restarts
3. **CyclicLR**: Cyclic learning rate
4. **Cosine with Warmup**: Cosine annealing with warmup phase

Example with custom scheduler:

```python
hyperparameters = {
    'hidden_dim': 38,
    'lr': 0.0008,
    'n_epochs': 434,
    'batch_size': 104,
    'use_cosine_warm_restarts': True,
    'cosine_restart_params': {
        'T_0': 20,
        'T_mult': 2,
        'eta_min': 1e-6
    }
}
```

## Model Files Structure

After training, the following files are generated:

```
{model_name}/
├── checkpoint/
│   ├── {model_name}_checkpoint.pth          # Latest checkpoint
│   └── {model_name}_checkpoint_epoch{}.pth  # Epoch-specific checkpoints
├── {model_name}.pth                         # Final model weights
├── Report_training_{model_name}.txt        # Training log
└── Training_files/                          # Additional training artifacts
```

## Evaluation Metrics

The model reports the following metrics during training and evaluation:

- **MAE** (Mean Absolute Error): Primary evaluation metric
- **RMSE** (Root Mean Squared Error): Secondary metric
- **R²** (Coefficient of Determination): Goodness of fit
- **MSE** (Mean Squared Error): Loss metric

## Feature Ablation Study

The model supports feature ablation studies to evaluate the importance of different molecular descriptors:

```python
from ablation_results.ablation_study import AblationStudy

# Initialize ablation study
ablation = AblationStudy(model, device)

# Run single feature ablation
single_feature_df = ablation.single_feature_ablation(data_loader)

# Run progressive ablation
progressive_df = ablation.progressive_ablation(data_loader)

# Run feature combination ablation
combination_df = ablation.feature_combination_ablation(data_loader)
```

See `feature_ablation/feature_ablation.md` for detailed documentation.

## Numerical Stability

The model includes several numerical stability measures:

- **Clamping**: Output values are clamped to prevent overflow/underflow
- **Temperature protection**: Minimum temperature threshold (10K) to prevent division by zero
- **B/T protection**: B parameter and B/T ratio are clamped to reasonable ranges
- **NaN/Inf detection**: Automatic detection and handling of invalid values

## Performance Tips

1. **GPU Usage**: The model automatically uses CUDA if available. Ensure sufficient GPU memory for large batch sizes.

2. **Batch Size**: Larger batch sizes (64-128) generally improve training stability but require more memory.

3. **Learning Rate**: Start with the default learning rate and adjust based on training dynamics. Use learning rate scheduling for better convergence.

4. **Early Stopping**: Enable early stopping to prevent overfitting. Typical patience values are 20-50 epochs.

5. **Checkpointing**: Regular checkpointing allows resuming training and prevents loss of progress.

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training (if supported)

2. **NaN/Inf Values**
   - Check input data for invalid values
   - Reduce learning rate
   - Enable gradient clipping

3. **Poor Convergence**
   - Adjust learning rate
   - Check data quality and distribution
   - Verify hyperparameters match training configuration

4. **Slow Training**
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Reduce number of epochs or enable early stopping

## Citation

If you use GHGEAT in your research, please cite:

```
@article{ghgeat2024,
  title={GHGEAT: Graph Neural Network for Activity Coefficient Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

**Please also cite the original GH-GNN paper:**

```
@article{sanchez2023ghgnn,
  title={Gibbs-Helmholtz graph neural network: capturing the temperature dependency of activity coefficients at infinite dilution},
  author={Sanchez Medina, Edgar Ivan and Linke, Steffen and Stoll, Martin and Sundmacher, Kai},
  journal={Digital Discovery},
  volume={2},
  pages={781--798},
  year={2023},
  publisher={Royal Society of Chemistry},
  doi={10.1039/d2dd00142j}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact:
- Email: [your-email@example.com]
- GitHub Issues: [repository URL]

## Acknowledgments

- This work is based on the **Gibbs-Helmholtz Graph Neural Network (GH-GNN)** model developed by Edgar Ivan Sanchez Medina, Steffen Linke, Martin Stoll, and Kai Sundmacher (Digital Discovery, 2023, 2, 781-798)
- Uses PyTorch Geometric for graph neural network operations
- RDKit for molecular processing

---

**Last Updated**: 2024

