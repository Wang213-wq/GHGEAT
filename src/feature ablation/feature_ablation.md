## Feature Ablation Guide

### 1. Overview

Feature ablation is a systematic procedure to quantify how much each input feature (or group of features) contributes to model performance.  
The core idea is simple: **select one or more features, mask or remove them, re‑evaluate the model, and measure how much the error changes**.

In this project, feature ablation is mainly implemented for the `GHGEAT` model via the `AblationStudy` class defined in `ablation_results/ablation_study.py`.  
It operates on four high‑level features:
- `ap`
- `bp`
- `topopsa`
- `intra_hb`

### 2. Types of Feature Ablation

- **Single‑feature ablation**  
  Remove one feature at a time (e.g. remove `ap` only) and compare the performance with the full model.  
  - Large performance drop ⇒ that feature is important  
  - Small performance change ⇒ that feature is less important

- **Progressive ablation**  
  Remove features step by step (e.g. first remove `ap`, then remove `ap+bp`, then `ap+bp+topopsa`, …) and observe the cumulative effect.  
  This reveals how performance degrades as more information is removed.

- **Feature‑combination ablation**  
  Keep only specific feature subsets (e.g. only `ap`, or `ap+bp`, etc.) and compare performance.  
  This helps identify **useful combinations** and **synergies** between features.

### 3. How It Works in Code

The main implementation is in `ablation_results/ablation_study.py`:

- **`AblationStudy` class**
  - Holds a trained `GHGEAT` model and a list of feature names.
  - Provides methods:
    - `single_feature_ablation(data_loader)`: one‑feature‑removed experiments.
    - `progressive_ablation(data_loader)`: stepwise feature removal.
    - `feature_combination_ablation(data_loader)`: keep specific feature subsets.
    - `comprehensive_ablation(data_loader, save_dir)`: runs all above and saves results and plots.

- **`mask_feature(solvent, solute, feature_name)`**
  - Creates deep copies of solvent/solute objects.
  - Sets the specified feature to zero (or another masking strategy if customized).
  - Returns masked solvent/solute to be used in a forward pass.

- **Metrics**
  - MAE
  - RMSE
  - MSE
  - R² and adjusted R² (with some numerical‑stability protection)

### 4. Running Feature Ablation

#### 4.1 Quick usage via helper function

```python
from ablation_results.ablation_study import run_ablation_study
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = run_ablation_study(
    model_path="path/to/your_trained_GHGEAT_model.pth",
    data_loader=your_data_loader,      # e.g. built with get_dataloader_pairs_T
    device=device
)
```

This will:
- Load the `GHGEAT` model from `model_path`.
- Run comprehensive ablation (single‑feature, progressive, combination).
- Save CSV, JSON and plots into the target directory (by default under `ablation_results/`).

#### 4.2 Fine‑grained control

```python
from ablation_results.ablation_study import AblationStudy
from scr.models.GH_pyGEAT_architecture_0615_v0 import GHGEAT
from scr.models.utilities_v2.mol2graph import (
    get_dataloader_pairs_T,
    n_atom_features,
    n_bond_features,
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build and load model
v_in = n_atom_features()
e_in = n_bond_features()
u_in = 3
hidden_dim = 64

model = GHGEAT(v_in, e_in, u_in, hidden_dim)
state = torch.load("path/to/your_trained_GHGEAT_model.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# 2. Build data loader
data_loader = get_dataloader_pairs_T(
    # fill in your dataset arguments here
)

# 3. Create ablation object
ablation = AblationStudy(model, device)

# 4. Run specific experiments
single_df = ablation.single_feature_ablation(data_loader)
progressive_df = ablation.progressive_ablation(data_loader)
combination_df = ablation.feature_combination_ablation(data_loader)

# 5. Save / visualize
ablation.save_results(single_df, save_dir="ablation_results", experiment_name="single_feature_ablation")
ablation.visualize_results(single_df, save_path="ablation_results/single_feature_ablation.png")
```

### 5. Output Files and Structure

By default, results are written under `ablation_results/`, for example:

```text
ablation_results/
├── single_feature_ablation_results.csv
├── single_feature_ablation_summary.json
├── single_feature_ablation.png
├── progressive_ablation_results.csv
├── progressive_ablation_summary.json
├── progressive_ablation.png
├── feature_combination_ablation_results.csv
├── feature_combination_ablation_summary.json
└── feature_combination_ablation.png
```

- **`*_results.csv`**: detailed metrics for each ablation configuration.  
- **`*_summary.json`**: compact summary (best/worst MAE, RMSE, etc.).  
- **`*.png`**: bar charts and heatmaps for quick visual comparison.

### 6. Interpreting Results

- **MAE / RMSE / MSE**
  - Higher values mean worse performance.
  - Compare each ablation configuration with the **baseline (full model)**.

- **Performance degradation**
  - Often represented as `MSE_delta = MSE_ablation − MSE_baseline` (or analogous for MAE).
  - Large positive `MSE_delta` ⇒ the removed feature(s) are important.

- **R² / adjusted R²**
  - Close to 1 ⇒ predictions closely match targets.
  - When many features are removed, R² often decreases; adjusted R² accounts for the effective number of features.

Typical analysis steps:
- Identify features whose removal causes the largest degradation ⇒ **key features**.
- Identify features whose removal has minimal effect ⇒ candidates for **pruning** or simplification.
- Use feature combinations to detect interactions or redundancy among features.

### 7. Customizing Feature Ablation

- **Masking strategy**
  - Default: set feature tensors to zero (`torch.zeros_like`).
  - Alternative: use random noise:
    ```python
    masked_solvent.ap = torch.randn_like(solvent.ap).to(self.device)
    ```
  - Alternative: partially scale features instead of fully removing:
    ```python
    masked_solvent.ap = solvent.ap * 0.1  # keep only 10% of the signal
    ```

- **Ablation scope**
  - You can choose to mask:
    - Only solvent features,
    - Only solute features,
    - Or both (default implementation does both).
  - This requires small modifications inside `mask_feature`.

- **Sampling strategy**
  - To reduce computation, you may restrict ablation to a subset of the training/validation data (e.g. 300–1000 samples).
  - For large models and long training sets, consider:
    - Smaller batch size.
    - Only running single‑feature and progressive ablation (skip full combination search).

### 8. Practical Tips and Troubleshooting

- **Memory issues**
  - Reduce batch size in the data loader.
  - Use fewer samples for the ablation study.

- **Computation time too long**
  - Start only with single‑feature ablation.
  - Run progressive ablation next if needed.
  - Treat feature‑combination ablation as optional because it is combinatorial and may be expensive.

- **Unexpected model behavior**
  - Ensure the model is in evaluation mode: `model.eval()`.
  - Make sure all data tensors are moved to the correct device (`cuda` or `cpu`).
  - Confirm that feature names used in ablation (`ap`, `bp`, `topopsa`, `intra_hb`) match the model’s actual attributes.

### 9. When to Use Feature Ablation

Feature ablation is particularly useful when you want to:
- Understand which features drive most of the predictive power.
- Justify model design choices (e.g. why include certain descriptors).
- Simplify the model by removing weak or redundant features.
- Design more efficient models for deployment (fewer features, lower cost).

By applying feature ablation systematically, you obtain a **quantitative, experiment‑based view** of feature importance that goes beyond simple correlation or heuristic intuition.


