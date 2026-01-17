from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader

try:  # pragma: no cover
    from .data_utils import (  # type: ignore
        NCFDataset,
        compute_temperature_stats,
        encode_components,
        load_dataset,
        split_by_combination,
    )
    from .model import NCFConfig, NCFModel
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    from data_utils import (  # type: ignore  # noqa: E402
        NCFDataset,
        compute_temperature_stats,
        encode_components,
        load_dataset,
        split_by_combination,
    )
    from model import NCFConfig, NCFModel  # type: ignore  # noqa: E402

DEFAULT_SEED = 42
BEST_HYPERPARAMS: Dict[str, Any] = {
    "embedding_dim": 128,
    "repr_layer_sizes": [128, 128],
    "cf_layer_sizes": [256, 256, 256],
    "dropout": 0.0,
    "activation": "relu",
    "learning_rate": 0.001,
    "batch_size": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural collaborative filtering model for γ∞ prediction.")
    parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"), help="Path to dataset CSV.")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=BEST_HYPERPARAMS["batch_size"], help="Training batch size.")
    parser.add_argument("--lr", type=float, default=BEST_HYPERPARAMS["learning_rate"], help="Learning rate for Adam optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (in epochs).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to store checkpoints.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training using existing checkpoints in the output directory.",
    )
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_dataloaders(
    train_frame,
    val_frame,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Tuple[float, float]]:
    temp_mean, temp_std = compute_temperature_stats(train_frame)
    train_dataset = NCFDataset(train_frame, temp_mean, temp_std)
    val_dataset = NCFDataset(val_frame, temp_mean, temp_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, (temp_mean, temp_std)


def train_fold(
    fold_idx: int,
    model: NCFModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        improved = val_metrics["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        print(
            f"[Fold {fold_idx:02d}] Epoch {epoch:03d} | "
            f"Train MAE: {train_metrics['mae']:.4f} | Train R2: {train_metrics['r2']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | Val R2: {val_metrics['r2']:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"[Fold {fold_idx:02d}] Early stopping triggered after {epoch} epochs.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=False)
    final_val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    return {
        "train_mse": final_train_metrics["mse"],
        "train_mae": final_train_metrics["mae"],
        "train_r2": final_train_metrics["r2"],
        "val_mse": final_val_metrics["mse"],
        "val_mae": final_val_metrics["mae"],
        "val_r2": final_val_metrics["r2"],
    }


def run_epoch(
    model: NCFModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    residual_sum_squares = 0.0
    target_sum = 0.0
    target_sum_squares = 0.0
    total_samples = 0

    for solute_ids, solvent_ids, temperatures, targets in dataloader:
        solute_ids = solute_ids.to(device)
        solvent_ids = solvent_ids.to(device)
        temperatures = temperatures.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            predictions = model(solute_ids, solvent_ids, temperatures)
            loss = criterion(predictions, targets)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        differences = predictions - targets
        total_abs_error += torch.sum(torch.abs(differences)).item()
        residual_sum_squares += torch.sum(differences.pow(2)).item()
        target_sum += torch.sum(targets).item()
        target_sum_squares += torch.sum(targets.pow(2)).item()
        total_samples += batch_size

    mse = total_loss / total_samples
    mae = total_abs_error / total_samples
    target_mean = target_sum / total_samples
    ss_tot = target_sum_squares - total_samples * target_mean * target_mean
    if ss_tot <= 1e-12:
        r2 = 0.0
    else:
        r2 = 1.0 - (residual_sum_squares / ss_tot)
    return {"mse": mse, "mae": mae, "r2": r2}


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]], total_training_time: float) -> Dict[str, Any]:
    def _mean_std(key: str) -> Tuple[float, float]:
        values = [m[key] for m in fold_metrics]
        return float(np.mean(values)), float(np.std(values))

    computed_total = 0.0
    for metrics in fold_metrics:
        try:
            computed_total += float(metrics.get("duration_seconds", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue
    if computed_total > 0.0 and total_training_time <= 0.0:
        total_training_time = computed_total

    train_mse_mean, train_mse_std = _mean_std("train_mse")
    train_mae_mean, train_mae_std = _mean_std("train_mae")
    train_r2_mean, train_r2_std = _mean_std("train_r2")
    val_mse_mean, val_mse_std = _mean_std("val_mse")
    val_mae_mean, val_mae_std = _mean_std("val_mae")
    val_r2_mean, val_r2_std = _mean_std("val_r2")

    return {
        "train_mse_mean": train_mse_mean,
        "train_mse_std": train_mse_std,
        "train_mae_mean": train_mae_mean,
        "train_mae_std": train_mae_std,
        "train_r2_mean": train_r2_mean,
        "train_r2_std": train_r2_std,
        "val_mse_mean": val_mse_mean,
        "val_mse_std": val_mse_std,
        "val_mae_mean": val_mae_mean,
        "val_mae_std": val_mae_std,
        "val_r2_mean": val_r2_mean,
        "val_r2_std": val_r2_std,
        "fold_metrics": fold_metrics,
        "total_training_time_seconds": total_training_time,
    }


def run_cross_validation(
    config: NCFConfig,
    encoded,
    combination_keys,
    unique_combinations,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    output_dir: Path | None,
    save_checkpoints: bool,
    metrics_output_path: Path | None,
    resume: bool = False,
) -> Dict[str, Any]:
    if save_checkpoints and output_dir is None:
        raise ValueError("output_dir must be provided when save_checkpoints is True.")

    if save_checkpoints and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []
    total_training_time = 0.0

    for fold_number, (train_indices, val_indices) in enumerate(kfold.split(unique_combinations), start=1):
        checkpoint_path: Optional[Path] = None
        if save_checkpoints and output_dir is not None:
            checkpoint_path = output_dir / f"fold_{fold_number:02d}.pt"

        if resume and checkpoint_path is not None and checkpoint_path.exists():
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            except Exception as error:  # pragma: no cover
                print(
                    f"[Fold {fold_number:02d}] Failed to load checkpoint {checkpoint_path}: {error}. "
                    "Re-training this fold."
                )
            else:
                stored_metrics = checkpoint_data.get("metrics")
                if isinstance(stored_metrics, dict):
                    metrics_copy = dict(stored_metrics)
                    duration = metrics_copy.get("duration_seconds")
                    if duration is None:
                        duration = checkpoint_data.get("duration_seconds", 0.0)
                        try:
                            duration = float(duration)
                        except (TypeError, ValueError):
                            duration = 0.0
                        metrics_copy["duration_seconds"] = duration
                    try:
                        total_training_time += float(metrics_copy.get("duration_seconds", 0.0))
                    except (TypeError, ValueError):
                        pass
                    fold_metrics.append(metrics_copy)
                    print(
                        f"[Fold {fold_number:02d}] Loaded existing checkpoint and metrics; skipping retraining."
                    )
                    continue
                else:
                    print(
                        f"[Fold {fold_number:02d}] Existing checkpoint lacks metrics; re-training this fold."
                    )

        fold_start = time.time()
        train_combos = set(unique_combinations[train_indices])
        val_combos = set(unique_combinations[val_indices])

        train_mask = combination_keys.isin(train_combos)
        val_mask = combination_keys.isin(val_combos)

        train_frame = encoded.frame[train_mask].reset_index(drop=True)
        val_frame = encoded.frame[val_mask].reset_index(drop=True)

        train_loader, val_loader, (temp_mean, temp_std) = create_dataloaders(
            train_frame, val_frame, batch_size, num_workers
        )

        model = NCFModel(config).to(device)

        metrics = train_fold(
            fold_number,
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
        )
        fold_metrics.append(metrics)

        fold_duration = time.time() - fold_start
        metrics["duration_seconds"] = fold_duration

        if save_checkpoints and output_dir is not None:
            checkpoint_path = output_dir / f"fold_{fold_number:02d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "temp_mean": temp_mean,
                    "temp_std": temp_std,
                    "metrics": metrics,
                    "duration_seconds": fold_duration,
                    "training_params": {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "patience": patience,
                    },
                },
                checkpoint_path,
            )
            print(f"[Fold {fold_number:02d}] Saved checkpoint to {checkpoint_path}")
        print(f"[Fold {fold_number:02d}] Duration: {fold_duration:.2f} seconds")

        total_training_time += fold_duration

    aggregate = aggregate_fold_metrics(fold_metrics, total_training_time)
    aggregate.update(
        {
            "config": asdict(config),
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "total_training_time_seconds": total_training_time,
        }
    )

    if metrics_output_path is not None:
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output_path.open("w", encoding="utf-8") as fp:
            json.dump(aggregate, fp, indent=2)
        print(f"Stored aggregate metrics at {metrics_output_path}")
        print(f"Total training time: {total_training_time:.2f} seconds")

    return aggregate


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    raw_df = load_dataset(args.data)
    encoded = encode_components(raw_df)
    combination_keys = split_by_combination(encoded.frame)
    unique_combinations = combination_keys.unique()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = NCFConfig(
        num_solutes=len(encoded.solute_vocab),
        num_solvents=len(encoded.solvent_vocab),
        embedding_dim=BEST_HYPERPARAMS["embedding_dim"],
        representation_hidden_sizes=BEST_HYPERPARAMS["repr_layer_sizes"],
        collaborative_hidden_sizes=BEST_HYPERPARAMS["cf_layer_sizes"],
        dropout=BEST_HYPERPARAMS["dropout"],
        activation=BEST_HYPERPARAMS["activation"],
    )
    batch_size = args.batch_size
    lr = args.lr

    print(
        "Using fixed hyperparameters: "
        f"embedding_dim={config.embedding_dim}, "
        f"repr_layers={config.representation_hidden_sizes}, "
        f"cf_layers={config.collaborative_hidden_sizes}, "
        f"dropout={config.dropout}, activation={config.activation}. "
        f"Batch size={batch_size}, learning rate={lr}."
    )

    metrics_output_path = args.output_dir / "metrics.json"
    run_cross_validation(
        config=config,
        encoded=encoded,
        combination_keys=combination_keys,
        unique_combinations=unique_combinations,
        device=device,
        batch_size=batch_size,
        num_workers=args.num_workers,
        lr=lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir,
        save_checkpoints=True,
        metrics_output_path=metrics_output_path,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()


