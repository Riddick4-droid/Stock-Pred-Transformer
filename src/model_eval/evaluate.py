"""
Model Evaluation Script
Loads a trained checkpoint (pre‑trained or fine‑tuned) and evaluates it
on a test set, computing RMSE, MAE, MAPE (on actual prices), and directional accuracy.
For fine‑tuning mode the test set is the last 10% of the single stock’s data.
For pre‑training mode we evaluate on the validation split directly on log‑returns.
Results are logged and optionally saved to a JSON file.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import json

from src.logger import logger
from src.utils import load_yaml, get_device, plot_predictions_vs_true
from src.exceptions import StockTransformerException
from src.models.stock_transformer import StockTransformer
from src.data.dataset import load_scaler, FineTuneDataset
from src.compute_metrics.metrics import rmse, mae, mape, directional_accuracy


def evaluate(config_path: str = "configs/config.yaml",mode: str = "fine_tuning",checkpoint_path: Optional[str] = None,
    output_dir: str = "results"
) -> Dict[str, float]:
    """
    Run evaluation on test set.
    Parameters
    config_path : str
        Path to config.yaml.
    mode : str
        "pre_training" or "fine_tuning".
    checkpoint_path : str, optional
        If None, uses the checkpoint defined in config.
    output_dir : str
        Directory to save metrics JSON and optional plots.
    Returns
    dict with keys: rmse, mae, mape, direction_accuracy
    """
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    device = get_device()

    # Determine checkpoint
    if checkpoint_path is None:
        ckpt_dir = Path(data_cfg["checkpoints_dir"])
        if mode == "fine_tuning":
            ckpt_name = cfg["fine_tuning"]["checkpoint_name"]
        else:
            ckpt_name = cfg["pre_training"]["checkpoint_name"]
        checkpoint_path = ckpt_dir / ckpt_name
        if not checkpoint_path.exists():
            checkpoint_path = ckpt_dir / "best.pth"
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_features = ckpt["num_features"]
    model = StockTransformer(
        num_features=num_features,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scaler and feature columns
    scaler_path = "checkpoints/scaler.joblib"
    scaler, feature_cols = load_scaler(str(scaler_path))

    processed_dir = Path(data_cfg["processed_dir"])
    seq_len = ds_cfg["seq_len"]

    if mode == "fine_tuning":
        ticker = ds_cfg["fine_tune_ticker"]
        ticker_path = processed_dir / f"{ticker}.parquet"
        if not ticker_path.exists():
            raise StockTransformerException(f"Ticker file not found: {ticker_path}")
        df = pd.read_parquet(ticker_path)
        split_idx = int(len(df) * 0.9)
        test_df = df.iloc[split_idx:]

        close_raw = test_df["Close"].values
        test_dataset = FineTuneDataset(
            test_df, seq_len=seq_len, feature_cols=feature_cols, scaler=scaler,
            target_price_col="Target_Log_Ret_Next",
            target_dir_col="Target_Direction",
            target_vol_col="Target_Vol_Next"
        )
        pred_prices = []
        true_prices = []
        all_dir_probs = []
        true_dirs = []

        with torch.no_grad():
            for i in range(len(test_dataset)):
                x, _, dir_target, _ = test_dataset[i]
                x = x.unsqueeze(0).to(device)
                mu, _, dir_logits, _ = model(x)
                last_close = close_raw[i + seq_len - 1]
                true_close_next = close_raw[i + seq_len]
                pred_price = last_close * np.exp(mu.item())
                pred_prices.append(pred_price)
                true_prices.append(true_close_next)
                all_dir_probs.append(torch.softmax(dir_logits, dim=-1).cpu().numpy().flatten())
                true_dirs.append(dir_target.item())

        pred_prices = np.array(pred_prices)
        true_prices = np.array(true_prices)
        all_dir_probs = np.array(all_dir_probs)
        true_dirs = np.array(true_dirs)

        metrics = {
            "rmse": rmse(true_prices, pred_prices),
            "mae": mae(true_prices, pred_prices),
            "mape": mape(true_prices, pred_prices),
            "directional_accuracy": directional_accuracy(true_dirs, all_dir_probs)
        }
        # Plot
        plot_dates = test_df.index[seq_len:]
        plot_predictions_vs_true(
            plot_dates, true_prices, pred_prices,
            title=f"{ticker} – Evaluation Predictions",
            save_path=Path(output_dir) / f"{ticker}_eval.png"
        )

    else:  # pre_training mode
        from src.data.dataset import create_datasets
        _, val_dataset, _, _ = create_datasets(config_path, mode="pretrain")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        all_preds_logret = []
        all_targets_logret = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                mu, _, _, _ = model(x)
                all_preds_logret.append(mu.cpu().numpy())
                all_targets_logret.append(y.cpu().numpy())
        pred_log_returns = np.concatenate(all_preds_logret)
        true_log_returns = np.concatenate(all_targets_logret)
        metrics = {
            "rmse_logret": rmse(true_log_returns, pred_log_returns),
            "mae_logret": mae(true_log_returns, pred_log_returns),
            "mape_logret": mape(true_log_returns, pred_log_returns),
            "directional_accuracy": "N/A (no direction head in pretrain)"
        }

    # Save metrics
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(output_dir) / f"metrics_{mode}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Evaluation results: {metrics}")
    return metrics


if __name__ == "__main__":
    evaluate(mode="fine_tuning")