"""
Inference Module

Performs single‑step and multi‑step autoregressive prediction using a trained
StockTransformer checkpoint.

Key features:
- Loads the scaler + feature columns from the saved scaler file, guaranteeing
  the exact same feature set the model was trained on.
- Converts predicted log‑returns back to prices using the last known close.
- Supports multi‑step autoregressive generation where the model’s own
  predictions are fed back as input for subsequent steps.
- Outputs price predictions, direction probabilities, and volatility estimates.

All settings come from config.yaml.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

from src.logger import logger
from src.utils import load_yaml, get_device, plot_predictions_vs_true, plot_multistep_forecast
from src.exceptions import StockTransformerException
from src.models.stock_transformer import StockTransformer
from src.data.dataset import load_scaler
from src.compute_metrics.metrics import rmse, mae, mape, directional_accuracy

class InferenceEngine:
    """
    Handles model loading, data preparation, and prediction.
    Parameters
    config_path : str
        Path to config.yaml.
    checkpoint_path : str, optional
        Path to a specific checkpoint. If None, uses the one defined in config.
    """
    def __init__(self, config_path: str="configs/config.yaml", checkpoint_path: Optional[str]=None):
        self.cfg = load_yaml(config_path)
        self.device = get_device()

        #determine checkpoint
        if checkpoint_path is None:
            ckpt_dir = Path(self.cfg["data"]["checkpoints_dir"])
            if not os.path.exists(ckpt_dir):
                raise StockTransformerException(f"Checkpoint directory missing: {ckpt_dir.name}")
            ckpt_name = self.cfg["fine_tuning"]["checkpoint_name"]
            checkpoint_path = ckpt_dir / ckpt_name

            if not checkpoint_path.exists():
                checkpoint_path = ckpt_dir / "best.pth"
        self.checkpoint_path = Path(checkpoint_path)

        #load checkpoint (weights_only=False)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        #model architecture
        self.num_features = ckpt["num_features"]
        model_cfg = ckpt["config"]
        self.model = StockTransformer(
            num_features=self.num_features,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg.get("d_ff", 2048),
            max_seq_len=model_cfg.get("max_seq_len", 252),
            dropout=model_cfg.get("dropout", 0.1)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # loading scaler and the exact feature columns used during training
        scaler_path = "checkpoints/scaler.joblib"
        self.scaler, self.feature_cols = load_scaler(scaler_path)

        #setting up a variable for the processed dir
        self.seq_len = self.cfg["dataset"]["seq_len"]
        self.processed_dir = Path(self.cfg["data"]["processed_dir"])
        logger.info(f"Inference engine ready. Features: {len(self.feature_cols)}")

    def _load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """load the processed DataFrame for a single ticker."""
        path = self.processed_dir / f"{ticker}.parquet"
        if not path.exists():
            raise StockTransformerException(f"Ticker file not found: {path}")
        return pd.read_parquet(path)
    
    def _prepare_input(self, features_df: pd.DataFrame) -> torch.Tensor:
        """scale a feature window and convert to a batch of 1."""
        # Ensure only the trained feature columns are used, in the correct order
        X = features_df[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        return torch.from_numpy(X_scaled).unsqueeze(0).to(self.device)   # (1, T, F)
    
    def single_step_predict(self, ticker: str, recent_window: pd.DataFrame) -> Dict:
        """
        Predict the next day’s price, direction, and volatility.
        Parameters
        ticker : str
            Stock ticker (e.g., "AAPL.US").
        recent_window : pd.DataFrame
            DataFrame containing at least the last `seq_len` days of features.
            Must include a 'Close' column for price reconstruction.
        Returns
        dict with keys:
            - next_price (float)
            - price_uncertainty (float)   # sigma in price units
            - direction (str)            # "Down", "Flat", or "Up"
            - direction_probs (dict)     # probabilities for each class
            - volatility (float)         # predicted next‑day realised vol
        """
        if len(recent_window) < self.seq_len:
            raise StockTransformerException(
                f"recent_window must have at least {self.seq_len} rows; got {len(recent_window)}."
            )
        # Use the most recent seq_len days
        input_df = recent_window.iloc[-self.seq_len:]
        x = self._prepare_input(input_df)

        # Last known close price (raw column needed for price reconstruction)
        last_close = recent_window["Close"].iloc[-1]

        with torch.no_grad():
            mu_logret, logvar, dir_logits, vol_pred = self.model(x)

        # Convert log‑return to price
        mu = mu_logret.item()
        sigma_logret = torch.exp(0.5 * torch.clamp(logvar, -10, 5)).item()
        next_price = last_close * np.exp(mu)
        price_sigma = last_close * sigma_logret   # approximate

        # Direction
        dir_probs = F.softmax(dir_logits, dim=-1).cpu().numpy().flatten()
        dir_label = np.argmax(dir_probs)
        dir_names = ["Down", "Flat", "Up"]

        return {
            "next_price": next_price,
            "price_uncertainty": price_sigma,
            "direction": dir_names[dir_label],
            "direction_probs": {name: float(p) for name, p in zip(dir_names, dir_probs)},
            "volatility": vol_pred.item()
        }
    
    def autoregressive_predict(self, ticker: str,
                               initial_window: pd.DataFrame,
                               horizon: int = 21) -> Dict:
        """
        Generate multi‑step price forecasts autoregressively.
        After each step, the predicted log‑return is injected into the feature
        vector (updating Return_1d and Log_Ret_1d) and fed back as the next input.
        Parameters
        ticker : str
        initial_window : pd.DataFrame
            Last `seq_len` days of known data (must include raw 'Close').
        horizon : int
            Number of days to forecast.
        Returns
        dict with:
            - prices : list of predicted prices (length = horizon)
            - uncertainties : list of price sigmas (length = horizon)
            - directions : list of predicted direction strings
        """
        if len(initial_window) < self.seq_len:
            raise StockTransformerException("Not enough data for the initial window.")
        window_df = initial_window.iloc[-self.seq_len:].copy()

        last_close = window_df["Close"].iloc[-1]
        prices = []
        uncertainties = []
        directions = []

        # Locate the return columns in the feature list
        return1d_idx = self.feature_cols.index("Return_1d") if "Return_1d" in self.feature_cols else None
        logret_idx = self.feature_cols.index("Log_Ret_1d") if "Log_Ret_1d" in self.feature_cols else None

        model_input = self._prepare_input(window_df)

        for step in range(horizon):
            with torch.no_grad():
                mu_logret, logvar, dir_logits, _ = self.model(model_input)

            mu = mu_logret.item()
            sigma_logret = torch.exp(0.5 * torch.clamp(logvar, -10, 5)).item()
            next_price = last_close * np.exp(mu)
            price_sigma = last_close * sigma_logret

            prices.append(next_price)
            uncertainties.append(price_sigma)

            dir_probs = F.softmax(dir_logits, dim=-1).cpu().numpy().flatten()
            dir_label = np.argmax(dir_probs)
            dir_names = ["Down", "Flat", "Up"]
            directions.append(dir_names[dir_label])

            # Shift context: drop first day, append synthetic last day
            last_features = model_input[0, -1, :].clone()
            if logret_idx is not None:
                last_features[logret_idx] = mu
            if return1d_idx is not None:
                last_features[return1d_idx] = np.exp(mu) - 1.0
            next_input = torch.cat([model_input[:, 1:, :],
                                    last_features.unsqueeze(0).unsqueeze(0)], dim=1)
            model_input = next_input
            last_close = next_price

        return {
            "prices": prices,
            "uncertainties": uncertainties,
            "directions": directions
        }

def evaluate_on_test_set(self, ticker: str) -> Dict:
        """
        Run single‑step predictions over the entire test set (last 10% of data)
        and return regression / directional metrics.
        """
        df = self._load_ticker_data(ticker)
        split_idx = int(len(df) * 0.9)
        test_df = df.iloc[split_idx:].copy()
        if len(test_df) <= self.seq_len:
            raise StockTransformerException("Test set too small.")

        close_raw = test_df["Close"].values
        pred_prices = []
        true_prices = []
        all_dir_probs = []
        true_dirs = []

        for i in range(len(test_df) - self.seq_len):
            window = test_df.iloc[i : i + self.seq_len]
            x = self._prepare_input(window)
            last_close = window["Close"].iloc[-1]
            true_next_close = close_raw[i + self.seq_len]

            with torch.no_grad():
                mu_logret, _, dir_logits, _ = self.model(x)

            pred_price = last_close * np.exp(mu_logret.item())
            pred_prices.append(pred_price)
            true_prices.append(true_next_close)

            dir_probs = F.softmax(dir_logits, dim=-1).cpu().numpy().flatten()
            all_dir_probs.append(dir_probs)
            true_dirs.append(test_df["Target_Direction"].iloc[i + self.seq_len])

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

        # Optional plot
        plot_predictions_vs_true(
            dates=test_df.index[self.seq_len:],
            true=true_prices,
            pred=pred_prices,
            title=f"{ticker} – Single‑step Predictions"
        )
        return metrics



# convenience function for CLI / notebook
def run_inference(config_path: str = "configs/config.yaml",
                  checkpoint_path: Optional[str] = None,
                  ticker: Optional[str] = None) -> None:
    """
    High‑level function that creates an InferenceEngine and prints results
    for both single‑step and multi‑step forecasting on the given ticker.
    """
    engine = InferenceEngine(config_path, checkpoint_path)
    ticker = ticker or engine.cfg["dataset"]["fine_tune_ticker"]

    # Load full data to get a recent window (last seq_len days of training)
    df = engine._load_ticker_data(ticker)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    recent_window = train_df.iloc[-engine.seq_len:]

    # single step
    result = engine.single_step_predict(ticker, recent_window)
    print("Single‑step prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # multi-step
    multi = engine.autoregressive_predict(ticker, recent_window,
                                          horizon=engine.cfg["inference"]["forecast_horizon"])
    
    print(f"\nMulti‑step forecast ({len(multi['prices'])} days):")
    for i, p in enumerate(multi["prices"]):
        print(f" Day {i+1}: price={p:.2f} ± {multi['uncertainties'][i]:.2f}, "
              f"direction={multi['directions'][i]}")
    plot_multistep_forecast(multi["prices"], multi["uncertainties"],
                            title=f"{ticker} – Autoregressive Forecast")

    # full test evaluation
    print("\nEvaluating on test set...")
    metrics = engine.evaluate_on_test_set(ticker)
    for m, v in metrics.items():
        print(f"  {m}: {v:.3f}" if isinstance(v, float) else f"  {m}: {v}")


if __name__ == "__main__":
    run_inference()