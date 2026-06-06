import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Any

def set_seed(seed:int=42)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def get_device()->torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") #for apple users
    else:
        return torch.device("cpu")
    
def check_grad_norm(model:nn.Module, norm_type:float=2.0)->float:
    """Compute and return the total gradient norm (useful for logging)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(norm_type).item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def check_nan_in_gradients(model:nn.Module)->bool:
    """Return True if any parameter gradient contains NaN."""
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient detected in {name}")
            return True
    return False

class EarlyStopping:
    """
    Stop training when validation loss doesn't improve for a given patience.
    """
    def __init__(self, patience:int=5, min_delta:float=0.0): #small min_delta for strictness
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop= False
    def __call__(self, val_loss:float)->bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14,6)

def plot_training_curves(train_losses: List[float], val_losses: List[float], title:str = "Training Curves",
                         save_path: Optional[str] = None) -> None:
    """Plot training and validation loss over epochs."""
    plt.figure()
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_predictions_vs_true(dates, true: np.ndarray, pred:np.ndarray, uncertainty: Optional[np.ndarray]=None, title: str = "Predictions vs True",
                             save_path: Optional[str]=None)->None:
    """Plot ground truth, predictions, and optional ±2σ uncertainty band."""
    plt.figure()
    plt.plot(dates, true, lw=1, label="true", color="blue")
    plt.plot(dates, pred, lw=1, label="pred", color="red")
    if uncertainty is not None:
        plt.fill_between(dates, 
                         pred-2 * uncertainty,
                         pred + 2 * uncertainty,
                         alpha=0.2, color="red", label="±2σ")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpt=150)
    plt.show()

def plot_multistep_forecast(pred_prices:List[float],
                            uncertainty: Optional[List[float]] = None,
                            title: str = "Multi-step Forecast",
                            save_path: Optional[str]=None)->None:
    """Plot multi‑step autoregressive forecast with optional uncertainty.
    shows how uncertain the model gets overtime when it has to use autoregression from its one predictions 
    to predict future prices
    """
    plt.figure()
    steps = range(len(pred_prices))
    plt.plot(steps, pred_prices, marker="o", lw=1, label="Forcast")
    if uncertainty is not None:
        lower_bound = np.array(pred_prices)-2 * np.array(uncertainty)
        upper_bound = np.array(pred_prices)+2 * np.array(uncertainty)
    plt.xlabel("Days Ahead")
    plt.ylabel("Price")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def view_dataframe(df:pd.DataFrame,
                   rows: int = 10,
                   output_file:Optional[str]=None)->None:
    """Print the first `rows` of a DataFrame and optionally export to CSV / Excel."""
    print(df.head(rows))
    if output_file:
        path = Path(output_file)
        if path.suffix == ".csv":
            df.to_csv(path, index=True)
        elif path.suffix in (".xslx", ".xls"):
            df.to_excel(path, index=True)
        else:
            df.to_csv(path.with_suffix(".csv"), index=True)
        print(f"Dataframe saved to {path}")


def inspect_ticker_features(ticker:str, 
                            processed_dir: str="data/processed", 
                            output_file:Optional[str]=None)->None:
    """Load a single ticker's processed feature file and display it."""
    path = Path(processed_dir) / f"{ticker}.parquet"
    if not path.exists():
        print(f"File mot found: {path}")
        return
    df = pd.read_parquet(path)
    view_dataframe(df,rows=10, output_file=output_file)

def load_yaml(path:str)->Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save_json(data: Dict[str,Any], path:str)->None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path: str)->Dict[str,Any]:
    with open(path, "r") as f:
        return json.load(f)