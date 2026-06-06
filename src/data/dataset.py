import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Optional, Tuple, Dict

from src.exceptions import StockTransformerException
from src.logger import logger
from src.utils import load_yaml


#data loading helper
def load_all_data(processed_dir: str, tickers:Optional[List[str]]=None)->Dict[str, np.ndarray]:
    """
    Read all parquet files from processed_dir into a dictionary {ticker: array}.
    If tickers is provided, only those tickers are loaded.
    """
    proc = Path(processed_dir)
    files = list(proc.glob("*.parquet"))

    if not files:
        raise StockTransformerException(f"No processed files in {processed_dir}")
    
    data = {}
    for f in files:
        ticker = f.stem
        if tickers and ticker not in tickers:
            continue
        df = pd.read_parquet(f)
        data[ticker] = df.values.astype(np.float32)
    logger.info(f"Loaded {len(data)} tickers from {processed_dir}")
    return data

#scaler
def fit_and_save_scaler(
        processed_dir: str,
        feature_cols:List[str],
        scaler_path: str = "checkpoints/scaler.joblib",
        sample_size:int=2_000_000,
)->StandardScaler:
    """Fit a StandardScaler on a sample of all processed stocks and save it."""
    logger.info("fitting standardscaler...")
    scaler = StandardScaler()
    combined = []
    proc = Path(processed_dir)
    total = 0
    for f in proc.glob("*.parquet"):
        if total >= sample_size:
            break
        df = pd.read_parquet(f)[feature_cols]
        combined.append(df.values)
        total += len(df)
    if not combined:
        raise StockTransformerException("No data to fit scaler")
    data = np.concatenate(combined, axis=0)[:sample_size]
    scaler.fit(data)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler":scaler, "feature_cols":feature_cols},scaler_path)
    logger.info(f"Scaler + feature cols saved to {scaler_path} (fitted on {len(data)} samples)")
    return scaler

def load_scaler(scaler_path:str="checkpoints/scaler.joblib")->Tuple[StandardScaler,List[str]]:
    """Loads a saved scaler and the feature columns used in it"""
    if not Path(scaler_path).exists():
        raise StockTransformerException(f"Scaler file not found: {scaler_path}")
    bundle = joblib.load(scaler_path)
    return bundle["scaler"], bundle["feature_cols"]

class PreTrainingDataset(Dataset):
    """
    Dataset for pre‑training on many stocks.

    Each sample = (features_window, next_log_return).
    """
    def __init__(self, processed_dir:str="data/processed",
                 tickers:Optional[List[str]]=None,seq_len:int=60, 
                 feature_cols: Optional[List[str]]=None,
                 scaler: Optional[StandardScaler]=None, 
                 target_col: str="Target_Log_Ret_Next"):
        self.seq_len = seq_len
        self.target_col = target_col
        self.scaler = scaler

        self.data = load_all_data(processed_dir=processed_dir, tickers=tickers)
        if not self.data:
            raise StockTransformerException("No ticker data loaded")
        
        if feature_cols is None:
            sample_file = next(Path(processed_dir).glob("*.parquet"))
            sample_df = pd.read_parquet(sample_file)
            exclude = [self.target_col, "Target_Direction", "Target_Vol_Next", "Close"]
            feature_cols = [c for c in sample_df.columns if c not in exclude]
        self.feature_cols = feature_cols

        #map column names to indices in the numpy arrays
        first_ticker = list(self.data.keys())[0]
        sample_df = pd.read_parquet(Path(processed_dir)/f"{first_ticker}.parquet")
        all_cols = sample_df.columns.tolist()
        self.feature_indices = [all_cols.index(c) for c in self.feature_cols]
        self.target_idx = all_cols.index(self.target_col)

        #build index map for random access
        self.index_map = []
        for ticker, arr in self.data.items():
            if len(arr)>seq_len:
                self.index_map.append((ticker, len(arr)-seq_len))
        if not self.index_map:
            raise StockTransformerException("No ticker has enough data for the given seq_len")
        logger.info(f"Pretrainingdataset: {len(self.data)} tickers, {len(self.feature_cols)} features")

    def __len__(self)->int:
        return sum(m for _,m in self.index_map)
    
    def __getitem__(self, idx:int)->Tuple[torch.Tensor, torch.Tensor]:
        #locate ticker and start index
        cum = 0
        for ticker, max_start in self.index_map:
            if idx < cum + max_start:
                start = idx - cum
                break
            cum += max_start

        else:
            ticker, max_start = np.random.choice(list(self.data.keys()))
            start = np.random.randint(0, len(self.data[ticker])-self.seq_len)

        arr = self.data[ticker]
        x = arr[start:start + self.seq_len, self.feature_indices].copy()
        if self.scaler is not None:
            x = self.scaler.transform(x)
        y = arr[start + self.seq_len, self.target_idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)
    
#finetune dataset
class FineTuneDataset(Dataset):
    """
    Dataset for fine‑tuning on a single stock.

    Returns: (features, log_ret_target, direction_target, volatility_target)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        target_price_col: str = "Target_Log_Ret_Next",
        target_dir_col: str = "Target_Direction",
        target_vol_col: str = "Target_Vol_Next"
    ):
        self.seq_len = seq_len
        self.scaler = scaler

        if feature_cols is None:
            exclude = [target_price_col, target_dir_col, target_vol_col, "Close"]
            feature_cols = [c for c in df.columns if c not in exclude]
        self.feature_cols = feature_cols

        self.features = df[self.feature_cols].values.astype(np.float32)
        self.price_target = df[target_price_col].values.astype(np.float32)
        self.dir_target = df[target_dir_col].values.astype(np.int64)
        self.vol_target = df[target_vol_col].values.astype(np.float32)

        if self.scaler is not None:
            self.features = self.scaler.transform(self.features)

        self.valid_length = len(df) - seq_len
        if self.valid_length <= 0:
            raise StockTransformerException("DataFrame shorter than seq_len.")

    def __len__(self) -> int:
        return self.valid_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(self.price_target[idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.dir_target[idx + self.seq_len], dtype=torch.long),
            torch.tensor(self.vol_target[idx + self.seq_len], dtype=torch.float32)
        )

def create_datasets(config_path: str = "configs/config.yaml",mode: str = "pretrain") -> Tuple[Dataset, Dataset, StandardScaler, List[str]]:
    """
    Load config, fit/load scaler, and return train/val datasets + scaler + feature_cols.

    mode = 'pretrain' or 'finetune'.
    """
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    ds_cfg = cfg["dataset"]

    processed_dir = data_cfg["processed_dir"]
    seq_len = ds_cfg["seq_len"]
    fine_tune_ticker = ds_cfg["fine_tune_ticker"]

    # Load or fit scaler, retrieving the feature columns
    scaler_path = "checkpoints/scaler.joblib"
    if Path(scaler_path).exists():
        scaler, feature_cols = load_scaler(scaler_path)
        logger.info(f"Loaded scaler with {len(feature_cols)} features.")
    else:
        sample_file = next(Path(processed_dir).glob("*.parquet"))
        sample_df = pd.read_parquet(sample_file)
        exclude = ["Target_Log_Ret_Next", "Target_Direction", "Target_Vol_Next", "Close"]
        feature_cols = [c for c in sample_df.columns if c not in exclude]
        scaler = fit_and_save_scaler(processed_dir, feature_cols, scaler_path)

    if mode == "pretrain":
        full_dataset = PreTrainingDataset(
            processed_dir=processed_dir,
            tickers=ds_cfg.get("pre_train_tickers"),
            seq_len=seq_len,
            feature_cols=feature_cols,
            scaler=scaler,
            target_col="Target_Log_Ret_Next"
        )
        val_size = int(len(full_dataset) * ds_cfg["val_split"])
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset, scaler, feature_cols

    elif mode == "finetune":
        ticker_path = Path(processed_dir) / f"{fine_tune_ticker}.parquet"
        df = pd.read_parquet(ticker_path)
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        train_dataset = FineTuneDataset(
            train_df, seq_len=seq_len, feature_cols=feature_cols, scaler=scaler,
            target_price_col="Target_Log_Ret_Next",
            target_dir_col="Target_Direction",
            target_vol_col="Target_Vol_Next"
        )
        val_dataset = FineTuneDataset(
            val_df, seq_len=seq_len, feature_cols=feature_cols, scaler=scaler,
            target_price_col="Target_Log_Ret_Next",
            target_dir_col="Target_Direction",
            target_vol_col="Target_Vol_Next"
        )
        return train_dataset, val_dataset, scaler, feature_cols

    else:
        raise StockTransformerException(f"Unknown mode: {mode}")


#view what the processsed ticker looks like
def inspect_processed_ticker(
    ticker: str,
    processed_dir: str = "data/processed",
    feature_cols: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    rows: int = 10
) -> None:
    from src.utils import view_dataframe
    path = Path(processed_dir) / f"{ticker}.parquet"
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return
    df = pd.read_parquet(path)
    if df.empty:
        logger.warning(f"DataFrame for {ticker} is empty (0 rows). "
                       "The feature pipeline may not have completed successfully.")
        return
    if feature_cols is not None:
        available = [c for c in feature_cols if c in df.columns]
        for tcol in ["Target_Log_Ret_Next", "Target_Direction", "Target_Vol_Next"]:
            if tcol in df.columns:
                available.append(tcol)
        df = df[available]
    view_dataframe(df, rows=rows, output_file=output_file)

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from src.utils import view_dataframe
    processed_dir = "data/processed"
    ticker = "AAPL.US"
    df = pd.read_parquet(Path(processed_dir) / f"{ticker}.parquet")
    view_dataframe(df, rows=20, output_file="aapl_preview.csv")