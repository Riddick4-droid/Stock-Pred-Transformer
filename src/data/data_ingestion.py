import pandas as pd
import kagglehub
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

from src.exceptions import StockTransformerException
from src.logger import logger
from src.utils import load_yaml

def load_config(config_path:str = "configs/config.yaml")->dict:
    return load_yaml(config_path)

def download_kaggle_dataset(dataset: str)->Path:
    try:
        path = kagglehub.dataset_download(dataset)
        logger.info(f"kaggle dataset downloaded to {path}")
        return Path(path)
    except Exception as e:
        raise StockTransformerException("Failed to download kaggle dataset", str(e))
    
def find_stocks_folder(base_path: Path)->Path:
    """Locate the 'Stocks' subfolder inside the downloaded dataset."""
    stocks_folder = base_path / "Stocks"
    if not stocks_folder.exists():
        stocks_folder = base_path /  "Data" / "Stocks"
    if not stocks_folder.exists():
        raise StockTransformerException("stocks folder not found in dataset", str(base_path))
    return stocks_folder

def find_etfs_folder(base_path:Path)->Path:
    """Locate the 'ETFs' subfolder inside the downloaded dataset."""
    etfs_folder = base_path / "ETFs"
    if not etfs_folder.exists():
        etfs_folder  = base_path / "Data" / "ETFs"
    if not etfs_folder.exists():
        raise StockTransformerException("ETFs folder not found in dataset", str(base_path))
    return etfs_folder

def convert_file_to_parquet(src_path: Path, dst_dir: Path)->None:
    """
    Read a .txt file (CSV) and save it as a .parquet file.
    Skips if the destination parquet already exists.
    """
    ticker = src_path.stem.upper()
    dst_path = dst_dir / f"{ticker}.parquet"

    if dst_path.exists():
        logger.debug(f"{ticker} already cached, skipping")
        return 
    try:
        df = pd.read_csv(src_path, index_col=0, parse_dates=True)
        df.dropna(how="all", inplace=True)
        df.sort_index(inplace=True)
        df.to_parquet(dst_path)
        logger.debug(f"converted {ticker} to parquet")
    except Exception as e:
        logger.warning(f"skipping {ticker} due to error: {e}")

def convert_all_stocks(stocks_folder:Path, dst_dir:Path)->None:
    """Convert all .txt stock files in the folder to Parquet."""
    txt_files = list(stocks_folder.glob("*.txt"))
    if not txt_files:
        raise StockTransformerException("No .txt files found in stocks folder")
    logger.info(f"Found {len(txt_files)} stocks files. Converting...")
    for txt_path in tqdm(txt_files, desc="Stocks -> Parquet"):
        convert_file_to_parquet(txt_path, dst_dir)
    logger.info("stocks conversion completed")

def extract_etfs(etfs_folder: Path, dst_dir: Path, tickers: list)->None:
    """
    Extract specific ETF .txt files and convert them to Parquet.
    """
    for ticker in tickers:
        src_path = etfs_folder /  f"{ticker.lower()}.txt"
        if not src_path.exists():
            logger.warning(f"{ticker} not found in ETFs folder")
            continue
        convert_file_to_parquet(src_path, dst_dir)

def validate_parquet_files(dst_dir:Path, min_rows: int, required_cols:list)->None:
    """
    Remove Parquet files that don't meet quality criteria:
    - fewer rows than min_rows
    - missing any required column
    - completely empty
    """
    files = list(dst_dir.glob("*.parquet"))
    valid, removed = 0, 0
    for pq_path in files:
        ticker = pq_path.stem
        try:
            df = pd.read_parquet(pq_path)
            if df.shape[0] < min_rows or not all(c in df.columns for c in required_cols):
                pq_path.unlink()
                removed += 1
                logger.debug(f"removed {ticker} (rows={df.shape[0]}, cols={list(df.columns)})")
                continue
            if df.dropna(how="all").empty:
                pq_path.unlink()
                removed += 1
                continue
            valid += 1
        except Exception:
            pq_path.unlink()
            removed += 1
    logger.info(f"validation complete: {valid} valid, {removed} removed")

def download_vix(vix_ticker:str, output_dir:Path)->None:
    """
    Download VIX data from Yahoo Finance and save as Parquet.
    Idempotent: skips if VIX.parquet already exists.
    """
    vix_path = output_dir / "VIX.parquet"
    if vix_path.exists():
        logger.info("VIX data already cached")
        return
    try:
        vix = yf.download(vix_ticker, period="max", progress=False)
        if vix.empty:
            raise StockTransformerException("No vix data returned")
        vix = vix[["Close"]].rename(columns={"Close":"VIX"})
        vix.to_parquet(vix_path)
        logger.info(f"VIX data saved to {vix_path}")
    except Exception as e:
        raise StockTransformerException("Failed to download VIX", str(e))
    
def run_data_digestion(config_path: str="configs/config.yaml")->None:
    """
    Complete data ingestion pipeline driven by config.yaml.
    """
    logger.info("=== Data Ingestion Started ===")
    cfg = load_config(config_path=config_path)
    data_cfg = cfg["data"]

    raw_dir = Path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    external_dir = Path(data_cfg["external_dir"])
    external_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = download_kaggle_dataset(data_cfg["kaggle_dataset"])

    stocks_folder = find_stocks_folder(dataset_path)
    convert_all_stocks(stocks_folder, raw_dir)

    etfs_folder = find_etfs_folder(dataset_path)
    extract_etfs(etfs_folder, raw_dir, data_cfg["etf_tickers"])

    validate_parquet_files(raw_dir, data_cfg["min_rows"], data_cfg["required_columns"])

    download_vix(data_cfg["vix_ticker"], external_dir)

    logger.info("=== Data Ingestion Completed ===")

if __name__ == "__main__":
    run_data_digestion()