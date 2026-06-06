import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from arch import arch_model
from src.exceptions import StockTransformerException
from src.logger import logger
from src.utils import load_yaml

#creating the indicator helper functions

def compute_rsi(series: pd.Series, period: int=14)->pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 -(100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int=12, slow:int=26, signal: int=9):
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr(df:pd.DataFrame, period:int=14)->pd.Series:
    """Average True Range."""
    high,low, close = df["High"], df["Low"], df["Close"]
    tr1 = high-low
    tr2 = (high-close.shift()).abs()
    tr3 = (low-close.shift()).abs()
    true_range = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def compute_bollinger_band_width(series: pd.Series, period: int=20, num_std:float=2.0)->pd.Series:
    """Relative width of Bollinger Bands (top‑bottom) / middle."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ((sma + num_std * std)-(sma - num_std * std))/sma

def compute_parkinson_volatility(df:pd.DataFrame, period:int=21)->pd.Series:
    """Parkinson historical volatility estimator."""
    h1 = np.log(df["High"]/df["Low"])
    return np.sqrt((h1 ** 2).rolling(period).sum()/(4 * np.log(2) * period))

def compute_stochastic(df:pd.DataFrame, period:int=14)->Tuple[pd.Series,pd.Series]:
    """
    Stochastic Oscillator %K and %D.
    %K = 100 * (Close - Low_min) / (High_max - Low_min)
    %D = 3‑day SMA of %K
    """
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    pct_k = 100 * (df["Close"]-low_min)/(high_max - low_min + 1e-8)
    pct_d = pct_k.rolling(3).mean()
    return pct_k, pct_d

def compute_roc(series: pd.Series, period: int=10)->pd.Series:
    """Rate of Change: (price / price(period ago) - 1) * 100"""
    return (series / series.shift(period)-1) * 100

def compute_obv(df:pd.DataFrame)->pd.Series:
    """On‑Balance Volume: cumulative signed volume."""
    return (np.sign(df["Close"].diff())*df["Volume"]).fillna(0).cumsum()

def fit_garch_proxy(returns: pd.Series, alpha:float=0.06)->pd.Series:
    """EWMA fallback volatility."""
    return returns.ewm(alpha=alpha, adjust=False).std() * np.sqrt(252)

def fit_garch(returns:pd.Series, fallback_alpha:float=0.06)->pd.Series:
    """
    Fit GARCH(1,1) on log‑return series using an expanding window.
    Falls back to EWMA if fitting fails or insufficient data.
    """
    clean = returns.dropna()
    if len(clean) < 500:
        return fit_garch_proxy(returns, fallback_alpha)

    vol = pd.Series(np.nan, index=returns.index)
    min_window = 500
    refit_freq = 252  # re‑estimate every ~1 year

    for end_idx in range(min_window, len(clean) + 1, refit_freq):
        train = clean.iloc[:end_idx]
        try:
            model = arch_model(train, mean='Zero', vol='GARCH', p=1, q=1, dist='normal',rescale=False)
            res = model.fit(disp='off', show_warning=False)
            forecast_horizon = min(refit_freq, len(clean) - end_idx)
            forecast = res.forecast(horizon=forecast_horizon, reindex=False)
            cond_vol = np.sqrt(forecast.variance.values[-1, :])
            forecast_start = clean.index[end_idx]
            forecast_end = clean.index[end_idx + forecast_horizon - 1]
            vol.loc[forecast_start:forecast_end] = cond_vol[:forecast_horizon]
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}. Using EWMA fallback.")
            segment = clean.iloc[end_idx:end_idx + refit_freq]
            if len(segment) > 0:
                vol.loc[segment.index] = fit_garch_proxy(returns.loc[segment.index], fallback_alpha)

    # Fill remaining NaN (first min_window days) with EWMA
    nan_mask = vol.isna()
    if nan_mask.any():
        vol.loc[nan_mask] = fit_garch_proxy(returns.loc[nan_mask], fallback_alpha)

    return vol * np.sqrt(252)  # annualise

#finally the collate function for all the feature engnineering functions
def engineer_features(
        df:pd.DataFrame,
        spy_df: pd.DataFrame=None,
        vix_df: pd.DataFrame=None,
        cfg_features:dict=None
)->pd.DataFrame:
    """
    Build the full feature set from raw OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data for a single stock (index=DateTime).
    spy_df : pd.DataFrame, optional
        SPY data (must contain 'Close') for beta computation.
    vix_df : pd.DataFrame, optional
        VIX data (must contain 'VIX') to be merged.
    cfg_features : dict
        Feature parameters from config.yaml.

    Returns
    -------
    pd.DataFrame with all feature and target columns.
    """
    if cfg_features is None:
        cfg_features = {}
    data = df.copy()

    #basic returns from the close price
    data["Return_1d"]=data["Close"].pct_change()
    data["Log_Ret_1d"] = np.log(data["Close"]/data["Close"].shift(1))
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Volume_Change"] = data["Volume"].pct_change()
    data["High_Low_Pct"] = (data["High"] - data["Low"]) / data["Close"]
    data["Close_Open_Pct"] = (data["Close"] - data["Open"]) / data["Open"]

    #realised volatility
    vol_windows = cfg_features.get("vol_windows", [5,21,63])
    for win in vol_windows:
        data[f"Real_vol_{win}d"] = data["Log_Ret_1d"].rolling(win).std() * np.sqrt(252)
    
    #garch(1,1)-conditional volatility
    if cfg_features.get("use_garch",True):
        data["GARCH_vol"] = fit_garch(data["Log_Ret_1d"], fallback_alpha=cfg_features.get("garch_ewma_alpha", 0.06))
    else:
        data["GARCH_vol"] = fit_garch_proxy(data["Log_Ret_1d"], alpha=cfg_features.get("garch_ewma_alpha",0.06))

    #rsi for multiple periods
    rsi_periods = cfg_features.get("rsi_periods",[14])
    for per in rsi_periods:
        data[f"RSI_{per}"] = compute_rsi(data["Close"],per)
    
    #macd variants
    macd_params = cfg_features.get("macd_params", {"standard":[12,26,9]})
    for name, (f,s,sig) in macd_params.items():
        macd_line, signal_line, hist = compute_macd(data["Close"], fast=f, slow=s, signal=sig)
        data[f"MACD_{name}"] = macd_line
        data[f"MACD_signal_{name}"] = signal_line
        data[f"MACD_hist_{name}"] = hist

    #atr
    data["ATR_14"] = compute_atr(data, cfg_features.get("atr_period", 14))

    #bollinger band width
    bb_periods = cfg_features.get("bb_periods",[20])
    for per in bb_periods:
        data[f"BB_width_{per}"] = compute_bollinger_band_width(data["Close"],period=per)

    #parkinson volatility
    data["Parkinson_vol_21"] = compute_parkinson_volatility(data, cfg_features.get("parkinson_period",21))

    #stochastic oscillator
    stoch_period = cfg_features.get("stoch_period",None)
    if stoch_period:
        stoch_k, stoch_d = compute_stochastic(data, stoch_period)
        data["Stoch_%K"] = stoch_k
        data["Stoch_%D"] = stoch_d

    #rate of change (roc)
    data["ROC_10"] = compute_roc(data["Close"], 10)

    #on-balance volume
    data["OBV"] = compute_obv(data)

    #moving averages and crosses
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["SMA_200"] = data["Close"].rolling(200).mean()
    data["SMA_20_50_cross"] = (data["SMA_20"] - data["SMA_50"]) / data["Close"]
    data["SMA_50_200_cross"] = (data["SMA_50"] - data["SMA_200"]) / data["Close"]

    #temporal features
    data["DayOfWeek"] = data.index.dayofweek
    data["Month"] = data.index.month
    data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 5)
    data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 5)

    #systematic risk: rolling beta to SPY
    if spy_df is not None:
        spy_ret = spy_df["Close"].pct_change()
        combined = pd.concat([data["Return_1d"], spy_ret], axis=1)
        combined.columns = ["stock_ret", "spy_ret"]
        combined.dropna(inplace=True)
        beta_window = cfg_features.get("beta_window", 63)
        rolling_cov = combined["stock_ret"].rolling(beta_window).cov(combined["spy_ret"])
        rolling_var = combined["spy_ret"].rolling(beta_window).var()
        beta = rolling_cov / rolling_var
        data["Beta_63"] = beta
    else:
        data["Beta_63"] = np.nan

    #VIX
    if vix_df is not None:
        #build a clean daily vix series with a sample datetimeindex
        vix_series = vix_df.copy()

        if isinstance(vix_series.index, pd.MultiIndex):
            vix_series.index = vix_series.index.get_level_values(0)
        vix_series.index = pd.to_datetime(vix_series.index)

        if "VIX" in vix_series.columns:
            vix_series = vix_series["VIX"]
        else:
            vix_series = vix_series.iloc[:,-1]
        vix_series = vix_series.sort_index()
        data["VIX"] = data.index.to_series().map(vix_series)
    else:
        data["VIX"] = np.nan
    #target columns
    data["Target_Log_Ret_Next"] = data["Log_Ret_1d"].shift(-1)
    adaptive_factor = cfg_features.get("direction_adaptive_factor",0.7)
    threshold = adaptive_factor * data["Real_vol_21d"]  # 21-day realised vol must exist
    next_ret = data["Return_1d"].shift(-1)
    data["Target_Direction"] = np.where(next_ret > threshold, 2,
                                        np.where(next_ret < -threshold, 0, 1))
    data["Target_Vol_Next"] = data["Real_vol_21d"].shift(-1)

    #drop rows with NaN
    data.dropna(inplace=True)
    return data

def run_feature_pipeline(config_path: str = "configs/config.yaml") -> None:
    """
    Load raw data from data/raw, engineer features for every stock,
    and save to data/processed.
    """
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    external_dir = Path(data_cfg["external_dir"])

    # load SPY for beta calculation
    spy_path = raw_dir / "SPY.US.parquet"
    spy_df = pd.read_parquet(spy_path) if spy_path.exists() else None
    if spy_df is None:
        logger.warning("SPY.US not found; rolling beta will be NaN.")

    # load VIX
    vix_path = external_dir / "VIX.parquet"
    vix_df  =None
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        if isinstance(vix_df.index, pd.MultiIndex):
            vix_df.index = vix_df.index.get_level_values(0)
        vix_df.index = pd.to_datetime(vix_df.index)
        logger.info("VIX Loaded and index cleaned")
    else:
        logger.warning("VIX data not found; VIX features will be NaN")

    stock_files = list(raw_dir.glob("*.parquet"))
    logger.info(f"Processing {len(stock_files)} stock files...")

    for pq_path in tqdm(stock_files, desc="Feature Engineering"):
        ticker = pq_path.stem
        out_path = processed_dir / f"{ticker}.parquet"
        if out_path.exists():
            logger.debug(f"{ticker} already processed, skipping.")
            continue
        try:
            df = pd.read_parquet(pq_path)
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required):
                logger.warning(f"{ticker} missing required columns, skipping.")
                continue
            feat_df = engineer_features(df, spy_df=spy_df, vix_df=vix_df, cfg_features=feat_cfg)
            feat_df.to_parquet(out_path)
            logger.debug(f"{ticker} processed and saved.")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            raise StockTransformerException(f"Feature engineering failed for {ticker}.", str(e))

    logger.info(f"All processed features saved to {processed_dir}")

if __name__ == "__main__":
    run_feature_pipeline()