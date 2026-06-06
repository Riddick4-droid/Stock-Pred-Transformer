import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering.features import (
    compute_atr,
    compute_bollinger_band_width,
    compute_macd,
    compute_obv,
    compute_parkinson_volatility,
    compute_roc,
    compute_rsi,
    compute_stochastic,
    engineer_features,
    fit_garch_proxy,
)


def make_sample_ohlcv(days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    close = np.linspace(100.0, 120.0, days)
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = np.linspace(1_000, 1_500, days).astype(int)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


class TestFeatureEngineering(unittest.TestCase):
    def test_compute_rsi(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        rsi = compute_rsi(series, period=3)
        self.assertEqual(len(rsi), len(series))
        self.assertTrue((rsi.dropna() >= 0).all())
        self.assertTrue((rsi.dropna() <= 100).all())

    def test_compute_macd(self):
        series = pd.Series(np.arange(1, 21), dtype=float)
        macd_line, signal_line, hist = compute_macd(series, fast=3, slow=6, signal=2)
        self.assertEqual(len(macd_line), len(series))
        self.assertEqual(len(signal_line), len(series))
        self.assertEqual(len(hist), len(series))
        np.testing.assert_allclose(hist, macd_line - signal_line, rtol=1e-6)

    def test_compute_atr(self):
        df = make_sample_ohlcv(10)
        atr = compute_atr(df, period=3)
        self.assertEqual(len(atr), len(df))
        self.assertTrue((atr.dropna() >= 0).all())

    def test_compute_bollinger_band_width(self):
        series = pd.Series(np.linspace(1, 10, 20), dtype=float)
        width = compute_bollinger_band_width(series, period=5, num_std=2.0)
        self.assertEqual(len(width), len(series))
        self.assertTrue((width.dropna() >= 0).all())

    def test_compute_parkinson_volatility(self):
        df = make_sample_ohlcv(22)
        volatility = compute_parkinson_volatility(df, period=10)
        self.assertEqual(len(volatility), len(df))
        self.assertTrue((volatility.dropna() >= 0).all())

    def test_compute_stochastic(self):
        df = make_sample_ohlcv(20)
        pct_k, pct_d = compute_stochastic(df, period=5)
        self.assertEqual(len(pct_k), len(df))
        self.assertEqual(len(pct_d), len(df))
        self.assertTrue((pct_k.dropna() >= 0).all())
        self.assertTrue((pct_k.dropna() <= 100).all())

    def test_compute_roc(self):
        series = pd.Series([100.0, 105.0, 110.0, 121.0])
        roc = compute_roc(series, period=1)
        expected = pd.Series([np.nan, 5.0, 4.761904761904762, 10.0])
        np.testing.assert_allclose(roc, expected, rtol=1e-6, equal_nan=True)

    def test_compute_obv(self):
        df = make_sample_ohlcv(10)
        obv = compute_obv(df)
        self.assertEqual(len(obv), len(df))
        self.assertEqual(obv.iloc[0], 0)

    def test_fit_garch_proxy(self):
        returns = pd.Series(np.random.normal(0, 0.01, size=30))
        vol = fit_garch_proxy(returns, alpha=0.1)
        self.assertEqual(len(vol), len(returns))
        self.assertTrue((vol.dropna() >= 0).all())

    def test_engineer_features_basic(self):
        df = make_sample_ohlcv(80)
        spy_df = df.copy()
        vix_df = pd.DataFrame({"VIX": np.linspace(15.0, 20.0, len(df))}, index=df.index)
        cfg = {
            "use_garch": False,
            "garch_ewma_alpha": 0.06,
            "vol_windows": [5, 21],
            "rsi_periods": [5],
            "macd_params": {"standard": [2, 4, 3]},
            "bb_periods": [5],
            "stoch_period": None,
            "beta_window": 5,
            "direction_adaptive_factor": 0.7,
        }
        features = engineer_features(df, spy_df=spy_df, vix_df=vix_df, cfg_features=cfg)
        expected_columns = [
            "Return_1d",
            "Log_Ret_1d",
            "Return_5d",
            "Volume_Change",
            "High_Low_Pct",
            "Close_Open_Pct",
            "Real_vol_5d",
            "Real_vol_21d",
            "GARCH_vol",
            "RSI_5",
            "MACD_standard",
            "MACD_signal_standard",
            "MACD_hist_standard",
            "ATR_14",
            "BB_width_5",
            "Parkinson_vol_21",
            "ROC_10",
            "OBV",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "SMA_20_50_cross",
            "SMA_50_200_cross",
            "DayOfWeek",
            "Month",
            "DayOfWeek_sin",
            "DayOfWeek_cos",
            "Beta_63",
            "VIX",
            "Target_Log_Ret_Next",
            "Target_Direction",
            "Target_Vol_Next",
        ]
        for col in expected_columns:
            self.assertIn(col, features.columns)
        self.assertFalse(features.isna().any().any())
        self.assertTrue((features["VIX"] >= 15.0).all())


if __name__ == "__main__":
    unittest.main()
