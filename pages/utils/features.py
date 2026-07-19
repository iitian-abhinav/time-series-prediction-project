"""
Feature engineering utilities shared by the gradient boosting models
(XGBoost / LightGBM). Builds lag features, rolling statistics and
common technical indicators (RSI, MACD, Bollinger Bands) from a
single 'Close' price series.
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta

LAG_COUNT = 10
ROLL_WINDOWS = (7, 14, 30)


def build_features(close: pd.Series) -> pd.DataFrame:
    """Build a feature matrix from a Close-price series.

    Returns a DataFrame indexed like `close` containing lag features,
    rolling mean/std, and RSI / MACD / Bollinger Band indicators.
    Rows with NaNs (from warm-up windows) are dropped.
    """
    close = close.astype(float)
    df = pd.DataFrame(index=close.index)
    df['Close'] = close

    # Lag features
    for lag in range(1, LAG_COUNT + 1):
        df[f'lag_{lag}'] = close.shift(lag)

    # Rolling statistics
    for w in ROLL_WINDOWS:
        df[f'roll_mean_{w}'] = close.rolling(w).mean()
        df[f'roll_std_{w}'] = close.rolling(w).std()

    # Technical indicators
    rsi = ta.rsi(close, length=14)
    df['RSI_14'] = rsi

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_hist'] = macd.iloc[:, 1]
        df['MACD_signal'] = macd.iloc[:, 2]

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        df['BB_lower'] = bb.iloc[:, 0]
        df['BB_mid'] = bb.iloc[:, 1]
        df['BB_upper'] = bb.iloc[:, 2]

    # Daily return, as an extra momentum signal
    df['return_1d'] = close.pct_change()

    return df


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ('Close', 'target')]


def make_supervised(close: pd.Series):
    """Build (X, y) for next-day-close regression, NaN rows dropped."""
    feat = build_features(close)
    feat['target'] = feat['Close'].shift(-1)
    feat = feat.dropna()
    X = feat[feature_columns(feat)]
    y = feat['target']
    return X, y, feat
