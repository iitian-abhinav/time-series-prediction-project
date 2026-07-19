"""
Volatility forecasting with GARCH / EGARCH (via the `arch` package).
Price-only forecasts ignore risk; pairing a price forecast with a
volatility forecast is the more finance-relevant deliverable.
"""
import numpy as np
import pandas as pd
from arch import arch_model


def compute_returns(close: pd.Series) -> pd.Series:
    """Percentage daily returns, scaled to roughly O(1) for numerical
    stability, as is conventional for GARCH fitting."""
    returns = 100 * close.pct_change().dropna()
    return returns


def fit_volatility_model(close: pd.Series, model_type: str = 'GARCH'):
    """model_type: 'GARCH' or 'EGARCH'."""
    returns = compute_returns(close)
    if model_type.upper() == 'EGARCH':
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
    else:
        am = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    return res, returns


def forecast_volatility(close: pd.Series, model_type: str = 'GARCH', horizon: int = 30) -> pd.DataFrame:
    """Returns a DataFrame with forecast daily volatility (% std dev)
    and annualized volatility for the next `horizon` trading days."""
    res, returns = fit_volatility_model(close, model_type)
    # EGARCH has no closed-form multi-step-ahead variance forecast,
    # so it needs simulation-based forecasting; GARCH can use the
    # faster analytic method.
    method = 'simulation' if model_type.upper() == 'EGARCH' else 'analytic'
    f = res.forecast(horizon=horizon, reindex=False, method=method)
    variance = f.variance.values[-1]  # forecasted daily variance (in % units)
    daily_vol = np.sqrt(variance)
    annualized_vol = daily_vol * np.sqrt(252)

    last_date = close.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    forecast_index = pd.date_range(start=last_date + pd.offsets.BDay(), periods=horizon, freq='B')

    return pd.DataFrame({
        'Daily Volatility (%)': daily_vol,
        'Annualized Volatility (%)': annualized_vol,
    }, index=forecast_index)


def realized_volatility(close: pd.Series, window: int = 30) -> pd.Series:
    """Rolling realized (historical) daily volatility, % units, for
    plotting alongside the GARCH forecast."""
    returns = compute_returns(close)
    return returns.rolling(window).std()
